"""
Twin-Track (multi-round): online planning with tool-state simulation.

This entrypoint mirrors `planner/twin_track.py` in spirit (task-agnostic core),
but delegates the multi-round execution/search logic to a per-task hook:

  `artifacts/input/<task>/planner.json`:
    {
      "multi_round": {
        "hooks": { "execute": "package.module:callable" }
      }
    }

The hook is responsible for:
  - running a multi-round (Check/Do)* loop, executing tools as needed
  - minimizing tool cost / unnecessary side effects
  - monitoring GT-free constraints
  - returning a filled Plan/Phase/Action template dict
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

from dataclasses import dataclass
from typing import Any, List, Protocol, Tuple

from planner.config import ConfigError, load_hook, load_planner_config
from planner.parallel import available_cpu_count


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
OUTPUT_ROOT = ARTIFACTS_ROOT / "output"


# ---------------------------------------------------------------------------
# JSON utilities (strict JSON; no NaN/Infinity)
# ---------------------------------------------------------------------------


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_for_json(v) for v in value]
    return value


def _json_dumps_strict(value: Any) -> str:
    return json.dumps(_sanitize_for_json(value), ensure_ascii=False, allow_nan=False)


# ---------------------------------------------------------------------------
# Generic multi-round UCT/MCTS engine (task-agnostic; adapter-driven)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundAction:
    key: str
    stop: bool = False
    payload: Any = None


class MultiRoundAdapter(Protocol):
    def initial_history(self) -> List[Any]: ...

    def available_actions(self, *, history: List[Any], round_idx: int) -> List[RoundAction]: ...

    def apply(self, *, history: List[Any], round_idx: int, action: RoundAction) -> List[Any]: ...

    def rollout(self, *, history: List[Any], start_round: int, max_rounds: int, rng: random.Random) -> List[Any]: ...

    def score(self, *, history: List[Any]) -> Tuple[float, Mapping[str, object]]: ...

    def render(self, *, history: List[Any], details: Mapping[str, object]) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class MultiRoundMCTSConfig:
    iterations: int = 150
    uct_c: float = 1.4
    seed: Optional[int] = None
    semantic_tolerance: Optional[float] = None
    early_stop: bool = True
    progress: bool = False
    progress_every: int = 25
    dump_tree: bool = False
    dump_tree_max_depth: int = 6
    dump_tree_max_children: int = 25


class _SearchNode:
    def __init__(self, *, history: List[Any], next_round: int, parent: Optional["_SearchNode"], action: Optional[RoundAction]) -> None:
        self.history = history
        self.next_round = next_round
        self.parent = parent
        self.action = action
        self.children: List["_SearchNode"] = []
        self.untried: List[RoundAction] = []
        self.visits = 0
        self.value = 0.0

    def best_child(self, *, uct_c: float, rng) -> "_SearchNode":
        if not self.children:
            return self
        best_score = -float("inf")
        best: Optional[_SearchNode] = None
        for child in self.children:
            if child.visits == 0:
                score = float("inf")
            else:
                exploitation = child.value / child.visits
                exploration = uct_c * math.sqrt(math.log(self.visits + 1.0) / child.visits)
                score = exploitation + exploration
            if score > best_score:
                best_score = score
                best = child
        return best or rng.choice(self.children)


def _effective_max_rounds(template: Mapping[str, object], *, default_max_rounds: int) -> int:
    """
    Per-template override for multi-round planning.

    If template.plan.runs is a positive integer, cap rounds to min(runs, default_max_rounds).
    If runs is 0 or missing/invalid, use default_max_rounds (caller provides the safety cap).
    """

    base = max(1, int(default_max_rounds))
    plan = template.get("plan") if isinstance(template.get("plan"), dict) else {}
    raw = plan.get("runs") if isinstance(plan, dict) else None
    try:
        runs = int(raw) if raw is not None else None
    except Exception:
        runs = None
    if runs is None:
        return base
    if runs <= 0:
        return base
    return max(1, min(base, int(runs)))


def run_multi_round_mcts(
    *,
    adapter: MultiRoundAdapter,
    max_rounds: int,
    config: MultiRoundMCTSConfig,
) -> Tuple[List[Any], Dict[str, object]]:
    max_rounds_i = max(1, int(max_rounds))
    iterations = max(1, int(config.iterations))
    rng = random.Random(config.seed)
    started_at = time.perf_counter()

    root = _SearchNode(history=list(adapter.initial_history()), next_round=1, parent=None, action=None)
    best_reward = -float("inf")
    best_history: List[Any] = []
    best_details: Dict[str, object] = {}
    early_stopped = False
    iterations_used = iterations

    for it in range(1, iterations + 1):
        node = root

        # Selection
        while node.children and not node.untried and node.next_round <= max_rounds_i:
            node = node.best_child(uct_c=float(config.uct_c), rng=rng)

        # Expansion
        if node.next_round <= max_rounds_i and not node.untried:
            node.untried = list(adapter.available_actions(history=list(node.history), round_idx=int(node.next_round)))

        expanded = node
        if node.next_round <= max_rounds_i and node.untried:
            choice = node.untried.pop(0)
            new_history = adapter.apply(history=list(node.history), round_idx=int(node.next_round), action=choice)
            child = _SearchNode(
                history=list(new_history),
                next_round=(node.next_round + 1) if not choice.stop else (max_rounds_i + 1),
                parent=node,
                action=choice,
            )
            node.children.append(child)
            expanded = child

        # Rollout
        rollout_history = adapter.rollout(
            history=list(expanded.history),
            start_round=int(expanded.next_round),
            max_rounds=max_rounds_i,
            rng=rng,
        )
        reward, details_raw = adapter.score(history=list(rollout_history))
        details = dict(details_raw) if isinstance(details_raw, Mapping) else {}

        if reward > best_reward:
            best_reward = float(reward)
            best_history = list(rollout_history)
            best_details = dict(details)

        tol = config.semantic_tolerance
        if tol is None:
            tol_val = None
        else:
            tol_val = float(tol)
        hard_pass = details.get("hard_pass")
        sem = details.get("semantic_rate")
        if (
            config.early_stop
            and tol_val is not None
            and isinstance(hard_pass, bool)
            and hard_pass
            and isinstance(sem, (int, float))
            and float(sem) >= tol_val
        ):
            best_reward = float(reward)
            best_history = list(rollout_history)
            best_details = dict(details)
            early_stopped = True
            iterations_used = it
            break

        # Backprop
        cur: Optional[_SearchNode] = expanded
        while cur is not None:
            cur.visits += 1
            cur.value += float(reward)
            cur = cur.parent

        if config.progress and it % max(1, int(config.progress_every)) == 0:
            hard = best_details.get("hard_pass")
            sem = best_details.get("semantic_rate")
            cost = best_details.get("tool_cost") or best_details.get("total_cost")
            elapsed_s = time.perf_counter() - started_at
            hard_s = "?" if hard is None else ("T" if bool(hard) else "F")
            sem_s = "?" if sem is None else f"{float(sem):.3f}"
            cost_s = "?" if cost is None else f"{float(cost):.3f}"
            print(
                f"[multi_round_mcts] it={it}/{iterations} best_cost={cost_s} best_sem={sem_s} hard={hard_s} elapsed={elapsed_s:.1f}s"
            )

    if not early_stopped:
        iterations_used = iterations

    best_details = dict(best_details)
    best_details["mcts_iterations_used"] = int(iterations_used)
    best_details["mcts_iterations_total"] = int(iterations)
    best_details["mcts_early_stop"] = bool(early_stopped)

    if bool(config.dump_tree):
        max_depth = max(1, int(config.dump_tree_max_depth))
        max_children = max(1, int(config.dump_tree_max_children))

        def _serialize(node: _SearchNode, depth: int) -> Dict[str, object]:
            act = node.action
            out: Dict[str, object] = {
                "action": (act.key if act is not None else None),
                "stop": (bool(act.stop) if act is not None else False),
                "next_round": int(node.next_round),
                "visits": int(node.visits),
                "value": float(node.value),
                "avg_value": float(node.value / node.visits) if node.visits > 0 else 0.0,
                "history_len": int(len(node.history)),
            }
            if depth >= max_depth:
                return out
            children = sorted(node.children, key=lambda c: c.visits, reverse=True)
            out["children"] = [_serialize(c, depth + 1) for c in children[:max_children]]
            return out

        best_details["mcts_tree"] = _serialize(root, 0)
    return best_history, best_details


# Back-compat: some legacy hooks imported the private name.
_run_multi_round_mcts = run_multi_round_mcts


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Twin-Track (multi-round): execute per-task multi-round hook.")
    parser.add_argument("--task", type=str, default="travel", help="Task name (defaults to travel).")
    parser.add_argument("--config", type=Path, default=None, help="Override per-task planner config JSON (planner.json).")
    parser.add_argument("--tree", type=Path, required=True, help="AND–OR tree JSON file (tree_<split>.json).")
    parser.add_argument("--template-id", type=str, default=None, help="Run only a single template_id.")
    parser.add_argument("--out", type=Path, default=None, help="Output JSONL path (filled templates).")
    parser.add_argument("--mcts-tree-out", type=Path, default=None, help="Optional output JSON path for per-template MCTS search trees.")
    parser.add_argument("--enriched-tree-out", type=Path, default=None, help="Optional output JSON path for per-template enriched AND–OR trees.")
    parser.add_argument("--max-rounds", type=int, default=10, help="Max Check/Do rounds per query.")
    parser.add_argument("--workers", type=int, default=0, help="Parallel workers (processes). 0 => all CPUs.")
    # Optional knobs forwarded to the multi_round execute hook (hook may ignore them).
    parser.add_argument("--iterations", type=int, default=0, help="Hook search iterations (0 => hook default).")
    parser.add_argument("--uct-c", type=float, default=0.0, help="Hook UCT exploration constant (0 => hook default).")
    parser.add_argument("--seed", type=int, default=None, help="Hook RNG seed (optional).")
    parser.add_argument("--semantic-tolerance", type=float, default=0.8, help="Hook semantic tolerance (0 => hook default).")
    parser.add_argument(
        "--semantic-shortfall-weight",
        type=float,
        default=0.0,
        help="Hook penalty weight for semantic shortfall (0 => hook default).",
    )
    parser.add_argument("--hard-fail-penalty", type=float, default=0.0, help="Hook hard-fail penalty (0 => hook default).")
    parser.add_argument(
        "--max-do-calls-per-round",
        type=int,
        default=0,
        help="Hook cap for Do tool calls per round (0 => hook default).",
    )
    parser.add_argument(
        "--max-ids-per-action",
        type=int,
        default=0,
        help="Hook cap for ID expansions per action (0 => hook default).",
    )
    parser.add_argument("--hook-progress", action="store_true", help="Enable hook-level progress output.")
    parser.add_argument("--no-progress", action="store_true", help="Disable per-template progress output.")
    return parser.parse_args(argv)


def _iter_entries(payload: Mapping[str, object], template_id: Optional[str]) -> Iterable[tuple[str, Mapping[str, object]]]:
    if template_id:
        entry = payload.get(template_id)
        if isinstance(entry, dict):
            yield template_id, entry
        return
    for key, value in payload.items():
        if isinstance(value, dict):
            yield str(key), value


_WORKER_ADAPTER_FACTORY = None
_WORKER_EXECUTE = None


def _worker_init(task: str, config_path: Optional[str]) -> None:
    global _WORKER_ADAPTER_FACTORY, _WORKER_EXECUTE
    cfg = load_planner_config(task, config_path=(Path(config_path) if config_path else None))
    adapter_spec = cfg.multi_round.adapter_hook
    execute_spec = cfg.multi_round.execute_hook
    if adapter_spec:
        _WORKER_ADAPTER_FACTORY = load_hook(adapter_spec)
        _WORKER_EXECUTE = None
        return
    if execute_spec:
        _WORKER_EXECUTE = load_hook(execute_spec)
        _WORKER_ADAPTER_FACTORY = None
        return
    raise RuntimeError(f"Missing multi_round.hooks.adapter or multi_round.hooks.execute for task '{task}'")


def _worker_run_one(
    template_id: str,
    entry: Mapping[str, object],
    max_rounds: int,
    hook_config: Mapping[str, object],
) -> tuple[str, Dict[str, object], float]:
    started_at = time.perf_counter()
    template = entry.get("template") if isinstance(entry.get("template"), dict) else {}
    effective_max_rounds = _effective_max_rounds(template, default_max_rounds=int(max_rounds))

    adapter_factory = _WORKER_ADAPTER_FACTORY
    execute = _WORKER_EXECUTE

    if adapter_factory is not None:
        try:
            adapter = adapter_factory(tree=dict(entry), config=dict(hook_config))
        except TypeError:
            match_entry = entry.get("match") if isinstance(entry.get("match"), dict) else {}
            adapter = adapter_factory(template=template, match_entry=match_entry, config=dict(hook_config))
        mcts_cfg = MultiRoundMCTSConfig(
            iterations=int(hook_config.get("iterations") or 200),
            uct_c=float(hook_config.get("uct_c") or 1.4),
            seed=(int(hook_config["seed"]) if "seed" in hook_config and hook_config["seed"] is not None else None),
            semantic_tolerance=(
                float(hook_config["semantic_tolerance"])
                if "semantic_tolerance" in hook_config and hook_config["semantic_tolerance"] is not None
                else None
            ),
            progress=bool(hook_config.get("progress", False)),
            dump_tree=bool(hook_config.get("export_mcts_tree", False)),
        )
        best_history, best_details = run_multi_round_mcts(
            adapter=adapter, max_rounds=int(effective_max_rounds), config=mcts_cfg
        )
        reward, score_details = adapter.score(history=best_history)
        merged = dict(best_details)
        merged.update(dict(score_details))
        merged["reward"] = float(reward)
        result = dict(adapter.render(history=best_history, details=merged))
        return template_id, result, (time.perf_counter() - started_at)

    if execute is None:
        raise RuntimeError("Worker missing both adapter and execute hooks.")

    match_entry = entry.get("match") if isinstance(entry.get("match"), dict) else {}
    try:
        result_obj = execute(
            template=template, match_entry=match_entry, max_rounds=int(effective_max_rounds), config=dict(hook_config)
        )
    except TypeError:
        result_obj = execute(template=template, match_entry=match_entry, max_rounds=int(effective_max_rounds))
    if not isinstance(result_obj, dict):
        raise RuntimeError(f"multi_round execute hook returned non-dict for template_id={template_id}")
    return template_id, dict(result_obj), (time.perf_counter() - started_at)


def _extract_progress_metrics(result: Mapping[str, object]) -> Tuple[Optional[bool], Optional[float], Optional[float]]:
    notes = result.get("notes") if isinstance(result.get("notes"), dict) else {}
    # Prefer Work-style key; fall back to any nested notes dict that exposes these fields.
    for key in ("multi_round_mcts", "multi_round", "mcts"):
        blob = notes.get(key) if isinstance(notes.get(key), dict) else None
        if not isinstance(blob, dict):
            continue
        hard = blob.get("hard_pass")
        sem = blob.get("semantic_rate")
        cost = blob.get("tool_cost") if "tool_cost" in blob else blob.get("total_cost")
        return (bool(hard) if isinstance(hard, bool) else None, float(sem) if isinstance(sem, (int, float)) else None, float(cost) if isinstance(cost, (int, float)) else None)
    return None, None, None


def _extract_mcts_meta(result: Mapping[str, object]) -> Tuple[Optional[int], Optional[int], Optional[bool]]:
    notes = result.get("notes") if isinstance(result.get("notes"), dict) else {}
    for key in ("multi_round_mcts", "multi_round", "mcts"):
        blob = notes.get(key) if isinstance(notes.get(key), dict) else None
        if not isinstance(blob, dict):
            continue
        used = blob.get("mcts_iterations_used")
        total = blob.get("mcts_iterations_total")
        early = blob.get("mcts_early_stop")
        used_i = int(used) if isinstance(used, (int, float)) else None
        total_i = int(total) if isinstance(total, (int, float)) else None
        early_b = bool(early) if isinstance(early, bool) else None
        return used_i, total_i, early_b
    return None, None, None


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    task = str(args.task)
    cfg = load_planner_config(task, config_path=args.config)
    adapter_spec = cfg.multi_round.adapter_hook
    execute_spec = cfg.multi_round.execute_hook
    config_label = str(args.config) if args.config is not None else f"artifacts/input/{task}/planner.json"
    if not adapter_spec and not execute_spec:
        raise SystemExit(f"Missing multi_round.hooks.adapter or multi_round.hooks.execute for task '{task}' in {config_label}")
    adapter_factory = None
    execute = None
    if adapter_spec:
        try:
            adapter_factory = load_hook(adapter_spec)
        except ConfigError as exc:
            raise SystemExit(str(exc))
    if execute_spec and adapter_factory is None:
        try:
            execute = load_hook(execute_spec)
        except ConfigError as exc:
            raise SystemExit(str(exc))

    tree_path = args.tree
    if not tree_path.is_absolute():
        tree_path = (PROJECT_ROOT / tree_path).resolve()
    if not tree_path.exists():
        raise SystemExit(f"Tree JSON not found: {tree_path}")

    payload = json.loads(tree_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Tree JSON must be an object mapping template_id -> entry: {tree_path}")

    out_path = args.out
    if out_path is None:
        out_path = (OUTPUT_ROOT / task / f"optimized_{tree_path.stem.split('_')[-1]}.jsonl").resolve()
    if not out_path.is_absolute():
        out_path = (PROJECT_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mcts_tree_out = args.mcts_tree_out
    if mcts_tree_out is not None and not mcts_tree_out.is_absolute():
        mcts_tree_out = (PROJECT_ROOT / mcts_tree_out).resolve()
    enriched_tree_out = args.enriched_tree_out
    if enriched_tree_out is not None and not enriched_tree_out.is_absolute():
        enriched_tree_out = (PROJECT_ROOT / enriched_tree_out).resolve()

    written = 0
    hook_config: Dict[str, object] = {}
    if int(args.iterations) > 0:
        hook_config["iterations"] = int(args.iterations)
    if float(args.uct_c) > 0:
        hook_config["uct_c"] = float(args.uct_c)
    if args.seed is not None:
        hook_config["seed"] = int(args.seed)
    if float(args.semantic_tolerance) > 0:
        hook_config["semantic_tolerance"] = float(args.semantic_tolerance)
    if float(args.semantic_shortfall_weight) > 0:
        hook_config["semantic_shortfall_weight"] = float(args.semantic_shortfall_weight)
    if float(args.hard_fail_penalty) != 0.0:
        hook_config["hard_fail_penalty"] = float(args.hard_fail_penalty)
    if int(args.max_do_calls_per_round) > 0:
        hook_config["max_do_calls_per_round"] = int(args.max_do_calls_per_round)
    if int(args.max_ids_per_action) > 0:
        hook_config["max_ids_per_action"] = int(args.max_ids_per_action)
    if bool(args.hook_progress):
        hook_config["progress"] = True
    if mcts_tree_out is not None:
        hook_config["export_mcts_tree"] = True
    if enriched_tree_out is not None:
        hook_config["export_enriched_tree"] = True

    mcts_trees: Dict[str, object] = {}
    enriched_trees: Dict[str, object] = {}

    entries = list(_iter_entries(payload, args.template_id))
    total = len(entries)
    workers = int(args.workers)
    if workers == 0:
        workers = max(1, available_cpu_count())

    def _run_sequential() -> int:
        written_local = 0
        with out_path.open("w", encoding="utf-8") as fp:
            for idx, (template_id, entry) in enumerate(entries, start=1):
                started_at = time.perf_counter()
                template = entry.get("template") if isinstance(entry.get("template"), dict) else {}
                effective_max_rounds = _effective_max_rounds(template, default_max_rounds=int(args.max_rounds))
                if adapter_factory is not None:
                    try:
                        adapter = adapter_factory(tree=dict(entry), config=hook_config)
                    except TypeError:
                        match_entry = entry.get("match") if isinstance(entry.get("match"), dict) else {}
                        adapter = adapter_factory(template=template, match_entry=match_entry, config=hook_config)
                    mcts_cfg = MultiRoundMCTSConfig(
                        iterations=int(hook_config.get("iterations") or 200),
                        uct_c=float(hook_config.get("uct_c") or 1.4),
                        seed=(
                            int(hook_config["seed"])
                            if "seed" in hook_config and hook_config["seed"] is not None
                            else None
                        ),
                        semantic_tolerance=(
                            float(hook_config["semantic_tolerance"])
                            if "semantic_tolerance" in hook_config and hook_config["semantic_tolerance"] is not None
                            else None
                        ),
                        progress=bool(hook_config.get("progress", False)),
                        dump_tree=bool(mcts_tree_out is not None),
                    )
                    best_history, best_details = run_multi_round_mcts(
                        adapter=adapter,
                        max_rounds=int(effective_max_rounds),
                        config=mcts_cfg,
                    )
                    reward, score_details = adapter.score(history=best_history)
                    merged = dict(best_details)
                    merged.update(dict(score_details))
                    merged["reward"] = float(reward)
                    result = dict(adapter.render(history=best_history, details=merged))
                else:
                    assert execute is not None
                    match_entry = entry.get("match") if isinstance(entry.get("match"), dict) else {}
                    try:
                        result = execute(
                            template=template,
                            match_entry=match_entry,
                            max_rounds=int(effective_max_rounds),
                            config=hook_config,
                        )
                    except TypeError:
                        result = execute(
                            template=template,
                            match_entry=match_entry,
                            max_rounds=int(effective_max_rounds),
                        )
                if not isinstance(result, dict):
                    raise SystemExit(f"multi_round execute hook returned non-dict for template_id={template_id}")
                fp.write(_json_dumps_strict(result) + "\n")
                if mcts_tree_out is not None or enriched_tree_out is not None:
                    notes = result.get("notes") if isinstance(result.get("notes"), dict) else {}
                    blob = notes.get("multi_round_mcts") if isinstance(notes.get("multi_round_mcts"), dict) else {}
                    if mcts_tree_out is not None and isinstance(blob, dict) and "mcts_tree" in blob:
                        mcts_trees[template_id] = blob.get("mcts_tree")
                    if enriched_tree_out is not None and isinstance(blob, dict) and "tree_enriched" in blob:
                        enriched_trees[template_id] = blob.get("tree_enriched")
                written_local += 1
                if not args.no_progress:
                    elapsed_s = time.perf_counter() - started_at
                    hard, sem, cost = _extract_progress_metrics(result)
                    it_used, it_total, early = _extract_mcts_meta(result)
                    hard_s = "?" if hard is None else ("T" if hard else "F")
                    sem_s = "?" if sem is None else f"{sem:.3f}"
                    cost_s = "?" if cost is None else f"{cost:.3f}"
                    if it_used is not None and it_total is not None and it_total > 0:
                        frac = (it_used / it_total) * 100.0
                        mcts_s = f" mcts={it_used}/{it_total}({frac:.1f}%)"
                    else:
                        mcts_s = ""
                    if early is True:
                        mcts_s = mcts_s + " early=T"
                    print(
                        f"[{idx}/{total}] template_id={template_id}: best_cost={cost_s} best_sem={sem_s} hard={hard_s}{mcts_s} elapsed={elapsed_s:.1f}s"
                    )
        return written_local

    if workers <= 1 or total <= 1:
        written = _run_sequential()
    else:
        results: List[Optional[Dict[str, object]]] = [None] * total
        template_ids: List[str] = [tid for tid, _ in entries]
        config_path_str = str(args.config) if args.config is not None else None

        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                initializer=_worker_init,
                initargs=(task, config_path_str),
            ) as ex:
                future_to_index: Dict[concurrent.futures.Future[tuple[str, Dict[str, object], float]], int] = {}
                for idx0, (template_id, entry) in enumerate(entries):
                    fut = ex.submit(
                        _worker_run_one,
                        template_id,
                        entry,
                        int(args.max_rounds),
                        hook_config,
                    )
                    future_to_index[fut] = idx0

                completed = 0
                for fut in concurrent.futures.as_completed(future_to_index):
                    idx0 = future_to_index[fut]
                    template_id, result, elapsed_s = fut.result()
                    results[idx0] = result
                    completed += 1
                    if not args.no_progress:
                        hard, sem, cost = _extract_progress_metrics(result)
                        it_used, it_total, early = _extract_mcts_meta(result)
                        hard_s = "?" if hard is None else ("T" if hard else "F")
                        sem_s = "?" if sem is None else f"{sem:.3f}"
                        cost_s = "?" if cost is None else f"{cost:.3f}"
                        if it_used is not None and it_total is not None and it_total > 0:
                            frac = (it_used / it_total) * 100.0
                            mcts_s = f" mcts={it_used}/{it_total}({frac:.1f}%)"
                        else:
                            mcts_s = ""
                        if early is True:
                            mcts_s = mcts_s + " early=T"
                        print(
                            f"[{completed}/{total}] template_id={template_id}: best_cost={cost_s} best_sem={sem_s} hard={hard_s}{mcts_s} elapsed={elapsed_s:.1f}s"
                        )
        except (PermissionError, OSError) as exc:
            msg = str(exc)
            print(f"Warning: process pool unavailable ({msg}); falling back to sequential (use `--workers 1` to force).")
            written = _run_sequential()
            print(f"Wrote {written} filled template(s) to {out_path}")
            return

        with out_path.open("w", encoding="utf-8") as fp:
            for idx0, result in enumerate(results):
                if not isinstance(result, dict):
                    raise SystemExit(f"Missing result for template_id={template_ids[idx0]}")
                fp.write(_json_dumps_strict(result) + "\n")
                written += 1
                if mcts_tree_out is not None or enriched_tree_out is not None:
                    notes = result.get("notes") if isinstance(result.get("notes"), dict) else {}
                    blob = notes.get("multi_round_mcts") if isinstance(notes.get("multi_round_mcts"), dict) else {}
                    template_id = template_ids[idx0]
                    if mcts_tree_out is not None and isinstance(blob, dict) and "mcts_tree" in blob:
                        mcts_trees[template_id] = blob.get("mcts_tree")
                    if enriched_tree_out is not None and isinstance(blob, dict) and "tree_enriched" in blob:
                        enriched_trees[template_id] = blob.get("tree_enriched")

    if mcts_tree_out is not None:
        mcts_tree_out.parent.mkdir(parents=True, exist_ok=True)
        mcts_tree_out.write_text(_json_dumps_strict(mcts_trees), encoding="utf-8")
        print(f"Wrote MCTS trees to {mcts_tree_out}")
    if enriched_tree_out is not None:
        enriched_tree_out.parent.mkdir(parents=True, exist_ok=True)
        enriched_tree_out.write_text(_json_dumps_strict(enriched_trees), encoding="utf-8")
        print(f"Wrote enriched trees to {enriched_tree_out}")

    print(f"Wrote {written} filled template(s) to {out_path}")


if __name__ == "__main__":
    main()
