"""
Twin-Track MCTS: Monte Carlo Tree Search over the Step-2 AND–OR trees.

This module consumes the AND–OR plan space produced by `planner/compose_match.py`
(e.g. `artifacts/output/<task>/tree_<split>.json`) and selects one candidate per
OR-slot using MCTS (UCT) with lightweight online pruning.

Important: the output keeps the **Plan → Phase → Action** template structure.

How to add a new task
  1) Create `artifacts/input/<task>/planner.json` (version=1) with `twin_track.hooks`.
  2) Implement the hook callables under `task_helper/<task>/` and reference them as "module:callable".
  3) Run: `python -m planner.twin_track --task <task> --tree ...` (or override with `--config`).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import re
import sys
import time
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from planner.config import ConfigError, load_hook, load_planner_config
from planner.parallel import available_cpu_count


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOG_PATH = PROJECT_ROOT / "artifacts" / "logs" / "log.txt"


def _append_run_log(
    *,
    argv: List[str],
    started_at: float,
    status: str,
    extra: Optional[Dict[str, object]] = None,
    error: Optional[str] = None,
) -> None:
    try:
        payload: Dict[str, object] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "module": "planner.twin_track",
            "status": status,
            "elapsed_s": round(time.perf_counter() - started_at, 6),
            "argv": list(argv),
        }
        if extra:
            payload.update(dict(extra))
        if error:
            payload["error"] = error
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        return


# ---------------------------------------------------------------------------
# Core data structures (task-agnostic)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Slot:
    slot_index: int
    slot_edge: str
    phase_key: str
    phase_index: int
    query_action_node_id: str
    query_action: Dict[str, object]
    candidates: List[Dict[str, object]]

    @property
    def action_type(self) -> str:
        return str(self.query_action.get("action_type") or "")


@dataclass(frozen=True)
class Phase:
    phase_key: str
    phase_type: str
    phase_index: int
    attrs: Dict[str, object]
    slots: List[Slot]


@dataclass(frozen=True)
class ScoreResult:
    reward: float
    hard_pass: bool
    semantic_rate: float
    total_cost: float
    hard_details: Dict[str, Tuple[Optional[bool], Optional[str]]] = field(default_factory=dict)
    semantic_details: Dict[str, Tuple[Optional[bool], Optional[str]]] = field(default_factory=dict)


@dataclass(frozen=True)
class MCTSConfig:
    iterations: int = 400
    uct_c: float = 1.4
    seed: Optional[int] = None
    semantic_tolerance: float = 0.8
    semantic_shortfall_weight: float = 1e6
    hard_fail_penalty: float = -1e12
    # Stop MCTS early once we have a "good enough" complete rollout:
    # hard_pass == True and semantic_rate >= semantic_tolerance.
    early_stop: bool = True
    progress: bool = True
    progress_every: int = 0  # 0 => auto


@dataclass
class MonitorState:
    slot_index: int
    chosen_indices: List[int]
    cost_lb: float
    plan: Mapping[str, object]
    task_state: Dict[str, object] = field(default_factory=dict)

    @staticmethod
    def _clone_task_state(task_state: Mapping[str, object]) -> Dict[str, object]:
        copied: Dict[str, object] = {}
        for key, value in task_state.items():
            if isinstance(value, set):
                copied[key] = set(value)
            elif isinstance(value, dict):
                copied[key] = dict(value)
            elif isinstance(value, list):
                copied[key] = list(value)
            else:
                copied[key] = value
        return copied

    def clone(self) -> "MonitorState":
        return MonitorState(
            slot_index=self.slot_index,
            chosen_indices=self.chosen_indices.copy(),
            cost_lb=float(self.cost_lb),
            plan=self.plan,
            task_state=self._clone_task_state(self.task_state),
        )


class SearchNode:
    def __init__(
        self,
        state: MonitorState,
        slot_index: int,
        parent: Optional["SearchNode"],
        candidate_index: Optional[int],
        candidate_lists: Sequence[List[Dict[str, object]]],
    ) -> None:
        self.state = state
        self.slot_index = slot_index
        self.parent = parent
        self.candidate_index = candidate_index
        self.children: List["SearchNode"] = []
        self.untried: List[int] = (
            list(range(len(candidate_lists[slot_index]))) if slot_index < len(candidate_lists) else []
        )
        self.visits = 0
        self.value = 0.0

    def is_terminal(self, total_slots: int) -> bool:
        return self.slot_index >= total_slots

    def best_child(self, *, uct_c: float) -> "SearchNode":
        if not self.children:
            return self
        best_score = -float("inf")
        best_node: Optional[SearchNode] = None
        for child in self.children:
            if child.visits == 0:
                score = float("inf")
            else:
                exploitation = child.value / child.visits
                exploration = uct_c * math.sqrt(math.log(self.visits + 1) / child.visits)
                score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_node = child
        return best_node or self


class Monitor:
    def __init__(
        self,
        *,
        plan: Mapping[str, object],
        slots: Sequence[Slot],
        candidate_lists: Sequence[List[Dict[str, object]]],
        spec: "TaskSpec",
    ) -> None:
        self.plan = dict(plan)
        self.slots = list(slots)
        self.candidate_lists = list(candidate_lists)
        self.spec = spec

        # Suffix lower-bound cost for budget pruning (safe when costs are underestimates).
        n = len(self.slots)
        self.suffix_min_cost: List[float] = [0.0] * (n + 1)
        running = 0.0
        for idx in range(n - 1, -1, -1):
            candidates = self.candidate_lists[idx]
            min_cost = min((_safe_float(c.get("cost")) for c in candidates), default=0.0)
            running += min_cost
            self.suffix_min_cost[idx] = running
        self.suffix_min_cost[n] = 0.0

        budget = self.plan.get("budget")
        self.budget = _safe_float(budget, default=None)

    def initial_state(self) -> MonitorState:
        return MonitorState(
            slot_index=0,
            chosen_indices=[],
            cost_lb=0.0,
            plan=self.plan,
            task_state={},
        )

    def step(self, state: MonitorState, slot_idx: int, candidate_idx: int) -> Tuple[MonitorState, bool]:
        if slot_idx >= len(self.candidate_lists):
            return state, False
        candidates = self.candidate_lists[slot_idx]
        if candidate_idx >= len(candidates):
            return state, False

        slot = self.slots[slot_idx]
        candidate = candidates[candidate_idx]

        cost = _safe_float(candidate.get("cost"), default=0.0) or 0.0
        if self.budget is not None:
            projected = state.cost_lb + cost + self.suffix_min_cost[slot_idx + 1]
            if projected - 1e-6 > self.budget:
                return state, False

        # Task-specific online pruning + state updates.
        new_state = state.clone()
        if not self.spec.online_step(new_state, slot=slot, candidate=candidate):
            return state, False

        new_state.slot_index = slot_idx + 1
        new_state.cost_lb += cost
        new_state.chosen_indices.append(candidate_idx)
        return new_state, True


class TreeMCTS:
    def __init__(
        self,
        *,
        plan: Mapping[str, object],
        phases: Sequence[Phase],
        slots: Sequence[Slot],
        spec: "TaskSpec",
        config: MCTSConfig,
    ) -> None:
        self.plan = dict(plan)
        self.phases = list(phases)
        self.slots = list(slots)
        self.candidate_lists: List[List[Dict[str, object]]] = [slot.candidates for slot in self.slots]
        self.spec = spec
        self.config = config

        self.monitor = Monitor(plan=self.plan, slots=self.slots, candidate_lists=self.candidate_lists, spec=spec)

        self.best_reward = -float("inf")
        self.best_indices: Optional[List[int]] = None
        self.best_score: Optional[ScoreResult] = None

    def search(self, *, progress_prefix: str = "") -> Tuple[List[int], ScoreResult, List[str]]:
        if self.config.seed is not None:
            random.seed(self.config.seed)

        root_state = self.monitor.initial_state()
        root = SearchNode(root_state, 0, None, None, self.candidate_lists)
        total_slots = len(self.slots)

        iterations = max(int(self.config.iterations), 1)
        iterations_used = iterations
        early_stopped = False
        progress_every = int(self.config.progress_every)
        if progress_every <= 0:
            progress_every = max(1, iterations // 10)
        show_progress = bool(self.config.progress)
        progress_prefix = progress_prefix.strip()
        if progress_prefix:
            progress_prefix = progress_prefix + " "
        start = time.perf_counter()
        last_len = 0
        use_inplace = bool(getattr(sys.stderr, "isatty", lambda: False)())

        def _progress_message(iteration: int) -> str:
            elapsed = time.perf_counter() - start
            rate = (iteration / elapsed) if elapsed > 1e-9 else 0.0
            remaining = iterations - iteration
            eta = (remaining / rate) if rate > 1e-9 else float("inf")
            if self.best_score is None:
                return (
                    f"{progress_prefix}MCTS {iteration}/{iterations} "
                    f"(no feasible rollout yet) "
                    f"elapsed={elapsed:.1f}s"
                )
            return (
                f"{progress_prefix}MCTS {iteration}/{iterations} "
                f"best_cost={self.best_score.total_cost:.2f} "
                f"best_sem={self.best_score.semantic_rate:.3f} "
                f"hard={self.best_score.hard_pass} "
                f"elapsed={elapsed:.1f}s "
                f"eta={eta:.1f}s"
            )

        def _emit_progress(iteration: int, *, final: bool = False) -> None:
            nonlocal last_len
            if not show_progress:
                return
            msg = _progress_message(iteration)
            if use_inplace:
                pad = max(0, last_len - len(msg))
                sys.stderr.write("\r" + msg + (" " * pad) + ("\n" if final else ""))
                sys.stderr.flush()
                last_len = 0 if final else len(msg)
            else:
                print(msg, file=sys.stderr, flush=True)

        for iteration in range(1, iterations + 1):
            node = self._tree_policy(root, total_slots)
            reward, indices, score = self._default_policy(node, total_slots)
            self._backup(node, reward)
            if score is not None and reward > self.best_reward:
                self.best_reward = reward
                self.best_indices = indices
                self.best_score = score
            if (
                self.config.early_stop
                and score is not None
                and indices is not None
                and len(indices) == total_slots
                and score.hard_pass
                and score.semantic_rate >= float(self.config.semantic_tolerance)
            ):
                # Return immediately with a feasible plan that meets the semantic tolerance.
                self.best_reward = reward
                self.best_indices = indices
                self.best_score = score
                iterations_used = iteration
                early_stopped = True
                break
            if (
                show_progress
                and iterations > 1
                and iteration != iterations
                and (iteration == 1 or iteration % progress_every == 0)
            ):
                _emit_progress(iteration, final=False)

        if self.best_indices is None or self.best_score is None:
            indices = self._greedy_fallback(total_slots)
            score = self.spec.evaluate(plan=self.plan, phases=self.phases, slots=self.slots, chosen_indices=indices, config=self.config)
            self.best_indices = indices
            self.best_score = score

        if show_progress:
            _emit_progress(iterations_used, final=True)

        trace = [
            f"MCTS iterations: {int(iterations_used)}/{int(self.config.iterations)}",
            f"Early stop: {bool(early_stopped)}",
            f"Hard pass: {self.best_score.hard_pass}",
            f"Semantic rate: {self.best_score.semantic_rate:.3f}",
            f"Total cost: {self.best_score.total_cost:.2f}",
        ]
        return self.best_indices, self.best_score, trace

    def _tree_policy(self, node: SearchNode, total_slots: int) -> SearchNode:
        current = node
        while not current.is_terminal(total_slots):
            if current.untried:
                expanded = self._expand(current, total_slots)
                if expanded is not None:
                    return expanded
                continue
            next_node = current.best_child(uct_c=float(self.config.uct_c))
            if next_node is current:
                break
            current = next_node
        return current

    def _expand(self, node: SearchNode, total_slots: int) -> Optional[SearchNode]:
        while node.untried:
            cand_idx = node.untried.pop()
            new_state, valid = self.monitor.step(node.state, node.slot_index, cand_idx)
            if not valid:
                continue
            child = SearchNode(new_state, node.slot_index + 1, node, cand_idx, self.candidate_lists)
            node.children.append(child)
            return child
        return None

    def _default_policy(self, node: SearchNode, total_slots: int) -> Tuple[float, Optional[List[int]], Optional[ScoreResult]]:
        state = node.state.clone()
        slot_idx = node.slot_index
        while slot_idx < total_slots:
            candidates = self.candidate_lists[slot_idx]
            feasible: List[Tuple[int, MonitorState]] = []
            indices = list(range(len(candidates)))
            random.shuffle(indices)
            for idx in indices:
                new_state, valid = self.monitor.step(state, slot_idx, idx)
                if valid:
                    feasible.append((idx, new_state))
            if not feasible:
                return -float("inf"), None, None
            idx, state = random.choice(feasible)
            slot_idx += 1

        chosen_indices = state.chosen_indices.copy()
        score = self.spec.evaluate(
            plan=self.plan,
            phases=self.phases,
            slots=self.slots,
            chosen_indices=chosen_indices,
            config=self.config,
        )
        return score.reward, chosen_indices, score

    def _backup(self, node: SearchNode, reward: float) -> None:
        current: Optional[SearchNode] = node
        while current is not None:
            current.visits += 1
            current.value += reward
            current = current.parent

    def _greedy_fallback(self, total_slots: int) -> List[int]:
        state = self.monitor.initial_state()
        chosen: List[int] = []
        for slot_idx in range(total_slots):
            picked = 0
            for idx in range(len(self.candidate_lists[slot_idx])):
                new_state, valid = self.monitor.step(state, slot_idx, idx)
                if valid:
                    picked = idx
                    state = new_state
                    break
            chosen.append(picked)
        return chosen


# ---------------------------------------------------------------------------
# Task hooks (loaded from JSON config)
# ---------------------------------------------------------------------------


OnlineStep = Callable[..., bool]
EvaluateFn = Callable[..., ScoreResult]
ActionFillFn = Callable[..., Dict[str, object]]
EvalPlanFn = Callable[..., List[dict]]


@dataclass(frozen=True)
class TaskSpec:
    task_name: str
    required_action_types: Tuple[str, ...]
    online_step: OnlineStep
    fill_action: ActionFillFn
    build_eval_plan: EvalPlanFn
    evaluate: EvaluateFn


def _default_online_step(state: MonitorState, *, slot: Slot, candidate: Mapping[str, object]) -> bool:
    return True


def _default_build_eval_plan(
    plan: Mapping[str, object],
    phases: Sequence[Phase],
    slots: Sequence[Slot],
    chosen_actions: Sequence[Dict[str, object]],
) -> List[dict]:
    return []


def load_task_spec(task: str, *, config_path: Optional[Path] = None) -> TaskSpec:
    cfg = load_planner_config(task, config_path=config_path)
    hooks = cfg.twin_track
    config_label = str(config_path) if config_path is not None else f"artifacts/input/{task}/planner.json"

    online_step: OnlineStep
    if hooks.online_step_hook:
        online_step = load_hook(hooks.online_step_hook)
    else:
        online_step = _default_online_step

    if not hooks.fill_action_hook:
        raise ConfigError(f"Missing twin_track.hooks.fill_action for task '{task}' in {config_label}")
    fill_action: ActionFillFn = load_hook(hooks.fill_action_hook)

    build_eval_plan: EvalPlanFn
    if hooks.build_eval_plan_hook:
        build_eval_plan = load_hook(hooks.build_eval_plan_hook)
    else:
        build_eval_plan = _default_build_eval_plan

    if not hooks.evaluate_hook:
        raise ConfigError(f"Missing twin_track.hooks.evaluate for task '{task}' in {config_label}")
    evaluate: EvaluateFn = load_hook(hooks.evaluate_hook)

    return TaskSpec(
        task_name=str(cfg.task or task),
        required_action_types=tuple(hooks.required_action_types),
        online_step=online_step,
        fill_action=fill_action,
        build_eval_plan=build_eval_plan,
        evaluate=evaluate,
    )


def _safe_float(value: object, default: Optional[float] = 0.0) -> Optional[float]:
    if value is None:
        return default
    try:
        if isinstance(value, bool):
            return float(value)
        number = float(value)  # type: ignore[arg-type]
        if math.isnan(number) or math.isinf(number):
            return default
        return float(number)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Rendering output (keep template structure)
# ---------------------------------------------------------------------------


def _split_action_attrs(query_action: Mapping[str, object]) -> Tuple[Dict[str, object], Dict[str, object], int]:
    params = query_action.get("params") if isinstance(query_action.get("params"), dict) else {}
    order_index_raw = query_action.get("order_index")
    try:
        order_index = int(order_index_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        order_index = 0
    attrs = {k: v for k, v in query_action.items() if k not in {"action_type", "params", "order_index"}}
    return dict(params), attrs, order_index


def render_filled_template(
    *,
    tree: Mapping[str, object],
    phases: Sequence[Phase],
    slots: Sequence[Slot],
    chosen_indices: Sequence[int],
    score: ScoreResult,
    trace: Sequence[str],
    spec: TaskSpec,
) -> Dict[str, object]:
    root = tree.get("root") if isinstance(tree.get("root"), dict) else {}
    plan = root.get("query_plan") if isinstance(root.get("query_plan"), dict) else {}
    template_id = str(tree.get("template_id") or "")

    output_phases: List[Dict[str, object]] = []
    for phase in phases:
        attrs = dict(phase.attrs)
        # Keep non-core attrs under `attrs` like init_template output.
        phase_obj = {
            "phase_key": phase.phase_key,
            "phase_type": phase.phase_type,
            "phase_index": phase.phase_index,
            "attrs": {k: v for k, v in attrs.items() if k not in {"phase_key", "phase_type", "phase_index"}},
        }
        output_phases.append(phase_obj)

    chosen_actions: List[Dict[str, object]] = []
    for slot, idx in zip(slots, chosen_indices):
        candidate = slot.candidates[idx] if idx < len(slot.candidates) else {"source": "placeholder", "text": "-"}
        filled = spec.fill_action(slot.query_action, candidate, slot)
        params, attrs, order_index = _split_action_attrs(filled)
        chosen_actions.append(
            {
                "phase_key": slot.phase_key,
                "action_type": slot.action_type,
                "params": params,
                "attrs": attrs,
                "order_index": order_index,
            }
        )

    notes = dict(tree.get("notes") or {}) if isinstance(tree.get("notes"), dict) else {}
    notes.update(
        {
            "mcts": {
                "reward": score.reward,
                "hard_pass": score.hard_pass,
                "semantic_rate": score.semantic_rate,
                "total_cost": score.total_cost,
                "trace": list(trace),
                "chosen_indices": list(chosen_indices),
            }
        }
    )

    return {
        "version": int(tree.get("version") or 1),
        "template_id": template_id,
        "task_name": spec.task_name,
        "plan": dict(plan),
        "phases": output_phases,
        "actions": chosen_actions,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Tree parsing
# ---------------------------------------------------------------------------


def _load_tree_file(path: Path) -> Dict[str, Dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Tree JSON must be an object mapping template_id -> tree: {path}")
    trees: Dict[str, Dict[str, object]] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            trees[str(key)] = value
    return trees


def _parse_tree_entry(tree: Mapping[str, object]) -> Tuple[Dict[str, object], List[Phase], List[Slot]]:
    root = tree.get("root") if isinstance(tree.get("root"), dict) else {}
    plan = root.get("query_plan") if isinstance(root.get("query_plan"), dict) else {}
    phase_payloads = root.get("children") if isinstance(root.get("children"), list) else []

    phases: List[Phase] = []
    slots: List[Slot] = []
    slot_index = 0

    for phase_obj in phase_payloads:
        if not isinstance(phase_obj, dict):
            continue
        query_phase = phase_obj.get("query_phase") if isinstance(phase_obj.get("query_phase"), dict) else {}
        phase_key = str(query_phase.get("phase_key") or "")
        phase_type = str(query_phase.get("phase_type") or "")
        try:
            phase_index = int(query_phase.get("phase_index") or 0)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            phase_index = 0
        attrs = dict(query_phase)

        slot_objs = phase_obj.get("children") if isinstance(phase_obj.get("children"), list) else []
        phase_slots: List[Slot] = []
        for slot_obj in slot_objs:
            if not isinstance(slot_obj, dict):
                continue
            query_action = slot_obj.get("query_action") if isinstance(slot_obj.get("query_action"), dict) else {}
            candidates = slot_obj.get("candidates") if isinstance(slot_obj.get("candidates"), list) else []
            slot = Slot(
                slot_index=slot_index,
                slot_edge=str(slot_obj.get("slot_edge") or ""),
                phase_key=phase_key,
                phase_index=phase_index,
                query_action_node_id=str(slot_obj.get("query_action_node_id") or ""),
                query_action=dict(query_action),
                candidates=[c for c in candidates if isinstance(c, dict)],
            )
            slots.append(slot)
            phase_slots.append(slot)
            slot_index += 1

        phases.append(
            Phase(
                phase_key=phase_key,
                phase_type=phase_type,
                phase_index=phase_index,
                attrs=attrs,
                slots=phase_slots,
            )
        )

    return dict(plan), phases, slots


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Twin-Track: MCTS over Step-2 AND–OR trees.")
    parser.add_argument("--task", type=str, default="travel", help="Task name (defaults to travel).")
    parser.add_argument("--config", type=Path, default=None, help="Override per-task planner config JSON (planner.json).")
    parser.add_argument("--tree", type=Path, required=True, help="AND–OR tree JSON file (tree_<split>.json).")
    parser.add_argument("--template-id", type=str, default=None, help="Run only a single template_id.")
    parser.add_argument("--out", type=Path, default=None, help="Output JSONL path (filled templates).")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes (0 = all CPUs; 1 = disable parallelism).",
    )
    parser.add_argument("--iterations", type=int, default=400, help="MCTS rollouts per template.")
    parser.add_argument("--uct-c", type=float, default=1.4, help="UCT exploration constant.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--semantic-tolerance",
        type=float,
        default=0.8,
        help="Semantic rate tolerance used for scoring and (optional) early stop.",
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable early stopping when a hard-pass plan reaches the semantic tolerance.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress output (printed to stderr by default).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print progress every N iterations (0 = auto).",
    )
    return parser.parse_args(argv)


_WORKER_SPEC: Optional[TaskSpec] = None
_WORKER_CONFIG: Optional[MCTSConfig] = None
_WORKER_TREES: Optional[Dict[str, Dict[str, object]]] = None


def _mcts_worker_init(
    task: str,
    config_path: Optional[str],
    tree_path: str,
    mcts_config: MCTSConfig,
) -> None:
    global _WORKER_SPEC
    global _WORKER_CONFIG
    global _WORKER_TREES

    cfg_path = Path(config_path) if isinstance(config_path, str) and config_path else None
    _WORKER_SPEC = load_task_spec(task, config_path=cfg_path)
    _WORKER_CONFIG = mcts_config
    _WORKER_TREES = _load_tree_file(Path(tree_path))


def _mcts_worker(template_id: str) -> Tuple[str, str, bool, float, float]:
    spec = _WORKER_SPEC
    config = _WORKER_CONFIG
    trees = _WORKER_TREES
    if spec is None or config is None or trees is None:
        raise RuntimeError("Twin-Track worker not initialised (missing spec/config/trees).")

    tree = trees.get(str(template_id))
    if tree is None:
        raise KeyError(f"Missing template_id in tree: {template_id}")

    plan, phases, slots = _parse_tree_entry(tree)
    mcts = TreeMCTS(plan=plan, phases=phases, slots=slots, spec=spec, config=config)
    indices, score, trace = mcts.search(progress_prefix="")
    filled_template = render_filled_template(
        tree=tree,
        phases=phases,
        slots=slots,
        chosen_indices=indices,
        score=score,
        trace=trace,
        spec=spec,
    )
    return str(template_id), json.dumps(filled_template, ensure_ascii=False), bool(score.hard_pass), float(score.semantic_rate), float(score.total_cost)


def main(argv: Optional[Sequence[str]] = None) -> None:
    started_at = time.perf_counter()
    argv_snapshot = sys.argv[:]
    status = "ok"
    error: Optional[str] = None
    extra: Dict[str, object] = {}
    written = 0

    try:
        args = parse_args(argv)
        extra.update(
            {
                "task": args.task,
                "config": str(args.config) if args.config is not None else None,
                "workers": int(args.workers),
                "iterations": int(args.iterations),
                "uct_c": float(args.uct_c),
                "seed": args.seed,
                "semantic_tolerance": float(args.semantic_tolerance),
                "template_id": args.template_id,
                "progress": not bool(args.no_progress),
                "progress_every": int(args.progress_every),
            }
        )

        task = str(args.task)
        try:
            spec = load_task_spec(task, config_path=args.config)
        except ConfigError as exc:
            raise SystemExit(str(exc))

        tree_path = args.tree
        if not tree_path.is_absolute():
            tree_path = (PROJECT_ROOT / tree_path).resolve()
        if not tree_path.exists():
            raise SystemExit(f"Tree JSON not found: {tree_path}")
        extra.update({"tree": str(tree_path)})

        config_path = args.config
        if config_path is not None and not config_path.is_absolute():
            config_path = (PROJECT_ROOT / config_path).resolve()

        out_path = args.out
        if out_path is None:
            split = None
            match = re.match(r"^tree_(?P<split>[^.]+)$", tree_path.stem)
            if match:
                split = match.group("split")
            if split:
                out_path = tree_path.parent / f"optimized_{split}.jsonl"
            else:
                out_path = tree_path.with_suffix(".optimized.jsonl")
        if not out_path.is_absolute():
            out_path = (PROJECT_ROOT / out_path).resolve()
        extra.update({"out": str(out_path)})

        base_config = MCTSConfig(
            iterations=int(args.iterations),
            uct_c=float(args.uct_c),
            seed=args.seed,
            semantic_tolerance=float(args.semantic_tolerance),
            early_stop=not bool(args.no_early_stop),
            progress=not bool(args.no_progress),
            progress_every=int(args.progress_every),
        )

        trees = _load_tree_file(tree_path)
        ids = [args.template_id] if args.template_id is not None else list(trees.keys())
        extra.update({"templates_total": len(ids)})

        out_path.parent.mkdir(parents=True, exist_ok=True)
        requested_workers = int(args.workers)
        if args.template_id is not None or len(ids) <= 1:
            worker_count = 1
        elif requested_workers <= 0:
            worker_count = available_cpu_count()
        else:
            worker_count = requested_workers
        worker_count = max(1, min(worker_count, len(ids)))
        extra.update({"workers_effective": worker_count})

        if worker_count <= 1:
            config = base_config
            with out_path.open("w", encoding="utf-8") as fp:
                total_templates = len(ids)
                for idx, template_id in enumerate(ids, start=1):
                    tree = trees.get(str(template_id))
                    if tree is None:
                        continue
                    plan, phases, slots = _parse_tree_entry(tree)
                    mcts = TreeMCTS(plan=plan, phases=phases, slots=slots, spec=spec, config=config)
                    prefix = f"[{idx}/{total_templates}] template_id={template_id}:"
                    indices, score, trace = mcts.search(progress_prefix=prefix)
                    filled_template = render_filled_template(
                        tree=tree,
                        phases=phases,
                        slots=slots,
                        chosen_indices=indices,
                        score=score,
                        trace=trace,
                        spec=spec,
                    )
                    fp.write(json.dumps(filled_template, ensure_ascii=False) + "\n")
                    written += 1
        else:
            # Avoid interleaved per-iteration progress from multiple processes; emit per-template progress here instead.
            worker_config = replace(base_config, progress=False, progress_every=0)
            pending_lines: List[Optional[str]] = [None] * len(ids)
            next_write = 0
            completed = 0
            total_templates = len(ids)

            with out_path.open("w", encoding="utf-8") as fp:
                try:
                    with concurrent.futures.ProcessPoolExecutor(
                        max_workers=worker_count,
                        initializer=_mcts_worker_init,
                        initargs=(task, str(config_path) if config_path is not None else None, str(tree_path), worker_config),
                    ) as executor:
                        future_to_index = {
                            executor.submit(_mcts_worker, str(template_id)): idx for idx, template_id in enumerate(ids)
                        }
                        for future in concurrent.futures.as_completed(future_to_index):
                            idx = future_to_index[future]
                            template_id, line, hard_pass, semantic_rate, total_cost = future.result()
                            pending_lines[idx] = line
                            completed += 1
                            if not bool(args.no_progress):
                                sys.stderr.write(
                                    f"[{completed}/{total_templates}] template_id={template_id} "
                                    f"cost={total_cost:.2f} sem={semantic_rate:.3f} hard={hard_pass}\n"
                                )
                                sys.stderr.flush()
                            while next_write < total_templates and pending_lines[next_write] is not None:
                                fp.write(pending_lines[next_write] + "\n")
                                pending_lines[next_write] = None
                                next_write += 1
                                written += 1
                except BrokenProcessPool as exc:
                    raise SystemExit(
                        "Twin-Track workers crashed (BrokenProcessPool). "
                        "On HPC this is often an OOM-kill or native crash; try `--workers 1` "
                        "to debug, or reduce `--workers` / request more memory."
                    ) from exc

        print(f"Wrote {written} filled template(s) to {out_path}")
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            status = "interrupted"
        elif isinstance(exc, SystemExit):
            status = "exit"
            code = exc.code
            if isinstance(code, (int, float, str, bool)) or code is None:
                extra.update({"exit_code": code})
            else:
                extra.update({"exit_code": repr(code)})
        else:
            status = "error"
        error = repr(exc)
        raise
    finally:
        extra.setdefault("written", written)
        _append_run_log(argv=argv_snapshot, started_at=started_at, status=status, extra=extra, error=error)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Legacy compatibility
# ---------------------------------------------------------------------------


def run(*_args: object, **_kwargs: object) -> None:  # pragma: no cover
    """
    Legacy API placeholder.

    Previous iterations of the repo exposed `planner.twin_track.run(template, cover_result, ...)`
    which depended on PlanletCover/ComposeMatch classes. The current pipeline uses the Step-2
    AND–OR tree JSON as input instead.
    """

    raise RuntimeError(
        "Legacy `planner.twin_track.run(...)` is deprecated. "
        "Use `python -m planner.twin_track --tree artifacts/output/<task>/tree_<split>.json`."
    )
