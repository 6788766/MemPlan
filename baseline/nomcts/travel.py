from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import shutil
import time
import traceback
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from baseline.nomcts.common import (
    PROJECT_ROOT,
    TRAVEL_OUTPUT_ROOT,
    TokenUsage,
    append_total_cost_block,
    combine_price_keys,
    compute_price_usd,
    copy_if_exists,
    find_stage_usage,
    iso_now,
    iter_jsonl,
    load_init_template_records,
    load_init_template_usage_map,
    model_slug,
    parse_travel_eval_metrics,
    python_executable,
    read_cost_float,
    read_cost_lines,
    read_cost_value,
    resolve_price_key,
    resolve_workers,
    run_subprocess,
    stable_seed,
    sum_mapping_values,
    sum_stage_elapsed_seconds,
    summarize_travel_tool_calls,
    write_json,
    write_jsonl,
)
from planner.twin_track import MCTSConfig, Monitor, load_task_spec, render_filled_template
from planner.twin_track import _load_tree_file, _parse_tree_entry  # type: ignore


_WORKER_SPEC = None
_WORKER_CONFIG = None
_WORKER_TREES = None
_WORKER_BASE_SEED = None


def _random_feasible_selection(
    *,
    tree: Mapping[str, object],
    spec: object,
    config: MCTSConfig,
    seed: int,
) -> Tuple[Dict[str, object], float, bool, float, float, List[int], bool]:
    plan, phases, slots = _parse_tree_entry(tree)
    candidate_lists = [slot.candidates for slot in slots]
    monitor = Monitor(plan=plan, slots=slots, candidate_lists=candidate_lists, spec=spec)
    rng = random.Random(seed)
    started_at = time.perf_counter()

    state = monitor.initial_state()
    chosen_indices: List[int] = []
    fallback_used = False

    for slot_idx, candidates in enumerate(candidate_lists):
        feasible: List[Tuple[int, object]] = []
        order = list(range(len(candidates)))
        rng.shuffle(order)
        for candidate_idx in order:
            new_state, valid = monitor.step(state, slot_idx, candidate_idx)
            if valid:
                feasible.append((candidate_idx, new_state))
        if not feasible:
            fallback_used = True
            state = monitor.initial_state()
            chosen_indices = []
            for greedy_idx, greedy_candidates in enumerate(candidate_lists):
                picked_idx = 0
                picked_state = None
                for candidate_idx in range(len(greedy_candidates)):
                    new_state, valid = monitor.step(state, greedy_idx, candidate_idx)
                    if valid:
                        picked_idx = candidate_idx
                        picked_state = new_state
                        break
                chosen_indices.append(int(picked_idx))
                if picked_state is not None:
                    state = picked_state
            break
        picked_idx, picked_state = rng.choice(feasible)
        chosen_indices.append(int(picked_idx))
        state = picked_state

    score = spec.evaluate(  # type: ignore[attr-defined]
        plan=plan,
        phases=phases,
        slots=slots,
        chosen_indices=chosen_indices,
        config=config,
    )
    trace = [
        "NoMCTS random feasible rollout",
        f"random_seed={seed}",
        f"fallback_used={fallback_used}",
        f"hard_pass={bool(score.hard_pass)}",
        f"semantic_rate={float(score.semantic_rate):.3f}",
        f"total_cost={float(score.total_cost):.2f}",
    ]
    record = render_filled_template(
        tree=tree,
        phases=phases,
        slots=slots,
        chosen_indices=chosen_indices,
        score=score,
        trace=trace,
        spec=spec,  # type: ignore[arg-type]
    )
    notes = record.get("notes")
    if not isinstance(notes, dict):
        notes = {}
        record["notes"] = notes
    mcts = notes.get("mcts")
    if not isinstance(mcts, dict):
        mcts = {}
        notes["mcts"] = mcts
    mcts["selector"] = "random_feasible_rollout"
    mcts["source"] = "nomcts"
    mcts["random_seed"] = int(seed)
    mcts["fallback_used"] = bool(fallback_used)
    notes["nomcts"] = {
        "selector": "random_feasible_rollout",
        "random_seed": int(seed),
        "fallback_used": bool(fallback_used),
        "chosen_indices": list(chosen_indices),
    }
    elapsed_s = time.perf_counter() - started_at
    return (
        record,
        float(elapsed_s),
        bool(score.hard_pass),
        float(score.semantic_rate),
        float(score.total_cost),
        list(chosen_indices),
        bool(fallback_used),
    )


def _worker_init(task: str, config_path: Optional[str], tree_path: str, semantic_tolerance: float, base_seed: Optional[int]) -> None:
    global _WORKER_SPEC
    global _WORKER_CONFIG
    global _WORKER_TREES
    global _WORKER_BASE_SEED

    cfg_path = Path(config_path) if config_path else None
    _WORKER_SPEC = load_task_spec(task, config_path=cfg_path)
    _WORKER_CONFIG = MCTSConfig(semantic_tolerance=float(semantic_tolerance))
    _WORKER_TREES = _load_tree_file(Path(tree_path))
    _WORKER_BASE_SEED = base_seed


def _worker_run(template_id: str) -> Tuple[str, Dict[str, object], float, bool, float, float, List[int], bool]:
    if _WORKER_SPEC is None or _WORKER_CONFIG is None or _WORKER_TREES is None:
        raise RuntimeError("Worker state was not initialised.")
    tree = _WORKER_TREES[str(template_id)]
    seed = stable_seed(_WORKER_BASE_SEED, str(template_id))
    record, elapsed_s, hard_pass, semantic_rate, total_cost, chosen_indices, fallback_used = _random_feasible_selection(
        tree=tree,
        spec=_WORKER_SPEC,
        config=_WORKER_CONFIG,
        seed=seed,
    )
    return str(template_id), record, float(elapsed_s), bool(hard_pass), float(semantic_rate), float(total_cost), chosen_indices, bool(fallback_used)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-MCTS baseline for the Travel benchmark.")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--semantic-threshold", type=float, default=0.8)
    parser.add_argument("--repair-model", type=str, default=None)
    parser.add_argument("--repair-workers", type=int, default=0)
    parser.add_argument("--repair-max-slot-candidates", type=int, default=50)
    parser.add_argument("--output-root", type=Path, default=TRAVEL_OUTPUT_ROOT)
    parser.add_argument("--source-run-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-repair", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args(argv)


def _build_per_query_metrics(
    *,
    records: List[Mapping[str, object]],
    elapsed_by_template: Mapping[str, float],
    init_usage_by_template: Mapping[str, TokenUsage],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for record in records:
        template_id = str(record.get("template_id") or "")
        plan = record.get("plan") if isinstance(record.get("plan"), dict) else {}
        query = str(plan.get("query") or "")
        notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
        mcts = notes.get("mcts") if isinstance(notes.get("mcts"), dict) else {}
        nomcts = notes.get("nomcts") if isinstance(notes.get("nomcts"), dict) else {}
        init_usage = init_usage_by_template.get(template_id, TokenUsage())
        stage_times = {
            "init_template": 0.0,
            "nomcts": float(elapsed_by_template.get(template_id, 0.0)),
        }
        stage_usage = {
            "init_template": init_usage.to_dict(),
            "nomcts": TokenUsage().to_dict(),
        }
        rows.append(
            {
                "template_id": template_id,
                "query": query,
                "status": "ok",
                "time_s_total": float(stage_times["nomcts"]),
                "time_s_by_stage": stage_times,
                "token_usage_by_stage": stage_usage,
                "llm_calls": int(init_usage.calls),
                "prompt_cache_hit_tokens": int(init_usage.prompt_cache_hit_tokens),
                "prompt_cache_miss_tokens": int(init_usage.prompt_cache_miss_tokens),
                "completion_tokens": int(init_usage.completion_tokens),
                "total_tokens": int(init_usage.total_tokens),
                "selector": "random_feasible_rollout",
                "random_seed": nomcts.get("random_seed"),
                "chosen_indices": list(nomcts.get("chosen_indices") or []),
                "fallback_used": bool(nomcts.get("fallback_used", False)),
                "hard_pass": mcts.get("hard_pass"),
                "semantic_rate": mcts.get("semantic_rate"),
                "selected_plan_cost": mcts.get("total_cost"),
            }
        )
    return rows


def _build_repair_per_query_metrics(
    *,
    records: List[Mapping[str, object]],
    init_usage_by_template: Mapping[str, TokenUsage],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for record in records:
        template_id = str(record.get("template_id") or "")
        plan = record.get("plan") if isinstance(record.get("plan"), dict) else {}
        query = str(plan.get("query") or "")
        notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
        repair = notes.get("llm_repair") if isinstance(notes.get("llm_repair"), dict) else {}
        mcts = notes.get("mcts") if isinstance(notes.get("mcts"), dict) else {}
        init_usage = init_usage_by_template.get(template_id, TokenUsage())
        repair_usage = TokenUsage.from_mapping(repair.get("usage") if isinstance(repair, dict) else None, calls=1 if repair else 0)
        stage_usage = {
            "init_template": init_usage.to_dict(),
            "llm_repair": repair_usage.to_dict(),
        }
        rows.append(
            {
                "template_id": template_id,
                "query": query,
                "status": str(repair.get("status") or "not_attempted"),
                "time_s_total": 0.0,
                "time_s_by_stage": {
                    "init_template": 0.0,
                    "llm_repair": 0.0,
                },
                "token_usage_by_stage": stage_usage,
                "llm_calls": int(init_usage.calls + repair_usage.calls),
                "prompt_cache_hit_tokens": int(init_usage.prompt_cache_hit_tokens + repair_usage.prompt_cache_hit_tokens),
                "prompt_cache_miss_tokens": int(init_usage.prompt_cache_miss_tokens + repair_usage.prompt_cache_miss_tokens),
                "completion_tokens": int(init_usage.completion_tokens + repair_usage.completion_tokens),
                "total_tokens": int(init_usage.total_tokens + repair_usage.total_tokens),
                "repair_status": repair.get("status"),
                "repair_before": repair.get("before"),
                "repair_after": repair.get("after"),
                "hard_pass": mcts.get("hard_pass"),
                "semantic_rate": mcts.get("semantic_rate"),
            }
        )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    repair_model = args.repair_model or args.model
    source_run_dir = (
        args.source_run_dir.expanduser().resolve()
        if args.source_run_dir is not None
        else (PROJECT_ROOT / "artifacts" / "output" / "travel" / f"{model_slug(args.model)}_{args.split}").resolve()
    )
    run_dir = args.output_root.expanduser().resolve() / f"{model_slug(args.model)}_{args.split}"
    run_dir.mkdir(parents=True, exist_ok=True)

    init_templates_path = run_dir / f"init_templates_{args.split}.jsonl"
    match_path = run_dir / f"match_{args.split}.json"
    tree_path = run_dir / f"tree_{args.split}.json"
    tool_calls_path = run_dir / f"tool_calls_{args.split}.json"
    optimized_path = run_dir / f"optimized_{args.split}.jsonl"
    optimized_raw_path = run_dir / f"optimized_{args.split}.raw.jsonl"
    results_txt_path = run_dir / "results.txt"
    raw_results_txt_path = run_dir / "raw_results.txt"
    cost_txt_path = run_dir / "cost.txt"
    run_metrics_path = run_dir / "run_metrics.json"
    per_query_metrics_path = run_dir / "per_query_metrics.jsonl"

    source_init_templates = source_run_dir / f"init_templates_{args.split}.jsonl"
    source_match = source_run_dir / f"match_{args.split}.json"
    source_tree = source_run_dir / f"tree_{args.split}.json"
    source_tool_calls = source_run_dir / f"tool_calls_{args.split}.json"
    source_cost = source_run_dir / "cost.txt"

    for src in (source_init_templates, source_match, source_tree, source_tool_calls, source_cost):
        if not src.exists():
            raise FileNotFoundError(f"Required source artifact not found: {src}")

    copy_if_exists(source_init_templates, init_templates_path)
    copy_if_exists(source_match, match_path)
    copy_if_exists(source_tree, tree_path)
    copy_if_exists(source_tool_calls, tool_calls_path)

    source_cost_lines = read_cost_lines(source_cost)
    reused_init_elapsed = sum_stage_elapsed_seconds(source_cost_lines, "init_template")
    reused_view_elapsed = sum_stage_elapsed_seconds(source_cost_lines, "view_select")
    reused_compose_elapsed = sum_stage_elapsed_seconds(source_cost_lines, "compose_match")
    source_price_key = read_cost_value(source_cost_lines, "llm_price_model_key") or resolve_price_key(args.model)

    init_records = load_init_template_records(init_templates_path)
    init_usage_by_template, init_total_usage = load_init_template_usage_map(init_templates_path)
    tool_call_summary = summarize_travel_tool_calls(tool_calls_path)
    reused_compose_tool_cost = read_cost_float(source_cost_lines, "tool_calls_total_cost")
    if reused_compose_tool_cost is None:
        reused_compose_tool_cost = float(tool_call_summary["total_cost"])

    results_txt_path.write_text("", encoding="utf-8")
    raw_results_txt_path.write_text("", encoding="utf-8")

    trees = _load_tree_file(tree_path)
    template_ids = list(trees.keys())
    if args.limit > 0:
        template_ids = template_ids[: int(args.limit)]

    workers = resolve_workers(args.workers, len(template_ids))
    run_start_iso = iso_now()
    random_stage_start = time.perf_counter()

    config_path = str(args.config.expanduser().resolve()) if args.config is not None else None
    results_by_template: Dict[str, Dict[str, object]] = {}
    elapsed_by_template: Dict[str, float] = {}

    if workers <= 1:
        spec = load_task_spec("travel", config_path=(args.config.expanduser().resolve() if args.config is not None else None))
        config = MCTSConfig(semantic_tolerance=float(args.semantic_threshold))
        for index, template_id in enumerate(template_ids, start=1):
            seed = stable_seed(args.seed, template_id)
            record, elapsed_s, hard_pass, semantic_rate, total_cost, _, _ = _random_feasible_selection(
                tree=trees[template_id],
                spec=spec,
                config=config,
                seed=seed,
            )
            results_by_template[template_id] = record
            elapsed_by_template[template_id] = elapsed_s
            print(
                f"[{index}/{len(template_ids)}] template_id={template_id} cost={total_cost:.2f} sem={semantic_rate:.3f} hard={hard_pass}",
                flush=True,
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=("travel", config_path, str(tree_path), float(args.semantic_threshold), args.seed),
        ) as pool:
            future_to_template = {pool.submit(_worker_run, template_id): template_id for template_id in template_ids}
            completed = 0
            for future in concurrent.futures.as_completed(future_to_template):
                expected_template_id = future_to_template[future]
                try:
                    template_id, record, elapsed_s, hard_pass, semantic_rate, total_cost, _, _ = future.result()
                except Exception:
                    message = (
                        f"[nomcts][travel][fatal] template_id={expected_template_id}\n"
                        f"{traceback.format_exc()}"
                    )
                    with results_txt_path.open("a", encoding="utf-8") as fp:
                        fp.write(message)
                        if not message.endswith("\n"):
                            fp.write("\n")
                    print(message, end="" if message.endswith("\n") else "\n", flush=True)
                    raise
                results_by_template[template_id] = record
                elapsed_by_template[template_id] = float(elapsed_s)
                completed += 1
                print(
                    f"[{completed}/{len(template_ids)}] template_id={template_id} cost={total_cost:.2f} sem={semantic_rate:.3f} hard={hard_pass}",
                    flush=True,
                )

    random_stage_elapsed = time.perf_counter() - random_stage_start
    ordered_records = [results_by_template[template_id] for template_id in template_ids]
    with optimized_path.open("w", encoding="utf-8") as fp:
        for record in ordered_records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    shutil.copy2(optimized_path, optimized_raw_path)

    per_query_rows = _build_per_query_metrics(
        records=ordered_records,
        elapsed_by_template=elapsed_by_template,
        init_usage_by_template=init_usage_by_template,
    )
    write_jsonl(per_query_metrics_path, per_query_rows)

    eval_elapsed = 0.0
    eval_metrics: Dict[str, float] = {}
    eval_output = ""
    if not args.skip_eval:
        print("Running Travel evaluation for no-repair output...", flush=True)
        if args.split == "test":
            eval_elapsed, eval_output = run_subprocess(
                cmd=[
                    python_executable(),
                    "-m",
                    "task_helper.travel.runners.eval_leaderboard",
                    "--split",
                    "test",
                    "--eval-mode",
                    "two-stage",
                    "--submission",
                    str(optimized_path),
                ],
                out_path=results_txt_path,
            )
        else:
            eval_elapsed, eval_output = run_subprocess(
                cmd=[
                    python_executable(),
                    "-m",
                    "task_helper.travel.runners.eval_bridge",
                    "--set-type",
                    args.split,
                    "--submission",
                    str(optimized_path),
                ],
                out_path=results_txt_path,
            )
            eval_metrics = parse_travel_eval_metrics(eval_output)
        shutil.copy2(results_txt_path, raw_results_txt_path)
        submission_path = run_dir / f"optimized_{args.split}.submission.jsonl"
        raw_submission_path = run_dir / f"optimized_{args.split}.raw.submission.jsonl"
        if submission_path.exists():
            shutil.copy2(submission_path, raw_submission_path)

    comparable_no_eval = reused_init_elapsed + reused_view_elapsed + reused_compose_elapsed + random_stage_elapsed
    comparable_total = comparable_no_eval + eval_elapsed
    actual_no_eval = random_stage_elapsed
    actual_total = random_stage_elapsed + eval_elapsed

    init_price = read_cost_float(source_cost_lines, "llm_price_usd_init_template")
    if init_price is None:
        init_price = compute_price_usd(model_name=args.model, usage=init_total_usage)
    total_price = sum_mapping_values([init_price])
    price_key = source_price_key
    pipeline_cost_total = float(total_price) + float(reused_compose_tool_cost)

    cost_lines = [
        f"run_start={run_start_iso}",
        "task=travel",
        "variant=nomcts",
        f"split={args.split}",
        f"model={args.model}",
        f"model_slug={model_slug(args.model)}",
        f"seed={int(args.seed)}",
        f"workers={args.workers} (effective={workers})",
        f"source_run_dir={source_run_dir}",
        f"source_llm_price_model_key={source_price_key}",
        f"outputs_dir={run_dir}",
        f"queries_total={len(template_ids)}",
        f"queries_processed={len(ordered_records)}",
        f"limit={int(args.limit)}",
        "reuse_init_template=1",
        "reuse_view_select=1",
        "reuse_compose_match=1",
        f"reused_elapsed_s_init_template={reused_init_elapsed}",
        f"reused_elapsed_s_view_select={reused_view_elapsed}",
        f"reused_elapsed_s_compose_match={reused_compose_elapsed}",
        f"reused_tool_cost_compose_match={float(reused_compose_tool_cost)}",
        f"nomcts_elapsed_s={random_stage_elapsed}",
        f"actual_pipeline_elapsed_s_no_eval={actual_no_eval}",
        f"actual_pipeline_elapsed_s_total={actual_total}",
        f"pipeline_elapsed_s_no_eval={comparable_no_eval}",
        f"pipeline_elapsed_s_total={comparable_total}",
        f"eval_elapsed_s={eval_elapsed}",
        f"LLM token usage (init_template): {init_total_usage.to_cost_line()}",
        f"LLM token usage (nomcts): {TokenUsage().to_cost_line()}",
        (
            "Tool call usage (compose_match): "
            f"calls={int(tool_call_summary['total_calls'])} "
            f"cost={float(reused_compose_tool_cost)} "
            f"by_tool={json.dumps(tool_call_summary['by_tool'], ensure_ascii=False, sort_keys=True)}"
        ),
        f"tool_calls_total_calls={int(tool_call_summary['total_calls'])}",
        f"tool_calls_total_cost={float(reused_compose_tool_cost)}",
        f"pipeline_tool_cost_total={float(reused_compose_tool_cost)}",
        f"pipeline_cost_usd_total={pipeline_cost_total}",
        f"pipeline_total_tokens={int(init_total_usage.total_tokens)}",
        "actual_total_tokens=0",
        f"run_end={iso_now()}",
        f"summary: time_s_no_eval={comparable_no_eval} tokens_total={int(init_total_usage.total_tokens)} tool_calls_cost={float(tool_call_summary['total_cost'])}",
    ]
    cost_lines = append_total_cost_block(
        lines=cost_lines,
        total_model_key=source_price_key,
        total_tokens=int(init_total_usage.total_tokens),
        total_time_s=float(comparable_no_eval),
        stage_prices={
            "llm_price_usd_init_template": init_price,
            "llm_price_usd_nomcts": 0.0,
            "llm_price_usd_total": total_price,
        },
    )
    cost_txt_path.write_text("\n".join(cost_lines) + "\n", encoding="utf-8")

    run_metrics = {
        "metadata": {
            "task": "travel",
            "variant": "nomcts",
            "split": args.split,
            "model": args.model,
            "model_slug": model_slug(args.model),
            "seed": int(args.seed),
            "workers_requested": int(args.workers),
            "workers_effective": int(workers),
            "source_run_dir": str(source_run_dir),
            "outputs_dir": str(run_dir),
            "limit": int(args.limit),
        },
        "stages": {
            "init_template": {
                "reused": True,
                "elapsed_s": reused_init_elapsed,
                "token_usage": init_total_usage.to_dict(),
                "price_usd": init_price,
            },
            "view_select": {
                "reused": True,
                "elapsed_s": reused_view_elapsed,
                "token_usage": TokenUsage().to_dict(),
            },
            "compose_match": {
                "reused": True,
                "elapsed_s": reused_compose_elapsed,
                "token_usage": TokenUsage().to_dict(),
                "tool_calls_total": int(tool_call_summary["total_calls"]),
                "tool_cost_total": float(reused_compose_tool_cost),
                "tool_usage_by_tool": tool_call_summary["by_tool"],
            },
            "nomcts": {
                "elapsed_s": random_stage_elapsed,
                "elapsed_s_sum_per_query": sum(float(value) for value in elapsed_by_template.values()),
                "token_usage": TokenUsage().to_dict(),
                "selector": "random_feasible_rollout",
            },
            "eval": {
                "elapsed_s": eval_elapsed,
                "token_usage": TokenUsage().to_dict(),
            },
        },
        "totals": {
            "actual_pipeline_elapsed_s_no_eval": actual_no_eval,
            "actual_pipeline_elapsed_s_total": actual_total,
            "pipeline_elapsed_s_no_eval": comparable_no_eval,
            "pipeline_elapsed_s_total": comparable_total,
            "total_tokens": int(init_total_usage.total_tokens),
            "actual_total_tokens": 0,
            "llm_price_model_key": price_key,
            "llm_price_usd_total": total_price,
            "tool_cost_total": float(reused_compose_tool_cost),
            "pipeline_cost_usd_total": pipeline_cost_total,
        },
        "evaluation": eval_metrics,
    }
    write_json(run_metrics_path, run_metrics)

    if not args.run_repair:
        print(f"Travel NoMCTS baseline complete: {run_dir}", flush=True)
        return

    repair_dir = run_dir / "repair"
    repair_dir.mkdir(parents=True, exist_ok=True)
    repair_cost_txt = repair_dir / "cost.txt"
    repair_results_txt = repair_dir / "results.txt"
    repair_jsonl = repair_dir / f"repaired_{args.split}_{model_slug(repair_model)}.jsonl"
    repair_run_metrics_path = repair_dir / "run_metrics.json"
    repair_per_query_metrics_path = repair_dir / "per_query_metrics.jsonl"
    repair_log_path = repair_dir / "llm_repair.log"
    repair_results_txt.write_text("", encoding="utf-8")
    repair_log_path.write_text("", encoding="utf-8")

    print("Running single-invocation LLM repair...", flush=True)
    repair_elapsed_llm, repair_output = run_subprocess(
        cmd=[
            python_executable(),
            "-m",
            "planner.llm_repair",
            "--task",
            "travel",
            "--tree",
            str(tree_path),
            "--input",
            str(optimized_path),
            "--out",
            str(repair_jsonl),
            "--model",
            str(repair_model),
            "--semantic-threshold",
            str(args.semantic_threshold),
            "--max-slot-candidates",
            str(args.repair_max_slot_candidates),
            "--workers",
            str(resolve_workers(args.repair_workers, len(template_ids))),
            *([] if args.config is None else ["--config", str(args.config.expanduser().resolve())]),
        ],
        out_path=repair_log_path,
        stream_output=True,
    )
    repair_usage = find_stage_usage(repair_output.splitlines(), "llm_repair")
    repair_records = list(iter_jsonl(repair_jsonl))
    write_jsonl(
        repair_per_query_metrics_path,
        _build_repair_per_query_metrics(records=repair_records, init_usage_by_template=init_usage_by_template),
    )

    repair_eval_elapsed = 0.0
    repair_eval_metrics: Dict[str, float] = {}
    if not args.skip_eval:
        print("Running Travel evaluation for repaired output...", flush=True)
        if args.split == "test":
            repair_eval_elapsed, repair_eval_output = run_subprocess(
                cmd=[
                    python_executable(),
                    "-m",
                    "task_helper.travel.runners.eval_leaderboard",
                    "--split",
                    "test",
                    "--eval-mode",
                    "two-stage",
                    "--submission",
                    str(repair_jsonl),
                ],
                out_path=repair_results_txt,
            )
        else:
            repair_eval_elapsed, repair_eval_output = run_subprocess(
                cmd=[
                    python_executable(),
                    "-m",
                    "task_helper.travel.runners.eval_bridge",
                    "--set-type",
                    args.split,
                    "--submission",
                    str(repair_jsonl),
                ],
                out_path=repair_results_txt,
            )
            repair_eval_metrics = parse_travel_eval_metrics(repair_eval_output)

    repair_total_tokens = int(init_total_usage.total_tokens + repair_usage.total_tokens)
    repair_actual_no_eval = repair_elapsed_llm
    repair_actual_total = repair_elapsed_llm + repair_eval_elapsed
    repair_comparable_no_eval = comparable_no_eval + repair_elapsed_llm
    repair_comparable_total = repair_comparable_no_eval + repair_eval_elapsed
    base_price = init_price
    repair_price = compute_price_usd(model_name=repair_model, usage=repair_usage)
    repair_total_price = sum_mapping_values([base_price, repair_price])
    repair_pipeline_cost_total = float(repair_total_price) + float(reused_compose_tool_cost)

    repair_cost_lines = [
        f"run_start={iso_now()}",
        "task=travel",
        "variant=nomcts",
        "repair_enabled=1",
        f"split={args.split}",
        f"model={repair_model}",
        f"model_slug={model_slug(repair_model)}",
        f"base_model={args.model}",
        f"source_run_dir={source_run_dir}",
        f"source_llm_price_model_key={source_price_key}",
        f"base_run_dir={run_dir}",
        f"repair_tree={tree_path}",
        f"repair_input={optimized_path}",
        f"outputs_dir={repair_dir}",
        "reuse_init_template=1",
        "reuse_view_select=1",
        "reuse_compose_match=1",
        f"reused_elapsed_s_init_template={reused_init_elapsed}",
        f"reused_elapsed_s_view_select={reused_view_elapsed}",
        f"reused_elapsed_s_compose_match={reused_compose_elapsed}",
        f"reused_tool_cost_compose_match={float(reused_compose_tool_cost)}",
        f"reused_elapsed_s_nomcts={random_stage_elapsed}",
        f"llm_repair_elapsed_s={repair_elapsed_llm}",
        f"actual_pipeline_elapsed_s_no_eval={repair_actual_no_eval}",
        f"actual_pipeline_elapsed_s_total={repair_actual_total}",
        f"pipeline_elapsed_s_no_eval={repair_comparable_no_eval}",
        f"pipeline_elapsed_s_total={repair_comparable_total}",
        f"eval_elapsed_s={repair_eval_elapsed}",
        f"LLM token usage (init_template): {init_total_usage.to_cost_line()}",
        f"LLM token usage (llm_repair): {repair_usage.to_cost_line()}",
        f"pipeline_tool_cost_total={float(reused_compose_tool_cost)}",
        f"pipeline_cost_usd_total={repair_pipeline_cost_total}",
        f"pipeline_total_tokens={repair_total_tokens}",
        f"actual_total_tokens={int(repair_usage.total_tokens)}",
        f"run_end={iso_now()}",
    ]
    repair_cost_lines = append_total_cost_block(
        lines=repair_cost_lines,
        total_model_key=combine_price_keys(source_price_key, resolve_price_key(repair_model)),
        total_tokens=repair_total_tokens,
        total_time_s=float(repair_comparable_no_eval),
        stage_prices={
            "llm_price_usd_init_template": base_price,
            "llm_price_usd_llm_repair": repair_price,
            "llm_price_usd_total": repair_total_price,
        },
    )
    repair_cost_txt.write_text("\n".join(repair_cost_lines) + "\n", encoding="utf-8")

    repair_run_metrics = {
        "metadata": {
            "task": "travel",
            "variant": "nomcts",
            "split": args.split,
            "base_model": args.model,
            "repair_model": repair_model,
            "source_run_dir": str(source_run_dir),
            "base_run_dir": str(run_dir),
            "repair_dir": str(repair_dir),
        },
        "stages": {
            "llm_repair": {
                "elapsed_s": repair_elapsed_llm,
                "token_usage": repair_usage.to_dict(),
                "price_usd": repair_price,
            },
            "eval": {
                "elapsed_s": repair_eval_elapsed,
                "token_usage": TokenUsage().to_dict(),
            },
        },
        "totals": {
            "actual_pipeline_elapsed_s_no_eval": repair_actual_no_eval,
            "actual_pipeline_elapsed_s_total": repair_actual_total,
            "pipeline_elapsed_s_no_eval": repair_comparable_no_eval,
            "pipeline_elapsed_s_total": repair_comparable_total,
            "total_tokens": repair_total_tokens,
            "actual_total_tokens": int(repair_usage.total_tokens),
            "llm_price_model_key": combine_price_keys(source_price_key, resolve_price_key(repair_model)),
            "llm_price_usd_total": repair_total_price,
            "tool_cost_total": float(reused_compose_tool_cost),
            "pipeline_cost_usd_total": repair_pipeline_cost_total,
        },
        "evaluation_before_repair": eval_metrics,
        "evaluation_after_repair": repair_eval_metrics,
    }
    write_json(repair_run_metrics_path, repair_run_metrics)
    print(f"Travel NoMCTS baseline complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
