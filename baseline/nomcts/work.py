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
    WORK_OUTPUT_ROOT,
    TokenUsage,
    append_total_cost_block,
    combine_price_keys,
    compute_price_usd,
    copy_if_exists,
    find_stage_usage,
    iso_now,
    iter_jsonl,
    load_init_template_usage_map,
    model_slug,
    python_executable,
    read_cost_float,
    read_cost_lines,
    read_cost_value,
    read_json,
    resolve_price_key,
    resolve_workers,
    run_subprocess,
    stable_seed,
    sum_mapping_values,
    sum_stage_elapsed_seconds,
    summarize_work_tool_calls,
    write_json,
    write_jsonl,
)
from planner.twin_track_multi import _effective_max_rounds
from task_helper.work.multi_round import make_adapter


WORK_QUERY_ROOT = (PROJECT_ROOT / "artifacts" / "input" / "work" / "dataset" / "queries_and_answers").resolve()


def _append_trimmed_pass_rates(pass_rates_path: Path, results_txt_path: Path) -> None:
    if not pass_rates_path.exists():
        return
    payload = read_json(pass_rates_path)
    if isinstance(payload, dict):
        payload.pop("constraints", None)
    with results_txt_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _copy_prediction_aliases(results_dir: Path, alias_dir: Path) -> None:
    alias_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(results_dir.glob("predictions_*.csv")):
        shutil.copy2(path, alias_dir / path.name)


def _evaluate_predictions(results_dir: Path, results_txt_path: Path) -> Tuple[float, Dict[str, object]]:
    overall_elapsed, _ = run_subprocess(
        cmd=[
            python_executable(),
            "-m",
            "task_helper.work.evaluation.calculate_all_metrics",
            "--predictions_dir",
            str(results_dir),
            "--ground_truth_dir",
            str(WORK_QUERY_ROOT),
        ],
        out_path=results_txt_path,
        header="========== WorkBench official-style metrics ==========",
    )
    pass_rates_elapsed, _ = run_subprocess(
        cmd=[
            python_executable(),
            "-m",
            "task_helper.work.evaluation.calculate_pass_rates_dir",
            "--predictions_dir",
            str(results_dir),
            "--ground_truth_dir",
            str(WORK_QUERY_ROOT),
            "--json_out",
            str(results_dir / "pass_rates.json"),
        ],
        out_path=results_txt_path,
        header="\n========== Constraint pass rates (local/global) ==========",
    )
    pass_rates_path = results_dir / "pass_rates.json"
    _append_trimmed_pass_rates(pass_rates_path, results_txt_path)
    payload = read_json(pass_rates_path) if pass_rates_path.exists() else {}
    return (overall_elapsed + pass_rates_elapsed), (payload if isinstance(payload, dict) else {})


def _build_summary_payload(
    *,
    variant: str,
    model_name: str,
    split: str,
    seed: int,
    pass_rates_payload: Mapping[str, object],
    token_usage: TokenUsage,
    total_price: float,
) -> Dict[str, object]:
    return {
        "variant": variant,
        "model_name": model_name,
        "split": split,
        "seed": int(seed),
        "n_examples_total": int(pass_rates_payload.get("n_examples", 0) or 0),
        "Local Pass Rate": pass_rates_payload.get("Local Pass Rate", 0.0),
        "Global Pass Rate": pass_rates_payload.get("Global Pass Rate", 0.0),
        "WorkBench Accuracy": pass_rates_payload.get("WorkBench Accuracy", 0.0),
        "WorkBench Exact Match": pass_rates_payload.get("WorkBench Exact Match", 0.0),
        "WorkBench Unwanted Side Effects": pass_rates_payload.get("WorkBench Unwanted Side Effects", 0.0),
        "total_prompt_tokens": int(token_usage.prompt_tokens),
        "total_completion_tokens": int(token_usage.completion_tokens),
        "total_tokens": int(token_usage.total_tokens),
        "total_prompt_cache_hit_tokens": int(token_usage.prompt_cache_hit_tokens),
        "total_prompt_cache_miss_tokens": int(token_usage.prompt_cache_miss_tokens),
        "total_price_usd": float(total_price),
    }


def _run_random_rollout(
    *,
    entry: Mapping[str, object],
    max_rounds: int,
    seed: int,
) -> Tuple[Dict[str, object], float, bool, float, float]:
    started_at = time.perf_counter()
    template = entry.get("template") if isinstance(entry.get("template"), dict) else {}
    effective_max_rounds = _effective_max_rounds(template, default_max_rounds=int(max_rounds))
    adapter = make_adapter(tree=dict(entry), config={"export_enriched_tree": True})
    rng = random.Random(seed)
    history = adapter.rollout(
        history=list(adapter.initial_history()),
        start_round=1,
        max_rounds=int(effective_max_rounds),
        rng=rng,
    )
    reward, score_details = adapter.score(history=history)
    details = dict(score_details)
    details["reward"] = float(reward)
    details["selector"] = "random_rollout"
    details["random_seed"] = int(seed)
    details["mcts_iterations_used"] = 0
    details["mcts_iterations_total"] = 0
    details["mcts_early_stop"] = False
    result = dict(adapter.render(history=history, details=details))
    notes = result.get("notes")
    if not isinstance(notes, dict):
        notes = {}
        result["notes"] = notes
    blob = notes.get("multi_round_mcts")
    if not isinstance(blob, dict):
        blob = {}
        notes["multi_round_mcts"] = blob
    blob["selector"] = "random_rollout"
    blob["source"] = "nomcts"
    blob["random_seed"] = int(seed)
    blob["max_rounds"] = int(effective_max_rounds)
    elapsed_s = time.perf_counter() - started_at
    return (
        result,
        float(elapsed_s),
        bool(blob.get("hard_pass", False)),
        float(blob.get("semantic_rate", 0.0)),
        float(blob.get("tool_cost", 0.0)),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-MCTS baseline for the Work benchmark.")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--semantic-threshold", type=float, default=0.8)
    parser.add_argument("--repair-model", type=str, default=None)
    parser.add_argument("--repair-workers", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=WORK_OUTPUT_ROOT)
    parser.add_argument("--source-run-dir", type=Path, default=None)
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
        notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
        blob = notes.get("multi_round_mcts") if isinstance(notes.get("multi_round_mcts"), dict) else {}
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
                "query": str(plan.get("query") or ""),
                "domains": list(plan.get("domains") or []),
                "status": "ok",
                "time_s_total": float(stage_times["nomcts"]),
                "time_s_by_stage": stage_times,
                "token_usage_by_stage": stage_usage,
                "llm_calls": int(init_usage.calls),
                "prompt_cache_hit_tokens": int(init_usage.prompt_cache_hit_tokens),
                "prompt_cache_miss_tokens": int(init_usage.prompt_cache_miss_tokens),
                "completion_tokens": int(init_usage.completion_tokens),
                "total_tokens": int(init_usage.total_tokens),
                "tool_calls": int(blob.get("total_calls", 0)),
                "tool_cost": float(blob.get("tool_cost", 0.0)),
                "hard_pass": blob.get("hard_pass"),
                "semantic_rate": blob.get("semantic_rate"),
                "rounds": blob.get("rounds"),
                "selector": blob.get("selector"),
                "random_seed": blob.get("random_seed"),
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
        notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
        blob = notes.get("multi_round_mcts") if isinstance(notes.get("multi_round_mcts"), dict) else {}
        repair = notes.get("llm_repair") if isinstance(notes.get("llm_repair"), dict) else {}
        init_usage = init_usage_by_template.get(template_id, TokenUsage())
        repair_usage = TokenUsage.from_mapping(repair.get("usage") if isinstance(repair, dict) else None, calls=1 if repair else 0)
        rows.append(
            {
                "template_id": template_id,
                "query": str(plan.get("query") or ""),
                "domains": list(plan.get("domains") or []),
                "status": str(repair.get("status") or "not_attempted"),
                "time_s_total": 0.0,
                "time_s_by_stage": {
                    "init_template": 0.0,
                    "llm_repair": 0.0,
                },
                "token_usage_by_stage": {
                    "init_template": init_usage.to_dict(),
                    "llm_repair": repair_usage.to_dict(),
                },
                "llm_calls": int(init_usage.calls + repair_usage.calls),
                "prompt_cache_hit_tokens": int(init_usage.prompt_cache_hit_tokens + repair_usage.prompt_cache_hit_tokens),
                "prompt_cache_miss_tokens": int(init_usage.prompt_cache_miss_tokens + repair_usage.prompt_cache_miss_tokens),
                "completion_tokens": int(init_usage.completion_tokens + repair_usage.completion_tokens),
                "total_tokens": int(init_usage.total_tokens + repair_usage.total_tokens),
                "tool_calls": int(blob.get("total_calls", 0)),
                "tool_cost": float(blob.get("tool_cost", 0.0)),
                "hard_pass": blob.get("hard_pass"),
                "semantic_rate": blob.get("semantic_rate"),
                "repair_workbench_eval": repair.get("workbench_eval"),
            }
        )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    repair_model = args.repair_model or args.model
    source_run_dir = (
        args.source_run_dir.expanduser().resolve()
        if args.source_run_dir is not None
        else (PROJECT_ROOT / "artifacts" / "output" / "work" / f"{model_slug(args.model)}_{args.split}").resolve()
    )
    run_dir = args.output_root.expanduser().resolve() / f"{model_slug(args.model)}_{args.split}"
    run_dir.mkdir(parents=True, exist_ok=True)

    init_templates_path = run_dir / f"init_templates_{args.split}.jsonl"
    match_path = run_dir / f"match_{args.split}.json"
    tree_path = run_dir / f"tree_{args.split}.json"
    optimized_path = run_dir / f"optimized_{args.split}.jsonl"
    tree_enriched_path = run_dir / f"tree_enriched_{args.split}.json"
    results_dir = run_dir / "results"
    predictions_alias_dir = run_dir / "predictions"
    results_txt_path = run_dir / "results.txt"
    cost_txt_path = run_dir / "cost.txt"
    run_metrics_path = run_dir / "run_metrics.json"
    per_query_metrics_path = run_dir / "per_query_metrics.jsonl"
    summary_json_path = results_dir / "summary.json"

    source_init_templates = source_run_dir / f"init_templates_{args.split}.jsonl"
    source_match = source_run_dir / f"match_{args.split}.json"
    source_tree = source_run_dir / f"tree_{args.split}.json"
    source_cost = source_run_dir / "cost.txt"
    for src in (source_init_templates, source_match, source_tree, source_cost):
        if not src.exists():
            raise FileNotFoundError(f"Required source artifact not found: {src}")

    copy_if_exists(source_init_templates, init_templates_path)
    copy_if_exists(source_match, match_path)
    copy_if_exists(source_tree, tree_path)

    results_dir.mkdir(parents=True, exist_ok=True)
    predictions_alias_dir.mkdir(parents=True, exist_ok=True)
    results_txt_path.write_text(
        f"==============================\nWorkBench evaluation: {run_dir}\nmodel={args.model} split={args.split} variant=nomcts\n==============================\n",
        encoding="utf-8",
    )

    source_cost_lines = read_cost_lines(source_cost)
    reused_init_elapsed = sum_stage_elapsed_seconds(source_cost_lines, "init_template")
    reused_view_elapsed = sum_stage_elapsed_seconds(source_cost_lines, "view_select")
    reused_compose_elapsed = sum_stage_elapsed_seconds(source_cost_lines, "compose_match")
    source_price_key = read_cost_value(source_cost_lines, "llm_price_model_key") or resolve_price_key(args.model)

    init_usage_by_template, init_total_usage = load_init_template_usage_map(init_templates_path)
    entries_payload = read_json(tree_path)
    if not isinstance(entries_payload, dict):
        raise TypeError(f"Expected tree JSON object: {tree_path}")
    template_ids = list(entries_payload.keys())
    if args.limit > 0:
        template_ids = template_ids[: int(args.limit)]

    workers = resolve_workers(args.workers, len(template_ids))
    run_start_iso = iso_now()
    nomcts_stage_start = time.perf_counter()

    results_by_template: Dict[str, Dict[str, object]] = {}
    elapsed_by_template: Dict[str, float] = {}

    if workers <= 1:
        for index, template_id in enumerate(template_ids, start=1):
            seed = stable_seed(args.seed, template_id)
            record, elapsed_s, hard_pass, semantic_rate, total_cost = _run_random_rollout(
                entry=entries_payload[template_id],
                max_rounds=int(args.max_rounds),
                seed=seed,
            )
            results_by_template[template_id] = record
            elapsed_by_template[template_id] = float(elapsed_s)
            print(
                f"[{index}/{len(template_ids)}] template_id={template_id} cost={total_cost:.3f} sem={semantic_rate:.3f} hard={hard_pass}",
                flush=True,
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_template = {
                pool.submit(
                    _run_random_rollout,
                    entry=entries_payload[template_id],
                    max_rounds=int(args.max_rounds),
                    seed=stable_seed(args.seed, template_id),
                ): template_id
                for template_id in template_ids
            }
            completed = 0
            for future in concurrent.futures.as_completed(future_to_template):
                template_id = future_to_template[future]
                try:
                    record, elapsed_s, hard_pass, semantic_rate, total_cost = future.result()
                except Exception:
                    message = (
                        f"[nomcts][work][fatal] template_id={template_id}\n"
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
                    f"[{completed}/{len(template_ids)}] template_id={template_id} cost={total_cost:.3f} sem={semantic_rate:.3f} hard={hard_pass}",
                    flush=True,
                )

    nomcts_elapsed = time.perf_counter() - nomcts_stage_start
    ordered_records = [results_by_template[template_id] for template_id in template_ids]
    with optimized_path.open("w", encoding="utf-8") as fp:
        for record in ordered_records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    enriched_payload = {}
    for record in ordered_records:
        template_id = str(record.get("template_id") or "")
        notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
        blob = notes.get("multi_round_mcts") if isinstance(notes.get("multi_round_mcts"), dict) else {}
        tree_enriched = blob.get("tree_enriched")
        if template_id and tree_enriched is not None:
            enriched_payload[template_id] = tree_enriched
    tree_enriched_path.write_text(json.dumps(enriched_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    write_jsonl(
        per_query_metrics_path,
        _build_per_query_metrics(
            records=ordered_records,
            elapsed_by_template=elapsed_by_template,
            init_usage_by_template=init_usage_by_template,
        ),
    )

    print("Converting no-MCTS optimized output to Work predictions...", flush=True)
    convert_elapsed, _ = run_subprocess(
        cmd=[
            python_executable(),
            "task_helper/work/evaluation/convert_optimized_to_predictions.py",
            "--optimized",
            str(optimized_path),
            "--out-dir",
            str(results_dir),
        ],
        out_path=results_txt_path,
    )
    _copy_prediction_aliases(results_dir, predictions_alias_dir)

    eval_elapsed = 0.0
    pass_rates_payload: Dict[str, object] = {}
    if not args.skip_eval:
        print("Running Work evaluation for no-repair output...", flush=True)
        eval_elapsed, pass_rates_payload = _evaluate_predictions(results_dir, results_txt_path)

    comparable_no_eval = reused_init_elapsed + reused_view_elapsed + reused_compose_elapsed + nomcts_elapsed + convert_elapsed
    comparable_total = comparable_no_eval + eval_elapsed
    actual_no_eval = nomcts_elapsed + convert_elapsed
    actual_total = actual_no_eval + eval_elapsed
    tool_call_summary = summarize_work_tool_calls(optimized_path)
    init_price = read_cost_float(source_cost_lines, "llm_price_usd_init_template")
    if init_price is None:
        init_price = compute_price_usd(model_name=args.model, usage=init_total_usage)
    total_price = sum_mapping_values([init_price])
    price_key = source_price_key

    summary_payload = _build_summary_payload(
        variant="nomcts",
        model_name=args.model,
        split=args.split,
        seed=int(args.seed),
        pass_rates_payload=pass_rates_payload,
        token_usage=init_total_usage,
        total_price=total_price,
    )
    summary_json_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    cost_lines = [
        f"run_start={run_start_iso}",
        "task=work",
        "variant=nomcts",
        f"split={args.split}",
        f"model={args.model}",
        f"model_slug={model_slug(args.model)}",
        f"seed={int(args.seed)}",
        f"workers={args.workers} (effective={workers})",
        f"max_rounds={int(args.max_rounds)}",
        f"source_run_dir={source_run_dir}",
        f"source_llm_price_model_key={source_price_key}",
        f"outputs_dir={run_dir}",
        f"results_dir={results_dir}",
        f"queries_total={len(template_ids)}",
        f"queries_processed={len(ordered_records)}",
        f"limit={int(args.limit)}",
        "reuse_init_template=1",
        "reuse_view_select=1",
        "reuse_compose_match=1",
        f"reused_elapsed_s_init_template={reused_init_elapsed}",
        f"reused_elapsed_s_view_select={reused_view_elapsed}",
        f"reused_elapsed_s_compose_match={reused_compose_elapsed}",
        f"nomcts_elapsed_s={nomcts_elapsed}",
        f"convert_elapsed_s={convert_elapsed}",
        f"actual_pipeline_elapsed_s_no_eval={actual_no_eval}",
        f"actual_pipeline_elapsed_s_total={actual_total}",
        f"pipeline_elapsed_s_no_eval={comparable_no_eval}",
        f"pipeline_elapsed_s_total={comparable_total}",
        f"eval_elapsed_s={eval_elapsed}",
        f"LLM token usage (init_template): {init_total_usage.to_cost_line()}",
        f"LLM token usage (nomcts): {TokenUsage().to_cost_line()}",
        (
            "Tool call usage (optimized): "
            f"calls_total={int(tool_call_summary['calls_total'])} "
            f"calls_check={int(tool_call_summary['calls_check'])} "
            f"calls_do={int(tool_call_summary['calls_do'])} "
            f"by_domain={json.dumps(tool_call_summary['by_domain'], ensure_ascii=False, sort_keys=True)}"
        ),
        f"tool_calls_total_calls={int(tool_call_summary['calls_total'])}",
        f"pipeline_total_tokens={int(init_total_usage.total_tokens)}",
        "actual_total_tokens=0",
        f"run_end={iso_now()}",
        f"summary: time_s_total={comparable_total} tokens_total={int(init_total_usage.total_tokens)} tool_calls_total={int(tool_call_summary['calls_total'])}",
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
            "task": "work",
            "variant": "nomcts",
            "split": args.split,
            "model": args.model,
            "model_slug": model_slug(args.model),
            "seed": int(args.seed),
            "workers_requested": int(args.workers),
            "workers_effective": int(workers),
            "max_rounds": int(args.max_rounds),
            "source_run_dir": str(source_run_dir),
            "outputs_dir": str(run_dir),
            "results_dir": str(results_dir),
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
            },
            "nomcts": {
                "elapsed_s": nomcts_elapsed,
                "elapsed_s_sum_per_query": sum(float(value) for value in elapsed_by_template.values()),
                "token_usage": TokenUsage().to_dict(),
                "selector": "random_rollout",
            },
            "convert": {
                "elapsed_s": convert_elapsed,
                "token_usage": TokenUsage().to_dict(),
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
        },
        "tool_usage": tool_call_summary,
        "evaluation": pass_rates_payload,
    }
    write_json(run_metrics_path, run_metrics)

    if not args.run_repair:
        print(f"Work NoMCTS baseline complete: {run_dir}", flush=True)
        return

    repair_dir = run_dir / "repair"
    repair_dir.mkdir(parents=True, exist_ok=True)
    repair_results_dir = repair_dir / "results"
    repair_predictions_alias_dir = repair_dir / "predictions"
    repair_results_dir.mkdir(parents=True, exist_ok=True)
    repair_predictions_alias_dir.mkdir(parents=True, exist_ok=True)
    repair_results_txt = repair_dir / "results.txt"
    repair_results_txt.write_text(
        f"==============================\nWorkBench evaluation (repair): {repair_dir}\nbase_model={args.model} repair_model={repair_model} split={args.split} variant=nomcts\n==============================\n",
        encoding="utf-8",
    )
    repair_cost_txt = repair_dir / "cost.txt"
    repair_jsonl = repair_dir / f"repaired_{args.split}_{model_slug(repair_model)}.jsonl"
    repair_log_path = repair_dir / "llm_repair.log"
    repair_run_metrics_path = repair_dir / "run_metrics.json"
    repair_per_query_metrics_path = repair_dir / "per_query_metrics.jsonl"
    repair_summary_json_path = repair_results_dir / "summary.json"

    repair_tree_path = tree_enriched_path if tree_enriched_path.exists() else tree_path

    print("Running single-invocation Work LLM repair...", flush=True)
    repair_elapsed_llm, repair_output = run_subprocess(
        cmd=[
            python_executable(),
            "-m",
            "planner.llm_repair",
            "--task",
            "work",
            "--tree",
            str(repair_tree_path),
            "--input",
            str(optimized_path),
            "--model",
            str(repair_model),
            "--semantic-threshold",
            str(args.semantic_threshold),
            "--out",
            str(repair_jsonl),
            "--workers",
            str(resolve_workers(args.repair_workers, len(template_ids))),
            "--workbench-only-incorrect",
        ],
        out_path=repair_log_path,
    )
    repair_usage = find_stage_usage(repair_output.splitlines(), "llm_repair")

    print("Converting repaired Work output to predictions...", flush=True)
    repair_convert_elapsed, _ = run_subprocess(
        cmd=[
            python_executable(),
            "task_helper/work/evaluation/convert_optimized_to_predictions.py",
            "--optimized",
            str(repair_jsonl),
            "--out-dir",
            str(repair_results_dir),
        ],
        out_path=repair_results_txt,
    )
    _copy_prediction_aliases(repair_results_dir, repair_predictions_alias_dir)

    repair_eval_elapsed = 0.0
    repair_pass_rates_payload: Dict[str, object] = {}
    if not args.skip_eval:
        print("Running Work evaluation for repaired output...", flush=True)
        repair_eval_elapsed, repair_pass_rates_payload = _evaluate_predictions(repair_results_dir, repair_results_txt)

    repair_records = list(iter_jsonl(repair_jsonl))
    write_jsonl(
        repair_per_query_metrics_path,
        _build_repair_per_query_metrics(records=repair_records, init_usage_by_template=init_usage_by_template),
    )
    repair_summary_payload = _build_summary_payload(
        variant="nomcts_repair",
        model_name=repair_model,
        split=args.split,
        seed=int(args.seed),
        pass_rates_payload=repair_pass_rates_payload,
        token_usage=TokenUsage(
            calls=int(init_total_usage.calls + repair_usage.calls),
            prompt_cache_hit_tokens=int(init_total_usage.prompt_cache_hit_tokens + repair_usage.prompt_cache_hit_tokens),
            prompt_cache_miss_tokens=int(init_total_usage.prompt_cache_miss_tokens + repair_usage.prompt_cache_miss_tokens),
            completion_tokens=int(init_total_usage.completion_tokens + repair_usage.completion_tokens),
            total_tokens=int(init_total_usage.total_tokens + repair_usage.total_tokens),
            prompt_tokens=int(init_total_usage.prompt_tokens + repair_usage.prompt_tokens),
            approx_calls=int(init_total_usage.approx_calls + repair_usage.approx_calls),
        ),
        total_price=sum_mapping_values(
            [
                compute_price_usd(model_name=args.model, usage=init_total_usage),
                compute_price_usd(model_name=repair_model, usage=repair_usage),
            ]
        ),
    )
    repair_summary_json_path.write_text(json.dumps(repair_summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    repair_actual_no_eval = repair_elapsed_llm + repair_convert_elapsed
    repair_actual_total = repair_actual_no_eval + repair_eval_elapsed
    repair_comparable_no_eval = comparable_no_eval + repair_elapsed_llm + repair_convert_elapsed
    repair_comparable_total = repair_comparable_no_eval + repair_eval_elapsed
    base_price = init_price
    repair_price = compute_price_usd(model_name=repair_model, usage=repair_usage)
    repair_total_price = sum_mapping_values([base_price, repair_price])
    repair_total_tokens = int(init_total_usage.total_tokens + repair_usage.total_tokens)

    repair_cost_lines = [
        f"run_start={iso_now()}",
        "task=work",
        "variant=nomcts",
        "repair_enabled=1",
        f"split={args.split}",
        f"model={repair_model}",
        f"model_slug={model_slug(repair_model)}",
        f"base_model={args.model}",
        f"base_run_dir={run_dir}",
        f"source_llm_price_model_key={source_price_key}",
        f"repair_tree={repair_tree_path}",
        f"repair_input={optimized_path}",
        f"outputs_dir={repair_dir}",
        "reuse_init_template=1",
        "reuse_view_select=1",
        "reuse_compose_match=1",
        f"reused_elapsed_s_init_template={reused_init_elapsed}",
        f"reused_elapsed_s_view_select={reused_view_elapsed}",
        f"reused_elapsed_s_compose_match={reused_compose_elapsed}",
        f"reused_elapsed_s_nomcts={nomcts_elapsed}",
        f"reused_elapsed_s_convert={convert_elapsed}",
        f"llm_repair_elapsed_s={repair_elapsed_llm}",
        f"convert_elapsed_s={repair_convert_elapsed}",
        f"actual_pipeline_elapsed_s_no_eval={repair_actual_no_eval}",
        f"actual_pipeline_elapsed_s_total={repair_actual_total}",
        f"pipeline_elapsed_s_no_eval={repair_comparable_no_eval}",
        f"pipeline_elapsed_s_total={repair_comparable_total}",
        f"eval_elapsed_s={repair_eval_elapsed}",
        f"LLM token usage (init_template): {init_total_usage.to_cost_line()}",
        f"LLM token usage (llm_repair): {repair_usage.to_cost_line()}",
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
            "task": "work",
            "variant": "nomcts",
            "split": args.split,
            "base_model": args.model,
            "repair_model": repair_model,
            "base_run_dir": str(run_dir),
            "repair_dir": str(repair_dir),
        },
        "stages": {
            "llm_repair": {
                "elapsed_s": repair_elapsed_llm,
                "token_usage": repair_usage.to_dict(),
                "price_usd": repair_price,
            },
            "convert": {
                "elapsed_s": repair_convert_elapsed,
                "token_usage": TokenUsage().to_dict(),
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
        },
        "evaluation_before_repair": pass_rates_payload,
        "evaluation_after_repair": repair_pass_rates_payload,
    }
    write_json(repair_run_metrics_path, repair_run_metrics)
    print(f"Work NoMCTS baseline complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
