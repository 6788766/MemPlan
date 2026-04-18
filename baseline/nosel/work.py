from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from baseline.nomcts.common import (
    PROJECT_ROOT,
    TokenUsage,
    append_total_cost_block,
    combine_price_keys,
    compute_price_usd,
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
    run_subprocess,
    sum_mapping_values,
    sum_stage_elapsed_seconds,
    summarize_work_tool_calls,
    write_json,
    write_jsonl,
)
from baseline.nosel.common import WORK_OUTPUT_ROOT, build_empty_match_payload, write_work_view_metrics


WORK_QUERY_ROOT = (PROJECT_ROOT / "artifacts" / "input" / "work" / "dataset" / "queries_and_answers").resolve()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-selection baseline for the Work benchmark.")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mcts-iterations", type=int, default=100)
    parser.add_argument("--mcts-uct-c", type=float, default=1.4)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--semantic-threshold", type=float, default=0.8)
    parser.add_argument("--repair-model", type=str, default=None)
    parser.add_argument("--repair-workers", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=WORK_OUTPUT_ROOT)
    parser.add_argument("--source-run-dir", type=Path, default=None)
    parser.add_argument("--run-repair", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args(argv)


def _append_trimmed_pass_rates(pass_rates_path: Path, results_txt_path: Path) -> Dict[str, object]:
    payload = read_json(pass_rates_path) if pass_rates_path.exists() else {}
    if isinstance(payload, dict):
        payload.pop("constraints", None)
        with results_txt_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        return payload
    return {}


def _copy_prediction_aliases(results_dir: Path, alias_dir: Path) -> None:
    alias_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(results_dir.glob("predictions_*.csv")):
        shutil.copy2(path, alias_dir / path.name)


def _evaluate_predictions(results_dir: Path, results_txt_path: Path) -> tuple[float, Dict[str, object]]:
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
    payload = _append_trimmed_pass_rates(results_dir / "pass_rates.json", results_txt_path)
    return (overall_elapsed + pass_rates_elapsed), payload


def _build_summary_payload(
    *,
    variant: str,
    model_name: str,
    split: str,
    pass_rates_payload: Mapping[str, object],
    token_usage: TokenUsage,
    total_price: float,
) -> Dict[str, object]:
    return {
        "variant": variant,
        "model_name": model_name,
        "split": split,
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


def _build_per_query_metrics(
    *,
    init_usage_by_template: Mapping[str, TokenUsage],
    optimized_records: List[Mapping[str, object]],
    match_payload: Mapping[str, object],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for record in optimized_records:
        template_id = str(record.get("template_id") or "")
        plan = record.get("plan") if isinstance(record.get("plan"), dict) else {}
        notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
        blob = notes.get("multi_round_mcts") if isinstance(notes.get("multi_round_mcts"), dict) else {}
        match_entry = match_payload.get(template_id) if isinstance(match_payload.get(template_id), dict) else {}
        init_usage = init_usage_by_template.get(template_id, TokenUsage())
        rows.append(
            {
                "template_id": template_id,
                "query": str(plan.get("query") or ""),
                "domains": list(plan.get("domains") or []),
                "status": "ok",
                "per_query_time_unavailable": True,
                "time_s_total": None,
                "time_s_by_stage": {
                    "init_template": None,
                    "nosel_prepare": None,
                    "compose_match": None,
                    "mcts": None,
                    "convert": None,
                },
                "token_usage_by_stage": {
                    "init_template": init_usage.to_dict(),
                    "nosel_prepare": TokenUsage().to_dict(),
                    "compose_match": TokenUsage().to_dict(),
                    "mcts": TokenUsage().to_dict(),
                    "convert": TokenUsage().to_dict(),
                },
                "llm_calls": int(init_usage.calls),
                "prompt_cache_hit_tokens": int(init_usage.prompt_cache_hit_tokens),
                "prompt_cache_miss_tokens": int(init_usage.prompt_cache_miss_tokens),
                "completion_tokens": int(init_usage.completion_tokens),
                "total_tokens": int(init_usage.total_tokens),
                "selected_views_total": 0,
                "selected_views_memory": 0,
                "selected_views_tool": 0,
                "uncovered_required_edges": len(match_entry.get("uncovered_edges") or []),
                "tool_calls": int(blob.get("total_calls", 0)),
                "tool_cost": float(blob.get("tool_cost", 0.0)),
                "hard_pass": blob.get("hard_pass"),
                "semantic_rate": blob.get("semantic_rate"),
            }
        )
    return rows


def _build_repair_per_query_metrics(
    *,
    init_usage_by_template: Mapping[str, TokenUsage],
    repaired_records: List[Mapping[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for record in repaired_records:
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
                "per_query_time_unavailable": True,
                "time_s_total": None,
                "time_s_by_stage": {
                    "init_template": None,
                    "llm_repair": None,
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
    results_dir = run_dir / "results"
    predictions_alias_dir = run_dir / "predictions"
    mcts_tree_path = run_dir / f"mcts_tree_{args.split}.json"
    tree_enriched_path = run_dir / f"tree_enriched_{args.split}.json"
    results_txt_path = run_dir / "results.txt"
    cost_txt_path = run_dir / "cost.txt"
    run_metrics_path = run_dir / "run_metrics.json"
    per_query_metrics_path = run_dir / "per_query_metrics.jsonl"
    summary_json_path = results_dir / "summary.json"

    source_init_templates = source_run_dir / f"init_templates_{args.split}.jsonl"
    source_match = source_run_dir / f"match_{args.split}.json"
    source_cost = source_run_dir / "cost.txt"
    for src in (source_init_templates, source_match, source_cost):
        if not src.exists():
            raise FileNotFoundError(f"Required source artifact not found: {src}")

    source_init_records = list(iter_jsonl(source_init_templates))
    if args.limit > 0:
        source_init_records = source_init_records[: int(args.limit)]
    write_jsonl(init_templates_path, source_init_records)
    allowed_template_ids = {str(obj.get("template_id") or "") for obj in source_init_records}
    source_match_payload = read_json(source_match)
    if not isinstance(source_match_payload, Mapping):
        raise TypeError(f"Expected match JSON object: {source_match}")
    filtered_source_match_payload = {
        str(template_id): value
        for template_id, value in source_match_payload.items()
        if str(template_id) in allowed_template_ids
    }
    match_payload = build_empty_match_payload(filtered_source_match_payload)
    match_path.write_text(json.dumps(match_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    source_cost_lines = read_cost_lines(source_cost)
    reused_init_elapsed = 0.0
    for line in source_cost_lines:
        if "] END init_template " in line and "elapsed_s=" in line:
            try:
                reused_init_elapsed += float(line.split("elapsed_s=", 1)[1].split()[0].strip())
            except Exception:
                continue
    reused_view_elapsed = sum_stage_elapsed_seconds(source_cost_lines, "view_select")
    source_price_key = read_cost_value(source_cost_lines, "llm_price_model_key") or resolve_price_key(args.model)

    init_usage_by_template, init_total_usage = load_init_template_usage_map(init_templates_path)
    results_dir.mkdir(parents=True, exist_ok=True)
    predictions_alias_dir.mkdir(parents=True, exist_ok=True)
    results_txt_path.write_text(
        f"==============================\nWorkBench evaluation: {run_dir}\nmodel={args.model} split={args.split} variant=nosel\n==============================\n",
        encoding="utf-8",
    )

    run_start_iso = iso_now()
    nosel_prepare_elapsed = 0.0

    compose_cmd = [
        python_executable(),
        "-m",
        "planner.compose_match_multi",
        "--task",
        "work",
        "--split",
        args.split,
        "--templates",
        str(init_templates_path),
        "--match",
        str(match_path),
        "--out",
        str(tree_path),
        "--no-memory-graph-extension",
    ]
    print("Running Work ComposeMatch without selected views...", flush=True)
    compose_elapsed, _ = run_subprocess(cmd=compose_cmd, out_path=results_txt_path)

    mcts_cmd = [
        python_executable(),
        "-m",
        "planner.twin_track_multi",
        "--task",
        "work",
        "--tree",
        str(tree_path),
        "--max-rounds",
        str(args.max_rounds),
        "--workers",
        str(args.workers),
        "--iterations",
        str(args.mcts_iterations),
        "--uct-c",
        str(args.mcts_uct_c),
        "--semantic-tolerance",
        str(args.semantic_threshold),
        "--out",
        str(optimized_path),
        "--mcts-tree-out",
        str(mcts_tree_path),
        "--enriched-tree-out",
        str(tree_enriched_path),
    ]
    print("Running Work MCTS after no-selection ComposeMatch...", flush=True)
    mcts_elapsed, _ = run_subprocess(cmd=mcts_cmd, out_path=results_txt_path)

    convert_cmd = [
        python_executable(),
        "task_helper/work/evaluation/convert_optimized_to_predictions.py",
        "--optimized",
        str(optimized_path),
        "--out-dir",
        str(results_dir),
    ]
    print("Converting Work no-selection output to predictions...", flush=True)
    convert_elapsed, _ = run_subprocess(cmd=convert_cmd, out_path=results_txt_path)
    _copy_prediction_aliases(results_dir, predictions_alias_dir)

    optimized_records = list(iter_jsonl(optimized_path))
    write_jsonl(
        per_query_metrics_path,
        _build_per_query_metrics(
            init_usage_by_template=init_usage_by_template,
            optimized_records=optimized_records,
            match_payload=match_payload,
        ),
    )

    eval_elapsed = 0.0
    pass_rates_payload: Dict[str, object] = {}
    if not args.skip_eval:
        print("Running Work evaluation for no-selection output...", flush=True)
        eval_elapsed, pass_rates_payload = _evaluate_predictions(results_dir, results_txt_path)

    tool_call_summary = summarize_work_tool_calls(optimized_path)
    actual_no_eval = nosel_prepare_elapsed + compose_elapsed + mcts_elapsed + convert_elapsed
    actual_total = actual_no_eval + eval_elapsed
    comparable_no_eval = reused_init_elapsed + reused_view_elapsed + actual_no_eval
    comparable_total = comparable_no_eval + eval_elapsed
    init_price = read_cost_float(source_cost_lines, "llm_price_usd_init_template")
    if init_price is None:
        init_price = compute_price_usd(model_name=args.model, usage=init_total_usage)
    total_price = sum_mapping_values([init_price])
    price_key = source_price_key

    summary_payload = _build_summary_payload(
        variant="nosel",
        model_name=args.model,
        split=args.split,
        pass_rates_payload=pass_rates_payload,
        token_usage=init_total_usage,
        total_price=total_price,
    )
    summary_json_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    cost_lines = [
        f"run_start={run_start_iso}",
        "task=work",
        "variant=nosel",
        f"split={args.split}",
        f"model={args.model}",
        f"model_slug={model_slug(args.model)}",
        f"workers={args.workers}",
        f"source_run_dir={source_run_dir}",
        f"source_llm_price_model_key={source_price_key}",
        f"outputs_dir={run_dir}",
        f"results_dir={results_dir}",
        "reuse_init_template=1",
        "reuse_view_select=1",
        "run_view_select=0",
        f"queries_total={len(source_init_records)}",
        f"queries_processed={len(optimized_records)}",
        f"limit={int(args.limit)}",
        f"reused_elapsed_s_init_template={reused_init_elapsed}",
        f"reused_elapsed_s_view_select={reused_view_elapsed}",
        f"reused_llm_price_usd_init_template={init_price}",
        f"nosel_prepare_elapsed_s={nosel_prepare_elapsed}",
        f"compose_match_elapsed_s={compose_elapsed}",
        f"mcts_elapsed_s={mcts_elapsed}",
        f"convert_elapsed_s={convert_elapsed}",
        f"actual_pipeline_elapsed_s_no_eval={actual_no_eval}",
        f"actual_pipeline_elapsed_s_total={actual_total}",
        f"pipeline_elapsed_s_no_eval={comparable_no_eval}",
        f"pipeline_elapsed_s_total={comparable_total}",
        f"eval_elapsed_s={eval_elapsed}",
        f"LLM token usage (init_template): {init_total_usage.to_cost_line()}",
        f"LLM token usage (nosel): {TokenUsage().to_cost_line()}",
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
            "llm_price_usd_nosel": 0.0,
            "llm_price_usd_total": total_price,
        },
    )
    cost_txt_path.write_text("\n".join(cost_lines) + "\n", encoding="utf-8")

    run_metrics = {
        "metadata": {
            "task": "work",
            "variant": "nosel",
            "split": args.split,
            "model": args.model,
            "model_slug": model_slug(args.model),
            "workers_requested": int(args.workers),
            "source_run_dir": str(source_run_dir),
            "outputs_dir": str(run_dir),
            "results_dir": str(results_dir),
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
            "nosel_prepare": {
                "elapsed_s": nosel_prepare_elapsed,
                "token_usage": TokenUsage().to_dict(),
            },
            "compose_match": {
                "elapsed_s": compose_elapsed,
                "token_usage": TokenUsage().to_dict(),
            },
            "mcts": {
                "elapsed_s": mcts_elapsed,
                "token_usage": TokenUsage().to_dict(),
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
    write_work_view_metrics(run_dir=run_dir, split=args.split, model=args.model, model_slug=model_slug(args.model))

    if not args.run_repair:
        print(f"Work NoSel baseline complete: {run_dir}", flush=True)
        return

    repair_dir = run_dir / "repair"
    repair_results_dir = repair_dir / "results"
    repair_predictions_alias_dir = repair_dir / "predictions"
    repair_dir.mkdir(parents=True, exist_ok=True)
    repair_results_dir.mkdir(parents=True, exist_ok=True)
    repair_predictions_alias_dir.mkdir(parents=True, exist_ok=True)

    repair_results_txt = repair_dir / "results.txt"
    repair_results_txt.write_text(
        f"==============================\nWorkBench evaluation (repair): {repair_dir}\nbase_model={args.model} repair_model={repair_model} split={args.split} variant=nosel\n==============================\n",
        encoding="utf-8",
    )
    repair_log_path = repair_dir / "llm_repair.log"
    repair_cost_txt = repair_dir / "cost.txt"
    repair_jsonl = repair_dir / f"repaired_{args.split}_{model_slug(repair_model)}.jsonl"
    repair_run_metrics_path = repair_dir / "run_metrics.json"
    repair_per_query_metrics_path = repair_dir / "per_query_metrics.jsonl"
    repair_summary_json_path = repair_results_dir / "summary.json"

    repair_tree_path = tree_enriched_path if tree_enriched_path.exists() else tree_path
    repair_cmd = [
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
        str(args.repair_workers),
        "--workbench-only-incorrect",
    ]
    print("Running single-round Work repair for no-selection output...", flush=True)
    repair_elapsed_llm, repair_output = run_subprocess(cmd=repair_cmd, out_path=repair_log_path)
    repair_usage = find_stage_usage(repair_output.splitlines(), "llm_repair")

    repair_convert_cmd = [
        python_executable(),
        "task_helper/work/evaluation/convert_optimized_to_predictions.py",
        "--optimized",
        str(repair_jsonl),
        "--out-dir",
        str(repair_results_dir),
    ]
    print("Converting repaired Work no-selection output...", flush=True)
    repair_convert_elapsed, _ = run_subprocess(cmd=repair_convert_cmd, out_path=repair_results_txt)
    _copy_prediction_aliases(repair_results_dir, repair_predictions_alias_dir)

    repair_records = list(iter_jsonl(repair_jsonl))
    write_jsonl(
        repair_per_query_metrics_path,
        _build_repair_per_query_metrics(
            init_usage_by_template=init_usage_by_template,
            repaired_records=repair_records,
        ),
    )

    repair_eval_elapsed = 0.0
    repair_pass_rates_payload: Dict[str, object] = {}
    if not args.skip_eval:
        print("Running Work evaluation for repaired no-selection output...", flush=True)
        repair_eval_elapsed, repair_pass_rates_payload = _evaluate_predictions(repair_results_dir, repair_results_txt)

    base_price = init_price
    repair_price = compute_price_usd(model_name=repair_model, usage=repair_usage)
    repair_total_price = sum_mapping_values([base_price, repair_price])
    repair_total_tokens = int(init_total_usage.total_tokens + repair_usage.total_tokens)
    repair_actual_no_eval = repair_elapsed_llm + repair_convert_elapsed
    repair_actual_total = repair_actual_no_eval + repair_eval_elapsed
    repair_comparable_no_eval = comparable_no_eval + repair_elapsed_llm + repair_convert_elapsed
    repair_comparable_total = repair_comparable_no_eval + repair_eval_elapsed

    repair_summary_payload = _build_summary_payload(
        variant="nosel_repair",
        model_name=repair_model,
        split=args.split,
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
        total_price=repair_total_price,
    )
    repair_summary_json_path.write_text(json.dumps(repair_summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    repair_cost_lines = [
        f"run_start={iso_now()}",
        "task=work",
        "variant=nosel",
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
        f"reused_elapsed_s_init_template={reused_init_elapsed}",
        f"reused_elapsed_s_view_select={reused_view_elapsed}",
        f"reused_llm_price_usd_init_template={base_price}",
        f"reused_elapsed_s_nosel_prepare={nosel_prepare_elapsed}",
        f"reused_elapsed_s_compose_match={compose_elapsed}",
        f"reused_elapsed_s_mcts={mcts_elapsed}",
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
            "variant": "nosel",
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
    print(f"Work NoSel baseline complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
