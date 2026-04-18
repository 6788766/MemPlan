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
    parse_travel_eval_metrics,
    python_executable,
    read_cost_float,
    read_cost_lines,
    read_cost_value,
    read_json,
    resolve_price_key,
    run_subprocess,
    sum_mapping_values,
    sum_stage_elapsed_seconds,
    summarize_travel_tool_calls,
    write_json,
    write_jsonl,
)
from baseline.nosel.common import TRAVEL_OUTPUT_ROOT, build_empty_match_payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-selection baseline for the Travel benchmark.")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max-tool-candidates", type=int, default=5)
    parser.add_argument("--mcts-iterations", type=int, default=150)
    parser.add_argument("--mcts-uct-c", type=float, default=1.4)
    parser.add_argument("--semantic-threshold", type=float, default=0.8)
    parser.add_argument("--repair-model", type=str, default=None)
    parser.add_argument("--repair-workers", type=int, default=0)
    parser.add_argument("--repair-max-slot-candidates", type=int, default=50)
    parser.add_argument("--output-root", type=Path, default=TRAVEL_OUTPUT_ROOT)
    parser.add_argument("--source-run-dir", type=Path, default=None)
    parser.add_argument("--run-repair", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args(argv)


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
        query = str(plan.get("query") or "")
        match_entry = match_payload.get(template_id) if isinstance(match_payload.get(template_id), dict) else {}
        notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
        mcts = notes.get("mcts") if isinstance(notes.get("mcts"), dict) else {}
        init_usage = init_usage_by_template.get(template_id, TokenUsage())
        rows.append(
            {
                "template_id": template_id,
                "query": query,
                "status": "ok",
                "per_query_time_unavailable": True,
                "time_s_total": None,
                "time_s_by_stage": {
                    "init_template": None,
                    "nosel_prepare": None,
                    "compose_match": None,
                    "mcts": None,
                },
                "token_usage_by_stage": {
                    "init_template": init_usage.to_dict(),
                    "nosel_prepare": TokenUsage().to_dict(),
                    "compose_match": TokenUsage().to_dict(),
                    "mcts": TokenUsage().to_dict(),
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
                "hard_pass": mcts.get("hard_pass"),
                "semantic_rate": mcts.get("semantic_rate"),
                "selected_plan_cost": mcts.get("total_cost"),
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
        repair = notes.get("llm_repair") if isinstance(notes.get("llm_repair"), dict) else {}
        mcts = notes.get("mcts") if isinstance(notes.get("mcts"), dict) else {}
        init_usage = init_usage_by_template.get(template_id, TokenUsage())
        repair_usage = TokenUsage.from_mapping(repair.get("usage") if isinstance(repair, dict) else None, calls=1 if repair else 0)
        rows.append(
            {
                "template_id": template_id,
                "query": str(plan.get("query") or ""),
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
                "hard_pass": mcts.get("hard_pass"),
                "semantic_rate": mcts.get("semantic_rate"),
                "repair_before": repair.get("before"),
                "repair_after": repair.get("after"),
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
    optimized_path = run_dir / f"optimized_{args.split}.jsonl"
    tool_calls_path = run_dir / f"tool_calls_{args.split}.json"
    results_txt_path = run_dir / "results.txt"
    raw_results_txt_path = run_dir / "raw_results.txt"
    cost_txt_path = run_dir / "cost.txt"
    run_metrics_path = run_dir / "run_metrics.json"
    per_query_metrics_path = run_dir / "per_query_metrics.jsonl"

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
    results_txt_path.write_text("", encoding="utf-8")
    raw_results_txt_path.write_text("", encoding="utf-8")
    run_start_iso = iso_now()

    print("Building no-selection Travel match...", flush=True)
    nosel_prepare_elapsed = 0.0

    compose_cmd = [
        python_executable(),
        "-m",
        "planner.compose_match",
        "--task",
        "travel",
        "--split",
        args.split,
        "--templates",
        str(init_templates_path),
        "--match",
        str(match_path),
        "--out",
        str(tree_path),
        "--max-tool-candidates",
        str(args.max_tool_candidates),
        "--workers",
        str(args.workers),
    ]
    if args.config is not None:
        compose_cmd.extend(["--config", str(args.config.expanduser().resolve())])
    print("Running Travel ComposeMatch...", flush=True)
    compose_elapsed, _ = run_subprocess(cmd=compose_cmd, out_path=results_txt_path)

    tool_call_summary = summarize_travel_tool_calls(tool_calls_path)

    mcts_cmd = [
        python_executable(),
        "-m",
        "planner.twin_track",
        "--task",
        "travel",
        "--tree",
        str(tree_path),
        "--out",
        str(optimized_path),
        "--workers",
        str(args.workers),
        "--iterations",
        str(args.mcts_iterations),
        "--uct-c",
        str(args.mcts_uct_c),
        "--semantic-tolerance",
        str(args.semantic_threshold),
    ]
    if args.config is not None:
        mcts_cmd.extend(["--config", str(args.config.expanduser().resolve())])
    print("Running Travel MCTS...", flush=True)
    mcts_elapsed, _ = run_subprocess(cmd=mcts_cmd, out_path=results_txt_path)

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
    eval_metrics: Dict[str, float] = {}
    if not args.skip_eval:
        print("Running Travel evaluation for no-selection output...", flush=True)
        if args.split == "test":
            eval_elapsed, _ = run_subprocess(
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

    actual_no_eval = nosel_prepare_elapsed + compose_elapsed + mcts_elapsed
    actual_total = actual_no_eval + eval_elapsed
    comparable_no_eval = reused_init_elapsed + reused_view_elapsed + actual_no_eval
    comparable_total = comparable_no_eval + eval_elapsed
    init_price = read_cost_float(source_cost_lines, "llm_price_usd_init_template")
    if init_price is None:
        init_price = compute_price_usd(model_name=args.model, usage=init_total_usage)
    total_price = sum_mapping_values([init_price])
    price_key = source_price_key
    pipeline_cost_total = float(total_price) + float(tool_call_summary["total_cost"])

    cost_lines = [
        f"run_start={run_start_iso}",
        "task=travel",
        "variant=nosel",
        f"split={args.split}",
        f"model={args.model}",
        f"model_slug={model_slug(args.model)}",
        f"workers={args.workers}",
        f"source_run_dir={source_run_dir}",
        f"source_llm_price_model_key={source_price_key}",
        f"outputs_dir={run_dir}",
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
        f"actual_pipeline_elapsed_s_no_eval={actual_no_eval}",
        f"actual_pipeline_elapsed_s_total={actual_total}",
        f"pipeline_elapsed_s_no_eval={comparable_no_eval}",
        f"pipeline_elapsed_s_total={comparable_total}",
        f"eval_elapsed_s={eval_elapsed}",
        f"LLM token usage (init_template): {init_total_usage.to_cost_line()}",
        f"LLM token usage (nosel): {TokenUsage().to_cost_line()}",
        (
            "Tool call usage (compose_match): "
            f"calls={int(tool_call_summary['total_calls'])} "
            f"cost={float(tool_call_summary['total_cost'])} "
            f"by_tool={json.dumps(tool_call_summary['by_tool'], ensure_ascii=False, sort_keys=True)}"
        ),
        f"tool_calls_total_calls={int(tool_call_summary['total_calls'])}",
        f"tool_calls_total_cost={float(tool_call_summary['total_cost'])}",
        f"pipeline_tool_cost_total={float(tool_call_summary['total_cost'])}",
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
            "llm_price_usd_nosel": 0.0,
            "llm_price_usd_total": total_price,
        },
    )
    cost_txt_path.write_text("\n".join(cost_lines) + "\n", encoding="utf-8")

    run_metrics = {
        "metadata": {
            "task": "travel",
            "variant": "nosel",
            "split": args.split,
            "model": args.model,
            "model_slug": model_slug(args.model),
            "workers_requested": int(args.workers),
            "source_run_dir": str(source_run_dir),
            "outputs_dir": str(run_dir),
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
                "tool_calls_total": int(tool_call_summary["total_calls"]),
                "tool_cost_total": float(tool_call_summary["total_cost"]),
                "tool_usage_by_tool": tool_call_summary["by_tool"],
            },
            "mcts": {
                "elapsed_s": mcts_elapsed,
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
            "tool_cost_total": float(tool_call_summary["total_cost"]),
            "pipeline_cost_usd_total": pipeline_cost_total,
        },
        "evaluation": eval_metrics,
    }
    write_json(run_metrics_path, run_metrics)

    if not args.run_repair:
        print(f"Travel NoSel baseline complete: {run_dir}", flush=True)
        return

    repair_dir = run_dir / "repair"
    repair_dir.mkdir(parents=True, exist_ok=True)
    repair_results_txt = repair_dir / "results.txt"
    repair_log_path = repair_dir / "llm_repair.log"
    repair_cost_txt = repair_dir / "cost.txt"
    repair_jsonl = repair_dir / f"repaired_{args.split}_{model_slug(repair_model)}.jsonl"
    repair_run_metrics_path = repair_dir / "run_metrics.json"
    repair_per_query_metrics_path = repair_dir / "per_query_metrics.jsonl"
    repair_results_txt.write_text("", encoding="utf-8")
    repair_log_path.write_text("", encoding="utf-8")

    repair_cmd = [
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
        str(args.repair_workers),
    ]
    if args.config is not None:
        repair_cmd.extend(["--config", str(args.config.expanduser().resolve())])
    print("Running single-round Travel repair...", flush=True)
    repair_elapsed_llm, repair_output = run_subprocess(
        cmd=repair_cmd,
        out_path=repair_log_path,
        stream_output=True,
    )
    repair_usage = find_stage_usage(repair_output.splitlines(), "llm_repair")

    repaired_records = list(iter_jsonl(repair_jsonl))
    write_jsonl(
        repair_per_query_metrics_path,
        _build_repair_per_query_metrics(
            init_usage_by_template=init_usage_by_template,
            repaired_records=repaired_records,
        ),
    )

    repair_eval_elapsed = 0.0
    repair_eval_metrics: Dict[str, float] = {}
    if not args.skip_eval:
        print("Running Travel evaluation for repaired no-selection output...", flush=True)
        if args.split == "test":
            repair_eval_elapsed, _ = run_subprocess(
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

    base_price = init_price
    repair_price = compute_price_usd(model_name=repair_model, usage=repair_usage)
    repair_total_price = sum_mapping_values([base_price, repair_price])
    repair_pipeline_cost_total = float(repair_total_price) + float(tool_call_summary["total_cost"])
    repair_total_tokens = int(init_total_usage.total_tokens + repair_usage.total_tokens)
    repair_actual_no_eval = repair_elapsed_llm
    repair_actual_total = repair_actual_no_eval + repair_eval_elapsed
    repair_comparable_no_eval = comparable_no_eval + repair_elapsed_llm
    repair_comparable_total = repair_comparable_no_eval + repair_eval_elapsed

    repair_cost_lines = [
        f"run_start={iso_now()}",
        "task=travel",
        "variant=nosel",
        "repair_enabled=1",
        f"split={args.split}",
        f"model={repair_model}",
        f"model_slug={model_slug(repair_model)}",
        f"base_model={args.model}",
        f"base_run_dir={run_dir}",
        f"source_llm_price_model_key={source_price_key}",
        f"repair_tree={tree_path}",
        f"repair_input={optimized_path}",
        f"outputs_dir={repair_dir}",
        "reuse_init_template=1",
        f"reused_elapsed_s_init_template={reused_init_elapsed}",
        f"reused_elapsed_s_view_select={reused_view_elapsed}",
        f"reused_llm_price_usd_init_template={base_price}",
        f"reused_elapsed_s_nosel_prepare={nosel_prepare_elapsed}",
        f"reused_elapsed_s_compose_match={compose_elapsed}",
        f"reused_elapsed_s_mcts={mcts_elapsed}",
        f"llm_repair_elapsed_s={repair_elapsed_llm}",
        f"actual_pipeline_elapsed_s_no_eval={repair_actual_no_eval}",
        f"actual_pipeline_elapsed_s_total={repair_actual_total}",
        f"pipeline_elapsed_s_no_eval={repair_comparable_no_eval}",
        f"pipeline_elapsed_s_total={repair_comparable_total}",
        f"eval_elapsed_s={repair_eval_elapsed}",
        f"LLM token usage (init_template): {init_total_usage.to_cost_line()}",
        f"LLM token usage (llm_repair): {repair_usage.to_cost_line()}",
        f"pipeline_tool_cost_total={float(tool_call_summary['total_cost'])}",
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
            "tool_cost_total": float(tool_call_summary["total_cost"]),
            "pipeline_cost_usd_total": repair_pipeline_cost_total,
        },
        "evaluation_before_repair": eval_metrics,
        "evaluation_after_repair": repair_eval_metrics,
    }
    write_json(repair_run_metrics_path, repair_run_metrics)
    print(f"Travel NoSel baseline complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
