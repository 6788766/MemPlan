from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, MutableMapping

from analysis.work_tool_edge_rate import compute_stats as compute_edge_stats
from analysis.work_tree_candidate_stats import compute_stats as compute_tree_stats
from baseline.nomcts.common import (
    PROJECT_ROOT,
    read_cost_lines,
    read_cost_value,
    read_json,
    summarize_work_tool_calls,
)


TRAVEL_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "output" / "travel" / "nosel"
WORK_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "output" / "work" / "nosel"


def build_empty_match_payload(source_payload: Mapping[str, object]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for template_id, raw_entry in source_payload.items():
        if not isinstance(raw_entry, Mapping):
            continue
        lambda_raw = raw_entry.get("lambda") if isinstance(raw_entry.get("lambda"), Mapping) else {}
        edge_keys = [str(key) for key in lambda_raw.keys() if str(key).strip()]
        required_edges_raw = raw_entry.get("required_edges")
        try:
            required_edges = int(required_edges_raw) if required_edges_raw is not None else len(edge_keys)
        except Exception:
            required_edges = len(edge_keys)

        out[str(template_id)] = {
            "required_edges": int(required_edges),
            "uncovered_edges": list(edge_keys),
            "selected_views": [],
            "selected_views_by_source": {
                "memory": [],
                "tool": [],
            },
            "total_cost": 0.0,
            "lambda": {edge_key: [] for edge_key in edge_keys},
            "selection_mode": "none",
            "num_memory_views": 0,
            "num_tool_views": 0,
        }
    return out


def _parse_numeric(value: object) -> object:
    if isinstance(value, (int, float)):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text.isdigit():
        try:
            return int(text)
        except Exception:
            return value
    try:
        return float(text)
    except Exception:
        return value


def _set_if_present(out: MutableMapping[str, object], key: str, value: object) -> None:
    if value is None:
        return
    out[key] = _parse_numeric(value)


def build_work_view_metrics(
    *,
    run_dir: Path,
    split: str,
    model: str,
    model_slug: str,
) -> Dict[str, object]:
    run_dir = run_dir.expanduser().resolve()
    match_path = run_dir / f"match_{split}.json"
    tree_path = run_dir / f"tree_{split}.json"
    optimized_path = run_dir / f"optimized_{split}.jsonl"
    cost_path = run_dir / "cost.txt"
    pass_rates_path = run_dir / "results" / "pass_rates.json"

    match_payload = read_json(match_path) if match_path.exists() else {}
    tree_payload = read_json(tree_path) if tree_path.exists() else {}
    pass_rates_payload = read_json(pass_rates_path) if pass_rates_path.exists() else {}
    if isinstance(pass_rates_payload, dict):
        pass_rates_payload.pop("constraints", None)

    edge_stats = compute_edge_stats(match_payload) if isinstance(match_payload, Mapping) else None
    tree_stats = compute_tree_stats(tree_payload) if isinstance(tree_payload, Mapping) else None
    tool_stats = summarize_work_tool_calls(optimized_path)
    cost_lines = read_cost_lines(cost_path)

    out: Dict[str, object] = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "split": split,
        "match_path": str(match_path),
        "optimized_path": str(optimized_path),
        "num_memory_views": 0,
        "cost_path": str(cost_path),
        "task": "work",
        "model": model,
        "model_slug": model_slug,
        "pass_rates_path": str(pass_rates_path) if pass_rates_path.exists() else None,
    }

    for key in (
        "tolerance_rate",
        "mcts_iterations",
        "mcts_uct_c",
        "pipeline_elapsed_s_total",
        "pipeline_elapsed_s_no_eval",
        "llm_price_usd_total",
        "total_tokens",
        "total_time_s",
    ):
        _set_if_present(out, key, read_cost_value(cost_lines, key))

    if isinstance(pass_rates_payload, Mapping):
        for key in (
            "WorkBench Accuracy",
            "WorkBench Exact Match",
            "WorkBench Unwanted Side Effects",
            "Global Pass Rate",
            "Local Pass Rate",
            "Hard Constraint Micro Pass Rate",
            "Semantic Constraint Micro Pass Rate",
        ):
            if key in pass_rates_payload:
                out[key] = pass_rates_payload[key]

    if edge_stats is not None:
        out.update(
            {
                "templates": int(edge_stats.templates),
                "total_edges": int(edge_stats.total_edges),
                "tool_only_edges": int(edge_stats.tool_only_edges),
                "tool_only_rate": float(edge_stats.tool_only_rate),
                "edges_with_any_tool": int(edge_stats.edges_with_any_tool),
                "any_tool_rate": float(edge_stats.any_tool_rate),
                "uncovered_edges": int(edge_stats.uncovered_edges),
                "uncovered_rate": float(edge_stats.uncovered_rate),
            }
        )

    if tree_stats is not None:
        out.update(
            {
                "tree_path": str(tree_path),
                "tree_slots_total": int(tree_stats.slots_total),
                "tree_slots_with_any_tool": int(tree_stats.slots_with_any_tool),
                "tree_slots_with_effective_tool": int(tree_stats.slots_with_effective_tool),
                "tree_candidates_total": int(tree_stats.candidates_total),
                "tree_candidates_tool_total": int(tree_stats.candidates_tool_total),
                "tree_candidates_memory_total": int(tree_stats.candidates_memory_total),
                "tree_tool_shadowed_by_memory": int(tree_stats.tool_shadowed_by_memory),
                "tree_tool_effective_total": int(tree_stats.tool_effective_total),
            }
        )

    out.update(
        {
            "calls_total": int(tool_stats["calls_total"]),
            "calls_check": int(tool_stats["calls_check"]),
            "calls_do": int(tool_stats["calls_do"]),
            "calls_by_domain": tool_stats["by_domain"],
        }
    )

    tool_cost_total = 0.0
    try:
        for raw_line in optimized_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            notes = obj.get("notes") if isinstance(obj.get("notes"), dict) else {}
            blob = notes.get("multi_round_mcts") if isinstance(notes.get("multi_round_mcts"), dict) else {}
            cost = blob.get("tool_cost")
            if isinstance(cost, (int, float)):
                tool_cost_total += float(cost)
    except Exception:
        tool_cost_total = 0.0
    out["tool_cost_total"] = float(tool_cost_total)
    return out


def write_work_view_metrics(
    *,
    run_dir: Path,
    split: str,
    model: str,
    model_slug: str,
) -> Path:
    payload = build_work_view_metrics(run_dir=run_dir, split=split, model=model, model_slug=model_slug)
    out_path = run_dir / "view_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path
