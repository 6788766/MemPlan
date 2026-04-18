#!/usr/bin/env python3
"""
Summarize baseline failures / low-quality outputs.

Reads baseline outputs produced by `baseline/tool_agents.py` and reports:
- hard failures (exception in worker -> `error` field)
- empty/very short results
- missing Planner step (no `Planner[...]` action observed)
- empty or placeholder plans in converted submission JSONL (if present)

Usage:
  python baseline/summarize_failures.py --run-dir artifacts/output/travel/baseline/gpt52_validation
  python baseline/summarize_failures.py --run-dir ... --json-out failures.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def _parse_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return int(value)
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _safe_str(value: object) -> str:
    return str(value) if value is not None else ""


def _first_error_type(message: str) -> str:
    msg = (message or "").strip()
    if not msg:
        return "UnknownError"
    head = msg.split(":", 1)[0].strip()
    return head or "UnknownError"


def _has_planner_action(action_log: object) -> bool:
    if not isinstance(action_log, list):
        return False
    for step in action_log:
        if not isinstance(step, dict):
            continue
        raw = step.get("action")
        if not isinstance(raw, str):
            continue
        if "planner[" in raw.lower():
            return True
    return False


def _looks_like_placeholder_plan(plan: object) -> bool:
    if not isinstance(plan, list) or not plan:
        return False
    placeholder = True
    for day in plan:
        if not isinstance(day, dict):
            continue
        for key, value in day.items():
            if key == "days":
                continue
            if isinstance(value, str) and value.strip() and value.strip() != "-":
                placeholder = False
                break
        if not placeholder:
            break
    return placeholder


def _resolve_baseline_jsonl(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("two_stage_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No baseline JSONL found under {run_dir} (expected two_stage_*.jsonl).")
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Multiple baseline JSONLs found under {run_dir}: {[p.name for p in candidates]}")


def _resolve_submission_jsonl(run_dir: Path) -> Optional[Path]:
    candidates = sorted(run_dir.glob("submission_*.jsonl"))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # Prefer validation/train if multiple exist.
    for preferred in ("submission_validation.jsonl", "submission_train.jsonl", "submission_test.jsonl"):
        hit = run_dir / preferred
        if hit.exists():
            return hit
    return candidates[0]


@dataclass
class Summary:
    total: int = 0
    error_count: int = 0
    empty_result_count: int = 0
    no_planner_action_count: int = 0
    missing_submission_count: int = 0
    empty_plan_count: int = 0
    placeholder_plan_count: int = 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize baseline failures / low-quality outputs.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Baseline run directory (e.g. .../baseline/gpt52_validation).")
    parser.add_argument("--min-result-chars", type=int, default=32, help="Treat results shorter than this as empty.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to write a JSON report.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    baseline_jsonl = _resolve_baseline_jsonl(run_dir)
    submission_jsonl = _resolve_submission_jsonl(run_dir)

    submission_by_idx: Dict[int, Dict[str, object]] = {}
    if submission_jsonl is not None and submission_jsonl.exists():
        for row in _iter_jsonl(submission_jsonl):
            idx = _parse_int(row.get("idx"))
            if idx is None:
                continue
            submission_by_idx[idx] = row

    summary = Summary()
    error_types: Counter[str] = Counter()
    indices: Dict[str, List[int]] = defaultdict(list)

    token_totals = {"prompt_cache_hit": 0, "prompt_cache_miss": 0, "output": 0, "total": 0, "calls": 0}

    for row in _iter_jsonl(baseline_jsonl):
        idx = _parse_int(row.get("idx"))
        if idx is None:
            continue
        summary.total += 1

        err = row.get("error")
        if isinstance(err, str) and err.strip():
            summary.error_count += 1
            err_type = _first_error_type(err)
            error_types[err_type] += 1
            indices["error"].append(idx)

        result = _safe_str(row.get("result"))
        if len(result.strip()) < int(args.min_result_chars):
            summary.empty_result_count += 1
            indices["empty_result"].append(idx)

        if not _has_planner_action(row.get("action_log")):
            summary.no_planner_action_count += 1
            indices["no_planner_action"].append(idx)

        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        token_totals["calls"] += int(metrics.get("llm_calls") or 0)
        token_totals["prompt_cache_hit"] += int(metrics.get("prompt_cache_hit_tokens") or 0)
        token_totals["prompt_cache_miss"] += int(metrics.get("prompt_cache_miss_tokens") or 0)
        token_totals["output"] += int(metrics.get("output_tokens") or metrics.get("completion_tokens") or 0)
        token_totals["total"] += int(metrics.get("total_tokens") or 0)

        if submission_jsonl is None:
            summary.missing_submission_count = summary.total
            continue

        sub = submission_by_idx.get(idx)
        if not isinstance(sub, dict):
            summary.empty_plan_count += 1
            indices["missing_submission_row"].append(idx)
            continue
        plan = sub.get("plan")
        if not isinstance(plan, list) or not plan:
            summary.empty_plan_count += 1
            indices["empty_plan"].append(idx)
        elif _looks_like_placeholder_plan(plan):
            summary.placeholder_plan_count += 1
            indices["placeholder_plan"].append(idx)

    report: Dict[str, object] = {
        "run_dir": str(run_dir),
        "baseline_jsonl": str(baseline_jsonl),
        "submission_jsonl": str(submission_jsonl) if submission_jsonl is not None else None,
        "counts": {
            "total": summary.total,
            "error": summary.error_count,
            "empty_result": summary.empty_result_count,
            "no_planner_action": summary.no_planner_action_count,
            "empty_plan": summary.empty_plan_count,
            "placeholder_plan": summary.placeholder_plan_count,
        },
        "error_types": dict(error_types.most_common()),
        "token_totals": dict(token_totals),
        "indices": {k: v for k, v in indices.items()},
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json_out is not None:
        out_path = args.json_out.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"summarize_failures: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)

