#!/usr/bin/env python3
"""
Prepare inputs for the TravelPlanner postprocess scripts.

The upstream TravelPlanner postprocess expects files:
  <output_dir>/<set_type>/generated_plan_<idx>.json

with a JSON list whose last element contains a field like:
  "<model_name>_<mode>_results": "<natural language plan>"

Our MemPlan baseline writes per-query JSON objects under:
  artifacts/output/travel/baseline/<model_slug>_<split>/two_stage_<model_slug>_<split>/generated_plan_<idx>.json

This script bridges the formats so we can run the upstream postprocess
(parsing.py -> element_extraction.py -> combination.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from baseline.utils.dataset import load_travelplanner_dataset


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


def _result_key(*, model_name: str, mode: str, strategy: str) -> str:
    if mode == "two-stage":
        suffix = ""
    elif mode == "sole-planning":
        suffix = f"_{strategy}"
    else:
        suffix = ""
    return f"{model_name}{suffix}_{mode}_results"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare TravelPlanner postprocess inputs from MemPlan baseline JSONL.")
    parser.add_argument("--baseline-jsonl", type=Path, required=True, help="Path to two_stage_*.jsonl")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory root for postprocess inputs.")
    parser.add_argument("--set-type", type=str, default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--model-name", type=str, required=True, help="Model name used for postprocess keying/filenames.")
    parser.add_argument("--mode", type=str, default="two-stage", choices=("two-stage", "sole-planning"))
    parser.add_argument("--strategy", type=str, default="direct")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated_plan_<idx>.json files.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    baseline_jsonl = args.baseline_jsonl.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    set_type = str(args.set_type)
    model_name = str(args.model_name)
    mode = str(args.mode)
    strategy = str(args.strategy)

    dataset = load_travelplanner_dataset(set_type)
    total = len(dataset)
    if total <= 0:
        raise SystemExit(f"Dataset split is empty: {set_type}")

    idx_to_result: Dict[int, str] = {}
    for row in _iter_jsonl(baseline_jsonl):
        idx = _parse_int(row.get("idx"))
        if idx is None:
            continue
        result = row.get("result")
        idx_to_result[idx] = str(result or "")

    split_dir = out_dir / set_type
    split_dir.mkdir(parents=True, exist_ok=True)
    key = _result_key(model_name=model_name, mode=mode, strategy=strategy)

    written = 0
    skipped = 0
    for idx in range(1, total + 1):
        path = split_dir / f"generated_plan_{idx}.json"
        if path.exists() and not args.overwrite:
            skipped += 1
            continue
        payload: List[Dict[str, object]] = [{key: idx_to_result.get(idx, "")}]
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        written += 1

    print(
        f"prepare_tp_postprocess_input: out_dir={out_dir} split={set_type} total={total} written={written} skipped={skipped} key={key}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"prepare_tp_postprocess_input: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)

