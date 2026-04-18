#!/usr/bin/env python3
"""
Normalize a TravelPlanner submission JSONL so evaluation doesn't crash.

The official evaluator expects each line to have:
  {"idx": int, "query": str, "plan": list[dict]}

When upstream TravelPlanner postprocess fails to parse an output, it may set
`plan` to null. This script replaces non-list plans with a placeholder plan
inferred from the query duration (when possible), otherwise an empty list.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


_PLAN_KEYS = (
    "days",
    "current_city",
    "transportation",
    "breakfast",
    "attraction",
    "lunch",
    "dinner",
    "accommodation",
)

_INFER_DAYS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(\d+)\s*-\s*day\b"),
    re.compile(r"(?i)\bspanning\s+(\d+)\s+days?\b"),
    re.compile(r"(?i)\bfor\s+(\d+)\s+days?\b"),
    re.compile(r"(?i)\b(\d+)\s+day\s+trip\b"),
)


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


def _infer_days_from_query(query: str) -> Optional[int]:
    if not query:
        return None
    text = str(query)
    for pat in _INFER_DAYS_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        value = _parse_int(m.group(1))
        if value is not None and 0 < value <= 30:
            return value
    return None


def _placeholder_plan(days: int) -> List[Dict[str, object]]:
    plan: List[Dict[str, object]] = []
    for day in range(1, int(days) + 1):
        record: Dict[str, object] = {k: "-" for k in _PLAN_KEYS}
        record["days"] = day
        plan.append(record)
    return plan


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize TravelPlanner submission JSONL.")
    parser.add_argument("--in", dest="in_path", type=Path, required=True, help="Input submission JSONL.")
    parser.add_argument("--out", dest="out_path", type=Path, required=True, help="Output submission JSONL.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    in_path = args.in_path.expanduser()
    out_path = args.out_path.expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Avoid truncating the input when normalizing in-place.
    items = list(_iter_jsonl(in_path)) if in_path.resolve() == out_path.resolve() else None

    fixed = 0
    total = 0
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    source_iter = items if items is not None else _iter_jsonl(in_path)
    with tmp_path.open("w", encoding="utf-8") as out_fp:
        for obj in source_iter:
            total += 1
            plan = obj.get("plan")
            if isinstance(plan, list):
                out_fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            query = obj.get("query")
            days = _infer_days_from_query(str(query or ""))
            if days is not None:
                obj["plan"] = _placeholder_plan(days)
            else:
                obj["plan"] = []
            fixed += 1
            out_fp.write(json.dumps(obj, ensure_ascii=False) + "\n")

    tmp_path.replace(out_path)

    print(f"normalize_submission: in={in_path} out={out_path} fixed={fixed}/{total}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"normalize_submission: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
