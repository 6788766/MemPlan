#!/usr/bin/env python3
"""
Convert baseline output JSONL (two_stage_*.jsonl) to the TravelPlanner submission JSONL format.

Baseline records look like:
  {"idx": ..., "query": "...", "result": "<text plan>", ...}

Submission records look like:
  {"idx": ..., "query": "...", "plan": [ {"days": 1, ...}, ... ]}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


_DAY_SPLIT_RE = re.compile(r"(?im)^\s*day\s+(\d+)\s*:\s*$")
_FIELD_RE = re.compile(r"(?im)^\s*([a-z][a-z _-]*?)\s*:\s*(.*?)\s*$")

_KEY_MAP = {
    "current city": "current_city",
    "current_city": "current_city",
    "transportation": "transportation",
    "breakfast": "breakfast",
    "lunch": "lunch",
    "dinner": "dinner",
    "attraction": "attraction",
    "attractions": "attraction",
    "accommodation": "accommodation",
    "hotel": "accommodation",
}

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


def _iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def _normalise_key(raw: str) -> str:
    k = re.sub(r"[\t\r\n]+", " ", raw.strip().lower())
    k = re.sub(r"\s+", " ", k)
    return k.replace("-", " ")


def _parse_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return int(value)
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _split_by_day(text: str) -> List[Tuple[int, str]]:
    if not text:
        return []
    lines = str(text).replace("\r\n", "\n").replace("\r", "\n").splitlines()
    indices: List[Tuple[int, int]] = []
    for i, line in enumerate(lines):
        m = _DAY_SPLIT_RE.match(line)
        if not m:
            continue
        day = _parse_int(m.group(1))
        if day is None or day <= 0:
            continue
        indices.append((day, i))
    if not indices:
        return []

    blocks: List[Tuple[int, str]] = []
    for pos, (day, start_idx) in enumerate(indices):
        end_idx = indices[pos + 1][1] if pos + 1 < len(indices) else len(lines)
        block = "\n".join(lines[start_idx + 1 : end_idx]).strip()
        blocks.append((day, block))
    return blocks


def _parse_day_block(day: int, block: str) -> Dict[str, object]:
    record: Dict[str, object] = {k: "-" for k in _PLAN_KEYS}
    record["days"] = day
    if not block:
        return record

    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = _FIELD_RE.match(line)
        if not m:
            continue
        raw_key, raw_value = m.group(1), m.group(2)
        key_norm = _normalise_key(raw_key)
        out_key = _KEY_MAP.get(key_norm)
        if not out_key:
            continue
        value = raw_value.strip() if raw_value is not None else ""
        record[out_key] = value if value else "-"
    return record


def parse_plan(text: str) -> List[Dict[str, object]]:
    blocks = _split_by_day(text)
    if not blocks:
        return []
    day_to_block: Dict[int, str] = {}
    for day, block in blocks:
        day_to_block[day] = block
    max_day = max(day_to_block.keys())
    plan: List[Dict[str, object]] = []
    for day in range(1, max_day + 1):
        plan.append(_parse_day_block(day, day_to_block.get(day, "")))
    return plan


_INFER_DAYS_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(\d+)\s*-\s*day\b"),
    re.compile(r"(?i)\bspanning\s+(\d+)\s+days?\b"),
    re.compile(r"(?i)\bfor\s+(\d+)\s+days?\b"),
    re.compile(r"(?i)\b(\d+)\s+day\s+trip\b"),
)


def _infer_days_from_query(query: str) -> Optional[int]:
    if not query:
        return None
    text = str(query)
    for pat in _INFER_DAYS_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        value = _parse_int(m.group(1))
        if value is not None and value > 0 and value <= 30:
            return value
    return None


def _placeholder_plan(days: int) -> List[Dict[str, object]]:
    plan: List[Dict[str, object]] = []
    for day in range(1, int(days) + 1):
        record: Dict[str, object] = {k: "-" for k in _PLAN_KEYS}
        record["days"] = day
        plan.append(record)
    return plan


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert baseline output JSONL to TravelPlanner submission JSONL.")
    parser.add_argument("--in", dest="in_path", type=Path, required=True, help="Path to baseline JSONL.")
    parser.add_argument("--out", dest="out_path", type=Path, required=True, help="Path to write submission JSONL.")
    parser.add_argument("--drop-empty", action="store_true", help="Drop entries when plan parsing fails.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    in_path = args.in_path.expanduser()
    out_path = args.out_path.expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with out_path.open("w", encoding="utf-8") as out_fp:
        for obj in _iter_jsonl(in_path):
            idx = _parse_int(obj.get("idx"))
            query = obj.get("query")
            result = obj.get("result")
            if idx is None or not isinstance(query, str):
                skipped += 1
                continue
            plan = parse_plan(str(result or ""))
            if not plan:
                inferred = _infer_days_from_query(query)
                if inferred is not None:
                    plan = _placeholder_plan(inferred)
            if not plan and args.drop_empty:
                skipped += 1
                continue
            out_fp.write(json.dumps({"idx": idx, "query": query, "plan": plan}, ensure_ascii=False) + "\n")
            written += 1

    print(f"convert_baseline_to_submission: in={in_path} out={out_path} written={written} skipped={skipped}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"convert_baseline_to_submission: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
