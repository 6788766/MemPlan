"""
Append TravelPlanner "Local Pass Rate" to a run's cost.txt.

Given a JSON file containing the detailed constraint distribution (the JSON
object returned as `detailed_scores` by `task_helper.travel.evaluation.eval`),
this script computes the micro-averaged local pass rate across both:
  - Commonsense Constraint
  - Hard Constraint

It then appends a line to `cost.txt` in the same folder as the JSON file.

Example:
  python -m task_helper.travel.utils.append_local_pass_rate \
    artifacts/output/travel/gpt52_test/20251222172753.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple


def _as_int(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
    return None


def _iter_true_total(obj: Any) -> Iterable[Tuple[int, int]]:
    if isinstance(obj, dict):
        raw_true = obj.get("true")
        raw_total = obj.get("total")
        true_count = _as_int(raw_true)
        total_count = _as_int(raw_total)
        if true_count is not None and total_count is not None:
            yield true_count, total_count

        for value in obj.values():
            yield from _iter_true_total(value)
        return

    if isinstance(obj, list):
        for value in obj:
            yield from _iter_true_total(value)


def compute_local_pass_rate(payload: Any) -> Tuple[float, int, int]:
    if isinstance(payload, dict):
        passed = 0
        total = 0
        for key in ("Commonsense Constraint", "Hard Constraint"):
            section = payload.get(key)
            if section is None:
                continue
            for true_count, total_count in _iter_true_total(section):
                if total_count <= 0:
                    continue
                passed += true_count
                total += total_count
        if total > 0:
            return passed / total, passed, total

        direct = payload.get("Local Pass Rate") or payload.get("local_pass_rate")
        if direct is not None:
            try:
                return float(direct), 0, 0
            except (TypeError, ValueError):
                pass

    raise ValueError(
        "Could not compute Local Pass Rate from JSON. "
        "Expected either a top-level 'Local Pass Rate' field or a detailed_scores-style object "
        "with 'Commonsense Constraint'/'Hard Constraint' sections containing {true, total} counts."
    )


def _needs_newline(path: Path) -> bool:
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return False
    if size <= 0:
        return False
    with path.open("rb") as fp:
        fp.seek(-1, 2)
        last = fp.read(1)
    return last not in (b"\n", b"\r")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute and append Local Pass Rate to cost.txt.")
    parser.add_argument("json_path", type=Path, help="Path to the detailed constraint JSON file.")
    parser.add_argument(
        "--cost",
        type=Path,
        default=None,
        help="Override cost.txt path (default: alongside json_path).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the computed value but do not modify cost.txt.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    json_path: Path = args.json_path.expanduser()
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    rate, passed, total = compute_local_pass_rate(payload)
    percent = rate * 100.0

    suffix = ""
    if total > 0:
        suffix = f" ({passed}/{total})"
    line = f"Local Pass Rate: {percent}%{suffix}\n"

    print(line, end="")

    if args.dry_run:
        return

    cost_path = (args.cost.expanduser() if args.cost is not None else (json_path.parent / "cost.txt")).resolve()
    if _needs_newline(cost_path):
        line = "\n" + line
    cost_path.parent.mkdir(parents=True, exist_ok=True)
    with cost_path.open("a", encoding="utf-8") as fp:
        fp.write(line)


if __name__ == "__main__":
    main()
