"""
Utility to run the official TravelPlanner evaluator on MemPlan outputs.

Supports two input formats:

1) Standard submission JSONL: each line has ``{"idx": ..., "query": ..., "plan": [day_records...]}``.
2) Optimized template JSONL produced by ``python -m planner.twin_track`` (e.g. ``optimized_validation.jsonl``).

When given an optimized template file, this script converts it to the standard
submission format (best-effort) and then runs ``task_helper/travel/evaluation/eval.py``.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from task_helper.travel.utils.paths import project_root
PROJECT_ROOT = project_root()

_LOG_PATH = PROJECT_ROOT / "artifacts" / "logs" / "log.txt"


def _append_run_log(
    *,
    argv: List[str],
    started_at: float,
    status: str,
    extra: Optional[Dict[str, object]] = None,
    error: Optional[str] = None,
) -> None:
    try:
        payload: Dict[str, object] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "module": "task_helper.travel.runners.eval_bridge",
            "status": status,
            "elapsed_s": round(time.perf_counter() - started_at, 6),
            "argv": list(argv),
        }
        if extra:
            payload.update(dict(extra))
        if error:
            payload["error"] = error
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        return


def _resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TravelPlanner evaluation.")
    parser.add_argument(
        "--set-type",
        choices={"train", "validation"},
        default="validation",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--submission",
        type=Path,
        required=True,
        help=(
            "Path to a submission JSONL (idx/query/plan), or an optimized template JSONL "
            "from `python -m planner.twin_track` (optimized_<split>.jsonl)."
        ),
    )
    parser.add_argument(
        "--converted-out",
        type=Path,
        default=None,
        help=(
            "Optional path to write a converted submission JSONL when the input is an optimized template file. "
            "Defaults to <submission>.submission.jsonl."
        ),
    )
    return parser.parse_args()

def _iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _parse_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return int(value)
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _phase_index_map(phases: object) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    if not isinstance(phases, list):
        return mapping
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        key = phase.get("phase_key")
        if not isinstance(key, str) or not key:
            continue
        idx = _parse_int(phase.get("phase_index"))
        if idx is not None:
            mapping[key] = idx
    return mapping


def _day_from_phase_key(phase_key: str) -> Optional[int]:
    if not phase_key:
        return None
    # Expected keys like "day_3".
    tail = phase_key.rsplit("_", 1)[-1]
    try:
        day = int(tail)
    except ValueError:
        return None
    return day if day > 0 else None


def _build_eval_plan_from_optimized(record: Dict[str, object]) -> List[dict]:
    plan_obj = record.get("plan")
    if not isinstance(plan_obj, dict):
        return []
    days = _parse_int(plan_obj.get("days")) or 0
    if days <= 0:
        return []

    phases = record.get("phases")
    phase_to_index = _phase_index_map(phases)

    actions = record.get("actions")
    if not isinstance(actions, list):
        return []

    records: List[dict] = []
    attractions_by_day: Dict[int, List[str]] = {d: [] for d in range(1, days + 1)}
    for d in range(1, days + 1):
        records.append(
            {
                "days": d,
                "current_city": "-",
                "transportation": "-",
                "breakfast": "-",
                "lunch": "-",
                "dinner": "-",
                "attraction": "-",
                "accommodation": "-",
            }
        )

    for action in actions:
        if not isinstance(action, dict):
            continue
        phase_key = action.get("phase_key")
        phase_key = phase_key if isinstance(phase_key, str) else ""
        action_type = action.get("action_type")
        action_type = action_type if isinstance(action_type, str) else ""

        params = action.get("params")
        params = params if isinstance(params, dict) else {}
        attrs = action.get("attrs")
        attrs = attrs if isinstance(attrs, dict) else {}

        day = _parse_int(params.get("day"))
        if day is None:
            day = phase_to_index.get(phase_key) or _day_from_phase_key(phase_key)
        if day is None or day < 1 or day > days:
            continue
        record_out = records[day - 1]

        if action_type == "DayRecord":
            current_city = params.get("current_city")
            if isinstance(current_city, str) and current_city.strip():
                record_out["current_city"] = current_city.strip()
            continue

        if action_type == "Move":
            raw = attrs.get("raw_transportation")
            if isinstance(raw, str) and raw.strip():
                raw_text = raw.strip()
                if raw_text[:7].lower() == "driving":
                    raw_text = "Self-driving" + raw_text[7:]
                record_out["transportation"] = raw_text
                continue

            mode = attrs.get("mode") or params.get("mode") or ""
            mode_str = str(mode) if mode is not None else ""
            mode_norm = mode_str.strip()
            mode_lower = mode_norm.lower()
            if mode_lower in {"self-driving", "self driving", "driving"}:
                mode_norm = "Self-driving"
                mode_lower = "self-driving"
            elif mode_lower == "taxi":
                mode_norm = "Taxi"
            elif mode_lower == "flight":
                mode_norm = "Flight"
            origin = params.get("origin")
            destination = params.get("destination")
            if mode_lower == "flight":
                flight_no = params.get("Flight Number")
                dep = params.get("DepTime")
                arr = params.get("ArrTime")
                record_out["transportation"] = (
                    f"Flight Number: {flight_no}, from {origin} to {destination}, "
                    f"Departure Time: {dep}, Arrival Time: {arr}"
                )
            else:
                record_out["transportation"] = (
                    f"{mode_norm or 'Self-driving'}, from {origin} to {destination}, "
                    f"duration: {params.get('duration')}, distance: {params.get('distance')}, cost: {params.get('cost')}"
                )
            continue

        if action_type == "Eat":
            slot_name = attrs.get("slot")
            if not isinstance(slot_name, str):
                continue
            slot_name = slot_name.strip().lower()
            if slot_name not in {"breakfast", "lunch", "dinner"}:
                continue
            name = params.get("Name")
            city = params.get("City")
            text = f"{name}, {city}" if name and city else "-"
            record_out[slot_name] = text
            continue

        if action_type == "Visit":
            name = params.get("Name")
            city = params.get("City")
            if name and city:
                attractions_by_day[day].append(f"{name}, {city}")
            continue

        if action_type == "Stay":
            name = params.get("NAME")
            city = params.get("city") or params.get("City")
            if name and city:
                record_out["accommodation"] = f"{name}, {city}"
            continue

    for day in range(1, days + 1):
        items = attractions_by_day.get(day, [])
        if items:
            records[day - 1]["attraction"] = ";".join(items) + ";"
    return records


def _detect_format(sample: Dict[str, object]) -> str:
    plan = sample.get("plan")
    if isinstance(plan, list):
        return "submission"
    if isinstance(sample.get("actions"), list) and isinstance(plan, dict) and "template_id" in sample:
        return "optimized"
    return "unknown"


def _convert_optimized_submission(
    *,
    input_path: Path,
    output_path: Path,
) -> Path:
    items: List[Tuple[int, Dict[str, object]]] = []
    fallback_index = 1
    for record in _iter_jsonl(input_path):
        template_id = record.get("template_id")
        idx = _parse_int(template_id)
        if idx is None:
            idx = fallback_index - 1
        fallback_index += 1

        eval_plan = _build_eval_plan_from_optimized(record)
        plan_obj = record.get("plan") if isinstance(record.get("plan"), dict) else {}
        query_text = plan_obj.get("query") if isinstance(plan_obj, dict) else None
        output = {
            "idx": idx + 1,
            "query": query_text if isinstance(query_text, str) else "",
            "plan": eval_plan,
        }
        items.append((idx, output))

    items.sort(key=lambda pair: pair[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        for _idx, payload in items:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return output_path


def main() -> None:
    started_at = time.perf_counter()
    argv_snapshot = sys.argv[:]
    status = "ok"
    error: Optional[str] = None
    extra: Dict[str, object] = {}

    try:
        args = parse_args()
        extra.update({"set_type": args.set_type})

        submission_path = _resolve_path(args.submission)
        if submission_path.exists() and submission_path.is_dir():
            preferred = submission_path / f"optimized_{args.set_type}.jsonl"
            if preferred.exists():
                submission_path = preferred
            else:
                candidates = sorted(submission_path.glob("*.jsonl"))
                if len(candidates) == 1:
                    submission_path = candidates[0]
                else:
                    listing = "\n".join(f"  - {path}" for path in candidates[:20])
                    extra_suffix = "" if len(candidates) <= 20 else f"\n  ... ({len(candidates) - 20} more)"
                    raise SystemExit(
                        f"--submission points to a directory: {submission_path}\n"
                        f"Pass a JSONL file path instead (e.g. {preferred}).\n"
                        f"JSONL files in directory:\n{listing}{extra_suffix}"
                    )
        if not submission_path.exists():
            raise SystemExit(f"Submission file not found: {submission_path}")
        extra.update({"submission": str(submission_path)})

        submission_to_eval = submission_path
        sample: Optional[Dict[str, object]] = None
        for sample in _iter_jsonl(submission_path):
            break

        if sample is None:
            raise SystemExit(f"Submission file is empty: {submission_path}")

        fmt = _detect_format(sample)
        if fmt == "optimized":
            converted_out = args.converted_out
            if converted_out is None:
                converted_out = submission_path.with_name(submission_path.stem + ".submission.jsonl")
            converted_out = _resolve_path(converted_out)
            submission_to_eval = _convert_optimized_submission(
                input_path=submission_path,
                output_path=converted_out,
            )
            expected = {"train": 45, "validation": 180}.get(args.set_type)
            if expected is not None:
                converted_count = sum(1 for _ in _iter_jsonl(submission_to_eval))
                if converted_count != expected:
                    print(
                        f"Warning: converted file has {converted_count} line(s) but {expected} expected for {args.set_type}.",
                        file=sys.stderr,
                        flush=True,
                    )
            print(f"Converted optimized templates -> submission JSONL: {submission_to_eval}")
            extra.update({"converted_submission": str(submission_to_eval)})
        elif fmt != "submission":
            raise SystemExit(
                f"Unsupported submission format in {submission_path}. "
                "Expected either standard submission JSONL (plan is a list) or optimized template JSONL (has actions)."
            )

        argv_backup = sys.argv[:]
        try:
            sys.argv = [
                "task_helper.travel.evaluation.eval",
                "--set_type",
                args.set_type,
                "--evaluation_file_path",
                str(submission_to_eval),
            ]
            runpy.run_module("task_helper.travel.evaluation.eval", run_name="__main__")
        finally:
            sys.argv = argv_backup
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            status = "interrupted"
        elif isinstance(exc, SystemExit):
            status = "exit"
            code = exc.code
            if isinstance(code, (int, float, str, bool)) or code is None:
                extra.update({"exit_code": code})
            else:
                extra.update({"exit_code": repr(code)})
        else:
            status = "error"
        error = repr(exc)
        raise
    finally:
        _append_run_log(
            argv=argv_snapshot,
            started_at=started_at,
            status=status,
            extra=extra,
            error=error,
        )


if __name__ == "__main__":
    main()
