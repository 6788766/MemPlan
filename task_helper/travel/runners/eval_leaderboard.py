"""
Run the official TravelPlanner leaderboard evaluation via the HuggingFace Space API.

Why this exists
  - The local benchmark does not provide labels for the TravelPlanner TEST split.
  - The public leaderboard Space hosts an evaluator with access to hidden TEST labels.

This script uploads a submission JSONL to:
  https://osunlp-travelplannerleaderboard.hf.space/

Input formats
  1) Optimized template JSONL produced by `python -m planner.twin_track`
     (e.g., optimized_test.jsonl), which will be converted to submission JSONL.
  2) Submission JSONL where each line is a JSON object containing at least:
       { "plan": [ ...day records... ] }

Example
  python -m task_helper.travel.runners.eval_leaderboard \
    --split test \
    --eval-mode two-stage \
    --submission artifacts/output/travel/gpt52_test/optimized_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from task_helper.travel.utils.paths import project_root

PROJECT_ROOT = project_root()

DEFAULT_SPACE = "https://osunlp-travelplannerleaderboard.hf.space/"
EXPECTED_LINES = {"validation": 180, "test": 1000}


def _resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


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


def _day_from_phase_key(phase_key: str) -> Optional[int]:
    if not phase_key:
        return None
    tail = phase_key.rsplit("_", 1)[-1]
    try:
        day = int(tail)
    except ValueError:
        return None
    return day if day > 0 else None


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


def _convert_optimized_to_submission(
    *,
    input_path: Path,
    output_path: Path,
    expected_lines: int,
) -> Path:
    by_idx: Dict[int, Dict[str, object]] = {}
    for record in _iter_jsonl(input_path):
        template_id = record.get("template_id")
        idx = _parse_int(template_id)
        if idx is None or idx < 0:
            continue
        eval_plan = _build_eval_plan_from_optimized(record)
        plan_obj = record.get("plan") if isinstance(record.get("plan"), dict) else {}
        query_text = plan_obj.get("query") if isinstance(plan_obj, dict) else None
        by_idx[idx] = {
            "idx": idx + 1,
            "query": query_text if isinstance(query_text, str) else "",
            "plan": eval_plan,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        for idx in range(expected_lines):
            payload = by_idx.get(
                idx,
                {
                    "idx": idx + 1,
                    "query": "",
                    "plan": [],
                },
            )
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return output_path


def _ensure_submission_length(
    *,
    input_path: Path,
    output_path: Path,
    expected_lines: int,
) -> Path:
    items = list(_iter_jsonl(input_path))
    if len(items) == expected_lines:
        return input_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        for idx in range(expected_lines):
            if idx < len(items):
                obj = items[idx]
                plan = obj.get("plan") if isinstance(obj, dict) else None
                if not isinstance(plan, list):
                    obj = dict(obj) if isinstance(obj, dict) else {}
                    obj["plan"] = []
                fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
            else:
                fp.write(json.dumps({"idx": idx + 1, "query": "", "plan": []}, ensure_ascii=False) + "\n")
    return output_path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TravelPlanner leaderboard evaluation via HF Space API.")
    parser.add_argument(
        "--split",
        choices={"validation", "test"},
        default="test",
        help="Dataset split to evaluate (use 'test' for hidden-label official evaluation).",
    )
    parser.add_argument(
        "--eval-mode",
        choices={"two-stage", "sole-planning"},
        default="two-stage",
        help="Leaderboard eval mode label (does not affect scoring on the server).",
    )
    parser.add_argument(
        "--submission",
        type=Path,
        required=True,
        help="Submission JSONL (plan list per line) or optimized template JSONL from `planner.twin_track`.",
    )
    parser.add_argument(
        "--converted-out",
        type=Path,
        default=None,
        help="Optional path to write the submission JSONL before uploading (defaults to <submission>.submission.jsonl).",
    )
    parser.add_argument(
        "--space",
        type=str,
        default=DEFAULT_SPACE,
        help=f"Leaderboard Space URL (default: {DEFAULT_SPACE}).",
    )
    parser.add_argument(
        "--save-detail-json",
        type=Path,
        default=None,
        help="Optional destination path for the downloaded detailed JSON report (defaults to alongside the submission).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    split = str(args.split)
    expected = int(EXPECTED_LINES.get(split, 0) or 0)
    if expected <= 0:
        raise SystemExit(f"Unsupported split: {split}")

    submission_path = _resolve_path(args.submission)
    if submission_path.exists() and submission_path.is_dir():
        preferred = submission_path / f"optimized_{split}.jsonl"
        if preferred.exists():
            submission_path = preferred
        else:
            candidates = sorted(submission_path.glob("*.jsonl"))
            if len(candidates) == 1:
                submission_path = candidates[0]
            else:
                raise SystemExit(f"--submission points to a directory; pass a JSONL file path: {submission_path}")
    if not submission_path.exists():
        raise SystemExit(f"Submission file not found: {submission_path}")

    sample: Optional[Dict[str, object]] = None
    for sample in _iter_jsonl(submission_path):
        break
    if sample is None:
        raise SystemExit(f"Submission file is empty: {submission_path}")

    fmt = _detect_format(sample)
    converted_out = args.converted_out
    if converted_out is None:
        converted_out = submission_path.with_name(submission_path.stem + ".submission.jsonl")
    converted_out = _resolve_path(converted_out)

    upload_path = submission_path
    if fmt == "optimized":
        upload_path = _convert_optimized_to_submission(
            input_path=submission_path,
            output_path=converted_out,
            expected_lines=expected,
        )
        print(f"Converted optimized templates -> submission JSONL: {upload_path}", flush=True)
    elif fmt == "submission":
        upload_path = _ensure_submission_length(
            input_path=submission_path,
            output_path=converted_out,
            expected_lines=expected,
        )
        if upload_path != submission_path:
            print(f"Normalised submission length -> {upload_path}", flush=True)
    else:
        raise SystemExit(
            f"Unsupported submission format in {submission_path}. "
            "Expected either standard submission JSONL (plan is a list) or optimized template JSONL (has actions)."
        )

    try:
        from gradio_client import Client, handle_file  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: gradio_client. Install with `pip install gradio-client`.") from exc

    client = Client(str(args.space))
    result = client.predict(
        split,
        str(args.eval_mode),
        handle_file(str(upload_path)),
        api_name="/add_new_eval",
    )

    markdown: str = ""
    detailed_path: Optional[Path] = None
    if isinstance(result, (list, tuple)) and len(result) >= 1:
        markdown = str(result[0] or "")
        if len(result) >= 2 and result[1]:
            detailed_path = Path(str(result[1]))
    else:
        markdown = str(result or "")

    if markdown:
        print(markdown, flush=True)

    if detailed_path is None:
        return

    dest = args.save_detail_json
    if dest is None:
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        dest = upload_path.parent / f"official_detail_{split}_{ts}.json"
    dest = _resolve_path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(detailed_path, dest)
        print(f"Saved detailed report: {dest}", flush=True)
    except OSError:
        # Still print the original path for debugging.
        print(f"Detailed report path: {detailed_path}", flush=True)


if __name__ == "__main__":
    main()

