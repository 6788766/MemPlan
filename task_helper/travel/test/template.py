"""
Evaluate `planner/init_template.py` (LLM-based) on a dataset split.

This script can:
  1) Generate templates using the LLM prompt under `artifacts/input/<task>/`
  2) Evaluate the extracted `plan` fields vs the dataset columns

Notes:
  - `planner/init_template.py` ALWAYS overwrites these plan fields from the CSV row:
      org, dest, days, date, query
    so performance metrics focus on the remaining fields like:
      visiting_city_number, people_number, budget, local_constraint

Usage:
  # Generate + evaluate (requires OPENAI_API_KEY):
  python -m task_helper.travel.test_performance --split train --generate

  # Evaluate an existing JSONL output:
  python -m task_helper.travel.test_performance --split train --templates artifacts/output/travel/init_templates_train_llm.jsonl
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from planner import init_template as llm_init


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"


def _parse_list_cell(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            return [str(item) for item in parsed if item is not None]
    return [text]


def _read_rows(path: Path) -> List[Dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        return [{k: (v if v != "" else None) for k, v in row.items()} for row in reader]


def _load_jsonl_objects(path: Path) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise TypeError(f"Each line must be a JSON object: {path}")
            items.append(obj)
    return items


def _parse_local_constraint(value: object) -> Optional[Dict[str, object]]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _as_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _match_rate(numer: int, denom: int) -> str:
    return f"{numer}/{denom} ({(numer / denom * 100.0) if denom else 0.0:.1f}%)"


def _ensure_dict(value: object, *, label: str) -> Dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a dict, got {type(value).__name__}")
    return value


def _schema_paths_for_task(task: str) -> Tuple[Path, Path, Path, Optional[Path]]:
    input_dir = ARTIFACTS_ROOT / "input" / task
    prompt_path = input_dir / "template.txt"
    schema_path = input_dir / "schema.jsonl"
    example_path = input_dir / "example.jsonl"
    example_multicity_path = input_dir / "example_multicity.jsonl"
    return prompt_path, schema_path, example_path, (example_multicity_path if example_multicity_path.exists() else None)


def _generate_template_for_row(
    *,
    task: str,
    split: str,
    index: int,
    csv_path: Path,
    model: str,
    plan_fields: Sequence[str],
) -> Dict[str, object]:
    prompt_path, schema_path, example_path, example_multicity_path = _schema_paths_for_task(task)

    row = llm_init._read_csv_row(csv_path, index=index)  # type: ignore[attr-defined]
    if "days" in row and row["days"] is not None:
        try:
            row["days"] = int(row["days"])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
    if "date" in row:
        row["date"] = [str(item) for item in llm_init._parse_list_cell(row.get("date")) if item is not None]  # type: ignore[attr-defined]

    input_payload = {field: row.get(field) for field in plan_fields}
    input_payload["task_name"] = task

    schema_payload = llm_init._load_jsonl_object(schema_path)  # type: ignore[attr-defined]

    example_payload: Optional[Dict[str, object]] = None
    if example_path.exists():
        example_payload = llm_init._load_jsonl_object(example_path)  # type: ignore[attr-defined]
    example_multicity_payload: Optional[Dict[str, object]] = None
    if example_multicity_path is not None:
        example_multicity_payload = llm_init._load_jsonl_object(example_multicity_path)  # type: ignore[attr-defined]

    example_input_payload = llm_init._derive_example_input_payload(  # type: ignore[attr-defined]
        example_payload, plan_fields=plan_fields, task_name=task
    )
    example_multicity_input_payload = llm_init._derive_example_input_payload(  # type: ignore[attr-defined]
        example_multicity_payload, plan_fields=plan_fields, task_name=task
    )

    prompt = llm_init._render_prompt(  # type: ignore[attr-defined]
        prompt_path,
        replacements={
            "{{INPUT_JSON}}": input_payload,
            "{{SCHEMA_JSON}}": schema_payload,
            "{{EXAMPLE_JSON}}": example_payload or {},
            "{{EXAMPLE_INPUT_JSON}}": example_input_payload or {},
            "{{EXAMPLE_MULTICITY_JSON}}": example_multicity_payload or {},
            "{{EXAMPLE_MULTICITY_INPUT_JSON}}": example_multicity_input_payload or {},
        },
    )

    messages = [
        {"role": "system", "content": "You output valid JSON only. No markdown. No extra text."},
        {"role": "user", "content": prompt},
    ]

    content = llm_init._call_openai_chat(model=model, messages=messages)  # type: ignore[attr-defined]
    try:
        template_obj = json.loads(content)
    except json.JSONDecodeError:
        template_obj = json.loads(llm_init._extract_json_block(content))  # type: ignore[attr-defined]

    template = llm_init._ensure_dict(template_obj, label="LLM template output")  # type: ignore[attr-defined]
    template.setdefault("task_name", task)
    template.setdefault("template_id", f"{split}:{index}")
    llm_init._fill_plan_fields(template, source=row, fields=plan_fields)  # type: ignore[attr-defined]
    return template


def _evaluate_templates(
    *,
    rows: List[Mapping[str, object]],
    templates: List[Mapping[str, object]],
) -> None:
    total = min(len(rows), len(templates))
    structural_errors = 0

    ok_budget = 0
    ok_people = 0
    ok_visiting = 0
    ok_transport = 0
    ok_room = 0
    ok_house = 0
    ok_cuisine = 0
    ok_cuisine_set = 0

    for idx in range(total):
        row = rows[idx]
        template = templates[idx]
        try:
            plan = _ensure_dict(template.get("plan"), label=f"template[{idx}].plan")
        except Exception:
            structural_errors += 1
            continue

        # Field-match report (against dataset columns; generation uses only org/dest/days/date/query).
        gt_budget = _as_int(row.get("budget"))
        if plan.get("budget") == gt_budget:
            ok_budget += 1

        gt_people = _as_int(row.get("people_number"))
        if plan.get("people_number") == gt_people:
            ok_people += 1

        gt_visiting = _as_int(row.get("visiting_city_number"))
        if plan.get("visiting_city_number") == gt_visiting:
            ok_visiting += 1

        gt_lc = _parse_local_constraint(row.get("local_constraint")) or {}
        gen_lc = plan.get("local_constraint")
        if isinstance(gen_lc, dict):
            if gen_lc.get("transportation") == gt_lc.get("transportation"):
                ok_transport += 1
            if gen_lc.get("room type") == gt_lc.get("room type"):
                ok_room += 1
            if gen_lc.get("house rule") == gt_lc.get("house rule"):
                ok_house += 1
            if gen_lc.get("cuisine") == gt_lc.get("cuisine"):
                ok_cuisine += 1
            gt_cuisine = gt_lc.get("cuisine")
            gen_cuisine = gen_lc.get("cuisine")
            if (gt_cuisine is None and gen_cuisine is None) or (
                isinstance(gt_cuisine, list) and isinstance(gen_cuisine, list) and set(gt_cuisine) == set(gen_cuisine)
            ):
                ok_cuisine_set += 1

    print(f"rows: {len(rows)}")
    print(f"templates: {len(templates)}")
    print(f"structural_errors: {structural_errors}")
    print("match_vs_dataset:")
    print(f"  budget: {_match_rate(ok_budget, total)}")
    print(f"  people_number: {_match_rate(ok_people, total)}")
    print(f"  visiting_city_number: {_match_rate(ok_visiting, total)}")
    print(f"  local_constraint.transportation: {_match_rate(ok_transport, total)}")
    print(f"  local_constraint.room type: {_match_rate(ok_room, total)}")
    print(f"  local_constraint.house rule: {_match_rate(ok_house, total)}")
    print(f"  local_constraint.cuisine (exact): {_match_rate(ok_cuisine, total)}")
    print(f"  local_constraint.cuisine (set): {_match_rate(ok_cuisine_set, total)}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate planner.init_template performance (travel).")
    parser.add_argument("--task", type=str, default="travel")
    parser.add_argument("--split", type=str, default="train", choices=("train", "validation", "test"))
    parser.add_argument("--model", type=str, default=os.getenv("MEMPLAN_LLM_MODEL", "gpt-5-mini"))
    parser.add_argument("--input", type=Path, help="Override input CSV path.")
    parser.add_argument("--templates", type=Path, help="Existing template JSONL to evaluate.")
    parser.add_argument("--out", type=Path, help="Where to write generated templates (JSONL).")
    parser.add_argument("--generate", action="store_true", help="Generate templates with the LLM (requires OPENAI_API_KEY).")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit evaluation to the first N rows.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    task = str(args.task)
    split = str(args.split)

    input_dir = ARTIFACTS_ROOT / "input" / task
    default_csv = input_dir / "dataset" / f"{split}.csv"
    csv_path = args.input or default_csv
    if not csv_path.exists():
        raise SystemExit(f"Input CSV not found: {csv_path}")

    rows = _read_rows(csv_path)
    if args.max_rows is not None:
        rows = rows[: int(args.max_rows)]

    out_path = args.out
    if out_path is None:
        out_path = ARTIFACTS_ROOT / "output" / task / f"init_templates_{split}_{llm_init._model_slug(str(args.model))}.jsonl"  # type: ignore[attr-defined]

    templates_path = args.templates or out_path

    if args.generate:
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set. Export it before running with --generate.")

        plan_fields = ["org", "dest", "days", "date", "query"]
        generated: List[Dict[str, object]] = []
        failures: List[Tuple[int, str]] = []

        for idx in range(len(rows)):
            try:
                generated.append(
                    _generate_template_for_row(
                        task=task,
                        split=split,
                        index=idx,
                        csv_path=csv_path,
                        model=str(args.model),
                        plan_fields=plan_fields,
                    )
                )
            except Exception as exc:  # pragma: no cover
                failures.append((idx, f"{type(exc).__name__}: {exc}"))
                generated.append({"error": str(exc), "template_id": f"{split}:{idx}", "task_name": task})

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fp:
            for item in generated:
                fp.write(json.dumps(item, ensure_ascii=False) + "\n")

        if failures:
            print(f"generation_failures: {len(failures)}")
            for idx, msg in failures[:5]:
                print(f"  - row {idx}: {msg}")

    if not templates_path.exists():
        raise SystemExit(f"Template JSONL not found: {templates_path} (run with --generate or pass --templates)")

    templates = _load_jsonl_objects(templates_path)
    if args.max_rows is not None:
        templates = templates[: int(args.max_rows)]

    _evaluate_templates(rows=rows, templates=templates)


if __name__ == "__main__":
    main()
