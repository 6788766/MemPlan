#!/usr/bin/env python3
"""
WorkBench repair run reporting helpers.

Responsibilities:
  - Append a trimmed pass_rates.json (drop `constraints`) to results.txt.
  - Append a minimal baseline vs repair comparison (accuracy / side effects).
  - Append/replace a combined multi-model total-cost block in repair/cost.txt.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

from task_helper import money_memplan


TOTAL_COST_MARKER = "--------------total cost-----------------"
_TOTAL_TOKENS_RE = re.compile(r"\btotal=([0-9]+)\b")
_ELAPSED_S_RE = re.compile(r"\belapsed_s=([0-9]+)\b")


def _sum_stage_elapsed_seconds(lines: list[str], stage: str) -> int:
    total = 0
    token = f"] END {stage} "
    for line in lines:
        if token not in line:
            continue
        match = _ELAPSED_S_RE.search(line)
        if not match:
            continue
        try:
            total += int(match.group(1))
        except Exception:
            continue
    return int(total)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report WorkBench repair run results and combined costs.")
    parser.add_argument("--base-cost-txt", type=Path, required=True)
    parser.add_argument("--repair-cost-txt", type=Path, required=True)
    parser.add_argument("--repair-elapsed-s", type=float, required=True)
    parser.add_argument("--repair-model", type=str, default="", help="Override repair model for pricing (default: model= in repair cost.txt).")
    parser.add_argument("--base-pass-rates-json", type=Path, default=None)
    parser.add_argument("--repair-pass-rates-json", type=Path, default=None)
    parser.add_argument("--repair-results-txt", type=Path, default=None, help="Append human-readable summary to this file.")
    return parser.parse_args(argv)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_lines(path: Path) -> list[str]:
    return _read_text(path).splitlines() if path.exists() else []


def _token_total_for_stage(lines: list[str], stage: str) -> int:
    prefix = f"LLM token usage ({stage}):"
    for line in reversed(lines):
        if prefix not in line:
            continue
        match = _TOTAL_TOKENS_RE.search(line)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return 0
        return 0
    return 0


def _read_elapsed_seconds(lines: list[str]) -> int:
    pipeline_total = None
    pipeline_no_eval = None
    total_time = None
    summary = None
    for line in lines:
        if line.startswith("pipeline_elapsed_s_no_eval="):
            try:
                pipeline_no_eval = int(float(line.split("=", 1)[1].strip()))
            except Exception:
                pipeline_no_eval = None
        elif line.startswith("pipeline_elapsed_s_total="):
            try:
                pipeline_total = int(float(line.split("=", 1)[1].strip()))
            except Exception:
                pipeline_total = None
        elif line.startswith("total_time_s="):
            try:
                total_time = int(float(line.split("=", 1)[1].strip()))
            except Exception:
                total_time = None
        elif "summary: time_s_total=" in line:
            match = re.search(r"\btime_s_total=([0-9]+(?:\.[0-9]+)?)\b", line)
            if match:
                summary = int(float(match.group(1)))
    for candidate in (pipeline_no_eval, pipeline_total, summary, total_time):
        if candidate is not None and candidate > 0:
            return candidate
    return 0


def _find_model(lines: list[str]) -> str:
    for line in lines:
        if line.startswith("model="):
            return line.split("=", 1)[1].strip()
    return ""


def _rates_for(model: str) -> tuple[str, dict[str, Any]]:
    price_table = json.loads(money_memplan.DEFAULT_PRICE_PATH.read_text(encoding="utf-8"))
    model_key = money_memplan._resolve_price_key(model, price_table)
    rates = price_table.get(model_key)
    if not isinstance(rates, dict):
        rates = {}
    return model_key, rates


def _usd_total_for_cost(lines: list[str], model: str) -> float:
    if not model:
        return 0.0
    _model_key, rates = _rates_for(model)
    init_usage = money_memplan._find_usage(lines, "init_template")
    repair_usage = money_memplan._find_usage(lines, "llm_repair")
    init_cost = money_memplan._compute_cost_usd(usage=init_usage, rates=rates)
    repair_cost = money_memplan._compute_cost_usd(usage=repair_usage, rates=rates)
    return float(init_cost + repair_cost)


def _usd_llm_repair_only(lines: list[str], model: str) -> float:
    if not model:
        return 0.0
    _model_key, rates = _rates_for(model)
    usage = money_memplan._find_usage(lines, "llm_repair")
    cost = money_memplan._compute_cost_usd(usage=usage, rates=rates)
    return float(cost)


def _write_cost_block(
    *,
    base_cost_txt: Path,
    repair_cost_txt: Path,
    repair_elapsed_s: int,
    repair_model: str,
) -> None:
    base_lines = _read_lines(base_cost_txt)
    repair_lines = _read_lines(repair_cost_txt)

    base_init_tokens = _token_total_for_stage(base_lines, "init_template") or _token_total_for_stage(repair_lines, "init_template")
    base_repair_tokens = _token_total_for_stage(base_lines, "llm_repair")
    repair_tokens = _token_total_for_stage(repair_lines, "llm_repair")
    total_tokens = int(base_init_tokens + base_repair_tokens + repair_tokens)

    base_elapsed = _read_elapsed_seconds(base_lines)
    repair_eval_elapsed = _sum_stage_elapsed_seconds(repair_lines, "eval")
    repair_llm_elapsed = _sum_stage_elapsed_seconds(repair_lines, "llm_repair")
    repair_no_eval_elapsed = repair_llm_elapsed
    total_elapsed = int(base_elapsed + repair_llm_elapsed)

    base_model = _find_model(base_lines)
    merged_base_lines = list(base_lines)
    if base_init_tokens and _token_total_for_stage(base_lines, "init_template") == 0:
        for line in repair_lines:
            if "LLM token usage (init_template):" in line:
                merged_base_lines.append(line)
                break

    base_usd_total = _usd_total_for_cost(merged_base_lines, base_model)
    repair_usd = _usd_llm_repair_only(repair_lines, repair_model)
    total_usd = float(base_usd_total + repair_usd)

    text = _read_text(repair_cost_txt) if repair_cost_txt.exists() else ""
    marker_idx = text.find(TOTAL_COST_MARKER)
    if marker_idx != -1:
        text = text[:marker_idx].rstrip() + "\n"
    if text and not text.endswith("\n"):
        text += "\n"

    text += f"pipeline_total_tokens={total_tokens}\n"
    text += f"pipeline_elapsed_s_total={total_elapsed}\n"
    text += f"pipeline_elapsed_s_repair_total={int(repair_llm_elapsed)}\n"
    text += f"pipeline_elapsed_s_repair_no_eval={int(repair_no_eval_elapsed)}\n"
    text += f"{TOTAL_COST_MARKER}\n"
    text += f"llm_price_model_key={repair_model}\n"
    text += f"llm_price_usd_init_template={base_usd_total:.6f}\n"
    text += f"llm_price_usd_llm_repair={repair_usd:.6f}\n"
    text += f"llm_price_usd_total={total_usd:.6f}\n"
    text += f"total_tokens={total_tokens}\n"
    text += f"total_time_s={float(total_elapsed):.1f}\n"

    repair_cost_txt.write_text(text, encoding="utf-8")


def _append_trimmed_pass_rates(*, pass_rates_json: Path, out_path: Path) -> None:
    if not pass_rates_json.exists():
        return
    obj = json.loads(pass_rates_json.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        obj.pop("constraints", None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(obj, indent=2))
        fp.write("\n")


def _append_improvement_summary(*, base_path: Path, repair_path: Path, out_path: Path) -> None:
    if not base_path.exists() or not repair_path.exists():
        return
    base = json.loads(base_path.read_text(encoding="utf-8"))
    repair = json.loads(repair_path.read_text(encoding="utf-8"))
    if not isinstance(base, dict) or not isinstance(repair, dict):
        return

    def _num(obj: dict[str, Any], key: str) -> Optional[float]:
        v = obj.get(key)
        try:
            return float(v)  # type: ignore[arg-type]
        except Exception:
            return None

    base_acc = _num(base, "WorkBench Accuracy")
    repair_acc = _num(repair, "WorkBench Accuracy")
    base_se = _num(base, "WorkBench Unwanted Side Effects")
    repair_se = _num(repair, "WorkBench Unwanted Side Effects")

    lines: list[str] = []
    lines.append("")
    lines.append("========== Repair Improvement Summary ==========")
    lines.append(f"baseline_pass_rates_json={base_path}")
    lines.append(f"repair_pass_rates_json={repair_path}")

    if base_acc is not None and repair_acc is not None:
        lines.append(f"WorkBench Accuracy: {base_acc} -> {repair_acc} (improved={repair_acc > base_acc})")
    else:
        lines.append("WorkBench Accuracy: (missing)")

    if base_se is not None and repair_se is not None:
        lines.append(f"WorkBench Unwanted Side Effects: {base_se} -> {repair_se} (improved={repair_se < base_se})")
    else:
        lines.append("WorkBench Unwanted Side Effects: (missing)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    base_cost_txt: Path = args.base_cost_txt.expanduser().resolve()
    repair_cost_txt: Path = args.repair_cost_txt.expanduser().resolve()

    base_lines = _read_lines(base_cost_txt)
    repair_lines = _read_lines(repair_cost_txt)

    repair_model = str(args.repair_model or _find_model(repair_lines) or "")
    repair_elapsed_s = int(float(args.repair_elapsed_s))

    _write_cost_block(
        base_cost_txt=base_cost_txt,
        repair_cost_txt=repair_cost_txt,
        repair_elapsed_s=repair_elapsed_s,
        repair_model=repair_model,
    )

    out_txt = args.repair_results_txt
    if out_txt is not None:
        out_txt = out_txt.expanduser().resolve()
        if args.repair_pass_rates_json is not None:
            _append_trimmed_pass_rates(pass_rates_json=args.repair_pass_rates_json.expanduser().resolve(), out_path=out_txt)
        if args.base_pass_rates_json is not None and args.repair_pass_rates_json is not None:
            _append_improvement_summary(
                base_path=args.base_pass_rates_json.expanduser().resolve(),
                repair_path=args.repair_pass_rates_json.expanduser().resolve(),
                out_path=out_txt,
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"report_repair_run: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
