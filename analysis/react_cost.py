#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRICE_PATH = PROJECT_ROOT / "artifacts" / "input" / "price.json"

MODEL_ALIAS = {
    "gpt52": "gpt-5.2",
    "gpt5mini": "gpt-5-mini",
    "gpt5nano": "gpt-5-nano",
    "deepseekchat": "deepseek-chat",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate ReAct per-query token usage and price from generated_plan_*.json files."
    )
    parser.add_argument("--model", type=str, help="Model tag/name (e.g., gpt52).")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "output" / "travel" / "analysis" / "query_load",
        help="Root folder containing run outputs (default: artifacts/output/travel/analysis/query_load).",
    )
    parser.add_argument(
        "--per-query-dir",
        type=Path,
        default=None,
        help="Explicit per-query dir (overrides model/split lookup).",
    )
    parser.add_argument(
        "--include-errors",
        action="store_true",
        help="Include queries with non-empty error fields.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def _resolve_model_key(model: str, price_table: Dict[str, object]) -> str:
    if model in price_table:
        return model
    model_norm = model.lower()
    for key in price_table:
        if key.lower() in model_norm:
            return key
    if model_norm in MODEL_ALIAS:
        alias = MODEL_ALIAS[model_norm]
        if alias in price_table:
            return alias
    raise ValueError(f"Unknown model for price table: {model}")


def _resolve_run_root(runs_root: Path, model_tag: str) -> Path:
    pattern = f"{model_tag}_react_*"
    candidates = sorted(runs_root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No run folders found for pattern {pattern} under {runs_root}")
    return candidates[0]


def _resolve_per_query_dir(runs_root: Path, model_tag: str, split: str) -> Path:
    run_root = _resolve_run_root(runs_root, model_tag)
    per_query_dir = run_root / f"{model_tag}_{split}" / f"two_stage_{model_tag}_{split}"
    if not per_query_dir.exists():
        raise FileNotFoundError(f"Per-query dir not found: {per_query_dir}")
    return per_query_dir


def _normalize_usage(metrics: Dict[str, object]) -> Dict[str, int]:
    cached = int(metrics.get("prompt_cache_hit_tokens") or metrics.get("prompt_cache_hit") or 0)
    uncached = int(metrics.get("prompt_cache_miss_tokens") or metrics.get("prompt_cache_miss") or 0)
    output = int(metrics.get("completion_tokens") or metrics.get("output_tokens") or metrics.get("output") or 0)
    total = int(metrics.get("total_tokens") or metrics.get("total") or 0)
    prompt_tokens = int(metrics.get("prompt_tokens") or metrics.get("prompt") or 0)
    if cached and not uncached and prompt_tokens:
        uncached = max(0, prompt_tokens - cached)
    if total <= 0:
        total = cached + uncached + output
    return {
        "prompt_cache_hit": cached,
        "prompt_cache_miss": uncached,
        "output": output,
        "total": total,
    }


def _price_for_usage(usage: Dict[str, int], rates: Dict[str, object]) -> float:
    hit_rate = float(rates.get("prompt_cache_hit", rates.get("cached", 0.0)) or 0.0)
    miss_rate = float(rates.get("prompt_cache_miss", rates.get("input", 0.0)) or 0.0)
    out_rate = float(rates.get("output", 0.0) or 0.0)
    million = 1_000_000.0
    return (
        (usage.get("prompt_cache_hit", 0) / million) * hit_rate
        + (usage.get("prompt_cache_miss", 0) / million) * miss_rate
        + (usage.get("output", 0) / million) * out_rate
    )


def _accumulate(values: Iterable[Dict[str, int]]) -> Dict[str, int]:
    total = {"prompt_cache_hit": 0, "prompt_cache_miss": 0, "output": 0, "total": 0}
    for usage in values:
        for key in total:
            total[key] += int(usage.get(key) or 0)
    if total["total"] <= 0:
        total["total"] = total["prompt_cache_hit"] + total["prompt_cache_miss"] + total["output"]
    return total


def main() -> None:
    args = _parse_args()
    runs_root = args.runs_root.expanduser().resolve()

    if args.per_query_dir is not None:
        per_query_dir = args.per_query_dir.expanduser().resolve()
        model_tag = args.model or ""
    else:
        if not args.model:
            raise SystemExit("--model is required when --per-query-dir is not provided")
        model_tag = args.model
        per_query_dir = _resolve_per_query_dir(runs_root, model_tag, args.split)

    price_table = json.loads(PRICE_PATH.read_text(encoding="utf-8"))
    model_key = _resolve_model_key(model_tag, price_table)
    rates = price_table.get(model_key, {})

    usage_list = []
    error_count = 0
    total_count = 0
    for path in per_query_dir.glob("generated_plan_*.json"):
        total_count += 1
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("error") and not args.include_errors:
            error_count += 1
            continue
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        usage_list.append(_normalize_usage(metrics))

    usage_total = _accumulate(usage_list)
    price_total = _price_for_usage(usage_total, rates)

    result = {
        "model": model_tag,
        "model_price_key": model_key,
        "per_query_dir": str(per_query_dir),
        "queries_total": total_count,
        "queries_counted": len(usage_list),
        "errors_skipped": error_count,
        "tokens": usage_total,
        "price_usd": round(price_total, 6),
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)

    if args.out:
        out_path = args.out.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
