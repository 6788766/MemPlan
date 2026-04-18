#!/usr/bin/env python3
"""
Append LLM dollar cost to a MemPlan run's cost.txt.

Reads per-1M-token prices from `artifacts/input/price.json` and token usage from
`cost.txt` lines like:
  - LLM token usage (init_template): ... prompt_cache_hit=... prompt_cache_miss=... output=... total=...
  - LLM token usage (llm_repair): ... prompt_cache_hit=... prompt_cache_miss=... output=... total=...

Notes:
  - Cached/uncached input tokens are provider-specific; when only `prompt_tokens` is present,
    cached breakdown defaults to hit=0, miss=prompt_tokens.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, getcontext
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRICE_PATH = PROJECT_ROOT / "artifacts" / "input" / "price.json"

_KV_INT_RE = re.compile(r"\b([A-Za-z_]+)=([0-9]+)\b")
_TOTAL_COST_MARKER = "--------------total cost-----------------"


@dataclass(frozen=True)
class TokenUsage:
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0
    completion_tokens: int = 0
    prompt_tokens: int = 0  # legacy / fallback (may be total or miss, depending on the logger)
    cached_tokens: int = 0  # legacy / fallback


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


def _parse_usage_line(line: str) -> TokenUsage:
    values: Dict[str, int] = {}
    for key, raw_value in _KV_INT_RE.findall(line):
        k = key.strip().lower()
        try:
            v = int(raw_value)
        except ValueError:
            continue
        values[k] = v

    prompt_cache_hit = values.get("prompt_cache_hit")
    if prompt_cache_hit is None:
        prompt_cache_hit = values.get("prompt_cache_hit_tokens")

    prompt_cache_miss = values.get("prompt_cache_miss")
    if prompt_cache_miss is None:
        prompt_cache_miss = values.get("prompt_cache_miss_tokens")

    prompt = values.get("prompt")
    if prompt is None:
        prompt = values.get("prompt_tokens")
    if prompt is None:
        prompt = values.get("input")
    if prompt is None:
        prompt = values.get("input_tokens")

    completion = values.get("output")
    if completion is None:
        completion = values.get("output_tokens")
    if completion is None:
        completion = values.get("completion")
    if completion is None:
        completion = values.get("completion_tokens")

    cached = values.get("cached")
    if cached is None:
        cached = values.get("cache")
    if cached is None:
        cached = values.get("cached_tokens")
    if cached is None:
        cached = values.get("cached_prompt_tokens")

    return TokenUsage(
        prompt_cache_hit_tokens=int(prompt_cache_hit or 0),
        prompt_cache_miss_tokens=int(prompt_cache_miss or 0),
        completion_tokens=int(completion or 0),
        prompt_tokens=int(prompt or 0),
        cached_tokens=int(cached or 0),
    )

def _normalise_usage(usage: TokenUsage) -> TokenUsage:
    hit = usage.prompt_cache_hit_tokens or usage.cached_tokens
    miss = usage.prompt_cache_miss_tokens or usage.prompt_tokens
    if usage.prompt_cache_hit_tokens and not usage.prompt_cache_miss_tokens and usage.prompt_tokens:
        miss = max(0, int(usage.prompt_tokens) - int(usage.prompt_cache_hit_tokens))
    return TokenUsage(
        prompt_cache_hit_tokens=int(hit),
        prompt_cache_miss_tokens=int(miss),
        completion_tokens=int(usage.completion_tokens),
        prompt_tokens=int(usage.prompt_tokens),
        cached_tokens=int(usage.cached_tokens),
    )


def _find_usage(lines: list[str], stage: str) -> TokenUsage:
    prefix = f"LLM token usage ({stage}):"
    for line in reversed(lines):
        if prefix in line:
            return _normalise_usage(_parse_usage_line(line))
    return TokenUsage()

def _find_baseline_usage(lines: list[str]) -> TokenUsage:
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith("llm_tokens:"):
            return _normalise_usage(_parse_usage_line(stripped))
        if stripped.startswith("LLM token usage (baseline):"):
            return _normalise_usage(_parse_usage_line(stripped))
    return TokenUsage()

def _read_kv(lines: list[str], key: str) -> Optional[str]:
    prefix = f"{key}="
    for line in reversed(lines):
        if line.startswith(prefix):
            return line.split("=", 1)[1].strip()
    return None


def _read_model(lines: list[str]) -> str:
    for line in lines:
        if line.startswith("model="):
            return line.split("=", 1)[1].strip()
    raise ValueError("Could not find `model=...` in cost.txt.")


def _resolve_price_key(model: str, price_table: Mapping[str, object]) -> str:
    if model in price_table:
        return model

    model_norm = str(model or "").strip().lower()
    for key in price_table.keys():
        if str(key).lower() in model_norm:
            return str(key)

    if "deepseek" in model_norm and "deepseek" in price_table:
        return "deepseek"

    for candidate in ("gpt-5.2", "gpt-5-mini", "gpt-5-nano"):
        if candidate in model_norm and candidate in price_table:
            return candidate

    raise ValueError(f"Unknown model `{model}`; no matching key in {sorted(map(str, price_table.keys()))}.")


def _as_decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Invalid price value: {value!r}") from exc


def _compute_cost_usd(*, usage: TokenUsage, rates: Mapping[str, Any]) -> Decimal:
    getcontext().prec = 28
    million = Decimal(1_000_000)

    cache_hit_rate = _as_decimal(rates.get("prompt_cache_hit", rates.get("cached", 0)))
    cache_miss_rate = _as_decimal(
        rates.get("prompt_cache_miss", rates.get("cprompt_cache_miss", rates.get("input", 0)))
    )
    output_rate = _as_decimal(rates.get("output", 0))

    cache_hit = Decimal(int(usage.prompt_cache_hit_tokens))
    cache_miss = Decimal(int(usage.prompt_cache_miss_tokens))
    completion = Decimal(int(usage.completion_tokens))

    return (cache_hit / million) * cache_hit_rate + (cache_miss / million) * cache_miss_rate + (completion / million) * output_rate


def _fmt_usd(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append MemPlan LLM dollar cost to a run's cost.txt.")
    parser.add_argument("cost_path", type=Path, help="Path to cost.txt.")
    parser.add_argument(
        "--prices",
        type=Path,
        default=DEFAULT_PRICE_PATH,
        help="Path to price.json (default: artifacts/input/price.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the computed value but do not modify cost.txt.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    cost_path: Path = args.cost_path.expanduser().resolve()
    price_path: Path = args.prices.expanduser().resolve()

    original_text = cost_path.read_text(encoding="utf-8")
    lines = original_text.splitlines()
    model = _read_model(lines)

    price_table = json.loads(price_path.read_text(encoding="utf-8"))
    if not isinstance(price_table, dict):
        raise TypeError(f"Expected a JSON object in {price_path}")

    model_key = _resolve_price_key(model, price_table)
    rates = price_table.get(model_key)
    if not isinstance(rates, dict):
        raise TypeError(f"Expected a JSON object for model `{model_key}` in {price_path}")

    init_usage = _find_usage(lines, "init_template")
    repair_usage = _find_usage(lines, "llm_repair")
    baseline_usage = _find_baseline_usage(lines)

    init_cost = _compute_cost_usd(usage=init_usage, rates=rates)
    repair_cost = _compute_cost_usd(usage=repair_usage, rates=rates)
    baseline_cost = _compute_cost_usd(usage=baseline_usage, rates=rates)

    has_pipeline_usage = any(
        (
            init_usage.prompt_cache_hit_tokens,
            init_usage.prompt_cache_miss_tokens,
            init_usage.completion_tokens,
            repair_usage.prompt_cache_hit_tokens,
            repair_usage.prompt_cache_miss_tokens,
            repair_usage.completion_tokens,
        )
    )
    has_baseline_usage = any((baseline_usage.prompt_cache_hit_tokens, baseline_usage.prompt_cache_miss_tokens, baseline_usage.completion_tokens))

    if has_pipeline_usage:
        total_cost = init_cost + repair_cost
    elif has_baseline_usage:
        total_cost = baseline_cost
    else:
        total_cost = init_cost + repair_cost

    total_tokens = None
    raw_total_tokens = _read_kv(lines, "pipeline_total_tokens") or _read_kv(lines, "tokens_total")
    if raw_total_tokens is not None:
        try:
            total_tokens = int(raw_total_tokens)
        except ValueError:
            total_tokens = None
    # Treat an explicit 0 as "missing" when we have usage lines to recompute from.
    # This keeps old/buggy runner logs regeneratable without rerunning the pipeline.
    if total_tokens is not None and total_tokens <= 0 and has_pipeline_usage:
        total_tokens = None
    if total_tokens is None:
        if has_pipeline_usage:
            total_tokens = (
                int(init_usage.prompt_cache_hit_tokens)
                + int(init_usage.prompt_cache_miss_tokens)
                + int(init_usage.completion_tokens)
                + int(repair_usage.prompt_cache_hit_tokens)
                + int(repair_usage.prompt_cache_miss_tokens)
                + int(repair_usage.completion_tokens)
            )
        else:
            total_tokens = (
                int(baseline_usage.prompt_cache_hit_tokens)
                + int(baseline_usage.prompt_cache_miss_tokens)
                + int(baseline_usage.completion_tokens)
            )

    total_time_s = None
    raw_total_time = _read_kv(lines, "pipeline_elapsed_s_no_eval") or _read_kv(lines, "pipeline_elapsed_s_total") or _read_kv(lines, "elapsed_s_total")
    if raw_total_time is not None:
        try:
            total_time_s = float(raw_total_time)
        except ValueError:
            total_time_s = None

    out_lines = [_TOTAL_COST_MARKER, f"llm_price_model_key={model_key}"]
    if has_pipeline_usage:
        out_lines.extend(
            [
                f"llm_price_usd_init_template={_fmt_usd(init_cost)}",
                f"llm_price_usd_llm_repair={_fmt_usd(repair_cost)}",
            ]
        )
    elif has_baseline_usage:
        out_lines.append(f"llm_price_usd_baseline={_fmt_usd(baseline_cost)}")
    out_lines.extend([f"llm_price_usd_total={_fmt_usd(total_cost)}", f"total_tokens={total_tokens}"])
    if total_time_s is not None:
        out_lines.append(f"total_time_s={total_time_s}")
    text = "\n".join(out_lines) + "\n"

    print(text, end="")

    if args.dry_run:
        return

    # Make the output idempotent: if a previous total-cost block exists, replace it.
    base = original_text
    marker_idx = base.find(_TOTAL_COST_MARKER)
    if marker_idx != -1:
        base = base[:marker_idx].rstrip() + "\n"
    elif _needs_newline(cost_path):
        base = base + "\n"
    cost_path.write_text(base + text, encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"money_memplan: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
