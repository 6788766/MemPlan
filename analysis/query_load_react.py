#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = PROJECT_ROOT / "artifacts" / "output" / "travel" / "analysis" / "query_load"

MODEL_ALIAS = {
    "gpt52": "gpt-5.2",
    "gpt5mini": "gpt-5-mini",
    "gpt5nano": "gpt-5-nano",
    "deepseekchat": "deepseek-chat",
}

try:
    from task_helper.travel.utils.paths import travel_dataset_root
except Exception:  # pragma: no cover - fallback for partial installs
    travel_dataset_root = None

CHECKPOINTS = (200, 400, 600, 800, 1000)


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _resolve_model(model_tag: str) -> str:
    model_tag = str(model_tag or "").strip()
    if model_tag in MODEL_ALIAS:
        return MODEL_ALIAS[model_tag]
    return model_tag


def _dataset_count(split: str) -> int:
    if travel_dataset_root is not None:
        root = travel_dataset_root(PROJECT_ROOT)
    else:
        root = PROJECT_ROOT / "artifacts" / "input" / "travel" / "dataset"
    path = root / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split not found: {path}")
    return sum(1 for _ in path.read_text(encoding="utf-8").splitlines()) - 1


def _write_time_file(path: Path, times: Iterable[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for value in times:
            fp.write(f"{value:.6f}\n")


def _write_time_jsonl(path: Path, times: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for template_id, value in times.items():
            fp.write(json.dumps({"template_id": template_id, "time_s": float(value)}, ensure_ascii=False) + "\n")


def _run_cmd(cmd: Sequence[str], *, log_path: Path, append: bool = False) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with log_path.open(mode, encoding="utf-8") as log_fp:
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(PROJECT_ROOT),
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_fp.write(line)
            log_fp.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        return proc.wait()


def _react_success_count(per_query_dir: Path) -> Tuple[int, int]:
    paths = list(per_query_dir.glob("generated_plan_*.json"))
    success_count = 0
    error_count = 0
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            error_count += 1
            continue
        if payload.get("error"):
            error_count += 1
            continue
        success_count += 1
    return success_count, error_count


def _react_completion_order(per_query_dir: Path) -> List[str]:
    entries: List[Tuple[float, str]] = []
    for path in per_query_dir.glob("generated_plan_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("error"):
            continue
        template_id = str(payload.get("idx") or "").strip()
        if not template_id:
            continue
        entries.append((path.stat().st_mtime, template_id))
    entries.sort(key=lambda x: x[0])
    return [template_id for _, template_id in entries]


def _monitor_checkpoints(
    *,
    per_query_dir: Path,
    checkpoints: Sequence[int],
    stop_event: threading.Event,
    checkpoint_times: Dict[int, float],
    poll_s: float = 5.0,
) -> None:
    done: set[int] = set()
    checkpoints_sorted = sorted(set(int(x) for x in checkpoints))
    start = time.perf_counter()
    while not stop_event.is_set():
        count, _ = _react_success_count(per_query_dir)
        for threshold in checkpoints_sorted:
            if threshold in done:
                continue
            if count < threshold:
                continue
            checkpoint_times[threshold] = round(time.perf_counter() - start, 6)
            done.add(threshold)
        stop_event.wait(poll_s)


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


def _usage_sum(ids: Iterable[str], usage_by_id: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    total = {"prompt_cache_hit": 0, "prompt_cache_miss": 0, "output": 0, "total": 0}
    for template_id in ids:
        usage = usage_by_id.get(template_id)
        if not usage:
            continue
        for key in total:
            total[key] += int(usage.get(key) or 0)
    if total["total"] <= 0:
        total["total"] = total["prompt_cache_hit"] + total["prompt_cache_miss"] + total["output"]
    return total


def _parse_react_usage(per_query_dir: Path) -> Dict[str, Dict[str, int]]:
    usage_by_id: Dict[str, Dict[str, int]] = {}
    for path in per_query_dir.glob("generated_plan_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("error"):
            continue
        template_id = str(payload.get("idx") or "").strip()
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        if template_id and metrics:
            usage_by_id[template_id] = _normalize_usage(metrics)
    return usage_by_id


def _write_checkpoint_metrics(
    *,
    output_root: Path,
    model_tag: str,
    checkpoint_times: Dict[int, float],
    usage_by_threshold: Dict[int, Dict[str, int]],
    price_by_threshold: Dict[int, float],
    ids_by_threshold: Dict[int, List[str]],
) -> None:
    for threshold, time_s in checkpoint_times.items():
        payload = {
            "time_s": round(float(time_s), 6),
            "tokens": dict(usage_by_threshold.get(threshold, {})),
            "price_usd": round(float(price_by_threshold.get(threshold, 0.0)), 6),
        }
        out_path = output_root / f"{model_tag}_react_{threshold}.json"
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        ids_path = output_root / f"{model_tag}_react_{threshold}_ids.json"
        ids_path.write_text(
            json.dumps({"template_ids": list(ids_by_threshold.get(threshold, []))}, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ReAct only with checkpoint aggregation.")
    parser.add_argument("--model", type=str, required=True, help="Model tag/name (e.g., gpt52).")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--retry-sleep", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_tag = args.model
    model_name = _resolve_model(model_tag)
    split = args.split

    method_root = args.out_root.expanduser().resolve() / f"{model_tag}_react"
    method_root.mkdir(parents=True, exist_ok=True)
    log_path = method_root / f"{model_tag}_react_{_timestamp()}.log"

    cmd_base = [
        sys.executable,
        "-m",
        "baseline.tool_agents",
        "--set_type",
        split,
        "--model_name",
        model_name,
        "--workers",
        str(args.workers),
        "--output_dir",
        str(method_root),
        "--resume-failed",
    ]

    run_dir = method_root / f"{model_tag}_{split}"
    per_query_dir = run_dir / f"two_stage_{model_tag}_{split}"

    stop_event = threading.Event()
    checkpoint_times: Dict[int, float] = {}
    monitor = threading.Thread(
        target=_monitor_checkpoints,
        kwargs={
            "per_query_dir": per_query_dir,
            "checkpoints": CHECKPOINTS,
            "stop_event": stop_event,
            "checkpoint_times": checkpoint_times,
        },
        daemon=True,
    )
    monitor.start()

    total_queries = _dataset_count(split)
    attempt = 0
    while True:
        status = _run_cmd(cmd_base, log_path=log_path, append=attempt > 0)
        success_count, error_count = _react_success_count(per_query_dir)
        missing = max(0, total_queries - success_count)
        if status == 0 and error_count == 0 and missing == 0:
            break
        attempt += 1
        if args.max_retries > 0 and attempt > args.max_retries:
            stop_event.set()
            monitor.join(timeout=5.0)
            raise RuntimeError(
                f"ReAct failed after {attempt} attempts (errors={error_count}, missing={missing})."
            )
        time.sleep(max(1.0, float(args.retry_sleep)))

    stop_event.set()
    monitor.join(timeout=5.0)

    times: List[float] = []
    time_map: Dict[str, float] = {}
    for path in per_query_dir.glob("generated_plan_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        value = metrics.get("time_s")
        if isinstance(value, (int, float)):
            times.append(float(value))
            template_id = str(payload.get("idx") or "").strip()
            if template_id:
                time_map[template_id] = float(value)

    time_path = method_root / f"{model_tag}_react.txt"
    _write_time_file(time_path, times)
    _write_time_jsonl(time_path.with_suffix(".jsonl"), time_map)

    usage_by_id = _parse_react_usage(per_query_dir)
    order = _react_completion_order(per_query_dir)

    price_table = json.loads((PROJECT_ROOT / "artifacts" / "input" / "price.json").read_text(encoding="utf-8"))
    model_key = MODEL_ALIAS.get(model_tag, model_tag)
    if model_key not in price_table and model_name in price_table:
        model_key = model_name
    rates = price_table.get(model_key, {})

    usage_by_threshold: Dict[int, Dict[str, int]] = {}
    price_by_threshold: Dict[int, float] = {}
    ids_by_threshold: Dict[int, List[str]] = {}
    for threshold in CHECKPOINTS:
        ids = order[:threshold]
        usage_sum = _usage_sum(ids, usage_by_id)
        usage_by_threshold[threshold] = usage_sum
        price_by_threshold[threshold] = _price_for_usage(usage_sum, rates)
        ids_by_threshold[threshold] = list(ids)

    _write_checkpoint_metrics(
        output_root=method_root,
        model_tag=model_tag,
        checkpoint_times=checkpoint_times,
        usage_by_threshold=usage_by_threshold,
        price_by_threshold=price_by_threshold,
        ids_by_threshold=ids_by_threshold,
    )

    print(f"Done. Outputs in {method_root}")


if __name__ == "__main__":
    main()
