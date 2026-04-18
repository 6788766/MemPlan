#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
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


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _resolve_model(model_tag: str) -> str:
    model_tag = str(model_tag or "").strip()
    if model_tag in MODEL_ALIAS:
        return MODEL_ALIAS[model_tag]
    return model_tag


def _dataset_path(split: str) -> Path:
    if travel_dataset_root is not None:
        root = travel_dataset_root(PROJECT_ROOT)
    else:
        root = PROJECT_ROOT / "artifacts" / "input" / "travel" / "dataset"
    return root / f"{split}.csv"


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        return [dict(row) for row in reader]


def _write_chunk_csv(
    *,
    rows: List[Dict[str, str]],
    fieldnames: Sequence[str],
    start_index: int,
    out_path: Path,
) -> None:
    needs_plan_id = "plan_id" not in fieldnames
    header = list(fieldnames)
    if needs_plan_id:
        header.append("plan_id")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=header)
        writer.writeheader()
        for offset, row in enumerate(rows):
            out_row = dict(row)
            if not out_row.get("plan_id"):
                out_row["plan_id"] = str(start_index + offset)
            writer.writerow(out_row)


def _run_cmd(
    cmd: Sequence[str],
    *,
    log_path: Path,
    env: Optional[Dict[str, str]] = None,
    line_handler=None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_fp:
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(PROJECT_ROOT),
            env=env or os.environ.copy(),
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
            if line_handler:
                line_handler(line)
        status = proc.wait()
    if status != 0:
        raise RuntimeError(f"Command failed ({status}): {' '.join(cmd)}")


def _chunk_ranges(total: int, chunk_size: int) -> List[Tuple[int, int]]:
    out = []
    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        out.append((start, end))
    return out


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


def _sum_usage(values: List[Dict[str, int]]) -> Dict[str, int]:
    total = {"prompt_cache_hit": 0, "prompt_cache_miss": 0, "output": 0, "total": 0}
    for usage in values:
        for key in total:
            total[key] += int(usage.get(key) or 0)
    if total["total"] <= 0:
        total["total"] = total["prompt_cache_hit"] + total["prompt_cache_miss"] + total["output"]
    return total


def _sum_init_usage(path: Path) -> Dict[str, int]:
    usages: List[Dict[str, int]] = []
    if not path.exists():
        return _sum_usage(usages)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        notes = obj.get("notes") if isinstance(obj.get("notes"), dict) else {}
        llm = notes.get("llm") if isinstance(notes.get("llm"), dict) else {}
        if llm:
            usages.append(_normalize_usage(llm))
    return _sum_usage(usages)


def _sum_repair_usage(path: Path) -> Dict[str, int]:
    usages: List[Dict[str, int]] = []
    if not path.exists():
        return _sum_usage(usages)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        notes = obj.get("notes") if isinstance(obj.get("notes"), dict) else {}
        repair = notes.get("llm_repair") if isinstance(notes.get("llm_repair"), dict) else {}
        usage = repair.get("usage") if isinstance(repair.get("usage"), dict) else {}
        if usage:
            usages.append(_normalize_usage(usage))
    return _sum_usage(usages)


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


def _read_price_table() -> Dict[str, Dict[str, float]]:
    path = PROJECT_ROOT / "artifacts" / "input" / "price.json"
    return json.loads(path.read_text(encoding="utf-8"))


class ChunkTracker:
    def __init__(self, *, start_index: int, total: int) -> None:
        self.start_index = int(start_index)
        self.total = int(total)
        self.stage_start: Dict[str, float] = {}
        self.stage_times: Dict[str, Dict[str, float]] = {
            "init_template": {},
            "compose_match": {},
            "mcts": {},
            "llm_repair": {},
        }
        self.view_select_elapsed_s: float = 0.0

        self._init_re = re.compile(r"Processing query\s+[^:]+:(\d+)")
        self._template_re = re.compile(r"template_id=([\w:-]+)")

    def start_stage(self, stage: str) -> None:
        self.stage_start[stage] = time.perf_counter()

    def end_stage(self, stage: str) -> None:
        if stage == "view_select" and stage in self.stage_start:
            self.view_select_elapsed_s = max(0.0, time.perf_counter() - self.stage_start[stage])

    def handle_line(self, stage: str, line: str) -> None:
        now = time.perf_counter()
        if stage == "llm_repair" and line.startswith("LLM repair:"):
            # Reset to exclude load/IO before repair compute begins.
            self.stage_start[stage] = now
            return
        if stage == "init_template":
            match = self._init_re.search(line)
            if not match or stage not in self.stage_start:
                return
            idx = int(match.group(1))
            template_id = str(self.start_index + idx)
            if template_id not in self.stage_times[stage]:
                self.stage_times[stage][template_id] = max(0.0, now - self.stage_start[stage])
            return

        if stage in ("compose_match", "mcts", "llm_repair"):
            match = self._template_re.search(line)
            if not match or stage not in self.stage_start:
                return
            template_id = match.group(1)
            if template_id not in self.stage_times[stage]:
                self.stage_times[stage][template_id] = max(0.0, now - self.stage_start[stage])

    def per_query_time_parts(self) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        norepair: Dict[str, float] = {}
        repair: Dict[str, float] = {}
        total: Dict[str, float] = {}
        view_share = self.view_select_elapsed_s / float(self.total) if self.total else 0.0
        for template_id in self.stage_times["mcts"].keys():
            init_time = self.stage_times["init_template"].get(template_id)
            compose_time = self.stage_times["compose_match"].get(template_id)
            mcts_time = self.stage_times["mcts"].get(template_id)
            if init_time is None or compose_time is None or mcts_time is None:
                continue
            base = view_share + float(init_time) + float(compose_time) + float(mcts_time)
            repair_time = float(self.stage_times["llm_repair"].get(template_id) or 0.0)
            norepair[template_id] = base
            repair[template_id] = repair_time
            total[template_id] = base + repair_time
        return norepair, repair, total


def _write_time_file(path: Path, times: Iterable[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for value in times:
            fp.write(f"{value:.6f}\\n")


def _write_time_jsonl(path: Path, times: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for template_id, value in times.items():
            fp.write(json.dumps({"template_id": template_id, "time_s": float(value)}, ensure_ascii=False) + "\\n")


def _write_time_parts_jsonl(
    path: Path,
    *,
    norepair: Dict[str, float],
    repair: Dict[str, float],
    total: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for template_id, value in total.items():
            fp.write(
                json.dumps(
                    {
                        "template_id": template_id,
                        "time_s_norepair": float(norepair.get(template_id, 0.0)),
                        "time_s_repair": float(repair.get(template_id, 0.0)),
                        "time_s_total": float(value),
                    },
                    ensure_ascii=False,
                )
                + "\\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MemPlan in chunks (N queries then repair, repeated)."
    )
    parser.add_argument("--model", type=str, required=True, help="Model tag/name (e.g., gpt52).")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max-tool-candidates", type=int, default=5)
    parser.add_argument("--mcts-iterations", type=int, default=150)
    parser.add_argument("--mcts-uct-c", type=float, default=1.4)
    parser.add_argument("--semantic-tolerance", type=float, default=0.8)
    parser.add_argument("--repair-max-slot-candidates", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name = _resolve_model(args.model)
    split = args.split
    price_table = _read_price_table()
    model_key = MODEL_ALIAS.get(args.model, args.model)
    if model_key not in price_table and model_name in price_table:
        model_key = model_name
    rates = price_table.get(model_key, {})

    dataset_path = _dataset_path(split)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    rows = _load_rows(dataset_path)
    if not rows:
        raise RuntimeError(f"No rows in dataset: {dataset_path}")
    fieldnames = list(rows[0].keys())

    base_dir = args.out_root.expanduser().resolve() / f"{args.model}_memplan_chunked"
    base_dir.mkdir(parents=True, exist_ok=True)
    log_path = base_dir / f"{args.model}_memplan_chunked_{_timestamp()}.log"

    ranges = _chunk_ranges(len(rows), int(args.chunk_size))
    summaries_path = base_dir / "chunk_summaries.jsonl"
    combined_time_map: Dict[str, float] = {}
    combined_norepair_map: Dict[str, float] = {}
    combined_repair_map: Dict[str, float] = {}
    for chunk_idx, (start, end) in enumerate(ranges, start=1):
        chunk_dir = base_dir / f"chunk_{start:04d}_{end:04d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_csv = chunk_dir / "chunk.csv"
        _write_chunk_csv(
            rows=rows[start:end],
            fieldnames=fieldnames,
            start_index=start,
            out_path=chunk_csv,
        )

        init_templates = chunk_dir / f"init_templates_{split}.jsonl"
        match_json = chunk_dir / f"match_{split}.json"
        tree_json = chunk_dir / f"tree_{split}.json"
        optimized_jsonl = chunk_dir / f"optimized_{split}.jsonl"

        print(f"==> Chunk {chunk_idx}/{len(ranges)}: rows {start}-{end - 1}")

        tracker = ChunkTracker(start_index=start, total=end - start)

        init_cmd = [
            sys.executable,
            "-m",
            "planner.init_template",
            "--task",
            "travel",
            "--split",
            split,
            "--all",
            "--input",
            str(chunk_csv),
            "--plan-fields",
            "org,dest,days,date,query",
            "--model",
            model_name,
            "--workers",
            str(args.workers),
            "--out",
            str(init_templates),
        ]
        tracker.start_stage("init_template")
        t0 = time.perf_counter()
        _run_cmd(init_cmd, log_path=log_path, line_handler=lambda line: tracker.handle_line("init_template", line))
        init_elapsed = time.perf_counter() - t0

        view_select_cmd = [
            sys.executable,
            "-m",
            "planner.view_select",
            "--task",
            "travel",
            "--split",
            split,
            "--templates",
            str(init_templates),
            "--out",
            str(match_json),
            "--workers",
            str(args.workers),
        ]
        if args.config:
            view_select_cmd += ["--config", str(args.config)]
        tracker.start_stage("view_select")
        t0 = time.perf_counter()
        _run_cmd(view_select_cmd, log_path=log_path)
        view_select_elapsed = time.perf_counter() - t0
        tracker.end_stage("view_select")

        compose_cmd = [
            sys.executable,
            "-m",
            "planner.compose_match",
            "--task",
            "travel",
            "--split",
            split,
            "--templates",
            str(init_templates),
            "--match",
            str(match_json),
            "--out",
            str(tree_json),
            "--max-tool-candidates",
            str(args.max_tool_candidates),
            "--workers",
            str(args.workers),
        ]
        if args.config:
            compose_cmd += ["--config", str(args.config)]
        tracker.start_stage("compose_match")
        t0 = time.perf_counter()
        _run_cmd(compose_cmd, log_path=log_path, line_handler=lambda line: tracker.handle_line("compose_match", line))
        compose_elapsed = time.perf_counter() - t0

        mcts_cmd = [
            sys.executable,
            "-m",
            "planner.twin_track",
            "--task",
            "travel",
            "--tree",
            str(tree_json),
            "--out",
            str(optimized_jsonl),
            "--workers",
            str(args.workers),
            "--iterations",
            str(args.mcts_iterations),
            "--uct-c",
            str(args.mcts_uct_c),
            "--semantic-tolerance",
            str(args.semantic_tolerance),
        ]
        if args.config:
            mcts_cmd += ["--config", str(args.config)]
        tracker.start_stage("mcts")
        t0 = time.perf_counter()
        _run_cmd(mcts_cmd, log_path=log_path, line_handler=lambda line: tracker.handle_line("mcts", line))
        mcts_elapsed = time.perf_counter() - t0

        repair_cmd = [
            sys.executable,
            "-m",
            "planner.llm_repair",
            "--task",
            "travel",
            "--tree",
            str(tree_json),
            "--input",
            str(optimized_jsonl),
            "--out",
            str(optimized_jsonl),
            "--model",
            model_name,
            "--semantic-threshold",
            str(args.semantic_tolerance),
            "--max-slot-candidates",
            str(args.repair_max_slot_candidates),
            "--workers",
            str(args.workers),
        ]
        if args.config:
            repair_cmd += ["--config", str(args.config)]
        tracker.start_stage("llm_repair")
        t0 = time.perf_counter()
        repair_elapsed: Optional[float] = None

        def _repair_handler(line: str) -> None:
            nonlocal repair_elapsed
            tracker.handle_line("llm_repair", line)
            match = re.search(r"LLM repair done:.*elapsed=([0-9.]+)s", line)
            if match:
                try:
                    repair_elapsed = float(match.group(1))
                except ValueError:
                    repair_elapsed = None

        _run_cmd(repair_cmd, log_path=log_path, line_handler=_repair_handler)
        if repair_elapsed is None:
            repair_elapsed = time.perf_counter() - t0

        norepair_usage = _sum_init_usage(init_templates)
        repair_usage = _sum_repair_usage(optimized_jsonl)
        norepair_price = _price_for_usage(norepair_usage, rates)
        repair_price = _price_for_usage(repair_usage, rates)
        combined_usage = {
            "prompt_cache_hit": norepair_usage["prompt_cache_hit"] + repair_usage["prompt_cache_hit"],
            "prompt_cache_miss": norepair_usage["prompt_cache_miss"] + repair_usage["prompt_cache_miss"],
            "output": norepair_usage["output"] + repair_usage["output"],
            "total": norepair_usage["total"] + repair_usage["total"],
        }

        summary = {
            "chunk_index": chunk_idx,
            "rows_start": start,
            "rows_end": end - 1,
            "rows_total": end - start,
            "model": model_name,
            "model_key": model_key,
            "split": split,
            "workers": args.workers,
            "time_s_norepair": init_elapsed + view_select_elapsed + compose_elapsed + mcts_elapsed,
            "time_s_repair": repair_elapsed,
            "time_s_total": init_elapsed + view_select_elapsed + compose_elapsed + mcts_elapsed + repair_elapsed,
            "tokens_norepair": norepair_usage,
            "tokens_repair": repair_usage,
            "tokens_total": combined_usage,
            "price_usd_norepair": round(float(norepair_price), 6),
            "price_usd_repair": round(float(repair_price), 6),
            "price_usd_total": round(float(norepair_price + repair_price), 6),
        }
        summary_path = chunk_dir / "chunk_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        with summaries_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(summary, ensure_ascii=False) + "\n")

        chunk_norepair, chunk_repair, chunk_total = tracker.per_query_time_parts()
        if chunk_total:
            _write_time_parts_jsonl(
                chunk_dir / "chunk_times.jsonl",
                norepair=chunk_norepair,
                repair=chunk_repair,
                total=chunk_total,
            )
            _write_time_file(chunk_dir / "chunk_times_norepair.txt", list(chunk_norepair.values()))
            _write_time_file(chunk_dir / "chunk_times_repair.txt", list(chunk_repair.values()))
            _write_time_file(chunk_dir / "chunk_times_total.txt", list(chunk_total.values()))
            combined_norepair_map.update(chunk_norepair)
            combined_repair_map.update(chunk_repair)
            combined_time_map.update(chunk_total)

    if combined_time_map:
        combined_jsonl = base_dir / f"{args.model}_memplan_times.jsonl"
        _write_time_parts_jsonl(
            combined_jsonl,
            norepair=combined_norepair_map,
            repair=combined_repair_map,
            total=combined_time_map,
        )
        _write_time_file(base_dir / f"{args.model}_memplan_times_norepair.txt", list(combined_norepair_map.values()))
        _write_time_file(base_dir / f"{args.model}_memplan_times_repair.txt", list(combined_repair_map.values()))
        _write_time_file(base_dir / f"{args.model}_memplan_times_total.txt", list(combined_time_map.values()))

    print(f"Done. Outputs in {base_dir}")


if __name__ == "__main__":
    main()
