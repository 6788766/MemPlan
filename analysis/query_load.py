#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT_DEFAULT = PROJECT_ROOT / "artifacts" / "output" / "travel" / "analysis"
RUNS_ROOT_DEFAULT = ANALYSIS_ROOT_DEFAULT / "query_load"

try:
    from task_helper.travel.utils.paths import travel_dataset_root
except Exception:  # pragma: no cover - fallback for partial installs
    travel_dataset_root = None

MODEL_ALIAS = {
    "gpt52": "gpt-5.2",
    "gpt5mini": "gpt-5-mini",
    "gpt5nano": "gpt-5-nano",
    "deepseekchat": "deepseek-chat",
}
MODEL_TAG_BY_NAME = {value: key for key, value in MODEL_ALIAS.items()}
MODEL_TAGS = tuple(MODEL_ALIAS.keys())
METHODS = ("react", "memplan", "norepair")
CHECKPOINTS = (200, 400, 600, 800, 1000)

_KV_INT_RE = re.compile(r"\b([A-Za-z_]+)=([0-9]+)\b")


def _model_slug(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "", str(name or "").lower())
    return cleaned or "model"


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _resolve_model(model_tag: str) -> str:
    if model_tag in MODEL_ALIAS:
        return MODEL_ALIAS[model_tag]
    raise ValueError(f"Unknown model tag: {model_tag}")


def _dataset_count(split: str) -> int:
    if travel_dataset_root is not None:
        root = travel_dataset_root(PROJECT_ROOT)
    else:
        root = PROJECT_ROOT / "artifacts" / "input" / "travel" / "dataset"
    path = root / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as fp:
        return sum(1 for _ in csv.DictReader(fp))


def _parse_kv_ints(line: str) -> Dict[str, int]:
    values: Dict[str, int] = {}
    for key, raw in _KV_INT_RE.findall(line):
        try:
            values[key.strip().lower()] = int(raw)
        except ValueError:
            continue
    return values


def _parse_usage_line(line: str) -> Dict[str, int]:
    values = _parse_kv_ints(line)
    prompt_cache_hit = values.get("prompt_cache_hit") or values.get("prompt_cache_hit_tokens") or 0
    prompt_cache_miss = values.get("prompt_cache_miss") or values.get("prompt_cache_miss_tokens") or 0
    output = values.get("output") or values.get("completion_tokens") or values.get("output_tokens") or 0
    total = values.get("total") or values.get("total_tokens") or 0

    if prompt_cache_hit and not prompt_cache_miss:
        prompt_tokens = values.get("prompt_tokens") or values.get("prompt") or 0
        prompt_cache_miss = max(0, int(prompt_tokens) - int(prompt_cache_hit))
    return {
        "prompt_cache_hit": int(prompt_cache_hit),
        "prompt_cache_miss": int(prompt_cache_miss),
        "output": int(output),
        "total": int(total),
    }


def _read_cost_file(cost_path: Path) -> Tuple[Dict[str, int], Optional[float], Optional[float]]:
    if not cost_path.exists():
        raise FileNotFoundError(f"Missing cost file: {cost_path}")
    lines = cost_path.read_text(encoding="utf-8").splitlines()
    usage = {"prompt_cache_hit": 0, "prompt_cache_miss": 0, "output": 0, "total": 0}
    price_usd: Optional[float] = None
    elapsed_s: Optional[float] = None

    for line in lines:
        if "LLM token usage (init_template):" in line:
            parsed = _parse_usage_line(line)
            for key in usage:
                usage[key] += parsed.get(key, 0)
        elif "LLM token usage (llm_repair):" in line:
            parsed = _parse_usage_line(line)
            for key in usage:
                usage[key] += parsed.get(key, 0)
        elif line.strip().startswith("llm_tokens:"):
            parsed = _parse_usage_line(line)
            for key in usage:
                usage[key] += parsed.get(key, 0)
        elif line.startswith("llm_price_usd_total="):
            try:
                price_usd = float(line.split("=", 1)[1].strip())
            except ValueError:
                price_usd = None
        elif line.startswith("pipeline_elapsed_s_no_eval="):
            try:
                elapsed_s = float(line.split("=", 1)[1].strip())
            except ValueError:
                elapsed_s = None
        elif line.startswith("pipeline_elapsed_s_total=") and elapsed_s is None:
            try:
                elapsed_s = float(line.split("=", 1)[1].strip())
            except ValueError:
                elapsed_s = None
        elif line.startswith("elapsed_s_total=") and elapsed_s is None:
            try:
                elapsed_s = float(line.split("=", 1)[1].strip())
            except ValueError:
                elapsed_s = None

    if usage["total"] <= 0:
        usage["total"] = usage["prompt_cache_hit"] + usage["prompt_cache_miss"] + usage["output"]
    return usage, price_usd, elapsed_s


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


class StageTracker:
    def __init__(
        self,
        *,
        split: str,
        total_queries: int,
        method: str,
        output_root: Path,
        model_tag: str,
        checkpoints: Sequence[int],
        repair_only: bool = False,
    ) -> None:
        self.split = split
        self.total_queries = total_queries
        self.method = method
        self.output_root = output_root
        self.model_tag = model_tag
        self.checkpoints = sorted(set(int(x) for x in checkpoints))
        self.repair_only = bool(repair_only)
        self.final_stage = "llm_repair" if (method == "memplan" or self.repair_only) else "mcts"

        self.active_stage: Optional[str] = None
        self.stage_start: Dict[str, float] = {}
        self.stage_times: Dict[str, Dict[str, float]] = {
            "init_template": {},
            "compose_match": {},
            "mcts": {},
            "llm_repair": {},
        }
        self.stage_completion_ts_by_id: Dict[str, Dict[str, float]] = {
            "init_template": {},
            "compose_match": {},
            "mcts": {},
            "llm_repair": {},
        }
        self.stage_completion_times: Dict[str, List[float]] = {
            "init_template": [],
            "compose_match": [],
            "mcts": [],
            "llm_repair": [],
        }
        self.view_select_elapsed_s: float = 0.0
        self._checkpoint_done: set[int] = set()
        self._checkpoint_lock = threading.Lock()
        self._checkpoint_times: Dict[int, float] = {}
        self.run_start: Optional[float] = None

        self._init_re = re.compile(r"Processing query\s+([^\s.]+)")
        self._template_re = re.compile(r"template_id=([^\s]+)")
        self._start_re = re.compile(r"START\s+([A-Za-z_]+)")
        self._end_re = re.compile(r"END\s+([A-Za-z_]+)")
        self._elapsed_re = re.compile(r"elapsed_s=([0-9.]+)")
        self._repair_start_re = re.compile(r"^LLM repair:")

    def handle_line(self, line: str) -> None:
        now = time.perf_counter()
        start_match = self._start_re.search(line)
        if start_match:
            stage = start_match.group(1)
            self.active_stage = stage
            self.stage_start[stage] = now
            if self.run_start is None:
                self.run_start = now
            return

        end_match = self._end_re.search(line)
        if end_match:
            stage = end_match.group(1)
            if stage == "view_select":
                elapsed_match = self._elapsed_re.search(line)
                if elapsed_match:
                    try:
                        self.view_select_elapsed_s = float(elapsed_match.group(1))
                    except ValueError:
                        self.view_select_elapsed_s = 0.0
                elif stage in self.stage_start:
                    self.view_select_elapsed_s = max(0.0, now - self.stage_start[stage])
            if self.active_stage == stage:
                self.active_stage = None
            return

        stage = self.active_stage
        if not stage:
            return

        if stage == "llm_repair" and self._repair_start_re.search(line):
            # Reset timing to avoid counting load/IO before repair compute begins.
            self.stage_start[stage] = now
            if self.run_start is None:
                self.run_start = now
            return

        if stage == "init_template":
            match = self._init_re.search(line)
            if match and stage in self.stage_start:
                template_id = match.group(1)
                elapsed = max(0.0, now - self.stage_start[stage])
                completion_ts = max(0.0, now - (self.run_start or self.stage_start[stage]))
                if template_id not in self.stage_times[stage]:
                    self.stage_times[stage][template_id] = elapsed
                    self.stage_completion_times[stage].append(completion_ts)
                    self.stage_completion_ts_by_id[stage][template_id] = completion_ts
            return

        if stage in ("compose_match", "mcts", "llm_repair"):
            match = self._template_re.search(line)
            if match and stage in self.stage_start:
                template_id = match.group(1)
                elapsed = max(0.0, now - self.stage_start[stage])
                completion_ts = max(0.0, now - (self.run_start or self.stage_start[stage]))
                if template_id not in self.stage_times[stage]:
                    self.stage_times[stage][template_id] = elapsed
                    self.stage_completion_times[stage].append(completion_ts)
                    self.stage_completion_ts_by_id[stage][template_id] = completion_ts
                if stage == self.final_stage:
                    self._maybe_checkpoint()

    def _total_time_for(self, template_id: str) -> Optional[float]:
        total = 0.0
        missing = False
        if not self.repair_only:
            for stage in ("init_template", "compose_match", "mcts"):
                value = self.stage_times[stage].get(template_id)
                if value is None:
                    missing = True
                    continue
                total += float(value)
            if self.view_select_elapsed_s and self.total_queries:
                total += self.view_select_elapsed_s / float(self.total_queries)
        if self.final_stage == "llm_repair":
            value = self.stage_times["llm_repair"].get(template_id)
            if value is None:
                missing = True
            else:
                total += float(value)
        return None if missing else total

    def _checkpoint_time_for(self, threshold: int) -> Optional[float]:
        times = sorted(self.stage_completion_times[self.final_stage])
        if len(times) < threshold:
            return None
        return float(times[threshold - 1])

    def _maybe_checkpoint(self) -> None:
        with self._checkpoint_lock:
            completed = len(self.stage_times[self.final_stage])
            for threshold in self.checkpoints:
                if threshold in self._checkpoint_done:
                    continue
                if completed < threshold:
                    continue
                checkpoint_time = self._checkpoint_time_for(threshold)
                if checkpoint_time is not None:
                    self._checkpoint_times[threshold] = checkpoint_time
                self._checkpoint_done.add(threshold)

    def finalize(self, out_path: Path) -> Tuple[List[float], int]:
        template_ids = list(self.stage_times[self.final_stage].keys())
        times: List[float] = []
        missing = 0
        for template_id in template_ids:
            total = self._total_time_for(template_id)
            if total is None:
                missing += 1
                continue
            times.append(total)
        _write_time_file(out_path, times)
        return times, missing

    def checkpoint_times(self) -> Dict[int, float]:
        return dict(self._checkpoint_times)

    def template_times(self) -> Dict[str, float]:
        results: Dict[str, float] = {}
        for template_id in self.stage_times[self.final_stage].keys():
            total = self._total_time_for(template_id)
            if total is None:
                continue
            results[template_id] = total
        return results

    def completion_order_ids(self) -> List[str]:
        items = list(self.stage_completion_ts_by_id[self.final_stage].items())
        items.sort(key=lambda x: x[1])
        return [template_id for template_id, _ in items]


def _tee_subprocess(
    cmd: Sequence[str],
    *,
    log_path: Path,
    cwd: Path,
    env: Dict[str, str],
    line_handler=None,
    append: bool = False,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with log_path.open(mode, encoding="utf-8") as log_fp:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
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
        return proc.wait()


def _monitor_react_checkpoints(
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
            elapsed = time.perf_counter() - start
            checkpoint_times[threshold] = round(elapsed, 6)
            done.add(threshold)
        stop_event.wait(poll_s)


def _react_error_summary(per_query_dir: Path, total_queries: int) -> Tuple[int, int]:
    paths = list(per_query_dir.glob("generated_plan_*.json"))
    error_count = 0
    success_count = 0
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
    missing = max(0, int(total_queries) - success_count)
    return error_count, missing


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


def _react_run(
    *,
    model_name: str,
    model_tag: str,
    split: str,
    workers: int,
    output_root: Path,
    log_path: Path,
    run_root: Path,
    checkpoints: Sequence[int],
    retry_sleep_s: float = 60.0,
    max_retries: int = 5,
) -> Tuple[Path, Path, Dict[int, float]]:
    run_root.mkdir(parents=True, exist_ok=True)
    cmd_base = [
        sys.executable,
        "-m",
        "baseline.tool_agents",
        "--set_type",
        split,
        "--model_name",
        model_name,
        "--workers",
        str(workers),
        "--output_dir",
        str(run_root),
        "--resume-failed",
    ]

    model_slug = _model_slug(model_name)
    run_dir = run_root / f"{model_slug}_{split}"
    per_query_dir = run_dir / f"two_stage_{model_slug}_{split}"

    stop_event = threading.Event()
    checkpoint_times: Dict[int, float] = {}
    monitor = threading.Thread(
        target=_monitor_react_checkpoints,
        kwargs={
            "per_query_dir": per_query_dir,
            "checkpoints": checkpoints,
            "stop_event": stop_event,
            "checkpoint_times": checkpoint_times,
        },
        daemon=True,
    )
    monitor.start()

    total_queries = _dataset_count(split)
    attempt = 0
    while True:
        status = _tee_subprocess(
            cmd_base,
            log_path=log_path,
            cwd=PROJECT_ROOT,
            env=os.environ.copy(),
            append=attempt > 0,
        )
        error_count, missing = _react_error_summary(per_query_dir, total_queries)
        if status == 0 and error_count == 0 and missing == 0:
            break
        attempt += 1
        if max_retries > 0 and attempt > max_retries:
            stop_event.set()
            monitor.join(timeout=5.0)
            raise RuntimeError(
                f"React run failed after {attempt} attempts (errors={error_count}, missing={missing})."
            )
        time.sleep(max(1.0, float(retry_sleep_s)))

    stop_event.set()
    monitor.join(timeout=5.0)

    cost_path = run_dir / "cost.txt"
    return cost_path, per_query_dir, checkpoint_times


def _memplan_run(
    *,
    model_name: str,
    model_tag: str,
    method: str,
    split: str,
    workers: int,
    output_root: Path,
    log_path: Path,
    run_root: Path,
    checkpoints: Sequence[int],
) -> Tuple[Path, StageTracker]:
    total_queries = _dataset_count(split)
    tracker = StageTracker(
        split=split,
        total_queries=total_queries,
        method=method,
        output_root=output_root,
        model_tag=model_tag,
        checkpoints=checkpoints,
    )

    env = os.environ.copy()
    env.update(
        {
            "TASK": "travel",
            "SPLIT": split,
            "MODEL": model_name,
            "CPUS": str(workers),
            "RUN_EVAL": "0",
        }
    )
    env["RUN_REPAIR"] = "0"

    cmd = ["bash", str(PROJECT_ROOT / "task_helper" / "travel" / "run.sh")]
    status = _tee_subprocess(
        cmd,
        log_path=log_path,
        cwd=PROJECT_ROOT,
        env=env,
        line_handler=tracker.handle_line,
    )
    if status != 0:
        raise RuntimeError(f"MemPlan run failed with status {status}")

    model_slug = _model_slug(model_name)
    source_dir = PROJECT_ROOT / "artifacts" / "output" / "travel" / f"{model_slug}_{split}"
    if source_dir.exists():
        run_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_dir, run_root, dirs_exist_ok=True)

    cost_path = run_root / "cost.txt"
    if not cost_path.exists() and source_dir.exists():
        cost_path = source_dir / "cost.txt"

    time_path = output_root / f"{model_tag}_{method}.txt"
    _, missing = tracker.finalize(time_path)
    _write_time_jsonl(time_path.with_suffix(".jsonl"), tracker.template_times())
    if missing:
        print(
            f"warning: {missing} template(s) missing stage timing for {model_tag}/{method}",
            file=sys.stderr,
        )
    return cost_path, tracker


def _repair_run(
    *,
    model_name: str,
    model_tag: str,
    split: str,
    workers: int,
    output_root: Path,
    log_path: Path,
    run_root: Path,
    checkpoints: Sequence[int],
) -> StageTracker:
    total_queries = _dataset_count(split)
    tracker = StageTracker(
        split=split,
        total_queries=total_queries,
        method="memplan",
        output_root=output_root,
        model_tag=model_tag,
        checkpoints=checkpoints,
        repair_only=True,
    )
    tracker.active_stage = "llm_repair"

    tree_path = run_root / f"tree_{split}.json"
    input_path = run_root / f"optimized_{split}.jsonl"
    if not tree_path.exists():
        raise FileNotFoundError(f"Missing tree JSON: {tree_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Missing optimized JSONL: {input_path}")

    cmd = [
        sys.executable,
        "-m",
        "planner.llm_repair",
        "--task",
        "travel",
        "--tree",
        str(tree_path),
        "--input",
        str(input_path),
        "--out",
        str(input_path),
        "--model",
        model_name,
        "--semantic-threshold",
        "0.8",
        "--max-slot-candidates",
        "50",
        "--workers",
        str(workers),
    ]
    status = _tee_subprocess(
        cmd,
        log_path=log_path,
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        line_handler=tracker.handle_line,
    )
    if status != 0:
        raise RuntimeError(f"LLM repair run failed with status {status}")
    return tracker


def _write_summary(
    *,
    output_root: Path,
    model_name: str,
    model_tag: str,
    method: str,
    split: str,
    workers: int,
    cost_path: Path,
) -> None:
    usage, price, elapsed = _read_cost_file(cost_path)
    summary = {
        "model": model_name,
        "model_tag": model_tag,
        "method": method,
        "split": split,
        "workers": workers,
        "time_s": elapsed,
        "tokens": usage,
        "price_usd": price,
        "cost_path": str(cost_path),
    }
    summary_path = output_root / f"{model_tag}_{method}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _collect_react_times(per_query_dir: Path, out_path: Path) -> None:
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
    _write_time_file(out_path, times)
    _write_time_jsonl(out_path.with_suffix(".jsonl"), time_map)


def _run_money(cost_path: Path) -> None:
    cmd = [sys.executable, "-m", "task_helper.money_memplan", str(cost_path)]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _read_price_table() -> Dict[str, Dict[str, float]]:
    path = PROJECT_ROOT / "artifacts" / "input" / "price.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _price_for_usage(model_name: str, usage: Dict[str, int]) -> float:
    price_table = _read_price_table()
    model_key = None
    if model_name in price_table:
        model_key = model_name
    else:
        model_norm = str(model_name).lower()
        for key in price_table.keys():
            if str(key).lower() in model_norm:
                model_key = key
                break
    if model_key is None:
        model_key = model_name
    rates = price_table.get(model_key, {})
    hit_rate = float(rates.get("prompt_cache_hit", rates.get("cached", 0.0)) or 0.0)
    miss_rate = float(rates.get("prompt_cache_miss", rates.get("input", 0.0)) or 0.0)
    out_rate = float(rates.get("output", 0.0) or 0.0)
    million = 1_000_000.0
    return (
        (usage.get("prompt_cache_hit", 0) / million) * hit_rate
        + (usage.get("prompt_cache_miss", 0) / million) * miss_rate
        + (usage.get("output", 0) / million) * out_rate
    )


def _parse_repair_usage_from_log(log_path: Path) -> Dict[str, int]:
    if not log_path.exists():
        return {"prompt_cache_hit": 0, "prompt_cache_miss": 0, "output": 0, "total": 0}
    usage = {"prompt_cache_hit": 0, "prompt_cache_miss": 0, "output": 0, "total": 0}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if "LLM token usage (llm_repair):" in line:
            parsed = _parse_usage_line(line)
            for key in usage:
                usage[key] += parsed.get(key, 0)
    if usage["total"] <= 0:
        usage["total"] = usage["prompt_cache_hit"] + usage["prompt_cache_miss"] + usage["output"]
    return usage


def _parse_repair_elapsed_from_log(log_path: Path) -> Optional[float]:
    if not log_path.exists():
        return None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if "LLM repair done:" in line and "elapsed=" in line:
            match = re.search(r"elapsed=([0-9.]+)s", line)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    return None
    return None


def _normalize_usage_fields(raw: Dict[str, object]) -> Dict[str, int]:
    prompt_cache_hit = int(raw.get("prompt_cache_hit_tokens") or raw.get("prompt_cache_hit") or 0)
    prompt_cache_miss = int(raw.get("prompt_cache_miss_tokens") or raw.get("prompt_cache_miss") or 0)
    output = int(raw.get("completion_tokens") or raw.get("output_tokens") or raw.get("output") or 0)
    total = int(raw.get("total_tokens") or raw.get("total") or 0)
    prompt_tokens = int(raw.get("prompt_tokens") or raw.get("prompt") or 0)
    if prompt_cache_hit and not prompt_cache_miss and prompt_tokens:
        prompt_cache_miss = max(0, prompt_tokens - prompt_cache_hit)
    if total <= 0:
        total = prompt_cache_hit + prompt_cache_miss + output
    return {
        "prompt_cache_hit": prompt_cache_hit,
        "prompt_cache_miss": prompt_cache_miss,
        "output": output,
        "total": total,
    }


def _parse_init_usage(path: Path) -> Dict[str, Dict[str, int]]:
    usage_by_id: Dict[str, Dict[str, int]] = {}
    if not path.exists():
        return usage_by_id
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        template_id = str(obj.get("template_id") or obj.get("plan_id") or obj.get("id") or "")
        if not template_id:
            continue
        notes = obj.get("notes") if isinstance(obj.get("notes"), dict) else {}
        llm = notes.get("llm") if isinstance(notes.get("llm"), dict) else {}
        if llm:
            usage_by_id[template_id] = _normalize_usage_fields(llm)
    return usage_by_id


def _parse_repair_usage(path: Path) -> Dict[str, Dict[str, int]]:
    usage_by_id: Dict[str, Dict[str, int]] = {}
    if not path.exists():
        return usage_by_id
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        template_id = str(obj.get("template_id") or obj.get("plan_id") or obj.get("id") or "")
        if not template_id:
            continue
        notes = obj.get("notes") if isinstance(obj.get("notes"), dict) else {}
        repair = notes.get("llm_repair") if isinstance(notes.get("llm_repair"), dict) else {}
        usage = repair.get("usage") if isinstance(repair.get("usage"), dict) else {}
        if usage:
            usage_by_id[template_id] = _normalize_usage_fields(usage)
    return usage_by_id


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
            usage_by_id[template_id] = _normalize_usage_fields(metrics)
    return usage_by_id


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


def _write_checkpoint_metrics_by_threshold(
    *,
    output_root: Path,
    model_tag: str,
    method: str,
    checkpoint_times: Dict[int, float],
    usage_by_threshold: Dict[int, Dict[str, int]],
    price_by_threshold: Dict[int, float],
    price_parts_by_threshold: Optional[Dict[int, Dict[str, float]]] = None,
    ids_by_threshold: Optional[Dict[int, List[str]]] = None,
) -> None:
    for threshold, time_s in checkpoint_times.items():
        payload = {
            "time_s": round(float(time_s), 6),
            "tokens": dict(usage_by_threshold.get(threshold, {})),
            "price_usd": round(float(price_by_threshold.get(threshold, 0.0)), 6),
        }
        if price_parts_by_threshold and threshold in price_parts_by_threshold:
            payload["price_usd_parts"] = {
                key: round(float(value), 6)
                for key, value in price_parts_by_threshold[threshold].items()
            }
        path = output_root / f"{model_tag}_{method}_{threshold}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        if ids_by_threshold and threshold in ids_by_threshold:
            ids_path = output_root / f"{model_tag}_{method}_{threshold}_ids.json"
            ids_path.write_text(
                json.dumps({"template_ids": list(ids_by_threshold[threshold])}, indent=2, sort_keys=True),
                encoding="utf-8",
            )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scalability experiments for Travel test set.")
    parser.add_argument("--method", choices=("all",) + METHODS, default="all")
    parser.add_argument("--model", choices=("all",) + MODEL_TAGS, default="all")
    parser.add_argument("--split", default="test")
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--output-root", type=Path, default=ANALYSIS_ROOT_DEFAULT)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT_DEFAULT)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    output_root = args.output_root.expanduser().resolve()
    runs_root = args.runs_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    methods = METHODS if args.method == "all" else (args.method,)
    model_tags = MODEL_TAGS if args.model == "all" else (args.model,)

    for model_tag in model_tags:
        model_name = _resolve_model(model_tag)
        want_react = "react" in methods
        want_norepair = "norepair" in methods
        want_memplan = "memplan" in methods

        if want_react:
            method = "react"
            run_root = runs_root / f"{model_tag}_{method}"
            method_root = run_root
            log_path = method_root / f"{model_tag}_{method}_{_timestamp()}.log"
            print(f"==> Running react with {model_name} on {args.split} ({args.workers} workers)")
            cost_path, per_query_dir, react_checkpoint_times = _react_run(
                model_name=model_name,
                model_tag=model_tag,
                split=args.split,
                workers=args.workers,
                output_root=method_root,
                log_path=log_path,
                run_root=run_root,
                checkpoints=CHECKPOINTS,
            )
            _run_money(cost_path)
            time_path = method_root / f"{model_tag}_{method}.txt"
            _collect_react_times(per_query_dir, time_path)
            usage, _price_full, _ = _read_cost_file(cost_path)
            react_usage_by_id = _parse_react_usage(per_query_dir)
            order = _react_completion_order(per_query_dir)
            usage_by_threshold: Dict[int, Dict[str, int]] = {}
            price_by_threshold: Dict[int, float] = {}
            ids_by_threshold: Dict[int, List[str]] = {}
            for threshold in CHECKPOINTS:
                ids = order[:threshold]
                usage_sum = _usage_sum(ids, react_usage_by_id)
                usage_by_threshold[threshold] = usage_sum
                price_by_threshold[threshold] = _price_for_usage(model_name, usage_sum)
                ids_by_threshold[threshold] = list(ids)
            _write_checkpoint_metrics_by_threshold(
                output_root=method_root,
                model_tag=model_tag,
                method=method,
                checkpoint_times=react_checkpoint_times,
                usage_by_threshold=usage_by_threshold,
                price_by_threshold=price_by_threshold,
                ids_by_threshold=ids_by_threshold,
            )
            _write_summary(
                output_root=method_root,
                model_name=model_name,
                model_tag=model_tag,
                method=method,
                split=args.split,
                workers=args.workers,
                cost_path=cost_path,
            )

        if want_norepair or want_memplan:
            method = "norepair"
            run_root = runs_root / f"{model_tag}_{method}"
            method_root = run_root
            log_path = method_root / f"{model_tag}_{method}_{_timestamp()}.log"
            print(f"==> Running norepair with {model_name} on {args.split} ({args.workers} workers)")
            norepair_cost_path, norepair_tracker = _memplan_run(
                model_name=model_name,
                model_tag=model_tag,
                method=method,
                split=args.split,
                workers=args.workers,
                output_root=method_root,
                log_path=log_path,
                run_root=run_root,
                checkpoints=CHECKPOINTS,
            )
            _run_money(norepair_cost_path)
            norepair_usage, _norepair_price_full, _ = _read_cost_file(norepair_cost_path)
            norepair_price = _price_for_usage(model_name, norepair_usage)
            init_usage_by_id = _parse_init_usage(run_root / f"init_templates_{args.split}.jsonl")
            order = norepair_tracker.completion_order_ids()
            usage_by_threshold = {}
            price_by_threshold = {}
            ids_by_threshold = {}
            for threshold in CHECKPOINTS:
                ids = order[:threshold]
                usage_sum = _usage_sum(ids, init_usage_by_id)
                usage_by_threshold[threshold] = usage_sum
                price_by_threshold[threshold] = _price_for_usage(model_name, usage_sum)
                ids_by_threshold[threshold] = list(ids)
            _write_checkpoint_metrics_by_threshold(
                output_root=method_root,
                model_tag=model_tag,
                method=method,
                checkpoint_times=norepair_tracker.checkpoint_times(),
                usage_by_threshold=usage_by_threshold,
                price_by_threshold=price_by_threshold,
                ids_by_threshold=ids_by_threshold,
            )
            if want_norepair:
                _write_summary(
                    output_root=method_root,
                    model_name=model_name,
                    model_tag=model_tag,
                    method=method,
                    split=args.split,
                    workers=args.workers,
                    cost_path=norepair_cost_path,
                )

            if want_memplan:
                method = "memplan"
                run_root = runs_root / f"{model_tag}_{method}"
                method_root = run_root
                log_path = method_root / f"{model_tag}_{method}_{_timestamp()}.log"
                if run_root.exists():
                    shutil.rmtree(run_root)
                run_root.mkdir(parents=True, exist_ok=True)
                shutil.copytree(runs_root / f"{model_tag}_norepair", run_root, dirs_exist_ok=True)

                print(f"==> Running repair with {model_name} on {args.split} ({args.workers} workers)")
                repair_tracker = _repair_run(
                    model_name=model_name,
                    model_tag=model_tag,
                    split=args.split,
                    workers=args.workers,
                    output_root=method_root,
                    log_path=log_path,
                    run_root=run_root,
                    checkpoints=CHECKPOINTS,
                )

                repair_usage = _parse_repair_usage_from_log(log_path)
                combined_usage = {
                    "prompt_cache_hit": norepair_usage.get("prompt_cache_hit", 0) + repair_usage.get("prompt_cache_hit", 0),
                    "prompt_cache_miss": norepair_usage.get("prompt_cache_miss", 0) + repair_usage.get("prompt_cache_miss", 0),
                    "output": norepair_usage.get("output", 0) + repair_usage.get("output", 0),
                    "total": norepair_usage.get("total", 0) + repair_usage.get("total", 0),
                }
                repair_price = _price_for_usage(model_name, repair_usage)
                if norepair_price is None:
                    norepair_price = _price_for_usage(model_name, norepair_usage)
                combined_price = (norepair_price or 0.0) + (repair_price or 0.0)

                combined_checkpoint_times: Dict[int, float] = {}
                for threshold in CHECKPOINTS:
                    t1 = norepair_tracker.checkpoint_times().get(threshold)
                    t2 = repair_tracker.checkpoint_times().get(threshold)
                    if t1 is None or t2 is None:
                        continue
                    combined_checkpoint_times[threshold] = float(t1) + float(t2)

                combined_times: List[float] = []
                combined_time_map: Dict[str, float] = {}
                norepair_times = norepair_tracker.template_times()
                repair_stage_times = repair_tracker.stage_times.get("llm_repair", {})
                for template_id, base_time in norepair_times.items():
                    repair_time = repair_stage_times.get(template_id)
                    if repair_time is None:
                        continue
                    combined_times.append(float(base_time) + float(repair_time))
                    combined_time_map[template_id] = float(base_time) + float(repair_time)
                if combined_times:
                    time_path = method_root / f"{model_tag}_{method}.txt"
                    _write_time_file(time_path, combined_times)
                    _write_time_jsonl(time_path.with_suffix(".jsonl"), combined_time_map)

                repair_usage_by_id = _parse_repair_usage(run_root / f"optimized_{args.split}.jsonl")
                order = repair_tracker.completion_order_ids()
                usage_by_threshold = {}
                price_by_threshold = {}
                price_parts_by_threshold: Dict[int, Dict[str, float]] = {}
                ids_by_threshold: Dict[int, List[str]] = {}
                for threshold in CHECKPOINTS:
                    ids = order[:threshold]
                    init_sum = _usage_sum(ids, init_usage_by_id)
                    repair_sum = _usage_sum(ids, repair_usage_by_id)
                    combined_sum = {
                        "prompt_cache_hit": init_sum["prompt_cache_hit"] + repair_sum["prompt_cache_hit"],
                        "prompt_cache_miss": init_sum["prompt_cache_miss"] + repair_sum["prompt_cache_miss"],
                        "output": init_sum["output"] + repair_sum["output"],
                        "total": init_sum["total"] + repair_sum["total"],
                    }
                    usage_by_threshold[threshold] = combined_sum
                    price_by_threshold[threshold] = _price_for_usage(model_name, combined_sum)
                    price_parts_by_threshold[threshold] = {
                        "norepair": _price_for_usage(model_name, init_sum),
                        "repair": _price_for_usage(model_name, repair_sum),
                    }
                    ids_by_threshold[threshold] = list(ids)
                _write_checkpoint_metrics_by_threshold(
                    output_root=method_root,
                    model_tag=model_tag,
                    method=method,
                    checkpoint_times=combined_checkpoint_times,
                    usage_by_threshold=usage_by_threshold,
                    price_by_threshold=price_by_threshold,
                    price_parts_by_threshold=price_parts_by_threshold,
                    ids_by_threshold=ids_by_threshold,
                )
            repair_elapsed = _parse_repair_elapsed_from_log(log_path)
            norepair_elapsed = _read_cost_file(norepair_cost_path)[2]
            total_elapsed = None
            if norepair_elapsed is not None and repair_elapsed is not None:
                total_elapsed = float(norepair_elapsed) + float(repair_elapsed)
            summary = {
                "model": model_name,
                "model_tag": model_tag,
                "method": method,
                "split": args.split,
                "workers": args.workers,
                "time_s": total_elapsed,
                "tokens": combined_usage,
                "price_usd": round(float(combined_price), 6),
                "price_usd_parts": {
                    "norepair": round(float(norepair_price or 0.0), 6),
                    "repair": round(float(repair_price or 0.0), 6),
                },
                "cost_path": str(norepair_cost_path),
                "repair_log": str(log_path),
            }
            summary_path = method_root / f"{model_tag}_{method}_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Done. Outputs saved under {output_root}")


if __name__ == "__main__":
    main()
