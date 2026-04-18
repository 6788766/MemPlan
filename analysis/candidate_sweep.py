#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = PROJECT_ROOT / "artifacts" / "output" / "travel" / "analysis" / "candidate"

MODEL_ALIAS = {
    "gpt52": "gpt-5.2",
    "gpt5mini": "gpt-5-mini",
    "gpt5nano": "gpt-5-nano",
    "deepseekchat": "deepseek-chat",
}
FULL_MODELS = ("gpt52", "gpt5mini", "gpt5nano", "deepseekchat")

_KV_INT_RE = re.compile(r"\b([A-Za-z_]+)=([0-9]+)\b")
_END_STAGE_RE = re.compile(r"\bEND\s+(?P<stage>[A-Za-z_]+)\s+status=(?P<status>[0-9]+)\s+elapsed_s=(?P<elapsed>[0-9]+)\b")
_EVAL_SUMMARY_ANCHOR = "{'Commonsense Constraint'"
_LEVELS = ("easy", "medium", "hard")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _available_cpu_count() -> int:
    try:
        affinity = os.sched_getaffinity(0)
    except Exception:
        affinity = None
    if affinity:
        return max(1, len(affinity))
    return max(1, int(os.cpu_count() or 1))


def _model_slug(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "", str(name or "").lower())
    return cleaned or "model"


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
    calls = values.get("calls") or 0

    if prompt_cache_hit and not prompt_cache_miss:
        prompt_tokens = values.get("prompt_tokens") or values.get("prompt") or 0
        prompt_cache_miss = max(0, int(prompt_tokens) - int(prompt_cache_hit))
    if total <= 0:
        total = int(prompt_cache_hit) + int(prompt_cache_miss) + int(output)
    return {
        "calls": int(calls),
        "prompt_cache_hit": int(prompt_cache_hit),
        "prompt_cache_miss": int(prompt_cache_miss),
        "output": int(output),
        "total": int(total),
    }


def _extract_last_stage_end(*, cost_path: Path, stage: str) -> Optional[Dict[str, object]]:
    if not cost_path.exists():
        return None
    last: Optional[Dict[str, object]] = None
    for line in cost_path.read_text(encoding="utf-8").splitlines():
        match = _END_STAGE_RE.search(line)
        if not match:
            continue
        if match.group("stage") != stage:
            continue
        try:
            last = {
                "line": line.strip(),
                "status": int(match.group("status")),
                "elapsed_s": int(match.group("elapsed")),
            }
        except ValueError:
            continue
    return last


def _extract_last_line_containing(*, path: Path, needle: str) -> Optional[str]:
    if not path.exists():
        return None
    last: Optional[str] = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if needle in line:
            last = line.strip()
    return last


def _parse_eval_stdout(text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.endswith("%"):
            continue
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        key = key.strip()
        raw = raw.strip().rstrip("%").strip()
        try:
            metrics[key] = float(raw) / 100.0
        except ValueError:
            continue
    return metrics


def _extract_eval_summary_dict(text: str) -> Optional[dict]:
    """
    Extract the printed `detailed_scores` dict from Travel eval output.

    The evaluator prints a Python literal dict like:
      {'Commonsense Constraint': {...}, 'Hard Constraint': {...}}
    """

    start = text.rfind(_EVAL_SUMMARY_ANCHOR)
    if start == -1:
        return None
    brace_count = 0
    end: Optional[int] = None
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            brace_count += 1
        elif ch == "}":
            brace_count -= 1
            if brace_count == 0:
                end = idx + 1
                break
    if end is None:
        return None
    try:
        return ast.literal_eval(text[start:end])
    except Exception:
        return None


def _safe_rate(true_count: int, total_count: int) -> float:
    return float(true_count) / float(total_count) if int(total_count) else 0.0


def _budget_satisfaction_from_eval_summary(summary: object) -> Optional[Dict[str, object]]:
    if not isinstance(summary, dict):
        return None
    hard = summary.get("Hard Constraint")
    if not isinstance(hard, dict):
        return None

    counts: Dict[str, Dict[str, int]] = {level: {"true": 0, "total": 0} for level in _LEVELS}
    for level in _LEVELS:
        level_obj = hard.get(level)
        if not isinstance(level_obj, dict):
            continue
        for day_data in level_obj.values():
            if not isinstance(day_data, dict):
                continue
            budget = day_data.get("Budget")
            if not isinstance(budget, dict):
                continue
            try:
                counts[level]["true"] += int(budget.get("true", 0) or 0)
                counts[level]["total"] += int(budget.get("total", 0) or 0)
            except (TypeError, ValueError):
                continue

    counts["all"] = {
        "true": sum(counts[level]["true"] for level in _LEVELS),
        "total": sum(counts[level]["total"] for level in _LEVELS),
    }
    rates = {level: _safe_rate(counts[level]["true"], counts[level]["total"]) for level in (*_LEVELS, "all")}
    return {"counts": counts, "rates": rates, "rate_all": rates.get("all", 0.0)}


@dataclass
class StageResult:
    status: int
    elapsed_s: int
    captured: str
    llm_usage_line: Optional[str] = None


def _run_stage(
    *,
    name: str,
    cmd: Sequence[str],
    cost_fp,
    log_fp,
    tee_path: Optional[Path] = None,
    extract_llm_usage: Optional[str] = None,
) -> StageResult:
    start_line = f"[{_iso_now()}] START {name}"
    cost_fp.write(start_line + "\n")
    cost_fp.flush()
    log_fp.write(start_line + "\n")
    log_fp.flush()
    sys.stdout.write(start_line + "\n")
    sys.stdout.flush()

    started = time.time()
    captured_lines: List[str] = []
    llm_usage_line: Optional[str] = None

    tee_fp = None
    if tee_path is not None:
        tee_path.parent.mkdir(parents=True, exist_ok=True)
        tee_fp = tee_path.open("w", encoding="utf-8")

    try:
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            captured_lines.append(line)
            log_fp.write(line)
            log_fp.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
            if tee_fp is not None:
                tee_fp.write(line)
                tee_fp.flush()
            if extract_llm_usage and extract_llm_usage in line:
                llm_usage_line = line.strip()
        status = int(proc.wait())
    finally:
        if tee_fp is not None:
            tee_fp.close()

    elapsed_s = max(0, int(time.time() - started))
    end_line = f"[{_iso_now()}] END {name} status={status} elapsed_s={elapsed_s}"
    cost_fp.write(end_line + "\n")
    if llm_usage_line is not None:
        cost_fp.write(f"{llm_usage_line}\n")
    cost_fp.flush()
    log_fp.write(end_line + "\n")
    log_fp.flush()
    sys.stdout.write(end_line + "\n")
    sys.stdout.flush()
    return StageResult(status=status, elapsed_s=elapsed_s, captured="".join(captured_lines), llm_usage_line=llm_usage_line)


def _load_tool_calls(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _resolve_model_arg(model_arg: str) -> Tuple[str, str]:
    model_arg = str(model_arg or "").strip()
    if model_arg in MODEL_ALIAS:
        return model_arg, MODEL_ALIAS[model_arg]
    if model_arg in MODEL_ALIAS.values():
        tag = next((k for k, v in MODEL_ALIAS.items() if v == model_arg), _model_slug(model_arg))
        return tag, model_arg
    # Allow passing a model slug (e.g., gpt5mini) as a tag.
    if model_arg in FULL_MODELS:
        return model_arg, MODEL_ALIAS[model_arg]
    raise ValueError(f"Unknown model: {model_arg} (expected one of {sorted(MODEL_ALIAS)} or a full model name)")


def _iter_models(model: str) -> List[Tuple[str, str]]:
    if model == "full":
        return [(tag, MODEL_ALIAS[tag]) for tag in FULL_MODELS]
    tag, name = _resolve_model_arg(model)
    return [(tag, name)]


def _parse_ks(values: str) -> List[int]:
    out: List[int] = []
    for part in str(values).split(","):
        part = part.strip()
        if not part:
            continue
        if part.lower() in {"unlimited", "inf", "infty"}:
            out.append(0)
            continue
        out.append(int(part))
    return out


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_money(cost_path: Path, *, log_fp) -> None:
    cmd = [sys.executable, "-m", "task_helper.money_memplan", str(cost_path)]
    try:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
    except Exception as exc:
        log_fp.write(f"[money] error: {type(exc).__name__}: {exc}\n")
        log_fp.flush()
        return
    if proc.stdout:
        log_fp.write(proc.stdout)
    if proc.stderr:
        log_fp.write(proc.stderr)
    log_fp.flush()


def run_one(
    *,
    model_tag: str,
    model_name: str,
    split: str,
    k: int,
    config: Optional[Path],
    workers: int,
    out_root: Path,
    repair_rounds: int,
    no_early_stop: bool,
) -> Path:
    task = "travel"
    model_slug = _model_slug(model_name)
    base_dir = PROJECT_ROOT / "artifacts" / "output" / task / f"{model_slug}_{split}"
    init_templates = base_dir / f"init_templates_{split}.jsonl"
    base_cost = base_dir / "cost.txt"
    if not init_templates.exists():
        raise FileNotFoundError(f"Missing init templates for {model_name}: {init_templates}")
    if not base_cost.exists():
        raise FileNotFoundError(f"Missing base cost.txt for {model_name}: {base_cost}")

    max_tool_candidates = int(k)
    # planner.llm_repair clamps max-slot-candidates to >= 1, so a literal 0 would become 1.
    # For the "unlimited" sweep point (k=0), keep tool candidates unlimited but use the default
    # llm_repair prompt cap (50) to avoid accidental prompt starvation.
    repair_max_slot_candidates = int(k) if int(k) > 0 else 50

    k_label = f"k{max_tool_candidates}" if max_tool_candidates > 0 else "kunlimited"
    run_dir = out_root / f"{model_slug}_{split}_{k_label}_{_utc_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    match_json = run_dir / f"match_{split}.json"
    tree_json = run_dir / f"tree_{split}.json"
    optimized_jsonl = run_dir / f"optimized_{split}.jsonl"
    optimized_raw_jsonl = run_dir / f"optimized_{split}.raw.jsonl"
    optimized_repair1_jsonl = run_dir / f"optimized_{split}.repair1.jsonl"
    optimized_repair2_jsonl = run_dir / f"optimized_{split}.repair2.jsonl"
    cost_txt = run_dir / "cost.txt"
    results1_txt = run_dir / "results_round1.txt"
    results2_txt = run_dir / "results_round2.txt"
    pipeline_log = run_dir / "pipeline.log"

    init_end = _extract_last_stage_end(cost_path=base_cost, stage="init_template") or {}
    init_usage_line = _extract_last_line_containing(path=base_cost, needle="LLM token usage (init_template):") or ""
    init_usage = _parse_usage_line(init_usage_line) if init_usage_line else {}
    init_total_tokens_line = _extract_last_line_containing(path=base_cost, needle="init_template_total_tokens=") or ""

    worker_flag = int(workers)
    if worker_flag < 0:
        worker_flag = 0

    cpu_count = _available_cpu_count()
    llm_repair_workers = cpu_count
    repair_rounds = max(1, int(repair_rounds))

    config_args: List[str] = []
    if config is not None:
        cfg_path = config
        if not cfg_path.is_absolute():
            cfg_path = (PROJECT_ROOT / cfg_path).resolve()
        config_args = ["--config", str(cfg_path)]

    with pipeline_log.open("w", encoding="utf-8") as log_fp, cost_txt.open("w", encoding="utf-8") as cost_fp:
        pipeline_start = int(time.time())

        cost_fp.write(f"run_start={_iso_now()}\n")
        cost_fp.write(f"task={task}\n")
        cost_fp.write(f"split={split}\n")
        cost_fp.write(f"model={model_name}\n")
        cost_fp.write(f"model_slug={model_slug}\n")
        cost_fp.write(f"cpus={worker_flag}\n")
        cost_fp.write(f"workers_effective_all_cpus={cpu_count}\n")
        cost_fp.write("tolerance_rate=0.8\n")
        cost_fp.write("mcts_iterations=50\n")
        cost_fp.write("mcts_uct_c=1.4\n")
        cost_fp.write(f"max_tool_candidates={max_tool_candidates}\n")
        cost_fp.write(f"repair_max_slot_candidates={repair_max_slot_candidates}\n")
        cost_fp.write(f"repair_tree={tree_json}\n")
        cost_fp.write("run_repair=1\n")
        cost_fp.write("run_eval=1\n")
        cost_fp.write("run_init_template=0\n")
        cost_fp.write(f"base_init_templates={init_templates}\n")
        cost_fp.write(f"base_cost_path={base_cost}\n")
        if init_end:
            cost_fp.write(f"base_init_template_end_line={init_end.get('line','')}\n")
        if init_usage_line:
            cost_fp.write(f"base_init_template_usage_line={init_usage_line}\n")
        if init_total_tokens_line:
            cost_fp.write(f"base_init_template_total_tokens_line={init_total_tokens_line}\n")
        cost_fp.flush()

        # Re-emit init_template metrics verbatim so downstream parsers (including money_memplan)
        # can treat this run like a full run while we reuse cached init templates.
        if init_end and isinstance(init_end.get("line"), str) and init_end.get("line"):
            cost_fp.write(f"{init_end['line']}\n")
        if init_usage_line:
            cost_fp.write(f"{init_usage_line}\n")
        if init_total_tokens_line:
            cost_fp.write(f"{init_total_tokens_line}\n")
        cost_fp.flush()

        # 1) ViewSelect
        view_select_cmd = [
            sys.executable,
            "-m",
            "planner.view_select",
            "--task",
            task,
            "--split",
            split,
            "--templates",
            str(init_templates),
            "--out",
            str(match_json),
            "--workers",
            str(worker_flag),
        ]
        view_select_cmd += config_args
        view_select = _run_stage(name="view_select", cmd=view_select_cmd, cost_fp=cost_fp, log_fp=log_fp)
        if view_select.status != 0:
            raise RuntimeError(f"view_select failed with status={view_select.status}")

        # 2) ComposeMatch
        compose_cmd = [
            sys.executable,
            "-m",
            "planner.compose_match",
            "--task",
            task,
            "--split",
            split,
            "--templates",
            str(init_templates),
            "--match",
            str(match_json),
            "--out",
            str(tree_json),
            "--max-tool-candidates",
            str(max_tool_candidates),
            "--workers",
            str(worker_flag),
        ]
        compose_cmd += config_args
        compose = _run_stage(name="compose_match", cmd=compose_cmd, cost_fp=cost_fp, log_fp=log_fp)
        if compose.status != 0:
            raise RuntimeError(f"compose_match failed with status={compose.status}")

        tool_calls_json = run_dir / f"tool_calls_{split}.json"
        tool_calls = _load_tool_calls(tool_calls_json)
        if tool_calls is not None:
            calls = int(tool_calls.get("total_calls") or 0)
            cost = float(tool_calls.get("total_cost") or 0.0)
            by_tool = tool_calls.get("by_tool") if isinstance(tool_calls.get("by_tool"), dict) else {}
            cost_fp.write(
                f"Tool call usage (compose_match): calls={calls} cost={cost} by_tool={json.dumps(by_tool, ensure_ascii=False)}\n"
            )
            cost_fp.write(f"tool_calls_total_calls={calls}\n")
            cost_fp.write(f"tool_calls_total_cost={cost}\n")
            cost_fp.flush()

        # 3) Twin-Track (MCTS)
        mcts_cmd = [
            sys.executable,
            "-m",
            "planner.twin_track",
            "--task",
            task,
            "--tree",
            str(tree_json),
            "--out",
            str(optimized_jsonl),
            "--workers",
            str(worker_flag),
            "--iterations",
            "50",
            "--uct-c",
            "1.4",
            "--semantic-tolerance",
            "0.8",
        ]
        if bool(no_early_stop):
            mcts_cmd.append("--no-early-stop")
        mcts_cmd += config_args
        mcts = _run_stage(name="mcts", cmd=mcts_cmd, cost_fp=cost_fp, log_fp=log_fp)
        if mcts.status != 0:
            raise RuntimeError(f"mcts failed with status={mcts.status}")

        if optimized_jsonl.exists():
            optimized_raw_jsonl.write_bytes(optimized_jsonl.read_bytes())

        # 4) LLM repair round(s) (in-place overwrite of optimized)
        llm_repair_cmd = [
            sys.executable,
            "-m",
            "planner.llm_repair",
            "--task",
            task,
            "--tree",
            str(tree_json),
            "--input",
            str(optimized_jsonl),
            "--out",
            str(optimized_jsonl),
            "--model",
            model_name,
            "--semantic-threshold",
            "0.8",
            "--max-slot-candidates",
            str(repair_max_slot_candidates),
            "--workers",
            str(llm_repair_workers),
        ]
        llm_repair_cmd += config_args
        repair_stage_results: List[StageResult] = []
        eval_stage_results: List[StageResult] = []
        optimized_snapshots: List[Path] = []
        results_paths: List[Path] = []

        for round_idx in range(1, repair_rounds + 1):
            stage_name = "llm_repair" if round_idx == 1 else f"llm_repair_round{round_idx}"
            stage = _run_stage(
                name=stage_name,
                cmd=llm_repair_cmd,
                cost_fp=cost_fp,
                log_fp=log_fp,
                extract_llm_usage="LLM token usage (llm_repair):",
            )
            if stage.status != 0:
                raise RuntimeError(f"{stage_name} failed with status={stage.status}")
            repair_stage_results.append(stage)

            snapshot_path = run_dir / f"optimized_{split}.repair{round_idx}.jsonl"
            if optimized_jsonl.exists():
                snapshot_path.write_bytes(optimized_jsonl.read_bytes())
            optimized_snapshots.append(snapshot_path)

            eval_name = "eval" if round_idx == 1 else f"eval_round{round_idx}"
            results_path = run_dir / f"results_round{round_idx}.txt"
            eval_stage = _run_stage(
                name=eval_name,
                cmd=[
                    sys.executable,
                    "-m",
                    "task_helper.travel.runners.eval_bridge",
                    "--set-type",
                    split,
                    "--submission",
                    str(optimized_jsonl),
                ],
                cost_fp=cost_fp,
                log_fp=log_fp,
                tee_path=results_path,
            )
            if eval_stage.status != 0:
                raise RuntimeError(f"{eval_name} failed with status={eval_stage.status}")
            eval_stage_results.append(eval_stage)
            results_paths.append(results_path)

        # Pipeline totals (exclude eval time for comparisons; init_template is reused and not timed here).
        pipeline_end = int(time.time())
        pipeline_elapsed_total_measured = max(0, pipeline_end - pipeline_start)
        eval_elapsed_total = sum(int(s.elapsed_s) for s in eval_stage_results)
        pipeline_elapsed_no_eval_measured = max(0, pipeline_elapsed_total_measured - eval_elapsed_total)

        base_init_elapsed_s = int(init_end.get("elapsed_s") or 0)

        cost_fp.write(f"run_end={_iso_now()}\n")
        cost_fp.write(f"base_init_template_elapsed_s={base_init_elapsed_s}\n")
        # Report sweep-stage wall-clock only; init_template is reused.
        cost_fp.write(f"pipeline_elapsed_s_total={pipeline_elapsed_total_measured}\n")
        cost_fp.write(f"pipeline_elapsed_s_no_eval={pipeline_elapsed_no_eval_measured}\n")
        cost_fp.write(f"eval_elapsed_s={eval_elapsed_total}\n")
        for idx, stage in enumerate(eval_stage_results, start=1):
            cost_fp.write(f"eval_elapsed_s_round{idx}={int(stage.elapsed_s)}\n")
        cost_fp.write(f"pipeline_elapsed_s_total_measured={pipeline_elapsed_total_measured}\n")
        cost_fp.write(f"pipeline_elapsed_s_no_eval_measured={pipeline_elapsed_no_eval_measured}\n")
        cost_fp.write(f"outputs_dir={run_dir}\n")
        cost_fp.flush()

        # Best-effort token totals (init_template from base + llm_repair from this run).
        repair_usages: List[Dict[str, int]] = []
        for stage in repair_stage_results:
            line = stage.llm_usage_line or ""
            repair_usages.append(_parse_usage_line(line) if line else {})
        pipeline_total_tokens = int(init_usage.get("total") or 0) + sum(int(u.get("total") or 0) for u in repair_usages)
        cost_fp.write(f"pipeline_total_tokens={pipeline_total_tokens}\n")

        tool_calls_cost = 0.0
        if tool_calls is not None:
            tool_calls_cost = float(tool_calls.get("total_cost") or 0.0)
        cost_fp.write(
            f"summary: time_s_no_eval={pipeline_elapsed_no_eval_measured} tokens_total={pipeline_total_tokens} tool_calls_cost={tool_calls_cost}\n"
        )
        cost_fp.flush()

        # Append LLM $ cost summary (best-effort).
        _run_money(cost_txt, log_fp=log_fp)

        eval_rounds: List[Dict[str, object]] = []
        for idx, stage in enumerate(eval_stage_results, start=1):
            results_path = results_paths[idx - 1]
            eval_text = results_path.read_text(encoding="utf-8") if results_path.exists() else stage.captured
            eval_summary = _extract_eval_summary_dict(eval_text)
            budget = _budget_satisfaction_from_eval_summary(eval_summary)
            metrics = _parse_eval_stdout(stage.captured)
            if budget is not None:
                metrics["Budget Satisfaction Rate"] = float(budget.get("rate_all") or 0.0)
            eval_rounds.append(
                {
                    "round": int(idx),
                    "elapsed_s": int(stage.elapsed_s),
                    "metrics": metrics,
                    "budget_satisfaction": budget,
                    "results_path": str(results_path),
                }
            )
        summary = {
            "version": 1,
            "task": task,
            "split": split,
            "k": int(k),
            "max_tool_candidates": int(max_tool_candidates),
            "repair_max_slot_candidates": int(repair_max_slot_candidates),
            "model_tag": model_tag,
            "model": model_name,
            "model_slug": model_slug,
            "workers": int(worker_flag),
            "workers_effective_all_cpus": int(cpu_count),
            "repair_rounds": int(repair_rounds),
            "paths": {
                "run_dir": str(run_dir),
                "base_dir": str(base_dir),
                "base_cost": str(base_cost),
                "init_templates": str(init_templates),
                "match": str(match_json),
                "tree": str(tree_json),
                "tool_calls": str(tool_calls_json),
                "optimized": str(optimized_jsonl),
                "optimized_raw": str(optimized_raw_jsonl),
                "optimized_repair1": str(optimized_repair1_jsonl),
                "optimized_repair2": str(optimized_repair2_jsonl),
                "optimized_snapshots": [str(p) for p in optimized_snapshots],
                "cost": str(cost_txt),
                "results_round1": str(results1_txt),
                "results_round2": str(results2_txt),
                "results_rounds": [str(p) for p in results_paths],
                "log": str(pipeline_log),
            },
            "init_template": {
                "base_cost_path": str(base_cost),
                "end": init_end,
                "usage_line": init_usage_line,
                "usage": init_usage,
            },
            "tool_calls": tool_calls,
            "llm_repair_rounds": [
                {"round": idx + 1, "usage_line": (stage.llm_usage_line or ""), "usage": repair_usages[idx]}
                for idx, stage in enumerate(repair_stage_results)
            ],
            "eval_rounds": eval_rounds,
            "pipeline_elapsed_s_total": int(pipeline_elapsed_total_measured),
            "pipeline_elapsed_s_no_eval": int(pipeline_elapsed_no_eval_measured),
            "pipeline_elapsed_s_total_measured": int(pipeline_elapsed_total_measured),
            "pipeline_elapsed_s_no_eval_measured": int(pipeline_elapsed_no_eval_measured),
            "base_init_template_elapsed_s": int(base_init_elapsed_s),
            "pipeline_total_tokens": int(pipeline_total_tokens),
        }
        _write_json(run_dir / "summary.json", summary)

    return run_dir


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Travel candidate-count sweep (ComposeMatch + LLM repair).")
    parser.add_argument("--model", type=str, default="full", help="Model tag (gpt52/gpt5mini/gpt5nano/deepseekchat) or 'full'.")
    parser.add_argument("--split", type=str, default="validation", choices=("train", "validation"), help="Dataset split (default: validation).")
    parser.add_argument(
        "--ks",
        type=str,
        default="2,4,6,8,10,unlimited",
        help="Comma-separated candidate counts to sweep (use 0 or 'unlimited' for unlimited tool candidates).",
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional override planner config JSON.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="Output root (defaults to artifacts/output/travel/analysis/candidate).")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Workers for view_select/compose_match/mcts (0 = all CPUs, matches task_helper/travel/run.sh default).",
    )
    parser.add_argument(
        "--repair-rounds",
        type=int,
        default=1,
        help="How many LLM repair rounds to run (1 = standard; 2 = run a second repair on the repaired plan and reevaluate).",
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable Twin-Track early stopping (always run full MCTS iterations).",
    )
    args = parser.parse_args(argv)

    out_root = (PROJECT_ROOT / args.out_root).resolve() if not args.out_root.is_absolute() else args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    ks = _parse_ks(args.ks)
    models = _iter_models(str(args.model))

    failures: List[str] = []
    for model_tag, model_name in models:
        for k in ks:
            print(f"\n=== candidate_sweep: model={model_name} split={args.split} k={k} ===")
            try:
                run_dir = run_one(
                    model_tag=model_tag,
                    model_name=model_name,
                    split=str(args.split),
                    k=int(k),
                    config=args.config,
                    workers=int(args.workers),
                    out_root=out_root,
                    repair_rounds=int(args.repair_rounds),
                    no_early_stop=bool(args.no_early_stop),
                )
                print(f"[done] {run_dir}")
            except Exception as exc:
                failures.append(f"model={model_name} k={k}: {type(exc).__name__}: {exc}")
                print(f"[failed] model={model_name} k={k}: {type(exc).__name__}: {exc}", file=sys.stderr)

    if failures:
        print("\nFailures:", file=sys.stderr)
        for item in failures:
            print(f"- {item}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
