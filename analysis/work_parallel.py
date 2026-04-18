#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = PROJECT_ROOT / "artifacts" / "output" / "work" / "analysis"
PARALLEL_PATH = ANALYSIS_ROOT / "parallel.json"
RUN_SCRIPT = PROJECT_ROOT / "task_helper" / "work" / "run.sh"

DEFAULT_WORKERS = (12, 24, 36, 48)
_REPAIR_DONE_RE = re.compile(r"LLM repair done: .* elapsed=([0-9.]+)s")


def _model_slug(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "", str(name or "").lower())
    return cleaned or "model"


def _unique(values: Iterable[int]) -> List[int]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _run_cmd(
    cmd: Sequence[str],
    *,
    env: Optional[Dict[str, str]] = None,
    log_path: Optional[Path] = None,
    line_handler=None,
) -> None:
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp = log_path.open("a", encoding="utf-8")
    else:
        log_fp = None
    status: Optional[int] = None
    try:
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
            if log_fp is not None:
                log_fp.write(line)
                log_fp.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
            if line_handler:
                line_handler(line)
        status = proc.wait()
    finally:
        if log_fp is not None:
            log_fp.close()
    if status is None:
        raise RuntimeError(f"Command failed to start: {' '.join(cmd)}")
    if status != 0:
        raise RuntimeError(f"Command failed ({status}): {' '.join(cmd)}")


def _read_pipeline_elapsed_no_eval(cost_path: Path) -> float:
    if not cost_path.exists():
        raise FileNotFoundError(f"Missing cost file: {cost_path}")
    value: Optional[float] = None
    for line in cost_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("pipeline_elapsed_s_no_eval="):
            raw = line.split("=", 1)[1].strip()
            try:
                value = float(raw)
            except ValueError:
                continue
    if value is None:
        raise RuntimeError(f"Missing pipeline_elapsed_s_no_eval in {cost_path}")
    return value


def _load_parallel(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"parallel.json must be a JSON object: {path}")
    return data


def _write_parallel(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _ensure_model_entry(data: Dict[str, object], model: str, workers: Sequence[int]) -> Dict[str, object]:
    entry = data.get(model)
    if not isinstance(entry, dict):
        entry = {}
        data[model] = entry
    for section in ("norepair", "repair"):
        section_map = entry.get(section)
        if not isinstance(section_map, dict):
            section_map = {}
            entry[section] = section_map
        for worker in workers:
            section_map.setdefault(str(worker), None)
    return entry


def _update_parallel(
    *,
    model: str,
    workers: Sequence[int],
    phase: str,
    worker_count: int,
    elapsed_s: float,
) -> None:
    data = _load_parallel(PARALLEL_PATH)
    entry = _ensure_model_entry(data, model, workers)
    if not isinstance(entry.get(phase), dict):
        raise RuntimeError(f"parallel.json entry missing {phase} map for {model}")
    entry[phase][str(worker_count)] = float(elapsed_s)
    _write_parallel(PARALLEL_PATH, data)


def _run_base_pipeline(
    *,
    model: str,
    split: str,
    mode: str,
    workers: int,
    log_dir: Path,
) -> float:
    env = os.environ.copy()
    env.update(
        {
            "TASK": "work",
            "SPLIT": split,
            "MODEL": model,
            "MODE": mode,
            "CPUS": str(workers),
            "MULTI_WORKERS": str(workers),
            "RUN_EVAL": "0",
            "RUN_LLM_REPAIR": "0",
        }
    )
    log_path = log_dir / f"{_model_slug(model)}_{split}_{workers}_base.log"
    _run_cmd(["bash", str(RUN_SCRIPT)], env=env, log_path=log_path)

    output_dir = PROJECT_ROOT / "artifacts" / "output" / "work" / f"{_model_slug(model)}_{split}"
    return _read_pipeline_elapsed_no_eval(output_dir / "cost.txt")


def _run_llm_repair(
    *,
    model: str,
    split: str,
    workers: int,
    semantic_threshold: float,
    run_convert: bool,
    log_dir: Path,
) -> float:
    output_dir = PROJECT_ROOT / "artifacts" / "output" / "work" / f"{_model_slug(model)}_{split}"
    tree_path = output_dir / f"tree_enriched_{split}.json"
    if not tree_path.exists():
        tree_path = output_dir / f"tree_{split}.json"
    optimized_path = output_dir / f"optimized_{split}.jsonl"
    if not tree_path.exists():
        raise FileNotFoundError(f"Missing tree JSON: {tree_path}")
    if not optimized_path.exists():
        raise FileNotFoundError(f"Missing optimized JSONL: {optimized_path}")

    repair_dir = output_dir / "repair"
    repair_dir.mkdir(parents=True, exist_ok=True)
    repair_slug = _model_slug(model)
    repair_out = repair_dir / f"repaired_{split}_{repair_slug}.jsonl"
    log_path = log_dir / f"{repair_slug}_{split}_{workers}_repair.log"

    elapsed_value: Dict[str, float] = {}

    def _capture_elapsed(line: str) -> None:
        match = _REPAIR_DONE_RE.search(line)
        if match:
            try:
                elapsed_value["elapsed"] = float(match.group(1))
            except ValueError:
                return

    cmd = [
        sys.executable,
        "-m",
        "planner.llm_repair",
        "--task",
        "work",
        "--tree",
        str(tree_path),
        "--input",
        str(optimized_path),
        "--model",
        model,
        "--semantic-threshold",
        str(semantic_threshold),
        "--out",
        str(repair_out),
        "--workers",
        str(workers),
        "--workbench-only-incorrect",
    ]
    _run_cmd(cmd, log_path=log_path, line_handler=_capture_elapsed)

    if "elapsed" not in elapsed_value:
        raise RuntimeError("Failed to capture LLM repair elapsed time from output.")

    if run_convert:
        results_dir = repair_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        convert_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "task_helper" / "work" / "evaluation" / "convert_optimized_to_predictions.py"),
            "--optimized",
            str(repair_out),
            "--out-dir",
            str(results_dir),
        ]
        convert_log = log_dir / f"{repair_slug}_{split}_{workers}_repair_convert.log"
        _run_cmd(convert_cmd, log_path=convert_log)

    return elapsed_value["elapsed"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run Work validation scalability experiments.")
    parser.add_argument(
        "--model",
        action="append",
        help="Model name (repeatable). Defaults to MODEL env or gpt-5-nano.",
    )
    parser.add_argument("--split", default="validation", help="Dataset split (default: validation).")
    parser.add_argument("--mode", default="multi", choices=("multi", "single"), help="Planner mode.")
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=list(DEFAULT_WORKERS),
        help="Worker counts to run (default: 12 24 36 48).",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.8,
        help="Semantic threshold for LLM repair.",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert repaired JSONL to predictions (default: skip).",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip converting repaired JSONL to predictions.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if parallel.json already has a value.",
    )
    args = parser.parse_args(argv)

    models = args.model or [os.environ.get("MODEL") or "gpt-5-nano"]
    workers_list = _unique(int(w) for w in args.workers if int(w) > 0)
    if not workers_list:
        raise SystemExit("No valid worker counts provided.")

    log_dir = ANALYSIS_ROOT / "parallel_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    run_convert = bool(args.convert)
    if args.skip_convert:
        run_convert = False

    for model in models:
        for worker_count in workers_list:
            data = _load_parallel(PARALLEL_PATH)
            entry = _ensure_model_entry(data, model, workers_list)
            norepair_map = entry.get("norepair")
            repair_map = entry.get("repair")
            if (
                not args.force
                and isinstance(norepair_map, dict)
                and isinstance(repair_map, dict)
                and norepair_map.get(str(worker_count)) is not None
                and repair_map.get(str(worker_count)) is not None
            ):
                print(f"Skipping {model} workers={worker_count}: already recorded.")
                continue

            print(f"==> Running base pipeline: model={model} workers={worker_count}")
            base_elapsed = _run_base_pipeline(
                model=model,
                split=args.split,
                mode=args.mode,
                workers=worker_count,
                log_dir=log_dir,
            )
            _update_parallel(
                model=model,
                workers=workers_list,
                phase="norepair",
                worker_count=worker_count,
                elapsed_s=base_elapsed,
            )

            print(f"==> Running LLM repair: model={model} workers={worker_count}")
            repair_elapsed = _run_llm_repair(
                model=model,
                split=args.split,
                workers=worker_count,
                semantic_threshold=args.semantic_threshold,
                run_convert=run_convert,
                log_dir=log_dir,
            )
            _update_parallel(
                model=model,
                workers=workers_list,
                phase="repair",
                worker_count=worker_count,
                elapsed_s=repair_elapsed,
            )

            print(
                f"Completed model={model} workers={worker_count} "
                f"norepair={base_elapsed:.2f}s repair={repair_elapsed:.2f}s"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
