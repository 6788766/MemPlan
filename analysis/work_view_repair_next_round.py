#!/usr/bin/env python3
"""
Run additional LLM repair rounds for Work view-sweep runs (15/30/45/60 by default).

For each run directory under:
  artifacts/output/work/analysis/view/<model_slug>_<num_views>_<split>_<tag>/

This script:
  - finds the latest existing repair round rN (or falls back to optimized_<split>.jsonl)
  - runs planner.llm_repair on that last output for `--rounds N` consecutive rounds
  - writes each new round under run_dir/repair/r{N+1}/, r{N+2}/, ...
  - runs convert + eval + pass_rates only for the final round by default

It only writes inside existing run folders (under artifacts/output/work/analysis/view).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]

# Avoid creating __pycache__ outside the analysis output tree.
sys.dont_write_bytecode = True

# Allow running as `python analysis/work_view_repair_next_round.py ...` from anywhere.
for parent in REPO_ROOT, *REPO_ROOT.parents:
    if (parent / "artifacts").is_dir():
        sys.path.insert(0, str(parent))
        break

from planner.parallel import available_cpu_count  # noqa: E402

OUTPUT_ROOT = REPO_ROOT / "artifacts" / "output" / "work" / "analysis" / "view"
GROUND_TRUTH_DIR = REPO_ROOT / "artifacts" / "input" / "work" / "dataset" / "queries_and_answers"


def _utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _model_slug(model: str) -> str:
    return "".join([c.lower() for c in str(model) if c.isalnum()]) or "model"


def _append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(line.rstrip() + "\n")


def _run_cmd(
    cmd: Sequence[str],
    *,
    log_path: Path,
    tee: bool,
    tee_prefix: str,
    cwd: Path = REPO_ROOT,
    extra_env: Optional[Mapping[str, str]] = None,
) -> None:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})

    cmd_line = " ".join(cmd)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fp:
        fp.write(f"[{_utc_ts()}] CMD: {cmd_line}\n")
        fp.flush()
        if tee:
            prefix = f"[{tee_prefix}] " if tee_prefix else ""
            print(f"{prefix}CMD: {cmd_line}", flush=True)

        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            fp.write(line)
            if tee:
                if tee_prefix:
                    print(f"[{tee_prefix}] {line}", end="", flush=True)
                else:
                    print(line, end="", flush=True)
        rc = proc.wait()
        fp.write(f"\n[{_utc_ts()}] EXIT: {rc}\n")
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)


def _extract_last_line(path: Path, *, needle: str) -> Optional[str]:
    try:
        last = None
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if needle in raw:
                last = raw.strip()
        return last
    except Exception:
        return None


def _find_last_repair_round(run_dir: Path) -> int:
    repair_root = run_dir / "repair"
    if not repair_root.exists():
        return 0
    best = 0
    for p in repair_root.iterdir():
        if not p.is_dir():
            continue
        name = p.name.strip().lower()
        if not (name.startswith("r") and name[1:].isdigit()):
            continue
        best = max(best, int(name[1:]))
    return int(best)


def _run_convert_and_eval(
    *,
    split: str,
    repaired_jsonl: Path,
    round_dir: Path,
    tee: bool,
) -> None:
    logs_dir = round_dir / "logs"
    results_dir = round_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    _run_cmd(
        [
            "python",
            "task_helper/work/evaluation/convert_optimized_to_predictions.py",
            "--optimized",
            str(repaired_jsonl),
            "--out-dir",
            str(results_dir),
        ],
        log_path=logs_dir / "convert.log",
        tee=tee,
        tee_prefix="convert",
    )

    _run_cmd(
        [
            "python",
            "task_helper/work/evaluation/calculate_all_metrics.py",
            "--predictions_dir",
            str(results_dir),
        ],
        log_path=logs_dir / "eval_metrics.log",
        tee=tee,
        tee_prefix="eval",
    )

    _run_cmd(
        [
            "python",
            "task_helper/work/evaluation/calculate_pass_rates_dir.py",
            "--predictions_dir",
            str(results_dir),
            "--ground_truth_dir",
            str(GROUND_TRUTH_DIR),
            "--json_out",
            str(results_dir / "pass_rates.json"),
        ],
        log_path=logs_dir / "pass_rates.log",
        tee=tee,
        tee_prefix="pass_rates",
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add one more WorkBench repair round for view-sweep runs (writes under analysis/view).")
    p.add_argument("--model", type=str, default="gpt-5.2", help="Model to select run folders (slugged).")
    p.add_argument("--split", type=str, default="validation", help="Split label (default: validation).")
    p.add_argument("--views", type=str, default="15,30,45,60", help="Comma-separated view counts.")
    p.add_argument("--tag", type=str, default="seed0", help="Run tag to match (default: seed0).")
    p.add_argument("--root", type=Path, default=OUTPUT_ROOT, help="Root folder containing the view-sweep run dirs.")

    p.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of additional sequential repair rounds to run (default: 1).",
    )
    p.add_argument(
        "--eval-mode",
        type=str,
        choices=("final", "all", "none"),
        default="final",
        help="Evaluation mode for these added rounds: final (default), all, none.",
    )
    p.add_argument("--repair-model", type=str, default="", help="LLM model for planner.llm_repair (default: same as --model).")
    p.add_argument("--tolerance", type=float, default=0.8, help="Semantic threshold passed to planner.llm_repair.")
    p.add_argument("--max-slot-candidates", type=int, default=50, help="Max slot candidates passed to planner.llm_repair.")
    p.add_argument("--workers", type=int, default=0, help="Repair workers (0 = all CPUs).")

    p.add_argument("--tee", action=argparse.BooleanOptionalAction, default=True, help="Stream stage output to stdout (default: enabled).")
    p.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable per-template progress output for llm_repair (default: enabled).",
    )
    p.add_argument("--force", action="store_true", help="Overwrite an existing next-round folder if present.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    model = str(args.model)
    model_slug = _model_slug(model)
    split = str(args.split)
    tag = str(args.tag).strip() or "seed0"
    root = Path(args.root).expanduser()
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()

    rounds = int(args.rounds)
    if rounds <= 0:
        raise SystemExit("--rounds must be >= 1")
    eval_mode = str(args.eval_mode)

    view_counts = [int(x.strip()) for x in str(args.views).split(",") if x.strip()]
    if not view_counts:
        raise SystemExit("No views provided.")

    repair_model = str(args.repair_model).strip() or model
    repair_workers = int(args.workers) if int(args.workers) > 0 else available_cpu_count()
    tee = bool(args.tee)
    show_progress = bool(args.show_progress)

    for v in view_counts:
        run_name = f"{model_slug}_{int(v)}_{split}_{tag}"
        run_dir = root / run_name
        if not run_dir.exists():
            print(f"[skip] missing run_dir: {run_dir}")
            continue

        last_round = _find_last_repair_round(run_dir)
        start_round = last_round + 1
        end_round = last_round + rounds

        tree = run_dir / f"tree_enriched_{split}.json"
        if not tree.exists():
            tree = run_dir / f"tree_{split}.json"
        if not tree.exists():
            raise SystemExit(f"Missing tree JSON for {run_dir} (expected tree_{split}.json or tree_enriched_{split}.json)")

        prev_dir = run_dir / "repair" / f"r{last_round}" if last_round > 0 else run_dir
        prev_jsonl = (
            (prev_dir / f"repaired_{split}_r{last_round}.jsonl") if last_round > 0 else (run_dir / f"optimized_{split}.jsonl")
        )

        if not prev_jsonl.exists():
            raise SystemExit(f"Missing previous output JSONL for {run_dir}: {prev_jsonl}")

        for cur_round in range(start_round, end_round + 1):
            is_final = cur_round == end_round
            do_eval = (eval_mode == "all") or (eval_mode == "final" and is_final)
            do_eval = do_eval and (eval_mode != "none")

            round_dir = run_dir / "repair" / f"r{cur_round}"
            out_jsonl = round_dir / f"repaired_{split}_r{cur_round}.jsonl"
            logs_dir = round_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            if out_jsonl.exists() and not bool(args.force):
                print(f"[skip] already exists: {out_jsonl} (use --force to overwrite)")
                prev_jsonl = out_jsonl
                continue
            if round_dir.exists() and bool(args.force):
                # Only remove files within this round folder (never touch sibling rounds).
                for p in list(round_dir.rglob("*")):
                    if p.is_file():
                        p.unlink()

            cost_txt = round_dir / "cost.txt"
            _append_line(cost_txt, f"run_start={_utc_ts()}")
            _append_line(cost_txt, "task=work")
            _append_line(cost_txt, f"split={split}")
            _append_line(cost_txt, f"model={repair_model}")
            _append_line(cost_txt, f"base_run_dir={str(run_dir.resolve())}")
            _append_line(cost_txt, f"prev_round={int(cur_round - 1)}")
            _append_line(cost_txt, f"next_round={int(cur_round)}")
            _append_line(cost_txt, f"tree={str(tree.resolve())}")
            _append_line(cost_txt, f"input={str(prev_jsonl.resolve())}")
            _append_line(cost_txt, f"semantic_threshold={float(args.tolerance)}")
            _append_line(cost_txt, f"max_slot_candidates={int(args.max_slot_candidates)}")
            _append_line(cost_txt, f"workers={int(args.workers)} (effective={repair_workers})")

            # Copy init_template token usage line if present (helps combined pricing).
            base_cost = run_dir / "cost.txt"
            if base_cost.exists():
                init_line = _extract_last_line(base_cost, needle="LLM token usage (init_template):")
                if init_line:
                    _append_line(cost_txt, init_line)
            else:
                init_templates = run_dir / f"init_templates_{split}.jsonl"
                if init_templates.exists():
                    try:
                        out = subprocess.check_output(
                            ["python", "-m", "task_helper.sum_init_tokens", str(init_templates)],
                            cwd=str(REPO_ROOT),
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                            text=True,
                        ).strip()
                        if out:
                            _append_line(cost_txt, f"LLM token usage (init_template): {out}")
                    except Exception:
                        pass

            print(f"[{run_name}] llm_repair r{cur_round - 1} → r{cur_round}", flush=True)
            cmd = [
                "python",
                "-m",
                "planner.llm_repair",
                "--task",
                "work",
                "--tree",
                str(tree),
                "--input",
                str(prev_jsonl),
                "--model",
                str(repair_model),
                "--semantic-threshold",
                str(float(args.tolerance)),
                "--max-slot-candidates",
                str(int(args.max_slot_candidates)),
                "--workers",
                str(int(repair_workers)),
                "--out",
                str(out_jsonl),
            ]
            if not show_progress:
                cmd.append("--no-progress")
            _run_cmd(cmd, log_path=logs_dir / "llm_repair.log", tee=tee, tee_prefix="llm_repair")

            usage = _extract_last_line(logs_dir / "llm_repair.log", needle="LLM token usage (llm_repair):")
            if usage:
                _append_line(cost_txt, usage)

            try:
                calls = subprocess.check_output(
                    ["python", "-m", "task_helper.work.summarize_tool_calls", str(out_jsonl)],
                    cwd=str(REPO_ROOT),
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                    text=True,
                ).strip()
                if calls:
                    _append_line(cost_txt, f"Tool call usage (optimized): {calls}")
            except Exception:
                pass

            if do_eval:
                print(f"[{run_name}] eval r{cur_round}", flush=True)
                _run_convert_and_eval(split=split, repaired_jsonl=out_jsonl, round_dir=round_dir, tee=tee)
            else:
                _append_line(cost_txt, "stage_skipped_convert_eval=1")

            # Append LLM dollar cost (best-effort).
            try:
                subprocess.check_call(
                    ["python", "-m", "task_helper.money_memplan", str(cost_txt)],
                    cwd=str(REPO_ROOT),
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                _append_line(cost_txt, "llm_money_error=1")

            _append_line(cost_txt, f"run_end={_utc_ts()}")
            print(f"Done: {round_dir}")
            prev_jsonl = out_jsonl


if __name__ == "__main__":
    main()
