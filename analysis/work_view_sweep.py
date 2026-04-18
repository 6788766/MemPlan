#!/usr/bin/env python3
"""
WorkBench view-count sweep runner.

Goal:
- Run the Work pipeline for different *memory view* counts (e.g., 15/30/45/60).
- Store *all* outputs (including the sampled `views/memory.jsonl`) under:
    artifacts/output/work/analysis/view/<model_slug>_<num_views>_<split>_<tag>/
- Avoid writing anywhere else by always providing explicit `--out` paths.

Parallelism:
- ViewSelect: uses `--workers 0` by default (all CPUs).
- ComposeMatch(multi): runs as-is (no extra parallel wrapper in this script).
- Twin-Track(multi): uses `--workers 0` by default (all CPUs).

Notes:
- This script does not modify any non-analysis files.
- It reads the canonical view libraries from:
    artifacts/input/work/views/{memory.jsonl,tool.json}
  and writes the sampled subset into each run folder.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]

# Avoid creating __pycache__ outside the analysis output tree.
sys.dont_write_bytecode = True

# Allow running as `python analysis/work_view_sweep.py ...` while importing repo modules.
for parent in REPO_ROOT, *REPO_ROOT.parents:
    if (parent / "artifacts").is_dir():
        sys.path.insert(0, str(parent))
        break

from planner.parallel import available_cpu_count  # noqa: E402


OUTPUT_ROOT = REPO_ROOT / "artifacts" / "output" / "work" / "analysis" / "view"
INPUT_VIEWS_DIR = REPO_ROOT / "artifacts" / "input" / "work" / "views"
GROUND_TRUTH_DIR = REPO_ROOT / "artifacts" / "input" / "work" / "dataset" / "queries_and_answers"


def _utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _model_slug(model: str) -> str:
    return "".join([c.lower() for c in str(model) if c.isalnum()]) or "model"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(line.rstrip() + "\n")

def _count_nonempty_lines(path: Path) -> int:
    return sum(1 for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip())


def _ensure_empty_dir(path: Path, *, force: bool) -> None:
    if path.exists():
        if not force:
            raise SystemExit(f"Run dir exists: {path} (use --force to overwrite)")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def _count_jsonl_rows(path: Path) -> int:
    return sum(1 for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip())


def _is_complete_jsonl(path: Path, *, expected_rows: int) -> bool:
    if not path.exists():
        return False
    try:
        return _count_jsonl_rows(path) >= int(expected_rows)
    except Exception:
        return False


def _run_cmd(
    cmd: Sequence[str],
    *,
    log_path: Path,
    tee: bool = False,
    tee_prefix: str = "",
    cwd: Path = REPO_ROOT,
    extra_env: Optional[Mapping[str, str]] = None,
) -> None:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fp:
        cmd_line = " ".join(cmd)
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
                    print(f"[{tee_prefix}] {line}", end="")
                else:
                    print(line, end="")
        rc = proc.wait()
        fp.write(f"\n[{_utc_ts()}] EXIT: {rc}\n")
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)


def _derive_run_name(*, model: str, views: int, split: str, tag: str) -> str:
    slug = _model_slug(model)
    safe_tag = str(tag).strip() or "run"
    safe_tag = safe_tag.replace(" ", "_")
    return f"{slug}_{int(views)}_{split}_{safe_tag}"


def _sample_memory_view_lines(*, src: Path, counts: Sequence[int], seed: int) -> Dict[int, List[str]]:
    """
    Nested sampling: shuffle once using `seed` then take prefixes for each K.
    Ensures the 15-view set is a subset of the 30-view set, etc.
    """
    lines = [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise SystemExit(f"Empty memory view library: {src}")
    max_k = max(int(k) for k in counts)
    if max_k > len(lines):
        raise SystemExit(f"Requested max views {max_k} but only {len(lines)} available in {src}")
    rng = random.Random(int(seed))
    idxs = list(range(len(lines)))
    rng.shuffle(idxs)
    chosen = [lines[i] for i in idxs]
    return {int(k): chosen[: int(k)] for k in counts}


def _write_views_subset(*, run_dir: Path, memory_lines: Sequence[str]) -> None:
    views_dir = run_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(INPUT_VIEWS_DIR / "tool.json", views_dir / "tool.json")
    _write_text(views_dir / "memory.jsonl", "\n".join(memory_lines).rstrip() + "\n")
    _write_text(
        views_dir / "memory_meta.json",
        json.dumps(
            {
                "source": str((INPUT_VIEWS_DIR / "memory.jsonl").resolve()),
                "total_available": _count_nonempty_lines(INPUT_VIEWS_DIR / "memory.jsonl"),
                "num_views": len(list(memory_lines)),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )


def _ensure_init_templates(
    *,
    out_path: Path,
    split: str,
    model: str,
    templates_in: Optional[Path],
    workers: int,
    max_rows: Optional[int],
    log_path: Path,
    tee: bool = False,
) -> None:
    if templates_in is not None:
        src = templates_in.expanduser()
        if not src.is_absolute():
            src = (REPO_ROOT / src).resolve()
        if not src.exists():
            raise SystemExit(f"Missing templates file: {src}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out_path)
        return

    worker_count = available_cpu_count() if int(workers) == 0 else max(1, int(workers))
    cmd = [
        "python",
        "-m",
        "planner.init_template",
        "--task",
        "work",
        "--split",
        str(split),
        "--all",
        "--plan-fields",
        "auto",
        "--model",
        str(model),
        "--workers",
        str(worker_count),
        "--out",
        str(out_path),
    ]
    if max_rows is not None:
        cmd += ["--max-rows", str(int(max_rows))]
    _run_cmd(cmd, log_path=log_path, tee=tee, tee_prefix="init_template")


def _default_templates_in(*, model: str, split: str) -> Optional[Path]:
    """
    Preferred reuse path for fair sweeps:
      artifacts/output/work/<model_slug>_<split>/init_templates_<split>.jsonl

    Example:
      artifacts/output/work/gpt52_validation/init_templates_validation.jsonl
    """

    slug = _model_slug(model)
    candidate = REPO_ROOT / "artifacts" / "output" / "work" / f"{slug}_{split}" / f"init_templates_{split}.jsonl"
    return candidate if candidate.exists() else None


def _trim_jsonl_in_place(path: Path, *, max_rows: int) -> None:
    lines = [ln for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    lines = lines[: int(max_rows)]
    _write_text(path, "\n".join(lines).rstrip() + "\n")


def _sum_init_tokens(init_templates_path: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["python", "-m", "task_helper.sum_init_tokens", str(init_templates_path)],
            cwd=str(REPO_ROOT),
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def _run_view_select(
    *,
    split: str,
    templates: Path,
    views_dir: Path,
    out_match: Path,
    workers: int,
    log_path: Path,
    tee: bool = False,
) -> None:
    cmd = [
        "python",
        "-m",
        "planner.view_select",
        "--task",
        "work",
        "--split",
        str(split),
        "--templates",
        str(templates),
        "--views-dir",
        str(views_dir),
        "--out",
        str(out_match),
        "--workers",
        str(int(workers)),
    ]
    _run_cmd(cmd, log_path=log_path, tee=tee, tee_prefix="view_select")

def _run_compose_match_multi(
    *,
    split: str,
    templates: Path,
    match_json: Path,
    views_dir: Path,
    out_tree: Path,
    max_id_expansions: int,
    memory_graph_extension: bool,
    memory_graph_path: Optional[Path],
    date_correction: bool,
    domain_correction: bool,
    log_path: Path,
    tee: bool = False,
) -> None:
    cmd = [
        "python",
        "-m",
        "planner.compose_match_multi",
        "--task",
        "work",
        "--split",
        str(split),
        "--templates",
        str(templates),
        "--match",
        str(match_json),
        "--tool-views",
        str(views_dir / "tool.json"),
        "--memory-views",
        str(views_dir / "memory.jsonl"),
        "--max-id-expansions",
        str(int(max_id_expansions)),
        "--out",
        str(out_tree),
    ]

    if memory_graph_path is not None:
        p = memory_graph_path.expanduser()
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        cmd += ["--memory-graph", str(p)]

    # BooleanOptionalAction defaults are True; only pass flags when disabling for clarity.
    if not bool(memory_graph_extension):
        cmd.append("--no-memory-graph-extension")
    if not bool(date_correction):
        cmd.append("--no-date-correction")
    if not bool(domain_correction):
        cmd.append("--no-domain-correction")

    _run_cmd(cmd, log_path=log_path, tee=tee, tee_prefix="compose_match")


def _run_twin_track_multi(
    *,
    tree_json: Path,
    out_optimized: Path,
    enriched_tree_out: Optional[Path],
    mcts_iterations: int,
    uct_c: float,
    tolerance: float,
    workers: int,
    max_rounds: int,
    log_path: Path,
    tee: bool,
    show_progress: bool,
    hook_progress: bool,
) -> None:
    cmd = [
        "python",
        "-m",
        "planner.twin_track_multi",
        "--task",
        "work",
        "--tree",
        str(tree_json),
        "--max-rounds",
        str(int(max_rounds)),
        "--workers",
        str(int(workers)),
        "--iterations",
        str(int(mcts_iterations)),
        "--uct-c",
        str(float(uct_c)),
        "--semantic-tolerance",
        str(float(tolerance)),
        "--out",
        str(out_optimized),
    ]
    if hook_progress:
        cmd.append("--hook-progress")
    if not show_progress:
        cmd.append("--no-progress")
    if enriched_tree_out is not None:
        cmd += ["--enriched-tree-out", str(enriched_tree_out)]
    _run_cmd(cmd, log_path=log_path, tee=tee, tee_prefix="twin_track")


def _extract_last_token_usage_line(log_path: Path, *, label: str) -> Optional[str]:
    """
    Extract a token usage line emitted by pipeline scripts.
    Example:
      "LLM token usage (llm_repair): calls=... prompt_cache_hit=... ..."
    """

    needle = f"LLM token usage ({label}):"
    last = None
    for raw in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if needle in raw:
            last = raw.strip()
    return last


def _run_repair_round(
    *,
    task: str,
    split: str,
    tree_json: Path,
    in_jsonl: Path,
    out_jsonl: Path,
    model: str,
    workers: int,
    semantic_threshold: float,
    max_slot_candidates: int,
    log_dir: Path,
    cost_txt: Path,
    init_usage_line: Optional[str],
    run_eval: bool,
    tee: bool,
    show_progress: bool,
) -> None:
    """
    Run one round of planner.llm_repair, then convert+eval, and log per-round metrics.
    """

    log_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    _append_line(cost_txt, f"run_start={_utc_ts()}")
    _append_line(cost_txt, f"task={task}")
    _append_line(cost_txt, f"split={split}")
    _append_line(cost_txt, f"model={model}")
    _append_line(cost_txt, f"semantic_threshold={float(semantic_threshold)}")
    _append_line(cost_txt, f"max_slot_candidates={int(max_slot_candidates)}")
    _append_line(cost_txt, f"workers={int(workers)}")
    _append_line(cost_txt, f"tree={str(tree_json.resolve())}")
    _append_line(cost_txt, f"input={str(in_jsonl.resolve())}")

    if init_usage_line:
        _append_line(cost_txt, init_usage_line)

    cmd = [
        "python",
        "-m",
        "planner.llm_repair",
        "--task",
        str(task),
        "--tree",
        str(tree_json),
        "--input",
        str(in_jsonl),
        "--model",
        str(model),
        "--semantic-threshold",
        str(float(semantic_threshold)),
        "--max-slot-candidates",
        str(int(max_slot_candidates)),
        "--workers",
        str(int(workers) if int(workers) > 0 else available_cpu_count()),
        "--out",
        str(out_jsonl),
    ]
    if not show_progress:
        cmd.append("--no-progress")
    _run_cmd(cmd, log_path=log_dir / "llm_repair.log", tee=tee, tee_prefix="llm_repair")

    usage = _extract_last_token_usage_line(log_dir / "llm_repair.log", label="llm_repair")
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

    # Convert + eval into this round folder (optional).
    _run_convert_and_eval(
        optimized_jsonl=out_jsonl,
        run_dir=cost_txt.parent,
        log_dir=log_dir,
        split=split,
        run_convert=bool(run_eval),
        run_eval=run_eval,
        tee=tee,
    )

    # Add derived per-round summary (view_metrics.json) under this round folder.
    try:
        subprocess.check_call(
            [
                "python",
                "analysis/work_view_collect.py",
                "--root",
                str(cost_txt.parent),
                "--split",
                split,
                "--write-per-run",
            ],
            cwd=str(REPO_ROOT),
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except Exception:
        pass

    # Price computation (appends llm_price_* lines when possible).
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


def _run_convert_and_eval(
    *,
    optimized_jsonl: Path,
    run_dir: Path,
    log_dir: Path,
    split: str,
    run_convert: bool,
    run_eval: bool,
    tee: bool = False,
) -> None:
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not run_convert:
        return

    _run_cmd(
        [
            "python",
            "task_helper/work/evaluation/convert_optimized_to_predictions.py",
            "--optimized",
            str(optimized_jsonl),
            "--out-dir",
            str(results_dir),
        ],
        log_path=log_dir / "convert.log",
        tee=tee,
        tee_prefix="convert",
    )

    if not run_eval:
        return

    # WorkBench metrics (prints to stdout).
    _run_cmd(
        [
            "python",
            "task_helper/work/evaluation/calculate_all_metrics.py",
            "--predictions_dir",
            str(results_dir),
        ],
        log_path=log_dir / "eval_metrics.log",
        tee=tee,
        tee_prefix="eval",
    )

    # Constraint pass rates JSON.
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
        log_path=log_dir / "pass_rates.log",
        tee=tee,
        tee_prefix="pass_rates",
    )


def _postprocess_run_dir(*, run_dir: Path, split: str) -> None:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}

    try:
        subprocess.check_call(
            [
                "python",
                "analysis/work_view_collect.py",
                "--root",
                str(run_dir),
                "--split",
                str(split),
                "--write-per-run",
            ],
            cwd=str(REPO_ROOT),
            env=env,
        )
    except Exception:
        pass

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorkBench view-count sweep runner (writes under analysis/view).")
    parser.add_argument("--model", type=str, default="gpt-5.2", help="LLM model for init_template only.")
    parser.add_argument("--split", type=str, default="validation", help="Split label (default: validation).")
    parser.add_argument("--views", type=str, default="15,30,45,60", help="Comma-separated view counts.")
    parser.add_argument("--seeds", type=str, default="0", help="Comma-separated RNG seeds for view sampling.")
    parser.add_argument("--tag", type=str, default="", help="Optional tag suffix (default: seed<seed>).")
    parser.add_argument("--templates-in", type=Path, default=None, help="Reuse an existing init_templates_<split>.jsonl (copied into run folder).")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit number of queries (applies to init_templates in run folder).")

    parser.add_argument("--workers", type=int, default=0, help="Worker processes per stage (0 = all CPUs).")
    parser.add_argument("--mcts-iterations", type=int, default=10, help="MCTS iterations per query.")
    parser.add_argument("--uct-c", type=float, default=1.4, help="UCT exploration constant.")
    parser.add_argument("--tolerance", type=float, default=0.8, help="Semantic tolerance.")
    parser.add_argument("--max-rounds", type=int, default=10, help="Max rounds per query (twin_track_multi).")
    parser.add_argument("--max-id-expansions", type=int, default=6, help="Max ID expansions per Do action (ComposeMatch).")
    parser.add_argument("--memory-graph-extension", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable memory_graph extension candidates.")
    parser.add_argument("--memory-graph", type=Path, default=None, help="Override memory_graph.jsonl path (read-only).")
    parser.add_argument("--date-correction", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable date normalization (ComposeMatch).")
    parser.add_argument("--domain-correction", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable domain correction (ComposeMatch).")

    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=("all", "final", "none"),
        default="final",
        help="Evaluation mode: all (baseline + every repair round), final (only final output; default), none (skip convert+eval).",
    )
    parser.add_argument("--skip-eval", action="store_true", help="Alias for --eval-mode none.")
    parser.add_argument(
        "--repair-rounds",
        type=int,
        default=4,
        help="Number of sequential repair rounds to run (each round takes the previous round output as input).",
    )
    parser.add_argument(
        "--repair-model",
        type=str,
        default="",
        help="LLM model for planner.llm_repair (default: same as --model).",
    )
    parser.add_argument(
        "--repair-max-slot-candidates",
        type=int,
        default=50,
        help="Max slot candidates passed to planner.llm_repair.",
    )
    parser.add_argument(
        "--tee",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream stage outputs to stdout while also saving logs under each run folder (default: enabled).",
    )
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable per-template progress output for Twin-Track and llm_repair (default: enabled).",
    )
    parser.add_argument(
        "--hook-progress",
        action="store_true",
        help="Enable hook-level MCTS progress output inside Twin-Track (can be verbose; note default progress prints every 25 iters).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume into existing run folders (skip completed stages; do not delete existing outputs).",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing run folders.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if bool(args.resume) and bool(args.force):
        raise SystemExit("Do not use --resume with --force (resume never deletes run folders).")

    split = str(args.split)
    model = str(args.model)
    eval_mode = "none" if bool(args.skip_eval) else str(args.eval_mode)

    view_counts = [int(x.strip()) for x in str(args.views).split(",") if x.strip()]
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    if not view_counts:
        raise SystemExit("No view counts provided.")
    if not seeds:
        raise SystemExit("No seeds provided.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    memory_src = INPUT_VIEWS_DIR / "memory.jsonl"
    if not memory_src.exists():
        raise SystemExit(f"Missing memory views: {memory_src}")
    tool_src = INPUT_VIEWS_DIR / "tool.json"
    if not tool_src.exists():
        raise SystemExit(f"Missing tool views: {tool_src}")

    for seed in seeds:
        samples = _sample_memory_view_lines(src=memory_src, counts=view_counts, seed=seed)
        for k in view_counts:
            tag = str(args.tag).strip() or f"seed{seed}"
            run_name = _derive_run_name(model=model, views=k, split=split, tag=tag)
            run_dir = OUTPUT_ROOT / run_name
            if run_dir.exists():
                if not bool(args.resume):
                    _ensure_empty_dir(run_dir, force=bool(args.force))
            else:
                run_dir.mkdir(parents=True, exist_ok=True)

            log_dir = run_dir / "logs"
            views_dir = run_dir / "views"
            # Only write views subset if missing (resume-safe).
            mem_subset_path = views_dir / "memory.jsonl"
            tool_subset_path = views_dir / "tool.json"
            if not mem_subset_path.exists() or not tool_subset_path.exists():
                _write_views_subset(run_dir=run_dir, memory_lines=samples[int(k)])
            else:
                # Best-effort sanity check (avoid silently mixing settings when resuming).
                try:
                    have = _count_jsonl_rows(mem_subset_path)
                    if have != int(k):
                        raise SystemExit(
                            f"[{run_name}] existing {mem_subset_path} has {have} rows but expected {int(k)} "
                            f"(delete the run dir or use the correct --views/--seeds/tag)"
                        )
                except Exception:
                    pass

            print(
                f"[{run_name}] settings: workers={int(args.workers)} mcts_iterations={int(args.mcts_iterations)} "
                f"max_rounds={int(args.max_rounds)} max_id_expansions={int(args.max_id_expansions)} "
                f"tolerance={float(args.tolerance)} uct_c={float(args.uct_c)}",
                flush=True,
            )

            init_templates = run_dir / f"init_templates_{split}.jsonl"
            templates_in = args.templates_in
            if templates_in is None:
                templates_in = _default_templates_in(model=model, split=split)
            if init_templates.exists() and _count_jsonl_rows(init_templates) > 0:
                pass
            else:
                _ensure_init_templates(
                    out_path=init_templates,
                    split=split,
                    model=model,
                    templates_in=templates_in,
                    workers=int(args.workers),
                    max_rows=args.max_rows,
                    log_path=log_dir / "init_template.log",
                    tee=bool(args.tee),
                )
                if args.max_rows is not None:
                    _trim_jsonl_in_place(init_templates, max_rows=int(args.max_rows))

            match_json = run_dir / f"match_{split}.json"
            tree_json = run_dir / f"tree_{split}.json"
            enriched_tree_json = run_dir / f"tree_enriched_{split}.json"
            optimized_jsonl = run_dir / f"optimized_{split}.jsonl"
            templates_count = _count_jsonl_rows(init_templates)

            cost_txt = run_dir / "cost.txt"
            if not cost_txt.exists():
                _append_line(cost_txt, f"run_start={_utc_ts()}")
                _append_line(cost_txt, f"task=work")
                _append_line(cost_txt, f"split={split}")
                _append_line(cost_txt, f"model={model}")
                _append_line(cost_txt, f"model_slug={_model_slug(model)}")
                _append_line(cost_txt, f"views={int(k)}")
                _append_line(cost_txt, f"seed={int(seed)}")
                _append_line(cost_txt, f"tag={tag}")
                _append_line(cost_txt, f"workers={int(args.workers)}")
                _append_line(cost_txt, f"mcts_iterations={int(args.mcts_iterations)}")
                _append_line(cost_txt, f"mcts_uct_c={float(args.uct_c)}")
                _append_line(cost_txt, f"tolerance_rate={float(args.tolerance)}")
                _append_line(cost_txt, f"max_id_expansions={int(args.max_id_expansions)}")
                _append_line(cost_txt, f"memory_graph_extension={bool(args.memory_graph_extension)}")
                _append_line(cost_txt, f"date_correction={bool(args.date_correction)}")
                _append_line(cost_txt, f"domain_correction={bool(args.domain_correction)}")

            init_tokens = _sum_init_tokens(init_templates)
            if init_tokens:
                _append_line(cost_txt, f"LLM token usage (init_template): {init_tokens}")

            # ViewSelect
            if match_json.exists():
                pass
            else:
                print(f"[{run_name}] view_select → {match_json.name}")
                t0 = time.time()
                _run_view_select(
                    split=split,
                    templates=init_templates,
                    views_dir=views_dir,
                    out_match=match_json,
                    workers=int(args.workers),
                    log_path=log_dir / "view_select.log",
                    tee=bool(args.tee),
                )
                _append_line(cost_txt, f"stage_elapsed_s_view_select={int(time.time() - t0)}")

            # ComposeMatch (multi)
            if tree_json.exists():
                pass
            else:
                print(f"[{run_name}] compose_match_multi → {tree_json.name}")
                t0 = time.time()
                _run_compose_match_multi(
                    split=split,
                    templates=init_templates,
                    match_json=match_json,
                    views_dir=views_dir,
                    out_tree=tree_json,
                    max_id_expansions=int(args.max_id_expansions),
                    memory_graph_extension=bool(args.memory_graph_extension),
                    memory_graph_path=(args.memory_graph if args.memory_graph is not None else None),
                    date_correction=bool(args.date_correction),
                    domain_correction=bool(args.domain_correction),
                    log_path=log_dir / "compose_match_multi.log",
                    tee=bool(args.tee),
                )
                _append_line(cost_txt, f"stage_elapsed_s_compose_match={int(time.time() - t0)}")

            # Twin-Track (multi)
            if _is_complete_jsonl(optimized_jsonl, expected_rows=templates_count):
                pass
            else:
                print(f"[{run_name}] twin_track_multi → {optimized_jsonl.name}")
                t0 = time.time()
                _run_twin_track_multi(
                    tree_json=tree_json,
                    out_optimized=optimized_jsonl,
                    enriched_tree_out=enriched_tree_json,
                    mcts_iterations=int(args.mcts_iterations),
                    uct_c=float(args.uct_c),
                    tolerance=float(args.tolerance),
                    workers=int(args.workers),
                    max_rounds=int(args.max_rounds),
                    log_path=log_dir / "twin_track_multi.log",
                    tee=bool(args.tee),
                    show_progress=bool(args.show_progress),
                    hook_progress=bool(args.hook_progress),
                )
                _append_line(cost_txt, f"stage_elapsed_s_twin_track={int(time.time() - t0)}")

            # Convert + eval (optionally deferred until final repair round)
            repair_rounds = int(args.repair_rounds)
            baseline_is_final = repair_rounds <= 0
            run_convert_eval = (
                eval_mode == "all" or (eval_mode == "final" and baseline_is_final)
            )
            if run_convert_eval:
                pass_rates = run_dir / "results" / "pass_rates.json"
                if pass_rates.exists():
                    pass
                else:
                    print(f"[{run_name}] convert/eval → results/")
                    t0 = time.time()
                    _run_convert_and_eval(
                        optimized_jsonl=optimized_jsonl,
                        run_dir=run_dir,
                        log_dir=log_dir,
                        split=split,
                        run_convert=True,
                        run_eval=(eval_mode != "none"),
                        tee=bool(args.tee),
                    )
                    _append_line(cost_txt, f"stage_elapsed_s_convert_eval={int(time.time() - t0)}")
            else:
                _append_line(cost_txt, "stage_skipped_convert_eval=1")

            # Extra derived metrics into run folder.
            _postprocess_run_dir(run_dir=run_dir, split=split)

            # Price computation (appends llm_price_* lines when possible).
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
            print(f"Done: {run_dir}")

            # Sequential repair rounds (each round refines the previous output).
            if repair_rounds > 0:
                repair_model = str(args.repair_model).strip() or model
                repair_root = run_dir / "repair"
                tree_for_repair = enriched_tree_json if enriched_tree_json.exists() else tree_json

                # Copy init_template token usage line if present (for consistent pricing).
                init_usage_line = None
                for raw in cost_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if raw.startswith("LLM token usage (init_template):"):
                        init_usage_line = raw.strip()
                        break

                prev = optimized_jsonl
                for r in range(1, repair_rounds + 1):
                    round_dir = repair_root / f"r{r}"
                    round_dir.mkdir(parents=True, exist_ok=True)
                    out_jsonl = round_dir / f"repaired_{split}_r{r}.jsonl"
                    is_final_round = (r == repair_rounds)
                    round_convert_eval = eval_mode == "all" or (eval_mode == "final" and is_final_round)
                    round_pass_rates = round_dir / "results" / "pass_rates.json"

                    if _is_complete_jsonl(out_jsonl, expected_rows=templates_count):
                        # If we're resuming and this is the final eval-only target, run eval if missing.
                        if bool(round_convert_eval and eval_mode != "none") and not round_pass_rates.exists():
                            print(f"[{run_name}] eval-only for existing r{r}", flush=True)
                            _run_convert_and_eval(
                                optimized_jsonl=out_jsonl,
                                run_dir=round_dir,
                                log_dir=round_dir / "logs",
                                split=split,
                                run_convert=True,
                                run_eval=True,
                                tee=bool(args.tee),
                            )
                        elif not bool(round_convert_eval):
                            _append_line(round_dir / "cost.txt", "stage_skipped_convert_eval=1")
                    else:
                        _run_repair_round(
                            task="work",
                            split=split,
                            tree_json=tree_for_repair,
                            in_jsonl=prev,
                            out_jsonl=out_jsonl,
                            model=repair_model,
                            workers=int(args.workers),
                            semantic_threshold=float(args.tolerance),
                            max_slot_candidates=int(args.repair_max_slot_candidates),
                            log_dir=round_dir / "logs",
                            cost_txt=round_dir / "cost.txt",
                            init_usage_line=init_usage_line,
                            run_eval=bool(round_convert_eval and eval_mode != "none"),
                            tee=bool(args.tee),
                            show_progress=bool(args.show_progress),
                        )

                        if not round_convert_eval:
                            _append_line(round_dir / "cost.txt", "stage_skipped_convert_eval=1")

                    prev = out_jsonl
                    print(f"Repair r{r} done: {round_dir}")


if __name__ == "__main__":
    main()
