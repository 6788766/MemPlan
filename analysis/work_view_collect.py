#!/usr/bin/env python3
"""
Collect per-run metrics for Work view-count experiments.

Expected layout:
  artifacts/output/work/analysis/view/<run_name>/
    - match_<split>.json
    - optimized_<split>.jsonl
    - cost.txt                         (optional, but recommended)
    - results/pass_rates.json          (optional; from calculate_pass_rates_dir.py)

This script adds two derived metrics:
  1) ViewSelect tool-only edge rate (from match_<split>.json)
  2) Real tool-call counts (from optimized_<split>.jsonl)

Outputs:
  - JSONL (default to stdout) with one object per run
  - Optional CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

# Allow `python analysis/work_view_collect.py ...` while importing sibling analysis modules.
for parent in Path(__file__).resolve().parents:
    if (parent / "artifacts").is_dir():
        sys.path.insert(0, str(parent))
        break

from analysis.work_tool_edge_rate import compute_stats  # type: ignore  # noqa: E402
from analysis.work_tree_candidate_stats import compute_stats as compute_tree_stats  # type: ignore  # noqa: E402


RUN_DIR_NAME_RE = re.compile(
    r"^(?P<model_slug>[a-z0-9]+)_(?P<num_views>\d+)_(?P<split>[a-z]+)(?:_(?P<tag>.*))?$",
    re.IGNORECASE,
)


def _iter_run_dirs(root: Path) -> Iterator[Path]:
    for p in sorted(root.iterdir()):
        if p.is_dir() and not p.name.startswith("."):
            yield p


def _read_json(path: Path) -> Dict[str, object]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return obj


def _count_jsonl_lines(path: Path) -> int:
    n = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip():
            n += 1
    return n


def _pick_by_split(run_dir: Path, *, split: Optional[str]) -> Tuple[Optional[Path], Optional[Path], Optional[str]]:
    if split:
        match_path = run_dir / f"match_{split}.json"
        opt_path = run_dir / f"optimized_{split}.jsonl"
        if match_path.exists() and opt_path.exists():
            return match_path, opt_path, split
        return (match_path if match_path.exists() else None), (opt_path if opt_path.exists() else None), split

    # Prefer validation, then test, then train, then any.
    for s in ("validation", "test", "train"):
        mp = run_dir / f"match_{s}.json"
        op = run_dir / f"optimized_{s}.jsonl"
        if mp.exists() and op.exists():
            return mp, op, s
    matches = sorted(run_dir.glob("match_*.json"))
    opts = sorted(run_dir.glob("optimized_*.jsonl"))
    if matches and opts:
        # Best-effort: align by split token.
        match_by_split = {m.stem.split("match_", 1)[1]: m for m in matches if "match_" in m.stem}
        for o in opts:
            sp = o.stem.split("optimized_", 1)[1] if "optimized_" in o.stem else ""
            if sp in match_by_split:
                return match_by_split[sp], o, sp
        return matches[0], opts[0], None
    return None, None, None


def _parse_cost_txt(path: Path) -> Dict[str, object]:
    """
    Parse simple key=value lines from cost.txt.

    We intentionally keep this tolerant: unknown lines are ignored.
    """

    out: Dict[str, object] = {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("["):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        # Keep numbers typed when easy.
        if value.isdigit():
            out[key] = int(value)
            continue
        try:
            out[key] = float(value)
            continue
        except Exception:
            out[key] = value
    return out


def _tool_calls_from_optimized(path: Path) -> Dict[str, object]:
    total = 0
    check_calls = 0
    do_calls = 0
    by_domain: Dict[str, int] = {}

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        actions = obj.get("actions")
        if not isinstance(actions, list):
            continue
        for action in actions:
            if not isinstance(action, dict):
                continue
            phase_key = str(action.get("phase_key") or "")
            phase_norm = phase_key.strip().lower()
            kind = "do" if phase_norm.startswith("do") else ("check" if phase_norm.startswith("check") else "")

            attrs = action.get("attrs") if isinstance(action.get("attrs"), dict) else {}
            qualified = attrs.get("qualified_tool_name")
            if isinstance(qualified, str) and qualified:
                domain = qualified.split(".", 1)[0] if "." in qualified else qualified
            else:
                domain = str(attrs.get("tool_name") or "")

            total += 1
            if domain:
                by_domain[domain] = by_domain.get(domain, 0) + 1
            if kind == "do":
                do_calls += 1
            elif kind == "check":
                check_calls += 1

    return {
        "calls_total": int(total),
        "calls_check": int(check_calls),
        "calls_do": int(do_calls),
        "calls_by_domain": dict(sorted(by_domain.items())),
    }


def _tool_cost_total_from_optimized(path: Path) -> float:
    total = 0.0
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        notes = obj.get("notes") if isinstance(obj.get("notes"), dict) else {}
        if not isinstance(notes, dict):
            continue
        blob = None
        for key in ("multi_round_mcts", "multi_round", "mcts"):
            cand = notes.get(key)
            if isinstance(cand, dict):
                blob = cand
                break
        if not isinstance(blob, dict):
            continue
        cost = blob.get("tool_cost", blob.get("total_cost"))
        if isinstance(cost, (int, float)):
            total += float(cost)
    return float(total)


def _infer_num_views(run_dir: Path) -> Optional[int]:
    # Preferred convention for view-sweep runs.
    preferred = run_dir / "views" / "memory.jsonl"
    if preferred.exists() and preferred.suffix == ".jsonl":
        return _count_jsonl_lines(preferred)

    for candidate in (
        run_dir / "memory.jsonl",
        run_dir / "views" / "memory.jsonl.gz",  # placeholder; not supported here
    ):
        if candidate.exists() and candidate.suffix == ".jsonl":
            return _count_jsonl_lines(candidate)
    return None


def _pick_tree_json(run_dir: Path, *, split: Optional[str]) -> Optional[Path]:
    if split:
        p = run_dir / f"tree_{split}.json"
        return p if p.exists() else None

    for s in ("validation", "test", "train"):
        p = run_dir / f"tree_{s}.json"
        if p.exists():
            return p

    candidates = sorted(run_dir.glob("tree_*.json"))
    return candidates[0] if candidates else None


def _flatten_for_csv(row: Mapping[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}

    def _copy(key: str) -> None:
        if key in row:
            out[key] = row[key]

    for key in (
        "run_name",
        "run_dir",
        "split",
        "num_memory_views",
        "templates",
        "tree_slots_total",
        "tree_slots_with_any_tool",
        "tree_slots_with_effective_tool",
        "tree_candidates_total",
        "tree_candidates_tool_total",
        "tree_candidates_memory_total",
        "tree_tool_shadowed_by_memory",
        "tree_tool_effective_total",
        "total_edges",
        "tool_only_edges",
        "tool_only_rate",
        "uncovered_edges",
        "calls_total",
        "calls_check",
        "calls_do",
        "tool_cost_total",
        "WorkBench Accuracy",
        "WorkBench Unwanted Side Effects",
        "Global Pass Rate",
        "Local Pass Rate",
        "pipeline_elapsed_s_total",
        "pipeline_elapsed_s_no_eval",
        "llm_price_usd_total",
        "total_tokens",
        "model",
        "model_slug",
        "mode",
        "mcts_iterations",
        "tolerance_rate",
    ):
        _copy(key)

    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect metrics from Work view analysis run folders.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/output/work/analysis/view"),
        help="Root directory containing one subfolder per run.",
    )
    parser.add_argument("--split", type=str, default=None, help="Optional split to select (validation/test/train).")
    parser.add_argument(
        "--model-slug",
        type=str,
        default=None,
        help="Optional filter: only include run dirs whose name parses and matches this model slug (e.g., gpt52).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional filter: only include run dirs whose name parses and matches this tag (e.g., seed0).",
    )
    parser.add_argument(
        "--views",
        type=str,
        default=None,
        help="Optional filter: comma-separated memory view counts to include (e.g., 15,30,45,60).",
    )
    parser.add_argument("--out-jsonl", type=Path, default=None, help="Write JSONL to this path (default: stdout).")
    parser.add_argument("--out-csv", type=Path, default=None, help="Write a flattened CSV summary to this path.")
    parser.add_argument("--strict", action="store_true", help="Fail if any run is missing required files.")
    parser.add_argument(
        "--include-cost-kv",
        action="store_true",
        help="Include parsed key/value lines from cost.txt under `cost_kv` (can be verbose).",
    )
    parser.add_argument(
        "--include-pass-rates",
        action="store_true",
        help="Include full pass_rates.json payload under `pass_rates` (constraints list is removed).",
    )
    parser.add_argument(
        "--write-per-run",
        action="store_true",
        help="Write a per-run JSON summary file into each run folder.",
    )
    parser.add_argument(
        "--per-run-filename",
        type=str,
        default="view_metrics.json",
        help="Filename for --write-per-run output (written under each run directory).",
    )
    return parser.parse_args(argv)


def _parse_views_filter(raw: Optional[str]) -> Optional[set[int]]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    out: set[int] = set()
    for part in text.split(","):
        s = part.strip()
        if not s:
            continue
        out.add(int(s))
    return out or None


def _filter_run_dirs(
    run_dirs: List[Path],
    *,
    model_slug: Optional[str],
    tag: Optional[str],
    views: Optional[set[int]],
) -> List[Path]:
    if model_slug is None and tag is None and views is None:
        return run_dirs

    want_model = str(model_slug).lower().strip() if model_slug is not None else None
    want_tag = str(tag).strip() if tag is not None else None
    out: List[Path] = []

    for d in run_dirs:
        m = RUN_DIR_NAME_RE.match(d.name)
        if not m:
            continue
        if want_model is not None:
            have = str(m.group("model_slug") or "").lower()
            if have != want_model:
                continue
        if want_tag is not None:
            have_tag = str(m.group("tag") or "")
            if have_tag != want_tag:
                continue
        if views is not None:
            try:
                have_views = int(m.group("num_views"))
            except Exception:
                continue
            if have_views not in views:
                continue
        out.append(d)
    return out


def _collect_one(
    run_dir: Path,
    *,
    split: Optional[str],
    strict: bool,
    include_cost_kv: bool,
    include_pass_rates: bool,
) -> Dict[str, object]:
    name_meta: Dict[str, object] = {}
    m = RUN_DIR_NAME_RE.match(run_dir.name)
    if m:
        name_meta = {
            "name_model_slug": str(m.group("model_slug") or "").lower(),
            "name_num_views": int(m.group("num_views")),
            "name_split": str(m.group("split") or "").lower(),
            "name_tag": (str(m.group("tag")) if m.group("tag") else ""),
        }

    match_path, opt_path, chosen_split = _pick_by_split(run_dir, split=split)
    if strict and (match_path is None or opt_path is None):
        raise SystemExit(f"Missing match/optimized for run: {run_dir}")

    out: Dict[str, object] = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "split": chosen_split,
        "match_path": str(match_path.resolve()) if match_path else None,
        "optimized_path": str(opt_path.resolve()) if opt_path else None,
    }
    out.update(name_meta)

    num_views = _infer_num_views(run_dir)
    out["num_memory_views"] = num_views
    if "name_num_views" in out and isinstance(out.get("name_num_views"), int) and isinstance(num_views, int):
        out["num_views_matches_name"] = bool(int(out["name_num_views"]) == int(num_views))

    cost_path = run_dir / "cost.txt"
    if cost_path.exists():
        cost_kv = _parse_cost_txt(cost_path)
        out["cost_path"] = str(cost_path.resolve())
        for key in (
            "task",
            "split",
            "model",
            "model_slug",
            "mode",
            "tolerance_rate",
            "mcts_iterations",
            "mcts_uct_c",
            "pipeline_elapsed_s_total",
            "pipeline_elapsed_s_no_eval",
            "llm_price_usd_total",
            "total_tokens",
            "total_time_s",
        ):
            if key in cost_kv:
                out[key] = cost_kv[key]
        if include_cost_kv:
            out["cost_kv"] = cost_kv

    pass_rates_path = run_dir / "results" / "pass_rates.json"
    if pass_rates_path.exists():
        try:
            payload = _read_json(pass_rates_path)
            payload.pop("constraints", None)  # big; the file path is enough for drill-down
            out["pass_rates_path"] = str(pass_rates_path.resolve())
            for key in (
                "WorkBench Accuracy",
                "WorkBench Exact Match",
                "WorkBench Unwanted Side Effects",
                "Global Pass Rate",
                "Local Pass Rate",
                "Hard Constraint Micro Pass Rate",
                "Semantic Constraint Micro Pass Rate",
            ):
                if key in payload:
                    out[key] = payload[key]
            if include_pass_rates:
                out["pass_rates"] = payload
        except Exception:
            out["pass_rates_parse_error"] = True

    if match_path and match_path.exists():
        payload = _read_json(match_path)
        stats = compute_stats(payload)
        out.update(
            {
                "templates": int(stats.templates),
                "total_edges": int(stats.total_edges),
                "tool_only_edges": int(stats.tool_only_edges),
                "tool_only_rate": float(stats.tool_only_rate),
                "edges_with_any_tool": int(stats.edges_with_any_tool),
                "any_tool_rate": float(stats.any_tool_rate),
                "uncovered_edges": int(stats.uncovered_edges),
                "uncovered_rate": float(stats.uncovered_rate),
            }
        )

    tree_path = _pick_tree_json(run_dir, split=chosen_split or split)
    if tree_path and tree_path.exists():
        try:
            payload = _read_json(tree_path)
            ts = compute_tree_stats(payload)
            out.update(
                {
                    "tree_path": str(tree_path.resolve()),
                    "tree_slots_total": int(ts.slots_total),
                    "tree_slots_with_any_tool": int(ts.slots_with_any_tool),
                    "tree_slots_with_effective_tool": int(ts.slots_with_effective_tool),
                    "tree_candidates_total": int(ts.candidates_total),
                    "tree_candidates_tool_total": int(ts.candidates_tool_total),
                    "tree_candidates_memory_total": int(ts.candidates_memory_total),
                    "tree_tool_shadowed_by_memory": int(ts.tool_shadowed_by_memory),
                    "tree_tool_effective_total": int(ts.tool_effective_total),
                }
            )
        except Exception:
            out["tree_parse_error"] = True

    if opt_path and opt_path.exists():
        out.update(_tool_calls_from_optimized(opt_path))
        out["tool_cost_total"] = float(_tool_cost_total_from_optimized(opt_path))

    return out


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    root = args.root.expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    views_filter = _parse_views_filter(args.views)

    # Auto-detect: if `root` itself looks like a run dir, summarize it; otherwise summarize subdirs.
    root_match, root_opt, _ = _pick_by_split(root, split=args.split)
    if root_match is not None and root_opt is not None:
        run_dirs = [root]
    else:
        run_dirs = _filter_run_dirs(
            list(_iter_run_dirs(root)),
            model_slug=(str(args.model_slug) if args.model_slug is not None else None),
            tag=(str(args.tag) if args.tag is not None else None),
            views=views_filter,
        )

    rows = [
        _collect_one(
            d,
            split=args.split,
            strict=bool(args.strict),
            include_cost_kv=bool(args.include_cost_kv),
            include_pass_rates=bool(args.include_pass_rates),
        )
        for d in run_dirs
    ]

    if bool(args.write_per_run):
        name = str(args.per_run_filename or "view_metrics.json").strip()
        if not name:
            raise SystemExit("--per-run-filename must be non-empty when --write-per-run is set.")
        for row in rows:
            run_dir_raw = row.get("run_dir")
            if not isinstance(run_dir_raw, str) or not run_dir_raw:
                continue
            out_path = Path(run_dir_raw) / name
            try:
                out_path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            except Exception:
                # Don't fail the whole run if one folder is not writable.
                continue

    if args.out_jsonl:
        out_path = args.out_jsonl
        if not out_path.is_absolute():
            out_path = (Path.cwd() / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    else:
        for r in rows:
            print(json.dumps(r, ensure_ascii=False))

    if args.out_csv:
        csv_path = args.out_csv
        if not csv_path.is_absolute():
            csv_path = (Path.cwd() / csv_path).resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        flat = [_flatten_for_csv(r) for r in rows]
        fieldnames: List[str] = []
        seen = set()
        for r in flat:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
        with csv_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for r in flat:
                writer.writerow(r)


if __name__ == "__main__":
    main()
