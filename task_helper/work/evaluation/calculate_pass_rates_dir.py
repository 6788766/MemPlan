from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task_helper.work.evaluation.constraints import DEFAULT_CONSTRAINTS, compute_example_context, parse_actions


FULL_TOOLS_LIST = [
    "multi_domain",
    "email",
    "calendar",
    "analytics",
    "project_management",
    "customer_relationship_manager",
]


def _canonical_query(q: str) -> str:
    return str(q).strip()


def _load_ground_truth(*, ground_truth_dir: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    query_to_tool: Dict[str, str] = {}
    for tool in FULL_TOOLS_LIST:
        path = ground_truth_dir / f"{tool}_queries_and_answers.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str)
        df["query"] = df["query"].map(_canonical_query)
        df["ground_truth"] = df["answer"].apply(ast.literal_eval)
        df["_tool"] = tool
        rows.append(df[["query", "ground_truth", "_tool"]])
        for q in df["query"].tolist():
            query_to_tool[q] = tool

    if not rows:
        raise SystemExit(f"No ground-truth files found under {ground_truth_dir}")
    gt = pd.concat(rows, ignore_index=True)
    if gt["query"].duplicated().any():
        dups = gt[gt["query"].duplicated(keep=False)]["query"].value_counts().head(10)
        raise SystemExit(f"Duplicate queries across ground truth files under {ground_truth_dir}: {dups.to_dict()}")
    return gt


def _load_predictions_dir(*, predictions_dir: Path) -> pd.DataFrame:
    pred_files = sorted(predictions_dir.glob("predictions_*.csv"))
    if not pred_files:
        raise SystemExit(f"No predictions_*.csv found under {predictions_dir}")

    rows: List[pd.DataFrame] = []
    for path in pred_files:
        df = pd.read_csv(path, dtype=str).fillna("")
        if "query" not in df.columns:
            raise SystemExit(f"Missing required column `query` in {path}")
        if "function_calls" not in df.columns:
            raise SystemExit(f"Missing required column `function_calls` in {path}")
        if "error" not in df.columns:
            df["error"] = ""
        df["query"] = df["query"].map(_canonical_query)
        df["prediction"] = df["function_calls"].apply(parse_actions)
        rows.append(df[["query", "prediction", "error"]])

    pred = pd.concat(rows, ignore_index=True)
    if pred["query"].duplicated().any():
        dups = pred[pred["query"].duplicated(keep=False)]["query"].value_counts().head(10)
        raise SystemExit(f"Duplicate queries across prediction files under {predictions_dir}: {dups.to_dict()}")
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute WorkBench constraint local/global pass rates for a predictions dir.")
    parser.add_argument("--predictions_dir", type=str, required=True)
    parser.add_argument("--ground_truth_dir", type=str, required=True)
    parser.add_argument(
        "--no_filter_to_predictions",
        action="store_false",
        default=True,
        dest="filter_to_predictions",
        help="Disable filtering ground truth to the queries present in the predictions.",
    )
    parser.add_argument("--constraints", action="append", default=[], help="Constraint names to include (repeatable).")
    parser.add_argument("--no_semantic", action="store_true", help="Exclude semantic constraints.")
    parser.add_argument("--json_out", type=str, default="", help="Optional JSON output path.")
    args = parser.parse_args()

    pred_dir = Path(args.predictions_dir)
    gt_dir = Path(args.ground_truth_dir)
    if not pred_dir.is_absolute():
        pred_dir = Path(__file__).resolve().parents[3] / pred_dir
    if not gt_dir.is_absolute():
        gt_dir = Path(__file__).resolve().parents[3] / gt_dir

    constraints = DEFAULT_CONSTRAINTS
    if args.no_semantic:
        constraints = [c for c in constraints if c.kind != "semantic"]
    if args.constraints:
        selected = set(args.constraints)
        constraints = [c for c in constraints if c.name in selected]
        missing = sorted(selected - {c.name for c in DEFAULT_CONSTRAINTS})
        if missing:
            raise SystemExit(f"Unknown constraints: {missing}")

    gt = _load_ground_truth(ground_truth_dir=gt_dir)
    pred = _load_predictions_dir(predictions_dir=pred_dir)

    if args.filter_to_predictions:
        gt = gt[gt["query"].isin(set(pred["query"].tolist()))].reset_index(drop=True)

    df = pred.merge(gt[["query", "ground_truth"]], on="query", how="inner")
    if len(df) != len(pred):
        missing = sorted(set(pred["query"].tolist()) - set(df["query"].tolist()))[:10]
        raise SystemExit(f"Predictions contain queries not found in ground truth (sample): {missing}")
    if (not args.filter_to_predictions) and len(df) != len(gt):
        missing = sorted(set(gt["query"].tolist()) - set(df["query"].tolist()))[:10]
        raise SystemExit(f"Ground truth contains queries not found in predictions (sample): {missing}")

    per_constraint_pass = {c.name: 0 for c in constraints}
    per_constraint_total = {c.name: len(df) for c in constraints}

    hard = [c for c in constraints if c.kind == "hard"]
    semantic = [c for c in constraints if c.kind == "semantic"]

    all_pass_count = 0
    hard_all_pass_count = 0
    semantic_all_pass_count = 0

    workbench_correct = 0
    workbench_exact_match = 0
    workbench_unwanted_side_effects = 0

    for _, row in df.iterrows():
        ctx = compute_example_context(
            query=row["query"],
            prediction=row["prediction"],
            ground_truth=row["ground_truth"],
            error_value=row.get("error", ""),
        )

        workbench_correct += int(ctx.correct)
        workbench_exact_match += int(ctx.exact_match)
        workbench_unwanted_side_effects += int(ctx.unwanted_side_effects)

        results = {c.name: bool(c.check(ctx)) for c in constraints}
        for name, passed in results.items():
            per_constraint_pass[name] += int(passed)

        if all(results.values()):
            all_pass_count += 1

        if hard and all(results[c.name] for c in hard):
            hard_all_pass_count += 1
        if semantic and all(results[c.name] for c in semantic):
            semantic_all_pass_count += 1

    n = len(df)
    k = len(constraints)
    hard_k = len(hard)
    semantic_k = len(semantic)

    def _micro(pass_counts: dict[str, int], names: list[str]) -> float:
        denom = n * len(names)
        return (sum(pass_counts[name] for name in names) / denom) if denom else 0.0

    payload = {
        "n_examples": n,
        "constraints": [{"name": c.name, "kind": c.kind, "description": c.description} for c in constraints],
        "constraint_pass_counts": per_constraint_pass,
        "constraint_pass_rates": {name: (per_constraint_pass[name] / per_constraint_total[name]) for name in per_constraint_pass},
        "Hard Constraint Micro Pass Rate": _micro(per_constraint_pass, [c.name for c in hard]),
        "Hard Constraint Macro Pass Rate": (hard_all_pass_count / n) if n and hard_k else 0.0,
        "Semantic Constraint Micro Pass Rate": _micro(per_constraint_pass, [c.name for c in semantic]),
        "Semantic Constraint Macro Pass Rate": (semantic_all_pass_count / n) if n and semantic_k else 0.0,
        "Local Pass Rate": (sum(per_constraint_pass.values()) / (n * k)) if n and k else 0.0,
        "Global Pass Rate": (all_pass_count / n) if n else 0.0,
        "WorkBench Accuracy": (workbench_correct / n) if n else 0.0,
        "WorkBench Exact Match": (workbench_exact_match / n) if n else 0.0,
        "WorkBench Unwanted Side Effects": (workbench_unwanted_side_effects / n) if n else 0.0,
        "filtered_to_predictions": bool(args.filter_to_predictions),
    }

    print(json.dumps(payload, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
