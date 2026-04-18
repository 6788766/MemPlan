from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd

from task_helper.work.evaluation.constraints import DEFAULT_CONSTRAINTS, compute_example_context, parse_actions


def _load_ground_truth(path: Path) -> pd.DataFrame:
    gt = pd.read_csv(path, dtype=str)
    gt["answer"] = gt["answer"].apply(ast.literal_eval)
    return gt


def _load_predictions(path: Path) -> pd.DataFrame:
    pred = pd.read_csv(path, dtype=str)
    pred["function_calls"] = pred["function_calls"].apply(parse_actions)
    pred["error"] = pred.get("error", "").fillna("")
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute WorkBench constraint local/global pass rates.")
    parser.add_argument("--predictions_path", type=str, required=True)
    parser.add_argument("--ground_truth_path", type=str, required=True)
    parser.add_argument("--constraints", action="append", default=[], help="Constraint names to include (repeatable).")
    parser.add_argument("--no_semantic", action="store_true", help="Exclude semantic constraints.")
    parser.add_argument("--json_out", type=str, default="", help="Optional JSON output path.")
    args = parser.parse_args()

    pred_path = Path(args.predictions_path)
    gt_path = Path(args.ground_truth_path)

    constraints = DEFAULT_CONSTRAINTS
    if args.no_semantic:
        constraints = [c for c in constraints if c.kind != "semantic"]
    if args.constraints:
        selected = set(args.constraints)
        constraints = [c for c in constraints if c.name in selected]
        missing = sorted(selected - {c.name for c in DEFAULT_CONSTRAINTS})
        if missing:
            raise SystemExit(f"Unknown constraints: {missing}")

    gt = _load_ground_truth(gt_path).rename(columns={"answer": "ground_truth"})
    pred = _load_predictions(pred_path).rename(columns={"function_calls": "prediction"})

    df = pred.merge(gt[["query", "ground_truth"]], on="query", how="inner")
    if len(df) != len(pred) or len(df) != len(gt):
        raise SystemExit(
            f"Query mismatch: predictions={len(pred)} ground_truth={len(gt)} merged={len(df)}; "
            "ensure both files refer to the same query set."
        )

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

        if hard:
            if all(results[c.name] for c in hard):
                hard_all_pass_count += 1
        if semantic:
            if all(results[c.name] for c in semantic):
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
    }

    print(json.dumps(payload, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

