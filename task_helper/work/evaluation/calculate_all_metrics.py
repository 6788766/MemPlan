import sys
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

import ast
import pandas as pd

from task_helper.work.evaluation.utils import DEFAULT_GROUND_TRUTH_DIR, DEFAULT_RESULTS_ROOT_DIR, calculate_metrics, get_latest_results_from_dir

# ignore pandas warning
import warnings

warnings.filterwarnings("ignore")

results_root_dir = str(DEFAULT_RESULTS_ROOT_DIR)
full_tools_list = [
    "multi_domain",
    "email",
    "calendar",
    "analytics",
    "project_management",
    "customer_relationship_manager",
]


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--predictions_dir",
    type=str,
    default="",
    help=(
        "Directory containing MemPlan-style WorkBench prediction CSVs named like "
        "`predictions_<tool>.csv`. If provided, this script evaluates those files (optionally on a split). "
        "Example: artifacts/output/work/gpt52_test/results"
    ),
)
arg_parser.add_argument(
    "--ground_truth_dir",
    type=str,
    default=str(DEFAULT_GROUND_TRUTH_DIR),
    help="Directory containing WorkBench ground truth CSVs like `<tool>_queries_and_answers.csv`.",
)
arg_parser.add_argument(
    "--no_filter_to_predictions",
    action="store_false",
    default=True,
    dest="filter_to_predictions",
    help="Disable filtering ground truth to the queries present in the predictions.",
)
arg_parser.add_argument(
    "--tools",
    action="append",
    default=[],
    help=f"Call with --tools <tool 1> --tools <tool 2> etc. Defaults to {full_tools_list}.",
)
arg_parser.add_argument(
    "--models",
    action="append",
    default=[],
    help="(Legacy only) WorkBench model name(s) to evaluate from the original results layout.",
)
arg_parser.add_argument(
    "--print_errors",
    action="store_true",
    help="Print errors when calculating metrics.",
    default=False,
)
arg_parser.add_argument(
    "--all_tools",
    action="store_true",
    help="Only consider domain specific tools.",
    default=False,
)

args = arg_parser.parse_args()
all_tools_in_prompt = args.all_tools

if __name__ == "__main__":
    tools = args.tools if len(args.tools) else full_tools_list

    def _canonical_query(q: str) -> str:
        # Some WorkBench CSVs contain leading/trailing whitespace in the query string.
        # Canonicalize to ensure split-extracted predictions still match ground truth.
        return str(q).strip()

    def _normalize_tool_name(name: str) -> str:
        if name == "crm":
            return "customer_relationship_manager"
        return name

    def _eval_predictions_dir(predictions_dir: Path) -> None:
        gt_root = Path(args.ground_truth_dir)
        if not gt_root.is_absolute():
            gt_root = project_root / gt_root

        pred_files = sorted(predictions_dir.glob("predictions_*.csv"))
        if not pred_files:
            raise SystemExit(f"No predictions_*.csv found under {predictions_dir}")

        preds = []
        for p in pred_files:
            df = pd.read_csv(p, dtype=str).fillna("")
            if "query" not in df.columns:
                raise SystemExit(f"Missing required column `query` in {p}")
            df["query"] = df["query"].map(_canonical_query)
            if "function_calls" not in df.columns:
                raise SystemExit(f"Missing required column `function_calls` in {p}")
            if "error" not in df.columns:
                df["error"] = ""
            if "full_response" not in df.columns:
                df["full_response"] = "{}"
            df["function_calls"] = df["function_calls"].apply(ast.literal_eval)
            df["_pred_file"] = p.name
            preds.append(df[["query", "function_calls", "error", "full_response", "_pred_file"]])

        pred_all = pd.concat(preds, ignore_index=True)
        if pred_all["query"].duplicated().any():
            dups = pred_all[pred_all["query"].duplicated(keep=False)]["query"].value_counts().head(10)
            raise SystemExit(f"Duplicate queries across prediction files under {predictions_dir}: {dups.to_dict()}")

        # Map each ground-truth query to exactly one tool/domain.
        query_to_tool: dict[str, str] = {}
        gt_by_tool: dict[str, pd.DataFrame] = {}
        for tool in full_tools_list:
            gt_path = gt_root / f"{tool}_queries_and_answers.csv"
            if not gt_path.exists():
                continue
            gt = pd.read_csv(gt_path, dtype=str)
            gt["query"] = gt["query"].map(_canonical_query)
            gt["answer"] = gt["answer"].apply(ast.literal_eval)
            gt_by_tool[tool] = gt
            for q in gt["query"].tolist():
                query_to_tool[q] = tool

        pred_all["_tool"] = pred_all["query"].map(query_to_tool).fillna("")
        unmatched = pred_all[pred_all["_tool"] == ""]
        if len(unmatched):
            print(f"\nWarning: {len(unmatched)} prediction row(s) not found in any ground-truth file; ignoring them.")
            sample = unmatched[["query", "_pred_file"]].head(10).to_dict(orient="records")
            print(f"Sample: {sample}")
        pred_all = pred_all[pred_all["_tool"] != ""].copy()

        total_correct = 0
        total_incorrect = 0
        total_side_effects = 0
        total_correct_no_actions = 0
        total_incorrect_no_actions = 0
        total_correct_non_zero_actions = 0
        total_incorrect_non_zero_actions = 0
        total_correct_two_or_more_actions = 0
        total_incorrect_two_or_more_actions = 0
        total_context_window_errors = 0

        for tool in tools:
            tool = _normalize_tool_name(tool)
            if tool not in gt_by_tool:
                gt_path = gt_root / f"{tool}_queries_and_answers.csv"
                print(f"\nNo ground truth found for {tool} at {gt_path}")
                continue

            ground_truth = gt_by_tool[tool]
            predictions = pred_all[pred_all["_tool"] == tool][["query", "function_calls", "error", "full_response"]].copy()

            if args.filter_to_predictions:
                ground_truth = ground_truth[ground_truth["query"].isin(set(predictions["query"].tolist()))].copy()
                predictions = predictions[predictions["query"].isin(set(ground_truth["query"].tolist()))].copy()

            if len(predictions) == 0:
                print(f"\nNo predictions for {tool} under {predictions_dir}")
                continue

            print(f"\nCalculating metrics for {tool} (n={len(predictions)}) from {predictions_dir}")
            df = calculate_metrics(ground_truth, predictions, print_errors=args.print_errors)

            num_correct = int(df["correct"].sum())
            num_incorrect = int(len(df) - num_correct)
            num_side_effects = int(df["unwanted_side_effects"].sum())
            num_correct_no_actions = int(df[df["ground_truth"].apply(len) == 0]["correct"].sum())
            num_incorrect_no_actions = int(len(df[df["ground_truth"].apply(len) == 0]) - num_correct_no_actions)
            num_correct_non_zero_actions = int(df[df["ground_truth"].apply(len) > 0]["correct"].sum())
            num_incorrect_non_zero_actions = int(len(df[df["ground_truth"].apply(len) > 0]) - num_correct_non_zero_actions)
            num_correct_two_or_more_actions = int(df[df["ground_truth"].apply(len) > 1]["correct"].sum())
            num_incorrect_two_or_more_actions = int(
                len(df[df["ground_truth"].apply(len) > 1]) - num_correct_two_or_more_actions
            )
            num_context_window_errors = int(len(df[df["error"] == "Context window exceeded"]))

            total_correct += num_correct
            total_incorrect += num_incorrect
            total_side_effects += num_side_effects
            total_correct_no_actions += num_correct_no_actions
            total_incorrect_no_actions += num_incorrect_no_actions
            total_correct_non_zero_actions += num_correct_non_zero_actions
            total_incorrect_non_zero_actions += num_incorrect_non_zero_actions
            total_correct_two_or_more_actions += num_correct_two_or_more_actions
            total_incorrect_two_or_more_actions += num_incorrect_two_or_more_actions
            total_context_window_errors += num_context_window_errors

        if total_correct + total_incorrect == 0:
            print("No results found.")
            return

        print()
        print(f"Calculating overall metrics for predictions in: {predictions_dir}")
        print(
            f"Accuracy (%): {total_correct / (total_correct + total_incorrect) * 100} "
            f"({total_correct} / {total_correct + total_incorrect})"
        )
        print(
            f"Side effects (%): {total_side_effects / (total_correct + total_incorrect) * 100} "
            f"({total_side_effects} / {total_correct + total_incorrect})"
        )
        print(f"Evaluated queries: {total_correct + total_incorrect} (filtered_to_predictions={args.filter_to_predictions})")
        if args.print_errors:
            if total_correct_no_actions + total_incorrect_no_actions:
                print(
                    f"Accuracy without actions (%): "
                    f"{total_correct_no_actions / (total_correct_no_actions + total_incorrect_no_actions) * 100} "
                    f"({total_correct_no_actions} / {total_correct_no_actions + total_incorrect_no_actions})"
                )
            if total_correct_non_zero_actions + total_incorrect_non_zero_actions:
                print(
                    f"Accuracy with non-zero actions (%): "
                    f"{total_correct_non_zero_actions / (total_correct_non_zero_actions + total_incorrect_non_zero_actions) * 100} "
                    f"({total_correct_non_zero_actions} / {total_correct_non_zero_actions + total_incorrect_non_zero_actions})"
                )
            if total_correct_two_or_more_actions + total_incorrect_two_or_more_actions:
                print(
                    f"Accuracy with two or more actions (%): "
                    f"{total_correct_two_or_more_actions / (total_correct_two_or_more_actions + total_incorrect_two_or_more_actions) * 100} "
                    f"({total_correct_two_or_more_actions} / {total_correct_two_or_more_actions + total_incorrect_two_or_more_actions})"
                )
            if total_correct + total_incorrect:
                print(
                    f"Context window errors (%): "
                    f"{total_context_window_errors / (total_correct + total_incorrect) * 100} "
                    f"({total_context_window_errors} / {total_correct + total_incorrect})"
                )
        print("==============================")

    if args.predictions_dir:
        pred_dir = Path(args.predictions_dir)
        if not pred_dir.is_absolute():
            pred_dir = project_root / pred_dir
        _eval_predictions_dir(pred_dir)
    else:
        # Backward compatible: evaluate original WorkBench-style results dir layout.
        # This expects `results_root_dir/<tool>/<model>_*.csv` and uses utils.get_latest_results_from_dir.
        from task_helper.work.evaluation.utils import AVAILABLE_LLMS  # keep import local for old behavior

        models = args.models if hasattr(args, "models") and len(args.models) else AVAILABLE_LLMS
        for model in models:
            total_correct = 0
            total_incorrect = 0
            total_side_effects = 0
            total_correct_no_actions = 0
            total_incorrect_no_actions = 0
            total_correct_non_zero_actions = 0
            total_incorrect_non_zero_actions = 0
            total_correct_two_or_more_actions = 0
            total_incorrect_two_or_more_actions = 0
            total_context_window_errors = 0
            for tool in tools:
                results = get_latest_results_from_dir(
                    results_root_dir, model, tool, args.print_errors, all_tools_in_prompt
                )
                if results is None:
                    continue
                (
                    correct,
                    incorrect,
                    side_effects,
                    correct_no_actions,
                    incorrect_no_actions,
                    correct_non_zero_actions,
                    incorrect_non_zero_actions,
                    correct_two_or_more_actions,
                    incorrect_two_or_more_actions,
                    num_context_window_errors,
                ) = results
                total_correct += correct
                total_incorrect += incorrect
                total_side_effects += side_effects
                total_correct_no_actions += correct_no_actions
                total_incorrect_no_actions += incorrect_no_actions
                total_correct_non_zero_actions += correct_non_zero_actions
                total_incorrect_non_zero_actions += incorrect_non_zero_actions
                total_correct_two_or_more_actions += correct_two_or_more_actions
                total_incorrect_two_or_more_actions += incorrect_two_or_more_actions
                total_context_window_errors += num_context_window_errors
            if total_correct + total_incorrect == 0:
                print(f"No results found for {model}.")
                continue
            print()
            print(f"Calculating overall metrics for {model}")
            print(f"Overall metrics for {model}:")
            print(
                f"Accuracy (%): {total_correct / (total_correct + total_incorrect) * 100} "
                f"({total_correct} / {total_correct + total_incorrect})"
            )
            print(
                f"Side effects (%): {total_side_effects / (total_correct + total_incorrect) * 100} "
                f"({total_side_effects} / {total_correct + total_incorrect})"
            )
            if args.print_errors:
                print(
                    f"Accuracy without actions (%): "
                    f"{total_correct_no_actions / (total_correct_no_actions + total_incorrect_no_actions) * 100} "
                    f"({total_correct_no_actions} / {total_correct_no_actions + total_incorrect_no_actions})"
                )
                print(
                    f"Accuracy with non-zero actions (%): "
                    f"{total_correct_non_zero_actions / (total_correct_non_zero_actions + total_incorrect_non_zero_actions) * 100} "
                    f"({total_correct_non_zero_actions} / {total_correct_non_zero_actions + total_incorrect_non_zero_actions})"
                )
                print(
                    f"Accuracy with two or more actions (%): "
                    f"{total_correct_two_or_more_actions / (total_correct_two_or_more_actions + total_incorrect_two_or_more_actions) * 100} "
                    f"({total_correct_two_or_more_actions} / {total_correct_two_or_more_actions + total_incorrect_two_or_more_actions})"
                )
                print(
                    f"Context window errors (%): "
                    f"{total_context_window_errors / (total_correct + total_incorrect) * 100} "
                    f"({total_context_window_errors} / {total_correct + total_incorrect})"
                )
            print("==============================")
