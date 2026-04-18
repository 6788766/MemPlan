import pandas as pd
import argparse
import warnings
import sys
import os
import ast

project_root = os.path.abspath(os.path.curdir)
sys.path.append(project_root)

from baseline.workbench.src.evals.utils import AVAILABLE_LLMS, generate_results, calculate_metrics

warnings.filterwarnings("ignore")  # suppress langchain deprecation warnings

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    help="model name. Must be one of " + ", ".join(AVAILABLE_LLMS),
    required=True,
)
parser.add_argument(
    "--queries_path",
    type=str,
    help="path to queries and answers csv. By default these are stored in data/processed/queries_and_answers/",
    default="artifacts/input/work/dataset/queries_and_answers/email_queries_and_answers.csv",
)

parser.add_argument(
    "--toolkits",
    action="append",
    nargs="*",
    help="toolkits to be used for generating answers. By default all toolkits are used: 'email', 'calendar', 'analytics', 'project_management', 'customer_relationship_manager'",
    default=[],
)

parser.add_argument(
    "--tool_selection", type=str, help="tool selection method. Must be one of 'all', 'domains'", default="all"
)

parser.add_argument(
    "--workers",
    type=int,
    default=1,
    help="Parallel workers (processes); one worker handles one query at a time. Use 1 for sequential.",
)

args = parser.parse_args()

if __name__ == "__main__":
    ground_truth = pd.read_csv(args.queries_path)
    ground_truth["answer"] = ground_truth["answer"].apply(ast.literal_eval)
    results = generate_results(args.queries_path, args.model_name, args.tool_selection, workers=args.workers)
    calculate_metrics(ground_truth, results)
