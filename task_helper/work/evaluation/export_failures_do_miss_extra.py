from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import io
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task_helper.work.evaluation.utils import calculate_metrics, get_function_name
from task_helper.work.tools.toolkits import tools_with_side_effects


FULL_TOOLS_LIST = (
    "multi_domain",
    "email",
    "calendar",
    "analytics",
    "project_management",
    "customer_relationship_manager",
)


def _canonical_query(q: object) -> str:
    return str(q or "").strip()


def _parse_actions(raw: object) -> List[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        return []
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    return []


def _load_ground_truth(ground_truth_dir: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    rows: List[pd.DataFrame] = []
    query_to_tool: Dict[str, str] = {}
    for tool in FULL_TOOLS_LIST:
        path = ground_truth_dir / f"{tool}_queries_and_answers.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        if "query" not in df.columns or "answer" not in df.columns:
            continue
        df["query"] = df["query"].map(_canonical_query)
        df["ground_truth"] = df["answer"].apply(_parse_actions)
        df["_tool"] = tool
        rows.append(df[["query", "ground_truth", "_tool"]])
        for q in df["query"].tolist():
            query_to_tool[q] = tool
    if not rows:
        raise SystemExit(f"No ground truth files found under {ground_truth_dir}")
    gt_all = pd.concat(rows, ignore_index=True)
    if gt_all["query"].duplicated().any():
        dups = gt_all[gt_all["query"].duplicated(keep=False)]["query"].value_counts().head(10)
        raise SystemExit(f"Duplicate queries across ground truth files under {ground_truth_dir}: {dups.to_dict()}")
    return gt_all, query_to_tool


def _iter_prediction_files(predictions_dir: Path) -> List[Path]:
    return sorted(predictions_dir.glob("predictions_*.csv"))


def _load_predictions_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    if "query" not in df.columns or "function_calls" not in df.columns:
        raise SystemExit(f"Missing required columns in {path} (need query,function_calls)")
    if "error" not in df.columns:
        df["error"] = ""
    if "full_response" not in df.columns:
        df["full_response"] = "{}"
    df["query"] = df["query"].map(_canonical_query)
    df["function_calls"] = df["function_calls"].apply(_parse_actions)
    df["_pred_file"] = path.name
    return df[["query", "function_calls", "error", "full_response", "_pred_file"]]


_SIDE_EFFECT_FNS = {str(f.name) for f in tools_with_side_effects}


def _side_effect_calls(actions: Sequence[str]) -> List[str]:
    out: List[str] = []
    for call in actions:
        try:
            name = get_function_name(str(call))
        except Exception:
            continue
        if name in _SIDE_EFFECT_FNS:
            out.append(str(call))
    return out


def _multiset_diff(
    *,
    ground_truth: Sequence[str],
    prediction: Sequence[str],
) -> Tuple[List[str], List[str]]:
    gt_norm = [str(x).lower() for x in ground_truth]
    pred_norm = [str(x).lower() for x in prediction]

    gt_counts = Counter(gt_norm)
    pred_counts = Counter(pred_norm)

    missing_norm: List[str] = []
    extra_norm: List[str] = []
    for key in sorted(set(gt_counts) | set(pred_counts)):
        diff = gt_counts.get(key, 0) - pred_counts.get(key, 0)
        if diff > 0:
            missing_norm.extend([key] * diff)
        elif diff < 0:
            extra_norm.extend([key] * (-diff))

    def _materialize(norm_list: List[str], originals: Sequence[str]) -> List[str]:
        buckets: Dict[str, List[str]] = {}
        for item in originals:
            buckets.setdefault(str(item).lower(), []).append(str(item))
        out: List[str] = []
        for key in norm_list:
            lst = buckets.get(key) or []
            if lst:
                out.append(lst.pop(0))
            else:
                out.append(key)
        return out

    missing = _materialize(missing_norm, ground_truth)
    extra = _materialize(extra_norm, prediction)
    return missing, extra


def _suppress_stdout() -> contextlib.AbstractContextManager[None]:
    return contextlib.redirect_stdout(io.StringIO())


@dataclass(frozen=True)
class FailureRow:
    tool: str
    pred_file: str
    query: str
    function_calls: List[str]
    ground_truth: List[str]
    miss: List[str]
    extra: List[str]
    error: str
    unwanted_side_effects: bool


def _write_failures_csv(path: Path, failures: Sequence[FailureRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "tool",
                "pred_file",
                "query",
                "ground_truth",
                "function_calls",
                "miss",
                "extra",
                "error",
                "unwanted_side_effects",
            ],
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for row in failures:
            writer.writerow(
                {
                    "tool": row.tool,
                    "pred_file": row.pred_file,
                    "query": row.query,
                    "ground_truth": repr(row.ground_truth),
                    "function_calls": repr(row.function_calls),
                    "miss": repr(row.miss),
                    "extra": repr(row.extra),
                    "error": row.error,
                    "unwanted_side_effects": "1" if row.unwanted_side_effects else "0",
                }
            )


def export_failures(
    *,
    predictions_dir: Path,
    ground_truth_dir: Path,
    out_dir: Path,
) -> None:
    gt_all, query_to_tool = _load_ground_truth(ground_truth_dir)

    pred_files = _iter_prediction_files(predictions_dir)
    if not pred_files:
        raise SystemExit(f"No predictions_*.csv found under {predictions_dir}")

    all_failures: List[FailureRow] = []
    for pred_path in pred_files:
        pred = _load_predictions_file(pred_path)
        pred["_tool"] = pred["query"].map(query_to_tool).fillna("")
        pred = pred[pred["_tool"] != ""].copy()
        if len(pred) == 0:
            continue

        for tool, pred_tool in pred.groupby("_tool"):
            gt_tool = gt_all[gt_all["_tool"] == tool][["query", "ground_truth"]].copy()
            pred_subset = pred_tool[["query", "function_calls", "error", "full_response", "_pred_file"]].copy()

            # Filter ground truth to exactly the predicted queries (run dir is typically a split subset).
            pred_queries = set(pred_subset["query"].tolist())
            gt_tool = gt_tool[gt_tool["query"].isin(pred_queries)].copy()
            gt_queries = set(gt_tool["query"].tolist())
            pred_subset = pred_subset[pred_subset["query"].isin(gt_queries)].copy()

            # calculate_metrics prints summary; keep the export script quiet.
            with _suppress_stdout():
                metrics = calculate_metrics(gt_tool, pred_subset, print_errors=False)

            # metrics columns: prediction, ground_truth, correct, unwanted_side_effects, error, _pred_file, ...
            for _, row in metrics[~metrics["correct"]].iterrows():
                query = str(row["query"])
                pred_actions = row["prediction"] if isinstance(row["prediction"], list) else []
                gt_actions = row["ground_truth"] if isinstance(row["ground_truth"], list) else []
                pred_do = _side_effect_calls(pred_actions)
                gt_do = _side_effect_calls(gt_actions)
                miss, extra = _multiset_diff(ground_truth=gt_do, prediction=pred_do)
                all_failures.append(
                    FailureRow(
                        tool=str(tool),
                        pred_file=str(row.get("_pred_file") or pred_path.name),
                        query=query,
                        function_calls=[str(x) for x in pred_actions],
                        ground_truth=[str(x) for x in gt_actions],
                        miss=miss,
                        extra=extra,
                        error=str(row.get("error") or ""),
                        unwanted_side_effects=bool(row.get("unwanted_side_effects")),
                    )
                )

        # Per-prediction-file output (keeps alignment with existing results layout).
        failures_for_file = [f for f in all_failures if f.pred_file == pred_path.name]
        if failures_for_file:
            out_path = out_dir / pred_path.name.replace(".csv", "_fail.csv")
            _write_failures_csv(out_path, failures_for_file)

    if all_failures:
        _write_failures_csv(out_dir / "failures_all.csv", all_failures)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate WorkBench predictions_*.csv, filter incorrect queries, and add DO-call miss/extra columns."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory like artifacts/output/work/gpt52_test (expects results/ under it).",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=Path("artifacts/input/work/dataset/queries_and_answers"),
        help="Ground truth directory containing *_queries_and_answers.csv files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <run-dir>/fail).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_dir = args.run_dir
    if not run_dir.exists():
        raise SystemExit(f"Missing run dir: {run_dir}")

    predictions_dir = run_dir / "results"
    if not predictions_dir.exists():
        raise SystemExit(f"Missing predictions dir: {predictions_dir}")

    gt_dir = args.ground_truth_dir
    if not gt_dir.is_absolute():
        gt_dir = (PROJECT_ROOT / gt_dir).resolve()

    out_dir = args.out_dir or (run_dir / "fail")
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()

    export_failures(predictions_dir=predictions_dir, ground_truth_dir=gt_dir, out_dir=out_dir)
    print(f"Wrote failure CSVs under {out_dir}")


if __name__ == "__main__":
    main()
