from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task_helper.work.evaluation.utils import calculate_metrics, get_function_name
from task_helper.work.tools.toolkits import tools_with_side_effects


FULL_TOOLS_LIST = [
    "multi_domain",
    "email",
    "calendar",
    "analytics",
    "project_management",
    "customer_relationship_manager",
]

_ARG_RE = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')


def _canonical_query(q: str) -> str:
    return str(q).strip()


def _iter_prediction_files(predictions_dir: Path) -> Iterable[Path]:
    return sorted(predictions_dir.glob("predictions_*.csv"))


def _iter_ground_truth_files(ground_truth_dir: Path) -> Iterable[Path]:
    for tool in FULL_TOOLS_LIST:
        p = ground_truth_dir / f"{tool}_queries_and_answers.csv"
        if p.exists():
            yield p


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


def _parse_call_args(call: str) -> Dict[str, str]:
    text = str(call or "")
    out: Dict[str, str] = {}
    for key, value in _ARG_RE.findall(text):
        try:
            out[key] = bytes(value, "utf-8").decode("unicode_escape")
        except Exception:
            out[key] = value
    return out


def _load_ground_truth(*, ground_truth_dir: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for path in _iter_ground_truth_files(ground_truth_dir):
        tool = path.name.replace("_queries_and_answers.csv", "")
        df = pd.read_csv(path, dtype=str)
        df["query"] = df["query"].map(_canonical_query)
        df["ground_truth"] = df["answer"].apply(_parse_actions)
        df["_tool"] = tool
        rows.append(df[["query", "ground_truth", "_tool"]])
    if not rows:
        raise SystemExit(f"No ground truth files found under {ground_truth_dir}")
    gt = pd.concat(rows, ignore_index=True)
    if gt["query"].duplicated().any():
        dups = gt[gt["query"].duplicated(keep=False)]["query"].value_counts().head(10)
        raise SystemExit(f"Duplicate queries across ground truth files under {ground_truth_dir}: {dups.to_dict()}")
    return gt


def _load_predictions(*, predictions_dir: Path) -> pd.DataFrame:
    pred_files = list(_iter_prediction_files(predictions_dir))
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
        if "full_response" not in df.columns:
            df["full_response"] = "{}"
        df["query"] = df["query"].map(_canonical_query)
        df["prediction"] = df["function_calls"].apply(_parse_actions)
        df["_pred_file"] = path.name
        rows.append(df[["query", "prediction", "error", "full_response", "_pred_file"]])

    pred = pd.concat(rows, ignore_index=True)
    if pred["query"].duplicated().any():
        dups = pred[pred["query"].duplicated(keep=False)]["query"].value_counts().head(10)
        raise SystemExit(f"Duplicate queries across prediction files under {predictions_dir}: {dups.to_dict()}")
    return pred


_SIDE_EFFECT_FNS = {str(f.name) for f in tools_with_side_effects}


def _side_effect_calls(actions: Sequence[str]) -> List[str]:
    out: List[str] = []
    for a in actions:
        try:
            name = get_function_name(str(a))
        except Exception:
            continue
        if name in _SIDE_EFFECT_FNS:
            out.append(str(a))
    return out


def _normalize_calls(actions: Sequence[str]) -> List[str]:
    return sorted([str(a).lower() for a in actions if isinstance(a, str)])


@dataclass(frozen=True)
class Example:
    tool: str
    query: str
    category: str
    prediction_side_effects: List[str]
    ground_truth_side_effects: List[str]
    extra: Dict[str, object]


def _category_for_row(*, tool: str, pred: Sequence[str], gt: Sequence[str], correct: bool) -> Tuple[str, Dict[str, object]]:
    if correct:
        return "correct", {}

    pred_se = _side_effect_calls(pred)
    gt_se = _side_effect_calls(gt)

    if gt_se and not pred_se:
        return "missing_required_side_effect", {"gt_side_effects": gt_se}
    if pred_se and not gt_se:
        return "unwanted_side_effects_on_noop_gt", {"pred_side_effects": pred_se}

    pred_fns = sorted({get_function_name(a) for a in pred_se})
    gt_fns = sorted({get_function_name(a) for a in gt_se})
    if pred_fns != gt_fns:
        return "wrong_side_effect_tool", {"pred_tools": pred_fns, "gt_tools": gt_fns}

    # Same tool(s); identify common subtypes.
    if len(_normalize_calls(pred_se)) != len(set(_normalize_calls(pred_se))):
        return "duplicate_side_effect_call", {"pred_side_effects": pred_se}

    if tool == "calendar":
        pred_ids = [(_parse_call_args(a).get("event_id") or "").strip() for a in pred_se]
        gt_ids = [(_parse_call_args(a).get("event_id") or "").strip() for a in gt_se]
        pred_ids = [x for x in pred_ids if x]
        gt_ids = [x for x in gt_ids if x]
        if pred_ids and gt_ids and pred_ids != gt_ids:
            return "wrong_calendar_event_id", {"pred_event_ids": pred_ids, "gt_event_ids": gt_ids}
        return "wrong_side_effect_params", {}

    if tool == "analytics":
        # Mostly about create_plot args/count.
        def _plot_key(call: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
            args = _parse_call_args(call)
            return (args.get("time_min"), args.get("time_max"), args.get("value_to_plot"), args.get("plot_type"))

        pk = sorted([_plot_key(a) for a in pred_se])
        gk = sorted([_plot_key(a) for a in gt_se])
        if pk != gk:
            return "wrong_plot_params", {"pred_plot_keys": pk, "gt_plot_keys": gk}
        return "wrong_side_effect_params", {}

    if tool == "project_management":
        # Board/list_name are case sensitive in evaluation.
        fields = ("board", "list_name", "assigned_to_email", "task_name", "due_date", "status", "task_id")

        def _key(call: str) -> Tuple[Tuple[str, str], ...]:
            args = _parse_call_args(call)
            return tuple(sorted((k, str(args.get(k) or "")) for k in fields))

        pk = sorted((get_function_name(a), _key(a)) for a in pred_se)
        gk = sorted((get_function_name(a), _key(a)) for a in gt_se)
        if pk != gk:
            # Case-only mismatch for board/list_name is a frequent culprit.
            def _lowered(sig):
                fn, pairs = sig
                pairs_l = tuple(sorted((k, v.lower()) for (k, v) in pairs))
                return (fn, pairs_l)

            if sorted(map(_lowered, pk)) == sorted(map(_lowered, gk)):
                return "pm_case_sensitive_field_mismatch", {"fields": ["board", "list_name", "status"]}
            return "wrong_side_effect_params", {}
        return "wrong_side_effect_params", {}

    if tool == "customer_relationship_manager":
        pred_ids = [(_parse_call_args(a).get("customer_id") or "").strip() for a in pred_se]
        gt_ids = [(_parse_call_args(a).get("customer_id") or "").strip() for a in gt_se]
        pred_ids = [x for x in pred_ids if x]
        gt_ids = [x for x in gt_ids if x]
        if pred_ids and gt_ids and set(pred_ids) != set(gt_ids):
            return "wrong_customer_id", {"pred_customer_ids": pred_ids, "gt_customer_ids": gt_ids}
        # Status casing mismatch is common ("Lost" vs "lost").
        pred_vals = [(_parse_call_args(a).get("new_value") or "").strip() for a in pred_se]
        gt_vals = [(_parse_call_args(a).get("new_value") or "").strip() for a in gt_se]
        if pred_vals and gt_vals and {v.lower() for v in pred_vals} == {v.lower() for v in gt_vals} and pred_vals != gt_vals:
            return "crm_status_case_mismatch", {"pred_new_value": pred_vals, "gt_new_value": gt_vals}
        return "wrong_side_effect_params", {}

    if tool == "email":
        pred_recips = [(_parse_call_args(a).get("recipient") or "").strip().lower() for a in pred_se]
        gt_recips = [(_parse_call_args(a).get("recipient") or "").strip().lower() for a in gt_se]
        if pred_recips and gt_recips and pred_recips != gt_recips:
            return "wrong_email_recipient", {"pred_recipient": pred_recips, "gt_recipient": gt_recips}
        return "wrong_side_effect_params", {}

    # multi_domain or unknown
    return "wrong_side_effect_params", {}


def _calendar_id_diagnostics(*, query: str, prediction: Sequence[str], ground_truth: Sequence[str]) -> Dict[str, object]:
    """
    Best-effort: check whether the ground-truth event_id appears in any predicted search_events results.

    This helps separate:
      - retrieval failure (GT id never retrieved by any check call)
      - selection failure (GT id is retrieved, but the chosen Do call uses a different id)
    """
    from task_helper.work.tools import calendar as cal

    gt_se = _side_effect_calls(ground_truth)
    gt_ids = []
    for a in gt_se:
        args = _parse_call_args(a)
        if "event_id" in args and args["event_id"].strip():
            gt_ids.append(args["event_id"].strip())
    gt_ids = [x for x in gt_ids if x]
    if not gt_ids:
        return {}

    cal.reset_state()
    searched: List[Dict[str, object]] = []
    all_ids: List[str] = []

    for a in prediction:
        if get_function_name(a) in _SIDE_EFFECT_FNS:
            break
        if get_function_name(a) != "calendar.search_events":
            continue
        kwargs = _parse_call_args(a)
        try:
            result = cal.search_events.func(**kwargs)
        except Exception as exc:
            searched.append({"call": a, "error": str(exc)})
            continue
        ids: List[str] = []
        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict) and str(item.get("event_id") or "").strip():
                    ids.append(str(item["event_id"]).strip())
        searched.append({"call": a, "result_ids": ids})
        all_ids.extend(ids)

    found = {gt_id: (gt_id in all_ids) for gt_id in gt_ids}
    first_pos: Dict[str, Optional[int]] = {}
    for gt_id in gt_ids:
        try:
            first_pos[gt_id] = all_ids.index(gt_id)
        except ValueError:
            first_pos[gt_id] = None

    return {"gt_event_ids": gt_ids, "retrieved_ids_union": sorted(set(all_ids)), "gt_id_found": found, "gt_id_first_index": first_pos}


def build_report(
    *,
    predictions_dir: Path,
    ground_truth_dir: Path,
    samples_per_category: int,
    include_calendar_diagnostics: bool,
    calendar_diag_limit: int,
    filter_to_predictions: bool = True,
) -> Dict[str, object]:
    pred = _load_predictions(predictions_dir=predictions_dir)
    gt = _load_ground_truth(ground_truth_dir=ground_truth_dir)

    df = pred.merge(gt[["query", "ground_truth", "_tool"]], on="query", how="inner")
    if len(df) != len(pred):
        missing = sorted(set(pred["query"].tolist()) - set(df["query"].tolist()))[:10]
        raise SystemExit(f"Predictions contain queries not found in ground truth (sample): {missing}")

    # Compute correctness using the canonical WorkBench metric implementation per tool.
    tool_dfs: List[pd.DataFrame] = []
    for tool in FULL_TOOLS_LIST:
        tool_pred = pred[df["_tool"] == tool][["query", "prediction", "error", "full_response"]].copy()
        if len(tool_pred) == 0:
            continue
        tool_gt = gt[gt["_tool"] == tool][["query", "ground_truth"]].copy()

        if filter_to_predictions:
            tool_gt = tool_gt[tool_gt["query"].isin(set(tool_pred["query"].tolist()))].copy()
            tool_pred = tool_pred[tool_pred["query"].isin(set(tool_gt["query"].tolist()))].copy()

        pred_for_metrics = tool_pred.rename(columns={"prediction": "function_calls"})
        gt_for_metrics = tool_gt.rename(columns={"ground_truth": "answer"})
        # `calculate_metrics` prints a summary unconditionally; silence it for a clean report.
        with contextlib.redirect_stdout(io.StringIO()):
            metrics_df = calculate_metrics(gt_for_metrics, pred_for_metrics, print_errors=False)
        metrics_df["_tool"] = tool
        tool_dfs.append(metrics_df)

    metrics_all = pd.concat(tool_dfs, ignore_index=True)
    if metrics_all["query"].duplicated().any():
        raise SystemExit("Duplicate queries encountered while building per-tool metrics.")

    merged = df.merge(metrics_all[["query", "_tool", "correct", "unwanted_side_effects"]], on=["query", "_tool"], how="left")
    if merged["correct"].isna().any():
        raise SystemExit("Failed to compute `correct` for some rows; check tool splits.")

    overall = Counter()
    per_tool: Dict[str, Counter] = defaultdict(Counter)
    examples: Dict[str, List[Example]] = defaultdict(list)

    cal_diag_done = 0
    for _, row in merged.iterrows():
        tool = str(row["_tool"])
        query = str(row["query"])
        pred_actions = list(row["prediction"]) if isinstance(row["prediction"], list) else []
        gt_actions = list(row["ground_truth"]) if isinstance(row["ground_truth"], list) else []
        correct = bool(row["correct"])

        cat, extra = _category_for_row(tool=tool, pred=pred_actions, gt=gt_actions, correct=correct)
        if cat == "correct":
            continue
        overall[cat] += 1
        per_tool[tool][cat] += 1

        pred_se = _side_effect_calls(pred_actions)
        gt_se = _side_effect_calls(gt_actions)

        if include_calendar_diagnostics and tool == "calendar" and cat in {"wrong_calendar_event_id", "missing_required_side_effect"}:
            if cal_diag_done < int(calendar_diag_limit):
                try:
                    extra["calendar_id_diagnostics"] = _calendar_id_diagnostics(
                        query=query, prediction=pred_actions, ground_truth=gt_actions
                    )
                except Exception as exc:
                    extra["calendar_id_diagnostics_error"] = str(exc)
                cal_diag_done += 1

        if len(examples[cat]) < int(samples_per_category):
            examples[cat].append(
                Example(
                    tool=tool,
                    query=query,
                    category=cat,
                    prediction_side_effects=pred_se,
                    ground_truth_side_effects=gt_se,
                    extra=dict(extra),
                )
            )

    payload: Dict[str, object] = {
        "predictions_dir": str(predictions_dir),
        "ground_truth_dir": str(ground_truth_dir),
        "n_examples": int(len(merged)),
        "n_correct": int(merged["correct"].sum()),
        "n_incorrect": int(len(merged) - int(merged["correct"].sum())),
        "overall_accuracy": float(merged["correct"].mean()),
        "overall_unwanted_side_effects_rate": float(merged["unwanted_side_effects"].mean()),
        "category_counts": dict(overall),
        "category_counts_by_tool": {tool: dict(cnt) for tool, cnt in per_tool.items()},
        "examples": {
            cat: [
                {
                    "tool": ex.tool,
                    "query": ex.query,
                    "prediction_side_effects": ex.prediction_side_effects,
                    "ground_truth_side_effects": ex.ground_truth_side_effects,
                    "extra": ex.extra,
                }
                for ex in xs
            ]
            for cat, xs in examples.items()
        },
    }
    return payload


def _format_text(report: Mapping[str, object], *, top_k: int) -> str:
    lines: List[str] = []
    lines.append(f"n_examples={report.get('n_examples')} accuracy={report.get('overall_accuracy'):.4f} unwanted_side_effects={report.get('overall_unwanted_side_effects_rate'):.4f}")  # type: ignore[arg-type]
    lines.append("")
    lines.append("Top categories:")
    cat_counts = report.get("category_counts") if isinstance(report.get("category_counts"), dict) else {}
    for cat, count in sorted(cat_counts.items(), key=lambda kv: int(kv[1]), reverse=True)[: int(top_k)]:
        lines.append(f"  {int(count):4d}  {cat}")
    lines.append("")
    lines.append("Top categories by tool:")
    by_tool = report.get("category_counts_by_tool") if isinstance(report.get("category_counts_by_tool"), dict) else {}
    for tool, counts in sorted(by_tool.items(), key=lambda kv: kv[0]):
        if not isinstance(counts, dict) or not counts:
            continue
        lines.append(f"  {tool}:")
        items = sorted(counts.items(), key=lambda kv: int(kv[1]), reverse=True)[: min(5, len(counts))]
        for cat, count in items:
            lines.append(f"    {int(count):4d}  {cat}")
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeatable WorkBench error pattern report for MemPlan predictions.")
    parser.add_argument("--predictions_dir", type=Path, required=True, help="Directory containing predictions_*.csv files.")
    parser.add_argument(
        "--ground_truth_dir",
        type=Path,
        default=(PROJECT_ROOT / "artifacts" / "input" / "work" / "dataset" / "queries_and_answers"),
        help="Directory containing *_queries_and_answers.csv files.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional output path (json).")
    parser.add_argument("--format", type=str, default="text", choices=("text", "json"), help="Output format.")
    parser.add_argument("--top_k", type=int, default=15, help="Top-K categories to print in text mode.")
    parser.add_argument("--samples_per_category", type=int, default=3, help="Example count per category.")
    parser.add_argument("--include_calendar_diagnostics", action="store_true", help="Try to diagnose calendar ID retrieval vs selection.")
    parser.add_argument("--calendar_diag_limit", type=int, default=25, help="Max calendar examples to run diagnostics for.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    pred_dir = args.predictions_dir
    gt_dir = args.ground_truth_dir
    if not pred_dir.is_absolute():
        pred_dir = (PROJECT_ROOT / pred_dir).resolve()
    if not gt_dir.is_absolute():
        gt_dir = (PROJECT_ROOT / gt_dir).resolve()

    report = build_report(
        predictions_dir=pred_dir,
        ground_truth_dir=gt_dir,
        samples_per_category=int(args.samples_per_category),
        include_calendar_diagnostics=bool(args.include_calendar_diagnostics),
        calendar_diag_limit=int(args.calendar_diag_limit),
    )

    if args.format == "json":
        text = json.dumps(report, indent=2)
    else:
        text = _format_text(report, top_k=int(args.top_k))

    if args.out is not None:
        out = args.out
        if not out.is_absolute():
            out = (PROJECT_ROOT / out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
        print(f"Wrote report to {out}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
