from __future__ import annotations

import argparse
import ast
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd


SIDE_EFFECT_TOOL_NAMES: Set[str] = {
    "calendar.create_event",
    "calendar.delete_event",
    "calendar.update_event",
    "email.send_email",
    "email.delete_email",
    "email.forward_email",
    "email.reply_email",
    "analytics.create_plot",
    "project_management.create_task",
    "project_management.delete_task",
    "project_management.update_task",
    "customer_relationship_manager.update_customer",
    "customer_relationship_manager.add_customer",
    "customer_relationship_manager.delete_customer",
}


@dataclass(frozen=True)
class SplitConfig:
    seed: int
    k_per_template: int
    train_fraction: float
    train_size: Optional[int]
    qa_dir: Path
    results_root: Path
    out_dir: Path


def _parse_action_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed if item is not None]
    return []


def _function_name(action: str) -> str:
    prefix = action.split("(", 1)[0]
    parts = prefix.split(".")
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return prefix


def _normalise_action(action: str) -> str:
    return str(action).replace("\n", "\\n").lower()


def _is_exact_match(*, predicted_actions: Sequence[str], ground_truth_actions: Sequence[str]) -> bool:
    predicted_with_side_effects = [
        action for action in predicted_actions if _function_name(action) in SIDE_EFFECT_TOOL_NAMES
    ]
    pred_norm = sorted(_normalise_action(action) for action in predicted_with_side_effects)
    gt_norm = sorted(_normalise_action(action) for action in ground_truth_actions)
    return pred_norm == gt_norm


def _load_queries_and_answers(qa_dir: Path) -> pd.DataFrame:
    paths = sorted(qa_dir.glob("*_queries_and_answers.csv"))
    if not paths:
        raise SystemExit(f"No queries_and_answers CSVs found under: {qa_dir}")
    frames: List[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path, dtype=str)
        domain = path.name.replace("_queries_and_answers.csv", "")
        df["_domain"] = domain
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    expected = {"query", "answer", "base_template", "chosen_template", "domains"}
    missing = sorted(expected - set(merged.columns))
    if missing:
        raise SystemExit(f"Missing required columns in queries_and_answers: {missing}")
    if merged["query"].duplicated().any():
        dups = merged[merged["query"].duplicated(keep=False)]["query"].head(5).tolist()
        raise SystemExit(f"Duplicate queries detected (need unique join key). Examples: {dups}")
    return merged


def _latest_gpt4_domains_results(results_root: Path) -> pd.DataFrame:
    domains = [
        "analytics",
        "calendar",
        "customer_relationship_manager",
        "email",
        "multi_domain",
        "project_management",
    ]
    frames: List[pd.DataFrame] = []
    for domain in domains:
        candidates = sorted((results_root / domain).glob("gpt-4_domains_*.csv"))
        if not candidates:
            raise SystemExit(f"Missing GPT-4 domains results for domain '{domain}' under {results_root}/{domain}/")
        path = candidates[-1]
        df = pd.read_csv(path, dtype=str)
        df["_domain"] = domain
        df["_results_path"] = str(path)
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    expected = {"query", "function_calls", "error"}
    missing = sorted(expected - set(merged.columns))
    if missing:
        raise SystemExit(f"Missing required columns in GPT-4 results: {missing}")
    if merged["query"].duplicated().any():
        dups = merged[merged["query"].duplicated(keep=False)]["query"].head(5).tolist()
        raise SystemExit(f"Duplicate queries detected in GPT-4 results. Examples: {dups}")
    return merged


def _iter_results_files(results_root: Path) -> Iterable[Tuple[str, str, Path]]:
    for domain_dir in sorted(results_root.iterdir()):
        if not domain_dir.is_dir():
            continue
        domain = domain_dir.name
        for path in sorted(domain_dir.glob("*.csv")):
            name = path.name
            parts = name.split("_")
            if len(parts) < 3:
                continue
            model = parts[0]
            tool_selection = parts[1]
            yield domain, tool_selection, path


def _source_rank(model: str, tool_selection: str) -> int:
    """
    Lower is better. Matches the split policy:
      1) Prefer GPT-4 with domain tools
      2) Then GPT-4 with all tools
      3) Then other models (domains first, then all if available)
    """
    model = str(model)
    tool_selection = str(tool_selection)

    model_order = {
        "gpt-4": 0,
        "claude-2": 1,
        "llama2-70b": 2,
        "mistral-8x7B": 3,
        "gpt-3.5": 4,
    }
    base = model_order.get(model, 100) * 10
    tool_bonus = 0 if tool_selection == "domains" else 1
    if model == "gpt-4" and tool_selection == "domains":
        return 0
    if model == "gpt-4" and tool_selection == "all":
        return 1
    return base + tool_bonus + 2


def _best_teacher_pass(qa: pd.DataFrame, *, results_root: Path) -> pd.DataFrame:
    """
    For each query, find the best (ranked) result file whose predicted actions
    are an exact-match pass vs the ground truth.

    Returns DataFrame with:
      query, teacher_exact_match, teacher_source, teacher_function_calls
    """
    gt = qa[["query", "answer"]].copy()
    gt["answer_list"] = gt["answer"].apply(_parse_action_list)

    best_rank: Dict[str, int] = {}
    best_source: Dict[str, str] = {}
    best_calls: Dict[str, str] = {}

    for _domain, tool_selection, path in _iter_results_files(results_root):
        parts = path.name.split("_")
        if len(parts) < 2:
            continue
        model = parts[0]
        source = f"{model}_{tool_selection}"
        rank = _source_rank(model, tool_selection)

        pred = pd.read_csv(path, dtype=str)[["query", "function_calls", "error"]]
        merged = pred.merge(gt, on="query", how="left")
        if merged["answer"].isna().any():
            # Result file contains queries we don't have ground truth for (shouldn't happen).
            merged = merged[~merged["answer"].isna()].copy()

        for row in merged.itertuples(index=False):
            query = str(row.query)
            err = "" if row.error is None else str(row.error)
            if err.lower() == "nan":
                err = ""
            if err.strip():
                continue
            predicted_actions = _parse_action_list(row.function_calls)
            ground_truth_actions = list(row.answer_list)
            if not _is_exact_match(predicted_actions=predicted_actions, ground_truth_actions=ground_truth_actions):
                continue
            prev = best_rank.get(query)
            if prev is None or rank < prev:
                best_rank[query] = rank
                best_source[query] = source
                best_calls[query] = str(row.function_calls)

    out = pd.DataFrame({"query": qa["query"]})
    out["teacher_exact_match"] = out["query"].map(lambda q: q in best_rank)
    out["teacher_source"] = out["query"].map(best_source).fillna("")
    out["teacher_function_calls"] = out["query"].map(best_calls).fillna("")
    return out


def _mark_gpt4_pass(qa: pd.DataFrame, gpt4: pd.DataFrame) -> pd.DataFrame:
    merged = qa.merge(gpt4[["query", "function_calls", "error"]], on="query", how="left")
    if merged["function_calls"].isna().any():
        missing = merged[merged["function_calls"].isna()]["query"].head(5).tolist()
        raise SystemExit(f"GPT-4 results missing for some queries. Examples: {missing}")

    def _row_pass(row: Mapping[str, object]) -> bool:
        err = row.get("error")
        if err is None:
            err_text = ""
        else:
            err_text = str(err)
            if err_text.lower() == "nan":
                err_text = ""
        if err_text.strip():
            return False
        pred = _parse_action_list(row.get("function_calls"))
        gt = _parse_action_list(row.get("answer"))
        return _is_exact_match(predicted_actions=pred, ground_truth_actions=gt)

    merged["gpt4_exact_match"] = merged.apply(_row_pass, axis=1)
    merged = merged.rename(columns={"function_calls": "gpt4_function_calls", "error": "gpt4_error"})
    return merged


def _pick_train_indices(
    df: pd.DataFrame,
    *,
    k_per_template: int,
    target_train_size: int,
    seed: int,
) -> Set[int]:
    rng = random.Random(int(seed))
    chosen: Set[int] = set()
    for base_template, group in df.groupby("base_template", sort=False):
        group_indices = list(group.index)
        k = min(int(k_per_template), len(group_indices))
        if k <= 0:
            continue
        picked: List[int] = []
        teacher_passed = [idx for idx in group_indices if bool(df.at[idx, "teacher_exact_match"])]
        if teacher_passed:
            # Prefer GPT-4 teacher sources first within the template.
            gpt4_candidates = [idx for idx in teacher_passed if str(df.at[idx, "teacher_source"]).startswith("gpt-4_")]
            non_gpt4_candidates = [idx for idx in teacher_passed if idx not in set(gpt4_candidates)]
            if gpt4_candidates:
                picked.extend(rng.sample(gpt4_candidates, k=min(k, len(gpt4_candidates))))
            if len(picked) < k and non_gpt4_candidates:
                picked.extend(
                    rng.sample(non_gpt4_candidates, k=min(k - len(picked), len(non_gpt4_candidates)))
                )
        # We only train on passed cases; if template has <k passed, we pick fewer.
        chosen.update(picked)

    if len(chosen) >= target_train_size:
        # Trim deterministically (prefer GPT-4 sources).
        picked = sorted(chosen)
        gpt4 = [idx for idx in picked if str(df.at[idx, "teacher_source"]).startswith("gpt-4_")]
        rest = [idx for idx in picked if idx not in set(gpt4)]
        ordered = gpt4 + rest
        return set(ordered[:target_train_size])

    # Fill remaining slots with additional passed examples, spreading across templates.
    remaining_needed = target_train_size - len(chosen)
    remaining_passed = df[df["teacher_exact_match"]].copy()
    remaining_passed = remaining_passed[~remaining_passed.index.isin(chosen)]

    # Prefer GPT-4 teacher sources for the fill set.
    remaining_passed["is_gpt4"] = remaining_passed["teacher_source"].astype(str).str.startswith("gpt-4_")
    remaining_passed = remaining_passed.sort_values(["is_gpt4"], ascending=[False])

    by_template: Dict[str, List[int]] = {}
    for idx, row in remaining_passed.iterrows():
        by_template.setdefault(str(row["base_template"]), []).append(int(idx))

    extras: List[int] = []
    template_keys = list(by_template.keys())
    rng.shuffle(template_keys)
    # Round-robin one extra per template to diversify.
    while remaining_needed > 0 and template_keys:
        progressed = False
        for key in list(template_keys):
            candidates = by_template.get(key) or []
            if not candidates:
                template_keys.remove(key)
                continue
            extras.append(candidates.pop(0))
            remaining_needed -= 1
            progressed = True
            if remaining_needed <= 0:
                break
        if not progressed:
            break

    if remaining_needed > 0:
        raise SystemExit(
            f"Not enough passed examples to reach target_train_size={target_train_size}. "
            f"Missing={remaining_needed}."
        )

    chosen.update(extras)
    return chosen


def _assign_plan_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "plan_id", [f"work_{i:06d}" for i in range(len(df))])
    return df


def _write_csv(path: Path, rows: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False, quoting=csv.QUOTE_ALL)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/test split for WorkBench queries_and_answers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (used only for tie-breaking).")
    parser.add_argument(
        "--k-per-template",
        type=int,
        default=2,
        help="How many examples per base_template to put in train.csv.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.2,
        help="Target train fraction of total rows (used when --train-size is not set).",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Exact target train size (overrides --train-fraction when set).",
    )
    parser.add_argument(
        "--qa-dir",
        type=Path,
        default=Path("artifacts/input/work/dataset/queries_and_answers"),
        help="Directory containing *_queries_and_answers.csv files.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("artifacts/input/work/results"),
        help="Root directory containing per-domain GPT-4 results.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/input/work/dataset"),
        help="Output directory for train.csv and test.csv.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cfg = SplitConfig(
        seed=int(args.seed),
        k_per_template=int(args.k_per_template),
        train_fraction=float(args.train_fraction),
        train_size=(int(args.train_size) if args.train_size is not None else None),
        qa_dir=Path(args.qa_dir),
        results_root=Path(args.results_root),
        out_dir=Path(args.out_dir),
    )

    qa = _load_queries_and_answers(cfg.qa_dir)
    teacher = _best_teacher_pass(qa, results_root=cfg.results_root)
    merged = qa.merge(teacher, on="query", how="left")
    merged = _assign_plan_ids(merged)

    total_rows = int(len(merged))
    if cfg.train_size is not None:
        target_train_size = max(1, min(total_rows, int(cfg.train_size)))
    else:
        frac = float(cfg.train_fraction)
        if not (0.0 < frac < 1.0):
            raise SystemExit("--train-fraction must be between 0 and 1 (exclusive).")
        target_train_size = max(1, min(total_rows, int(total_rows * frac)))

    picked = _pick_train_indices(
        merged,
        k_per_template=cfg.k_per_template,
        target_train_size=target_train_size,
        seed=cfg.seed,
    )
    train = merged.loc[sorted(picked)].copy()
    test = merged.drop(index=list(picked)).copy()

    # Keep dataset columns stable; drop helper columns.
    keep_cols = [
        "plan_id",
        "query",
        "answer",
        "base_template",
        "chosen_template",
        "domains",
    ]
    # Keep teacher calls only for train rows that have any exact-match pass (train must be all-pass).
    train["teacher_function_calls"] = train["teacher_function_calls"].where(train["teacher_exact_match"], "")
    train["teacher_source"] = train["teacher_source"].where(train["teacher_exact_match"], "")
    keep_train_extra = ["teacher_source", "teacher_function_calls"]
    extra_cols = [c for c in keep_cols if c not in merged.columns]
    if extra_cols:
        raise SystemExit(f"Missing expected columns for output: {extra_cols}")

    train_out = cfg.out_dir / "train.csv"
    test_out = cfg.out_dir / "test.csv"
    uncovered_out = cfg.out_dir / "uncovered_base_templates.csv"

    _write_csv(train_out, train[keep_cols + keep_train_extra])
    _write_csv(test_out, test[keep_cols])

    total_templates = int(merged["base_template"].nunique())
    train_rows = int(len(train))
    test_rows = int(len(test))
    teacher_passed_rows = int(merged["teacher_exact_match"].sum())
    teacher_passed_train_rows = int(train["teacher_exact_match"].sum())

    pass_counts = merged.groupby("base_template")["teacher_exact_match"].sum().reset_index()
    pass_counts = pass_counts.rename(columns={"teacher_exact_match": "teacher_pass_count"})
    total_counts = merged.groupby("base_template").size().reset_index(name="total_count")
    coverage = pass_counts.merge(total_counts, on="base_template", how="left")

    # Add domain summary for easier debugging of uncovered templates.
    domain_summary = (
        merged.groupby("base_template")["domains"]
        .apply(lambda col: "; ".join(sorted({str(v) for v in col.dropna().tolist()})))
        .reset_index(name="domains")
    )
    coverage = coverage.merge(domain_summary, on="base_template", how="left")
    uncovered = coverage[coverage["teacher_pass_count"] <= 0].copy()
    uncovered = uncovered.sort_values(["teacher_pass_count", "total_count", "base_template"], ascending=[True, True, True])
    _write_csv(uncovered_out, uncovered[["base_template", "domains", "teacher_pass_count", "total_count"]])
    templates_without_any_pass = int(len(uncovered))

    print(
        "\n".join(
            [
                f"Wrote {train_out} ({train_rows} rows)",
                f"Wrote {test_out} ({test_rows} rows)",
                f"Wrote {uncovered_out} ({len(uncovered)} templates)",
                f"Total rows: {total_rows}",
                f"Base templates: {total_templates}",
                f"Teacher exact-match passed rows: {teacher_passed_rows}",
                f"Teacher exact-match passed rows in train: {teacher_passed_train_rows}",
                f"Templates without any passed row: {templates_without_any_pass}",
            ]
        )
    )


if __name__ == "__main__":
    main()
