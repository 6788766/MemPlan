from __future__ import annotations

import ast
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODELS = {
    "gpt-5.2": "gpt52",
    "gpt-5-mini": "gpt5mini",
    "gpt-5-nano": "gpt5nano",
    "deepseek-chat": "deepseekchat",
}

BASELINES = {
    "RAG": "artifacts/output/travel/rag/{model}_validation/results.txt",
    # NoMCTS and NoSel should use the top-level pre-repair evaluation output.
    "NoMCTS": "artifacts/output/travel/nomcts/{model}_validation/results.txt",
    "NoSel": "artifacts/output/travel/nosel/{model}_validation/results.txt",
}

LEVELS = ["easy", "medium", "hard"]
ALL_LEVELS = [*LEVELS, "all"]


def find_baseline_results(model_key: str) -> dict[str, Path | None]:
    results: dict[str, Path | None] = {}
    for baseline, template in BASELINES.items():
        file_path = PROJECT_ROOT / template.format(model=model_key)
        results[baseline] = file_path if file_path.exists() else None
    return results


def extract_summary_dict(text: str) -> dict:
    start = text.rfind("{'Commonsense Constraint'")
    if start == -1:
        raise ValueError("Summary dict not found.")
    brace_count = 0
    end = None
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            brace_count += 1
        elif ch == "}":
            brace_count -= 1
            if brace_count == 0:
                end = idx + 1
                break
    if end is None:
        raise ValueError("Unbalanced braces in summary dict.")
    return ast.literal_eval(text[start:end])


def budget_counts(file_path: Path) -> dict:
    text = file_path.read_text(encoding="utf-8")
    summary = extract_summary_dict(text)
    hard = summary["Hard Constraint"]
    counts = {level: {"true": 0, "total": 0} for level in LEVELS}
    for level in LEVELS:
        for day_data in hard[level].values():
            budget = day_data.get("Budget")
            if not budget:
                continue
            counts[level]["true"] += budget.get("true", 0)
            counts[level]["total"] += budget.get("total", 0)
    counts["all"] = {
        "true": sum(counts[level]["true"] for level in LEVELS),
        "total": sum(counts[level]["total"] for level in LEVELS),
    }
    return counts


def safe_rate(true_count: int, total_count: int) -> float:
    return true_count / total_count if total_count else 0.0


def main() -> None:
    output_path = PROJECT_ROOT / "artifacts" / "output" / "travel" / "budget_satisfaction.csv"
    rows = []
    for model_name, model_key in MODELS.items():
        baseline_results = find_baseline_results(model_key)
        method_rates = {}
        for baseline, file_path in baseline_results.items():
            if file_path is None:
                print(f"Missing {baseline} results for {model_name}; leaving CSV cells blank.")
                method_rates[baseline] = {level: "" for level in ALL_LEVELS}
                continue
            counts = budget_counts(file_path)
            method_rates[baseline] = {
                level: safe_rate(counts[level]["true"], counts[level]["total"])
                for level in ALL_LEVELS
            }
        for level in ALL_LEVELS:
            row_id = f"{level}_{model_name}"
            rows.append(
                {
                    "set_model": row_id,
                    "RAG": method_rates["RAG"][level],
                    "NoMCTS": method_rates["NoMCTS"][level],
                    "NoSel": method_rates["NoSel"][level],
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["set_model", "RAG", "NoMCTS", "NoSel"],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
