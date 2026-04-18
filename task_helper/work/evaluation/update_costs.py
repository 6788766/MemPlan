#!/usr/bin/env python3
import json
import os
import re
import sys


PRICE_PATH = "artifacts/input/price.json"
OUTPUT_ROOT = "artifacts/output/work"
BASELINE_ROOT = os.path.join(OUTPUT_ROOT, "baseline")
TRAVEL_BASELINE_ROOT = "artifacts/output/travel/baseline"
TRAVEL_OUTPUT_ROOT = "artifacts/output/travel"

INIT_RE = re.compile(
    r"LLM token usage \(init_template\): .*prompt_cache_hit=(\d+) "
    r"prompt_cache_miss=(\d+) output=(\d+)"
)
REPAIR_RE = re.compile(
    r"LLM token usage \(llm_repair\): .*prompt_cache_hit=(\d+) "
    r"prompt_cache_miss=(\d+) output=(\d+)"
)
MODEL_RE = re.compile(r"^model=(.+)$", re.MULTILINE)
MODEL_NAME_RE = re.compile(r"^model_name=(.+)$", re.MULTILINE)
MODEL_LINE_RE = re.compile(r"^model=(.+)$", re.MULTILINE)
LLM_TOKENS_RE = re.compile(
    r"llm_tokens: .*prompt_cache_hit=(\d+) prompt_cache_miss=(\d+) output=(\d+) total=(\d+)"
)
ELAPSED_S_RE = re.compile(r"\belapsed_s=([0-9]+)\b")


def compute_cost(tokens, price):
    hit, miss, out = tokens
    total = hit * price["prompt_cache_hit"] + miss * price["prompt_cache_miss"] + out * price["output"]
    return total / 1_000_000.0


def parse_tokens(regex, text):
    match = regex.search(text)
    if not match:
        return None
    return tuple(int(x) for x in match.groups())


def parse_all_tokens(regex, text):
    return [tuple(int(x) for x in m.groups()) for m in regex.finditer(text)]


def read_elapsed_seconds(text):
    for key in ("pipeline_elapsed_s_no_eval=", "pipeline_elapsed_s_total=", "total_time_s="):
        for line in text.splitlines():
            if line.startswith(key):
                try:
                    return int(float(line.split("=", 1)[1].strip()))
                except Exception:
                    continue
    return 0


def sum_stage_elapsed_seconds(text, stage):
    total = 0
    token = f"] END {stage} "
    for line in text.splitlines():
        if token not in line:
            continue
        match = ELAPSED_S_RE.search(line)
        if not match:
            continue
        try:
            total += int(match.group(1))
        except Exception:
            continue
    return int(total)


def update_cost_block(text, init_cost, repair_cost):
    total_cost = init_cost + repair_cost
    text = re.sub(
        r"^llm_price_usd_init_template=.*$",
        f"llm_price_usd_init_template={init_cost:.6f}",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^llm_price_usd_llm_repair=.*$",
        f"llm_price_usd_llm_repair={repair_cost:.6f}",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^llm_price_usd_total=.*$",
        f"llm_price_usd_total={total_cost:.6f}",
        text,
        flags=re.MULTILINE,
    )
    return text


def main():
    price_path = PRICE_PATH
    output_root = OUTPUT_ROOT
    baseline_root = BASELINE_ROOT
    travel_baseline_root = TRAVEL_BASELINE_ROOT
    travel_output_root = TRAVEL_OUTPUT_ROOT

    with open(price_path, "r", encoding="utf-8") as f:
        prices = json.load(f)

    updated = []
    skipped = []

    for root, _, files in os.walk(output_root):
        if "validation" not in root:
            continue
        if root.startswith(baseline_root):
            continue
        if "cost.txt" not in files:
            continue
        path = os.path.join(root, "cost.txt")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        model_match = MODEL_RE.search(text)
        if not model_match:
            skipped.append((path, "no model line"))
            continue
        model = model_match.group(1).strip()
        if model not in prices:
            skipped.append((path, f"model not in price.json: {model}"))
            continue

        init_tokens = parse_tokens(INIT_RE, text)
        if not init_tokens:
            skipped.append((path, "no init_template token usage"))
            continue
        repair_tokens_list = parse_all_tokens(REPAIR_RE, text)
        repair_tokens = (0, 0, 0)
        for hit, miss, out in repair_tokens_list:
            repair_tokens = (repair_tokens[0] + hit, repair_tokens[1] + miss, repair_tokens[2] + out)

        price = prices[model]
        init_cost = compute_cost(init_tokens, price)
        repair_cost = compute_cost(repair_tokens, price)

        new_text = update_cost_block(text, init_cost, repair_cost)
        base_cost_line = None
        for line in text.splitlines():
            if line.startswith("base_cost_txt="):
                base_cost_line = line.split("=", 1)[1].strip()
                break
        if base_cost_line and os.path.exists(base_cost_line):
            base_text = open(base_cost_line, "r", encoding="utf-8").read()
            base_elapsed = read_elapsed_seconds(base_text)
            repair_llm_elapsed = sum_stage_elapsed_seconds(text, "llm_repair")
            total_elapsed = int(base_elapsed + repair_llm_elapsed)
            new_text = re.sub(
                r"^pipeline_elapsed_s_total=.*$",
                f"pipeline_elapsed_s_total={total_elapsed}",
                new_text,
                flags=re.MULTILINE,
            )
            new_text = re.sub(
                r"^total_time_s=.*$",
                f"total_time_s={float(total_elapsed):.1f}",
                new_text,
                flags=re.MULTILINE,
            )
            new_text = re.sub(
                r"^pipeline_elapsed_s_repair=.*$",
                f"pipeline_elapsed_s_repair={repair_llm_elapsed}",
                new_text,
                flags=re.MULTILINE,
            )
            new_text = re.sub(
                r"^pipeline_elapsed_s_repair_total=.*$",
                f"pipeline_elapsed_s_repair_total={repair_llm_elapsed}",
                new_text,
                flags=re.MULTILINE,
            )
            new_text = re.sub(
                r"^pipeline_elapsed_s_repair_no_eval=.*$",
                f"pipeline_elapsed_s_repair_no_eval={repair_llm_elapsed}",
                new_text,
                flags=re.MULTILINE,
            )
        if new_text != text:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_text)
            updated.append(path)
        else:
            skipped.append((path, "no changes"))

    # Baseline: update total_price_usd in cost.txt and results/summary.json.
    for root, _, files in os.walk(baseline_root):
        if "validation" not in root:
            continue

        if "cost.txt" in files:
            path = os.path.join(root, "cost.txt")
            text = open(path, "r", encoding="utf-8").read()
            model_match = MODEL_NAME_RE.search(text)
            if not model_match:
                skipped.append((path, "no model_name line"))
            else:
                model = model_match.group(1).strip()
                if model not in prices:
                    skipped.append((path, f"model not in price.json: {model}"))
                else:
                    price = prices[model]
                    new_lines = []
                    last_tokens = None
                    for line in text.splitlines():
                        m = LLM_TOKENS_RE.search(line)
                        if m:
                            hit, miss, out, _total = (int(x) for x in m.groups())
                            last_tokens = (hit, miss, out)
                            new_lines.append(line)
                            continue
                        if line.startswith("total_price_usd=") and last_tokens:
                            hit, miss, out = last_tokens
                            cost = compute_cost((hit, miss, out), price)
                            new_lines.append(f"total_price_usd={cost:.6f}")
                            continue
                        new_lines.append(line)
                    new_text = "\n".join(new_lines) + ("\n" if text.endswith("\n") else "")
                    if new_text != text:
                        with open(path, "w", encoding="utf-8") as f:
                            f.write(new_text)
                        updated.append(path)
                    else:
                        skipped.append((path, "no changes"))

        summary_path = os.path.join(root, "results", "summary.json")
        if os.path.exists(summary_path):
            summary = json.load(open(summary_path, "r", encoding="utf-8"))
            model = None
            if "model_name" in summary:
                model = summary["model_name"]
            else:
                model_match = MODEL_NAME_RE.search(
                    open(os.path.join(root, "cost.txt"), "r", encoding="utf-8").read()
                )
                if model_match:
                    model = model_match.group(1).strip()
            if not model:
                skipped.append((summary_path, "no model_name"))
            elif model not in prices:
                skipped.append((summary_path, f"model not in price.json: {model}"))
            else:
                price = prices[model]
                hit = summary["total_prompt_cache_hit_tokens"]
                miss = summary["total_prompt_cache_miss_tokens"]
                out = summary["total_completion_tokens"]
                cost = compute_cost((hit, miss, out), price)
                if summary.get("total_price_usd") != round(cost, 6):
                    summary["total_price_usd"] = round(cost, 6)
                    with open(summary_path, "w", encoding="utf-8") as f:
                        json.dump(summary, f, indent=2, sort_keys=False)
                        f.write("\n")
                    updated.append(summary_path)
                else:
                    skipped.append((summary_path, "no changes"))

    # Travel baseline: update total cost block in cost.txt only.
    for root, _, files in os.walk(travel_baseline_root):
        if "validation" not in root:
            continue
        if "cost.txt" not in files:
            continue
        path = os.path.join(root, "cost.txt")
        text = open(path, "r", encoding="utf-8").read()
        model_match = MODEL_LINE_RE.search(text)
        if not model_match:
            skipped.append((path, "no model line"))
            continue
        model = model_match.group(1).strip()
        if model not in prices:
            skipped.append((path, f"model not in price.json: {model}"))
            continue
        tokens_match = LLM_TOKENS_RE.search(text)
        if not tokens_match:
            skipped.append((path, "no llm_tokens line"))
            continue
        hit, miss, out, _total = (int(x) for x in tokens_match.groups())
        cost = compute_cost((hit, miss, out), prices[model])
        new_lines = []
        for line in text.splitlines():
            if line.startswith("llm_price_model_key="):
                new_lines.append(f"llm_price_model_key={model}")
                continue
            if line.startswith("llm_price_usd_baseline="):
                new_lines.append(f"llm_price_usd_baseline={cost:.6f}")
                continue
            if line.startswith("llm_price_usd_total="):
                new_lines.append(f"llm_price_usd_total={cost:.6f}")
                continue
            new_lines.append(line)
        new_text = "\n".join(new_lines) + ("\n" if text.endswith("\n") else "")
        if new_text != text:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_text)
            updated.append(path)
        else:
            skipped.append((path, "no changes"))

    # Travel non-baseline: update init_template + llm_repair totals (sum multiple repairs if present).
    for root, _, files in os.walk(travel_output_root):
        if "validation" not in root:
            continue
        if root.startswith(travel_baseline_root):
            continue
        if "cost.txt" not in files:
            continue
        path = os.path.join(root, "cost.txt")
        text = open(path, "r", encoding="utf-8").read()

        model_match = MODEL_LINE_RE.search(text)
        if not model_match:
            skipped.append((path, "no model line"))
            continue
        model = model_match.group(1).strip()
        if model not in prices:
            skipped.append((path, f"model not in price.json: {model}"))
            continue

        init_tokens = parse_tokens(INIT_RE, text)
        if not init_tokens:
            skipped.append((path, "no init_template token usage"))
            continue
        repair_tokens_list = parse_all_tokens(REPAIR_RE, text)
        total_repair_tokens = (0, 0, 0)
        for hit, miss, out in repair_tokens_list:
            total_repair_tokens = (
                total_repair_tokens[0] + hit,
                total_repair_tokens[1] + miss,
                total_repair_tokens[2] + out,
            )

        price = prices[model]
        init_cost = compute_cost(init_tokens, price)
        repair_cost = compute_cost(total_repair_tokens, price)
        new_text = update_cost_block(text, init_cost, repair_cost)
        if new_text != text:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_text)
            updated.append(path)
        else:
            skipped.append((path, "no changes"))

    for path in updated:
        print(f"updated: {path}")
    for path, reason in skipped:
        print(f"skipped: {path} ({reason})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
