#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# WorkBench ReAct baseline runner (MemPlan)
#
# Runs:
#   inference (per-domain) -> copy latest CSVs -> pass_rates (per-domain) -> summary
#
# Notes:
# - Inference writes raw CSVs under: data/results/<domain>/
# - This runner copies those CSVs under: artifacts/output/work/workbench_react/<run_id>/predictions/
# - Evaluation uses MemPlan's WorkBench evaluator + constraint pass rates:
#     python -m task_helper.work.evaluation.calculate_pass_rates
# -----------------------------------------------------------------------------

MODEL_NAME="${MODEL_NAME:-gpt-5.2}"                 # WorkBench model key (e.g., gpt-4, gpt-3.5)
SPLIT="${SPLIT:-validation}"                      # validation | test (label only; WorkBench Q/A CSVs are unsplit)
TOOL_SELECTION="${TOOL_SELECTION:-all}"           # all | domains
WORKERS="${WORKERS:-0}"                           # 0 = all CPUs; one worker per query at a time

RUN_INFERENCE="${RUN_INFERENCE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Load repo-root .env (keys) if present.
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${REPO_ROOT}/.env"
  set +a
fi

timestamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
seconds_now() { date +%s; }

MODEL_SLUG="$(echo "${MODEL_NAME}" | tr -cd '[:alnum:]' | tr '[:upper:]' '[:lower:]')"
if [[ -z "${MODEL_SLUG}" ]]; then
  MODEL_SLUG="model"
fi

OUTPUT_DIR="${REPO_ROOT}/artifacts/output/work/baseline/${MODEL_SLUG}_${SPLIT}"
PRED_DIR="${OUTPUT_DIR}/predictions"
RESULTS_DIR="${OUTPUT_DIR}/results"
mkdir -p "${PRED_DIR}" "${RESULTS_DIR}"

COST_TXT="${OUTPUT_DIR}/cost.txt"
RESULTS_TXT="${OUTPUT_DIR}/results.txt"
: >"${COST_TXT}"
: >"${RESULTS_TXT}"

INFERENCE_ELAPSED_S_TOTAL=0
EVAL_ELAPSED_S_TOTAL=0

run_stage() {
  local name="$1"
  shift

  local start end elapsed status
  start="$(seconds_now)"
  echo "[$(timestamp)] START ${name}" | tee -a "${COST_TXT}"

  set +e
  "$@" 2>&1 | tee -a "${RESULTS_TXT}"
  status="${PIPESTATUS[0]}"
  set -e

  end="$(seconds_now)"
  elapsed="$((end - start))"
  echo "[$(timestamp)] END ${name} status=${status} elapsed_s=${elapsed}" | tee -a "${COST_TXT}"
  if [[ "${name}" == eval_* ]]; then
    EVAL_ELAPSED_S_TOTAL="$((EVAL_ELAPSED_S_TOTAL + elapsed))"
  else
    INFERENCE_ELAPSED_S_TOTAL="$((INFERENCE_ELAPSED_S_TOTAL + elapsed))"
  fi
  if [[ "${status}" != "0" ]]; then
    exit "${status}"
  fi
}

DOMAINS=(
  email
  calendar
  analytics
  project_management
  customer_relationship_manager
  multi_domain
)

queries_path_for_domain() {
  local domain="$1"
  echo "${REPO_ROOT}/artifacts/input/work/dataset/queries_and_answers/${domain}_queries_and_answers.csv"
}

latest_results_csv() {
  local domain="$1"
  local dir="${REPO_ROOT}/data/results/${domain}"
  if [[ ! -d "${dir}" ]]; then
    return 1
  fi
  local pattern="${MODEL_NAME}_${TOOL_SELECTION}_"
  local f
  for f in $(ls -t "${dir}"/*.csv 2>/dev/null); do
    if [[ "$(basename "${f}")" == *"${pattern}"* ]]; then
      echo "${f}"
      return 0
    fi
  done
  return 0
}

if [[ "${RUN_INFERENCE}" == "1" ]]; then
  for domain in "${DOMAINS[@]}"; do
    qp="$(queries_path_for_domain "${domain}")"
    workers_effective="${WORKERS}"
    if [[ "${workers_effective}" == "0" ]]; then
      workers_effective="$(python -c 'import os; print(os.cpu_count() or 1)')"
    fi
    run_stage "inference_${domain}" \
      python baseline/workbench/generate_results.py \
        --model_name "${MODEL_NAME}" \
        --tool_selection "${TOOL_SELECTION}" \
        --workers "${workers_effective}" \
        --queries_path "${qp}"

    latest="$(latest_results_csv "${domain}")"
    if [[ -z "${latest}" ]]; then
      echo "Failed to find inference output CSV for domain=${domain} under data/results/${domain}" | tee -a "${RESULTS_TXT}"
      exit 1
    fi
    cp -f "${latest}" "${PRED_DIR}/${domain}.csv"
    echo "copied_predictions_${domain}=${PRED_DIR}/${domain}.csv" | tee -a "${COST_TXT}"
  done
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  for domain in "${DOMAINS[@]}"; do
    qp="$(queries_path_for_domain "${domain}")"
    pred="${PRED_DIR}/${domain}.csv"
    out_json="${RESULTS_DIR}/pass_rates_${domain}.json"
    run_stage "eval_${domain}" \
      python -m task_helper.work.evaluation.calculate_pass_rates \
        --predictions_path "${pred}" \
        --ground_truth_path "${qp}" \
        --json_out "${out_json}"
  done
fi

run_stage "summary" python - <<PY
import json
import re
from pathlib import Path
import ast
from typing import Any, Dict, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP, getcontext

run_dir = Path(${OUTPUT_DIR@Q})
results_dir = run_dir / "results"
out_path = results_dir / "summary.json"

payloads = []
for p in sorted(results_dir.glob("pass_rates_*.json")):
    payloads.append(json.loads(p.read_text(encoding="utf-8")))

if not payloads:
    out_path.write_text(json.dumps({"error": "no pass_rates_*.json found"}, indent=2) + "\\n", encoding="utf-8")
    raise SystemExit(0)

def wavg(key: str) -> float:
    total_n = sum(int(x.get("n_examples") or 0) for x in payloads)
    if total_n <= 0:
        return 0.0
    return sum((float(x.get(key) or 0.0) * int(x.get("n_examples") or 0)) for x in payloads) / total_n

def _find_token_usage(obj: Any) -> Optional[Dict[str, float]]:
    if obj is None:
        return None
    if isinstance(obj, dict):
        for k in ("token_usage", "usage", "tokenUsage"):
            v = obj.get(k)
            if isinstance(v, dict):
                # Common keys across providers.
                prompt = v.get("prompt_tokens") or v.get("input_tokens") or v.get("promptTokens")
                completion = v.get("completion_tokens") or v.get("output_tokens") or v.get("completionTokens")
                total = v.get("total_tokens") or v.get("totalTokens")
                out: Dict[str, float] = {}
                if prompt is not None:
                    out["prompt_tokens"] = float(prompt)
                if completion is not None:
                    out["completion_tokens"] = float(completion)
                if total is not None:
                    out["total_tokens"] = float(total)
                return out or None
        # Recurse common fields
        for k in ("response_metadata", "llm_output", "generations", "generation_info"):
            found = _find_token_usage(obj.get(k))
            if found:
                return found
        # Recurse all values
        for v in obj.values():
            found = _find_token_usage(v)
            if found:
                return found
    if isinstance(obj, list):
        for v in obj:
            found = _find_token_usage(v)
            if found:
                return found
    return None

def _extract_usage_from_csv(path: Path) -> Tuple[float, float, float]:
    # Prefer explicit columns if present; fallback to parsing `full_response`.
    import pandas as pd

    df = pd.read_csv(path, dtype=str)
    if {"prompt_tokens", "completion_tokens", "total_tokens"}.issubset(set(df.columns)):
        pt = pd.to_numeric(df["prompt_tokens"], errors="coerce").fillna(0).sum()
        ct = pd.to_numeric(df["completion_tokens"], errors="coerce").fillna(0).sum()
        tt = pd.to_numeric(df["total_tokens"], errors="coerce").fillna(0).sum()
        return float(pt), float(ct), float(tt)

    total_prompt = 0.0
    total_completion = 0.0
    total_total = 0.0
    for raw in df.get("full_response", "").fillna("").tolist():
        raw = str(raw)
        if not raw.strip():
            continue
        obj: Any = None
        try:
            obj = ast.literal_eval(raw)
        except Exception:
            # Some versions stringify objects; attempt to locate a usage dict substring.
            m = re.search(r"(token_usage|usage)\\s*[:=]\\s*(\\{.*?\\})", raw)
            if m:
                try:
                    obj = {m.group(1): ast.literal_eval(m.group(2))}
                except Exception:
                    obj = None
        usage = _find_token_usage(obj)
        if not usage:
            continue
        total_prompt += float(usage.get("prompt_tokens") or 0.0)
        total_completion += float(usage.get("completion_tokens") or 0.0)
        total_total += float(usage.get("total_tokens") or 0.0)
    return total_prompt, total_completion, total_total

summary = {
    "n_examples_total": sum(int(x.get("n_examples") or 0) for x in payloads),
    "Local Pass Rate": wavg("Local Pass Rate"),
    "Global Pass Rate": wavg("Global Pass Rate"),
    "WorkBench Accuracy": wavg("WorkBench Accuracy"),
    "WorkBench Exact Match": wavg("WorkBench Exact Match"),
    "WorkBench Unwanted Side Effects": wavg("WorkBench Unwanted Side Effects"),
}

pred_dir = run_dir / "predictions"
total_prompt = total_completion = total_tokens = 0.0
total_cache_hit = 0.0
total_cache_miss = 0.0
total_cost_usd = 0.0
for domain_csv in sorted(pred_dir.glob("*.csv")):
    p, c, t = _extract_usage_from_csv(domain_csv)
    total_prompt += p
    total_completion += c
    total_tokens += t
    import pandas as pd
    df = pd.read_csv(domain_csv, dtype=str)
    if {"prompt_cache_hit_tokens", "prompt_cache_miss_tokens"}.issubset(set(df.columns)):
        total_cache_hit += pd.to_numeric(df["prompt_cache_hit_tokens"], errors="coerce").fillna(0).sum()
        total_cache_miss += pd.to_numeric(df["prompt_cache_miss_tokens"], errors="coerce").fillna(0).sum()
    if "total_cost_usd" in df.columns:
        total_cost_usd += pd.to_numeric(df["total_cost_usd"], errors="coerce").fillna(0).sum()

summary["total_prompt_tokens"] = int(total_prompt) if total_prompt else 0
summary["total_completion_tokens"] = int(total_completion) if total_completion else 0
summary["total_tokens"] = int(total_tokens) if total_tokens else 0
summary["total_prompt_cache_hit_tokens"] = int(total_cache_hit) if total_cache_hit else 0
summary["total_prompt_cache_miss_tokens"] = int(total_cache_miss) if total_cache_miss else 0

def _resolve_price_key(model: str, table: dict) -> str:
    if model in table:
        return model
    m = (model or "").strip().lower()
    for k in table.keys():
        if str(k).strip().lower() in m:
            return str(k)
    if "deepseek" in m and "deepseek" in table:
        return "deepseek"
    for candidate in ("gpt-5.2", "gpt-5-mini", "gpt-5-nano"):
        if candidate in m and candidate in table:
            return candidate
    raise KeyError(model)

def _compute_cost_usd(*, model: str, hit: int, miss: int, out: int) -> float:
    prices_path = Path("artifacts/input/price.json")
    if not prices_path.exists():
        return 0.0
    table = json.loads(prices_path.read_text(encoding="utf-8"))
    key = _resolve_price_key(model, table)
    rates = table.get(key) or {}
    getcontext().prec = 28
    million = Decimal(1_000_000)
    hit_rate = Decimal(str(rates.get("prompt_cache_hit", 0)))
    miss_rate = Decimal(str(rates.get("prompt_cache_miss", 0)))
    out_rate = Decimal(str(rates.get("output", 0)))
    total = (Decimal(int(hit)) / million) * hit_rate + (Decimal(int(miss)) / million) * miss_rate + (Decimal(int(out)) / million) * out_rate
    return float(total.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))

model_name = ${MODEL_NAME@Q}
hit = int(summary["total_prompt_cache_hit_tokens"] or 0)
miss = int(summary["total_prompt_cache_miss_tokens"] or 0)
out_tok = int(summary["total_completion_tokens"] or 0)

# If hit/miss weren't provided by the provider, treat all prompt tokens as miss.
if hit == 0 and miss == 0:
    miss = int(summary["total_prompt_tokens"] or 0)

summary["total_price_usd"] = _compute_cost_usd(model=model_name, hit=hit, miss=miss, out=out_tok)
# Preserve callback-reported cost too (often 0 for non-OpenAI models).
summary["total_cost_usd_callback"] = float(total_cost_usd) if total_cost_usd else 0.0

out_path.write_text(json.dumps(summary, indent=2) + "\\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

# Append key numbers to cost.txt for quick grep.
python - <<PY >>"${COST_TXT}"
import json
from pathlib import Path

run_dir = Path(${OUTPUT_DIR@Q})
summary = json.loads((run_dir / "results" / "summary.json").read_text(encoding="utf-8"))

print(
    "llm_tokens: "
    f"prompt_cache_hit={summary.get('total_prompt_cache_hit_tokens', 0)} "
    f"prompt_cache_miss={summary.get('total_prompt_cache_miss_tokens', 0)} "
    f"output={summary.get('total_completion_tokens', 0)} "
    f"total={summary.get('total_tokens', 0)}"
)
print(f"total_price_usd={summary.get('total_price_usd')}")
print(f"total_cost_usd_callback={summary.get('total_cost_usd_callback')}")
PY

cat >>"${COST_TXT}" <<EOF
model_name=${MODEL_NAME}
split=${SPLIT}
tool_selection=${TOOL_SELECTION}
workers=${WORKERS}
total_time_s_excluding_eval=${INFERENCE_ELAPSED_S_TOTAL}
eval_time_s=${EVAL_ELAPSED_S_TOTAL}
total_time_s=$((INFERENCE_ELAPSED_S_TOTAL + EVAL_ELAPSED_S_TOTAL))
EOF

echo "outputs_dir=${OUTPUT_DIR}" | tee -a "${RESULTS_TXT}"
