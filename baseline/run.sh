#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Baseline runner (TravelPlanner ReAct + Planner) + postprocessing + evaluation.
#
# Outputs:
#   artifacts/output/travel/baseline/<model_slug>_<split>/
#     - cost.txt
#     - two_stage_<model_slug>_<split>.jsonl
#     - two_stage_<model_slug>_<split>/generated_plan_<idx>.json
#     - submission_<split>.jsonl              (converted for official evaluator)
#     - results.txt                           (evaluation output, when enabled)
# -----------------------------------------------------------------------------

# ----------------------------- Config (edit me) ------------------------------
SPLIT="${SPLIT:-validation}"          # train | validation | test
MODEL="${MODEL:-gpt-5-mini}"          # gpt-5.2 | gpt-5-mini | gpt-5-nano | deepseek-chat | ...
WORKERS="${WORKERS:-8}"               # baseline parallelism (threads); 0 = all CPUs (beware rate limits)
LIMIT="${LIMIT:-0}"                   # 0 = all queries
PYTHON="${PYTHON:-python}"            # set to python3 if `python` is unavailable

RUN_BASELINE="${RUN_BASELINE:-1}"     # 1 = run baseline; 0 = reuse existing outputs
RUN_CONVERT="${RUN_CONVERT:-1}"       # 1 = convert baseline JSONL to submission JSONL
RUN_EVAL="${RUN_EVAL:-1}"             # 1 = run evaluator (train/validation) or leaderboard (test)
RESUME_FAILED="${RESUME_FAILED:-0}"   # 1 = rerun only failed/empty-result queries (requires existing per-query outputs)

# Postprocess:
#   - simple: offline regex parser (baseline/convert_baseline_to_submission.py)
#   - tp: upstream TravelPlanner postprocess (GPT parsing.py -> element_extraction.py -> combination.py)
POSTPROCESS="${POSTPROCESS:-tp}"  # simple | tp
TP_MODEL_NAME="${TP_MODEL_NAME:-${MODEL}}"
TP_PARSE_MODEL="${TP_PARSE_MODEL:-gpt-5.2}"
TP_TEMPERATURE="${TP_TEMPERATURE:-0}"

# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

load_dotenv_if_needed() {
  local dotenv_path="${REPO_ROOT}/.env"
  if [[ ! -f "${dotenv_path}" ]]; then
    return 0
  fi
  # Load KEY=VALUE lines without clobbering already-set env vars.
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "${line}" ]] && continue
    [[ "${line}" == \#* ]] && continue
    if [[ "${line}" == export\ * ]]; then
      line="${line#export }"
      line="${line#"${line%%[![:space:]]*}"}"
    fi
    [[ "${line}" != *"="* ]] && continue
    local key="${line%%=*}"
    local value="${line#*=}"
    key="${key%"${key##*[![:space:]]}"}"
    key="${key#"${key%%[![:space:]]*}"}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    if [[ "${value}" == \"*\" && "${value}" == *\" ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "${value}" == \'*\' && "${value}" == *\' ]]; then
      value="${value:1:${#value}-2}"
    fi
    [[ -z "${key}" ]] && continue
    if [[ -z "${!key:-}" ]]; then
      export "${key}=${value}"
    fi
  done < "${dotenv_path}"
}

load_dotenv_if_needed

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
  fi
fi
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "Could not find a Python interpreter. Set PYTHON=python3 (or your env's python)." >&2
  exit 1
fi

MODEL_SLUG="$(echo "${MODEL}" | tr -cd '[:alnum:]' | tr '[:upper:]' '[:lower:]')"
if [[ -z "${MODEL_SLUG}" ]]; then
  MODEL_SLUG="model"
fi

OUT_ROOT="${REPO_ROOT}/artifacts/output/travel/baseline"
RUN_DIR="${OUT_ROOT}/${MODEL_SLUG}_${SPLIT}"
RUN_PREFIX="two_stage_${MODEL_SLUG}_${SPLIT}"

JSONL_PATH="${RUN_DIR}/${RUN_PREFIX}.jsonl"
PER_QUERY_DIR="${RUN_DIR}/${RUN_PREFIX}"
COST_TXT="${RUN_DIR}/cost.txt"
SUBMISSION_JSONL="${RUN_DIR}/submission_${SPLIT}.jsonl"
RESULTS_TXT="${RUN_DIR}/results.txt"

TP_POSTPROCESS_DIR="${RUN_DIR}/tp_postprocess"
TP_TMP_DIR="${RUN_DIR}/tp_tmp"

mkdir -p "${RUN_DIR}"
: >"${RESULTS_TXT}"

log_cost() {
  printf "%s\n" "$*" >>"${COST_TXT}"
}

strip_total_cost_block() {
  # If cost.txt already has a "total cost" block from a previous run, remove it
  # so any new log lines stay above the regenerated block.
  if [[ -f "${COST_TXT}" ]]; then
    awk 'BEGIN{keep=1} $0=="--------------total cost-----------------"{keep=0} keep{print}' "${COST_TXT}" > "${COST_TXT}.tmp"
    mv "${COST_TXT}.tmp" "${COST_TXT}"
  fi
}

strip_total_cost_block

iso_now() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

seconds_now() {
  date +%s
}

run_stage() {
  local name="$1"
  shift
  local start end elapsed
  start="$(seconds_now)"
  log_cost "[$(iso_now)] START ${name}"
  set +e
  (cd "${REPO_ROOT}" && "$@") 2>&1 | tee -a "${RESULTS_TXT}"
  local status="${PIPESTATUS[0]}"
  set -e
  end="$(seconds_now)"
  elapsed="$((end - start))"
  log_cost "[$(iso_now)] END ${name} status=${status} elapsed_s=${elapsed}"
  if [[ "${status}" != "0" ]]; then
    exit "${status}"
  fi
}

# 1) Run baseline
if [[ "${RUN_BASELINE}" == "1" ]]; then
  EXTRA_BASELINE_ARGS=()
  if [[ "${RESUME_FAILED}" == "1" ]]; then
    EXTRA_BASELINE_ARGS+=(--resume-failed)
  fi
  (cd "${REPO_ROOT}" && \
    "${PYTHON}" baseline/tool_agents.py \
        --set_type "${SPLIT}" \
        --model_name "${MODEL}" \
        --output_dir "${OUT_ROOT}" \
        --workers "${WORKERS}" \
        --limit "${LIMIT}" \
        "${EXTRA_BASELINE_ARGS[@]}")
else
  if [[ ! -f "${JSONL_PATH}" ]]; then
    echo "Missing baseline output: ${JSONL_PATH}" >&2
    exit 1
  fi
fi

# 2) Convert to official submission format (idx/query/plan)
if [[ "${RUN_CONVERT}" == "1" ]]; then
  if [[ "${POSTPROCESS}" == "tp" ]]; then
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
      echo "POSTPROCESS=tp requires OPENAI_API_KEY (used by baseline/postprocess/* for parsing)." >&2
      exit 1
    fi

    run_stage tp_prepare \
      "${PYTHON}" -m baseline.prepare_tp_postprocess_input \
        --baseline-jsonl "${JSONL_PATH}" \
        --out-dir "${TP_POSTPROCESS_DIR}" \
        --set-type "${SPLIT}" \
        --model-name "${TP_MODEL_NAME}" \
        --mode "two-stage" \
        --overwrite
    log_cost "tp_postprocess_dir=${TP_POSTPROCESS_DIR}"
    log_cost "tp_tmp_dir=${TP_TMP_DIR}"
    log_cost "tp_model_name=${TP_MODEL_NAME}"
    log_cost "tp_parse_model=${TP_PARSE_MODEL}"

    run_stage tp_parsing \
      bash -c "cd \"${REPO_ROOT}/baseline/postprocess\" && \"${PYTHON}\" parsing.py --set_type \"${SPLIT}\" --output_dir \"${TP_POSTPROCESS_DIR}\" --tmp_dir \"${TP_TMP_DIR}\" --model_name \"${TP_MODEL_NAME}\" --mode two-stage --parse_model_name \"${TP_PARSE_MODEL}\" --temperature \"${TP_TEMPERATURE}\""
    run_stage tp_extract \
      bash -c "cd \"${REPO_ROOT}/baseline/postprocess\" && \"${PYTHON}\" element_extraction.py --set_type \"${SPLIT}\" --output_dir \"${TP_POSTPROCESS_DIR}\" --tmp_dir \"${TP_TMP_DIR}\" --model_name \"${TP_MODEL_NAME}\" --mode two-stage"
    run_stage tp_combine \
      bash -c "cd \"${REPO_ROOT}/baseline/postprocess\" && \"${PYTHON}\" combination.py --set_type \"${SPLIT}\" --output_dir \"${TP_POSTPROCESS_DIR}\" --submission_file_dir \"${RUN_DIR}\" --model_name \"${TP_MODEL_NAME}\" --mode two-stage"

    TP_SUBMISSION="${RUN_DIR}/${SPLIT}_${TP_MODEL_NAME}_two-stage_submission.jsonl"
    if [[ ! -f "${TP_SUBMISSION}" ]]; then
      echo "Missing TravelPlanner postprocess submission file: ${TP_SUBMISSION}" >&2
      exit 1
    fi
    cp -f "${TP_SUBMISSION}" "${SUBMISSION_JSONL}"
    log_cost "converted_submission=${SUBMISSION_JSONL}"
    log_cost "tp_submission_raw=${TP_SUBMISSION}"
  else
    run_stage convert \
      "${PYTHON}" baseline/convert_baseline_to_submission.py --in "${JSONL_PATH}" --out "${SUBMISSION_JSONL}"
    log_cost "converted_submission=${SUBMISSION_JSONL}"
  fi

  run_stage normalize_submission \
    "${PYTHON}" -m baseline.normalize_submission --in "${SUBMISSION_JSONL}" --out "${SUBMISSION_JSONL}"
fi

# 3) Evaluate
if [[ "${RUN_EVAL}" == "1" ]]; then
  if [[ "${SPLIT}" == "test" ]]; then
    # Hidden-label evaluation for TEST is hosted on the public leaderboard Space.
    run_stage eval_leaderboard \
      "${PYTHON}" -m task_helper.travel.runners.eval_leaderboard --split test --eval-mode two-stage --submission "${SUBMISSION_JSONL}"
  else
    run_stage eval \
      "${PYTHON}" -m task_helper.travel.runners.eval_bridge --set-type "${SPLIT}" --submission "${SUBMISSION_JSONL}"
  fi
fi

# 4) LLM $ cost (best-effort; appends to cost.txt)
(cd "${REPO_ROOT}" && "${PYTHON}" -m task_helper.money_memplan "${COST_TXT}" >/dev/null 2>&1) || true

echo "Done."
echo "Outputs: ${RUN_DIR}"
