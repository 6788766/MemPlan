#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# MemPlan end-to-end runner (WorkBench default)
#
# Runs: init_template -> view_select -> compose_match(_multi) -> twin_track(_multi) -> convert -> eval
#
# Outputs (per run) are written under:
#   artifacts/output/<task>/<model_slug>_<split>/
#
# Output files:
#   - init_templates_<split>.jsonl
#   - match_<split>.json
#   - tree_<split>.json
#   - optimized_<split>.jsonl
#   - results/                      (predictions_*.csv + pass_rates.json)
#   - cost.txt                      (timings + token usage + LLM $ cost)
#   - results.txt                   (evaluation output)
# -----------------------------------------------------------------------------

TASK="${TASK:-work}"
SPLIT="${SPLIT:-validation}"                 # train | validation | test
MODEL="${MODEL:-gpt-5-nano}"              # OpenAI model name (used by init_template only)
MODE="${MODE:-multi}"                  # multi | single

# Workers:
#   - init_template expects a concrete worker count
#   - view_select / compose_match / twin_track_multi use their own worker flags
CPUS="${CPUS:-0}"                      # 0 = all available CPUs
MULTI_WORKERS="${MULTI_WORKERS:-0}"    # 0 = all CPUs (planner.twin_track_multi)

# Optional behaviour toggles
RUN_INIT_TEMPLATE="${RUN_INIT_TEMPLATE:-1}"  # 1 = run init_template (LLM); 0 = reuse existing init_templates_<split>.jsonl
RUN_VIEW_SELECT="${RUN_VIEW_SELECT:-1}"      # 1 = run view_select; 0 = skip
RUN_COMPOSE_MATCH="${RUN_COMPOSE_MATCH:-1}"  # 1 = run compose_match; 0 = skip
RUN_TWIN_TRACK="${RUN_TWIN_TRACK:-1}"        # 1 = run twin_track; 0 = skip
RUN_CONVERT="${RUN_CONVERT:-1}"              # 1 = convert optimized -> predictions_*.csv
RUN_EVAL="${RUN_EVAL:-1}"                    # 1 = run WorkBench evaluation + constraint pass rates

# Optional: LLM repair (writes under output_dir/repair; does not change baseline outputs)
RUN_LLM_REPAIR="${RUN_LLM_REPAIR:-0}"        # 1 = run planner.llm_repair on optimized JSONL
REPAIR_MODEL="${REPAIR_MODEL:-${MODEL}}"     # LLM model used by planner.llm_repair (repair is priced with REPAIR_MODEL rates)
REPAIR_WORKERS="${REPAIR_WORKERS:-0}"        # 0 = all CPUs (planner.llm_repair default), else explicit count
REPAIR_MAX_FAILED="${REPAIR_MAX_FAILED:-0}"  # 0 = all failing, else repair at most N failing templates
REPAIR_NO_LLM="${REPAIR_NO_LLM:-0}"          # 1 = run llm_repair in --no-llm mode (wiring/debug)
REPAIR_ONLY_INCORRECT="${REPAIR_ONLY_INCORRECT:-1}"  # 1 = only attempt repair for WorkBench-incorrect templates

# ComposeMatch corrections (multi only)
ENABLE_DATE_CORRECTION="${ENABLE_DATE_CORRECTION:-1}"      # 1 = apply date_correct in compose_match_multi
ENABLE_DOMAIN_CORRECTION="${ENABLE_DOMAIN_CORRECTION:-1}"  # 1 = apply domain_correct in compose_match_multi
ENABLE_MEMORY_GRAPH_EXTENSION="${ENABLE_MEMORY_GRAPH_EXTENSION:-1}"  # 1 = include memory_graph extension candidates in compose_match_multi

# Optional: show intra-MCTS progress (multi only)
HOOK_PROGRESS="${HOOK_PROGRESS:-0}"          # 1 = pass --hook-progress to twin_track_multi

# Optional: export debug trees (multi only)
EXPORT_MCTS_TREE="${EXPORT_MCTS_TREE:-1}"          # 1 = write mcts_tree_<split>.json
EXPORT_ENRICHED_TREE="${EXPORT_ENRICHED_TREE:-1}"  # 1 = write tree_enriched_<split>.json

# Optional: provide an existing init-template JSONL to reuse when RUN_INIT_TEMPLATE=0.
INIT_TEMPLATES_IN="${INIT_TEMPLATES_IN:-}"

# ComposeMatch / Twin-Track knobs
MAX_TOOL_CANDIDATES="${MAX_TOOL_CANDIDATES:-0}"  # single-only; 0 = unlimited
MCTS_ITERATIONS="${MCTS_ITERATIONS:-100}"
MCTS_UCT_C="${MCTS_UCT_C:-1.4}"
TOLERANCE_RATE="${TOLERANCE_RATE:-0.8}"
MCTS_SEED="${MCTS_SEED:-}"

MODEL_SLUG="$(echo "${MODEL}" | tr -cd '[:alnum:]' | tr '[:upper:]' '[:lower:]')"
if [[ -z "${MODEL_SLUG}" ]]; then
  MODEL_SLUG="model"
fi
REPAIR_MODEL_SLUG="$(echo "${REPAIR_MODEL}" | tr -cd '[:alnum:]' | tr '[:upper:]' '[:lower:]')"
if [[ -z "${REPAIR_MODEL_SLUG}" ]]; then
  REPAIR_MODEL_SLUG="model"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${CPUS}" == "0" ]]; then
  CPU_COUNT="$(python -c 'from planner.parallel import available_cpu_count; print(available_cpu_count())')"
else
  CPU_COUNT="${CPUS}"
fi

OUTPUT_DIR="${REPO_ROOT}/artifacts/output/${TASK}/${MODEL_SLUG}_${SPLIT}"
mkdir -p "${OUTPUT_DIR}"

INIT_TEMPLATES="${OUTPUT_DIR}/init_templates_${SPLIT}.jsonl"
MATCH_JSON="${OUTPUT_DIR}/match_${SPLIT}.json"
TREE_JSON="${OUTPUT_DIR}/tree_${SPLIT}.json"
OPTIMIZED_JSONL="${OUTPUT_DIR}/optimized_${SPLIT}.jsonl"
RESULTS_DIR="${OUTPUT_DIR}/results"
MCTS_TREE_JSON="${OUTPUT_DIR}/mcts_tree_${SPLIT}.json"
ENRICHED_TREE_JSON="${OUTPUT_DIR}/tree_enriched_${SPLIT}.json"

COST_TXT="${OUTPUT_DIR}/cost.txt"
RESULTS_TXT="${OUTPUT_DIR}/results.txt"

# Repair outputs (optional)
REPAIR_DIR="${OUTPUT_DIR}/repair"
REPAIR_RESULTS_DIR="${REPAIR_DIR}/results"
REPAIR_COST_TXT="${REPAIR_DIR}/cost.txt"
REPAIR_RESULTS_TXT="${REPAIR_DIR}/results.txt"
REPAIR_JSONL="${REPAIR_DIR}/repaired_${SPLIT}_${REPAIR_MODEL_SLUG}.jsonl"
REPAIR_PASS_RATES_JSON="${REPAIR_RESULTS_DIR}/pass_rates.json"

timestamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
seconds_now() { date +%s; }

log_cost() {
  printf "%s\n" "$*" >>"${COST_TXT}"
}

EVAL_ELAPSED_S_TOTAL="0"

run_stage() {
  local name="$1"
  shift

  local start end elapsed status
  start="$(seconds_now)"
  echo "[$(timestamp)] START ${name}" | tee -a "${COST_TXT}"

  local tmp
  tmp="$(mktemp)"
  local -a tee_args
  tee_args=("${tmp}")
  if [[ "${name}" == "eval" ]]; then
    tee_args+=("${RESULTS_TXT}")
  fi

  set +e
  (cd "${REPO_ROOT}" && "$@") 2>&1 | tee "${tee_args[@]}"
  status="${PIPESTATUS[0]}"
  set -e

  end="$(seconds_now)"
  elapsed="$((end - start))"
  echo "[$(timestamp)] END ${name} status=${status} elapsed_s=${elapsed}" | tee -a "${COST_TXT}"
  if [[ "${name}" == "eval" || "${name}" == "pass_rates" ]]; then
    EVAL_ELAPSED_S_TOTAL="$((EVAL_ELAPSED_S_TOTAL + elapsed))"
  fi

  rm -f "${tmp}"
  if [[ "${status}" != "0" ]]; then
    exit "${status}"
  fi
}

run_stage_repair() {
  local name="$1"
  shift

  local start end elapsed status
  start="$(seconds_now)"
  echo "[$(timestamp)] START ${name}" | tee -a "${REPAIR_COST_TXT}"

  local -a tee_args
  local log_path
  log_path="${REPAIR_DIR}/${name}.log"
  : >"${log_path}"
  tee_args=("${log_path}")
  if [[ "${name}" == "eval" ]]; then
    tee_args+=("${REPAIR_RESULTS_TXT}")
  fi

  set +e
  (cd "${REPO_ROOT}" && "$@") 2>&1 | tee -a "${tee_args[@]}"
  status="${PIPESTATUS[0]}"
  set -e

  end="$(seconds_now)"
  elapsed="$((end - start))"
  echo "[$(timestamp)] END ${name} status=${status} elapsed_s=${elapsed}" | tee -a "${REPAIR_COST_TXT}"

  if [[ "${status}" != "0" ]]; then
    exit "${status}"
  fi
}

RUN_BASELINE_PIPELINE="0"
if [[ "${RUN_INIT_TEMPLATE}" == "1" || "${RUN_VIEW_SELECT}" == "1" || "${RUN_COMPOSE_MATCH}" == "1" || "${RUN_TWIN_TRACK}" == "1" || "${RUN_CONVERT}" == "1" || "${RUN_EVAL}" == "1" ]]; then
  RUN_BASELINE_PIPELINE="1"
fi

if [[ "${RUN_BASELINE_PIPELINE}" == "1" ]]; then
  PIPELINE_START="$(seconds_now)"
  cat >"${COST_TXT}" <<EOF
run_start=$(timestamp)
task=${TASK}
split=${SPLIT}
model=${MODEL}
model_slug=${MODEL_SLUG}
mode=${MODE}
cpus=${CPUS} (effective=${CPU_COUNT})
multi_workers=${MULTI_WORKERS}
tolerance_rate=${TOLERANCE_RATE}
mcts_iterations=${MCTS_ITERATIONS}
mcts_uct_c=${MCTS_UCT_C}
max_tool_candidates=${MAX_TOOL_CANDIDATES}
run_init_template=${RUN_INIT_TEMPLATE}
run_view_select=${RUN_VIEW_SELECT}
run_compose_match=${RUN_COMPOSE_MATCH}
run_twin_track=${RUN_TWIN_TRACK}
run_convert=${RUN_CONVERT}
run_eval=${RUN_EVAL}
hook_progress=${HOOK_PROGRESS}
export_mcts_tree=${EXPORT_MCTS_TREE}
export_enriched_tree=${EXPORT_ENRICHED_TREE}
enable_date_correction=${ENABLE_DATE_CORRECTION}
enable_domain_correction=${ENABLE_DOMAIN_CORRECTION}
enable_memory_graph_extension=${ENABLE_MEMORY_GRAPH_EXTENSION}
EOF
  printf "" >"${RESULTS_TXT}"
  mkdir -p "${RESULTS_DIR}"
fi

# 1) InitTemplate (LLM)
if [[ "${RUN_BASELINE_PIPELINE}" == "1" && "${RUN_INIT_TEMPLATE}" == "1" ]]; then
  CMD=(
    python -m planner.init_template
    --task "${TASK}"
    --split "${SPLIT}"
    --all
    --plan-fields auto
    --model "${MODEL}"
    --workers "${CPU_COUNT}"
    --out "${INIT_TEMPLATES}"
  )
  run_stage init_template "${CMD[@]}"
elif [[ "${RUN_BASELINE_PIPELINE}" == "1" ]]; then
  if [[ -f "${INIT_TEMPLATES}" ]]; then
    :
  elif [[ -n "${INIT_TEMPLATES_IN}" && -f "${INIT_TEMPLATES_IN}" ]]; then
    INIT_TEMPLATES="${INIT_TEMPLATES_IN}"
  elif [[ -f "${REPO_ROOT}/artifacts/output/work/init_template.jsonl" ]]; then
    INIT_TEMPLATES="${REPO_ROOT}/artifacts/output/work/init_template.jsonl"
  else
    echo "Missing init templates file: ${INIT_TEMPLATES} (set RUN_INIT_TEMPLATE=1, or set INIT_TEMPLATES_IN to an existing JSONL)." | tee -a "${COST_TXT}"
    exit 1
  fi
  echo "Skipping init_template (RUN_INIT_TEMPLATE=${RUN_INIT_TEMPLATE}); using ${INIT_TEMPLATES}" | tee -a "${COST_TXT}"
fi

if [[ "${RUN_BASELINE_PIPELINE}" == "1" ]]; then
  INIT_TOKENS="$(python -m task_helper.sum_init_tokens "${INIT_TEMPLATES}" || true)"
  log_cost "LLM token usage (init_template): ${INIT_TOKENS:-}"
  INIT_TOTAL_TOKENS="$(echo "${INIT_TOKENS:-}" | sed -n 's/.*total=\\([0-9][0-9]*\\).*/\\1/p')"
  if [[ -n "${INIT_TOTAL_TOKENS}" ]]; then
    log_cost "init_template_total_tokens=${INIT_TOTAL_TOKENS}"
  fi
fi

# 2) ViewSelect
if [[ "${RUN_BASELINE_PIPELINE}" == "1" && "${RUN_VIEW_SELECT}" == "1" ]]; then
  CMD=(
    python -m planner.view_select
    --task "${TASK}"
    --split "${SPLIT}"
    --templates "${INIT_TEMPLATES}"
    --out "${MATCH_JSON}"
    --workers 1
  )
  run_stage view_select "${CMD[@]}"
elif [[ "${RUN_BASELINE_PIPELINE}" == "1" ]]; then
  echo "Skipping view_select (RUN_VIEW_SELECT=${RUN_VIEW_SELECT})" | tee -a "${COST_TXT}"
fi

# 3) ComposeMatch
if [[ "${RUN_BASELINE_PIPELINE}" == "1" && "${RUN_COMPOSE_MATCH}" == "1" ]]; then
  if [[ "${MODE}" == "multi" ]]; then
    CMD=(
      python -m planner.compose_match_multi
      --task "${TASK}"
      --split "${SPLIT}"
      --templates "${INIT_TEMPLATES}"
      --match "${MATCH_JSON}"
      --out "${TREE_JSON}"
    )
    if [[ "${ENABLE_DATE_CORRECTION}" != "1" ]]; then
      CMD+=(--no-date-correction)
    fi
    if [[ "${ENABLE_DOMAIN_CORRECTION}" != "1" ]]; then
      CMD+=(--no-domain-correction)
    fi
    if [[ "${ENABLE_MEMORY_GRAPH_EXTENSION}" != "1" ]]; then
      CMD+=(--no-memory-graph-extension)
    fi
  else
    CMD=(
      python -m planner.compose_match
      --task "${TASK}"
      --split "${SPLIT}"
      --templates "${INIT_TEMPLATES}"
      --match "${MATCH_JSON}"
      --max-tool-candidates "${MAX_TOOL_CANDIDATES}"
      --no-tools
      --workers 0
      --out "${TREE_JSON}"
    )
  fi
  run_stage compose_match "${CMD[@]}"
elif [[ "${RUN_BASELINE_PIPELINE}" == "1" ]]; then
  echo "Skipping compose_match (RUN_COMPOSE_MATCH=${RUN_COMPOSE_MATCH})" | tee -a "${COST_TXT}"
fi

# 4) Twin-Track / MCTS
if [[ "${RUN_BASELINE_PIPELINE}" == "1" && "${RUN_TWIN_TRACK}" == "1" ]]; then
  if [[ "${MODE}" == "multi" ]]; then
    CMD=(
      python -m planner.twin_track_multi
      --task "${TASK}"
      --tree "${TREE_JSON}"
      --max-rounds 10
      --workers "${MULTI_WORKERS}"
      --iterations "${MCTS_ITERATIONS}"
      --uct-c "${MCTS_UCT_C}"
      --semantic-tolerance "${TOLERANCE_RATE}"
      --out "${OPTIMIZED_JSONL}"
    )
    if [[ -n "${MCTS_SEED}" ]]; then
      CMD+=(--seed "${MCTS_SEED}")
    fi
    if [[ "${HOOK_PROGRESS}" == "1" ]]; then
      CMD+=(--hook-progress)
    fi
    if [[ "${EXPORT_MCTS_TREE}" == "1" ]]; then
      CMD+=(--mcts-tree-out "${MCTS_TREE_JSON}")
    fi
    if [[ "${EXPORT_ENRICHED_TREE}" == "1" ]]; then
      CMD+=(--enriched-tree-out "${ENRICHED_TREE_JSON}")
    fi
  else
    CMD=(
      python -m planner.twin_track
      --task "${TASK}"
      --tree "${TREE_JSON}"
      --workers 1
      --iterations "${MCTS_ITERATIONS}"
      --uct-c "${MCTS_UCT_C}"
      --semantic-tolerance "${TOLERANCE_RATE}"
      --out "${OPTIMIZED_JSONL}"
    )
  fi
  run_stage mcts "${CMD[@]}"
elif [[ "${RUN_BASELINE_PIPELINE}" == "1" ]]; then
  echo "Skipping twin_track (RUN_TWIN_TRACK=${RUN_TWIN_TRACK})" | tee -a "${COST_TXT}"
fi

if [[ "${RUN_BASELINE_PIPELINE}" == "1" && -f "${OPTIMIZED_JSONL}" ]]; then
  TOOL_CALLS_SUMMARY="$(python -m task_helper.work.summarize_tool_calls "${OPTIMIZED_JSONL}" || true)"
  log_cost "Tool call usage (optimized): ${TOOL_CALLS_SUMMARY}"
  TOOL_CALLS_TOTAL_CALLS="$(echo "${TOOL_CALLS_SUMMARY}" | sed -n 's/.*calls_total=\\([0-9][0-9]*\\).*/\\1/p')"
  if [[ -n "${TOOL_CALLS_TOTAL_CALLS}" ]]; then
    log_cost "tool_calls_total_calls=${TOOL_CALLS_TOTAL_CALLS}"
  fi
fi

# 5) Convert optimized output to WorkBench prediction CSV(s)
if [[ "${RUN_BASELINE_PIPELINE}" == "1" && "${RUN_CONVERT}" == "1" ]]; then
  mkdir -p "${RESULTS_DIR}"
  CMD=(
    python task_helper/work/evaluation/convert_optimized_to_predictions.py
    --optimized "${OPTIMIZED_JSONL}"
    --out-dir "${RESULTS_DIR}"
  )
  run_stage convert "${CMD[@]}"
fi

# 6) Evaluation (official-style metrics + constraint pass rates)
if [[ "${RUN_BASELINE_PIPELINE}" == "1" && "${RUN_EVAL}" == "1" ]]; then
  {
    echo "=============================="
    echo "WorkBench evaluation: ${OUTPUT_DIR}"
    echo "model=${MODEL} split=${SPLIT} mode=${MODE}"
    echo "=============================="
  } >>"${RESULTS_TXT}"

  echo "========== WorkBench official-style metrics ==========" >>"${RESULTS_TXT}"
  run_stage eval python task_helper/work/evaluation/calculate_all_metrics.py --predictions_dir "${RESULTS_DIR}"

  echo "========== Constraint pass rates (local/global) ==========" >>"${RESULTS_TXT}"
  CMD=(
    bash -lc
    "python task_helper/work/evaluation/calculate_pass_rates_dir.py --predictions_dir \"${RESULTS_DIR}\" --ground_truth_dir \"artifacts/input/work/dataset/queries_and_answers\" --json_out \"${RESULTS_DIR}/pass_rates.json\" >/dev/null"
  )
  run_stage pass_rates "${CMD[@]}"
  PYTHONDONTWRITEBYTECODE=1 python -c "import json; from pathlib import Path; p=Path('${RESULTS_DIR}/pass_rates.json'); obj=json.loads(p.read_text(encoding='utf-8')); obj.pop('constraints', None); print(json.dumps(obj, indent=2, ensure_ascii=False))" >>"${RESULTS_TXT}"
fi

# 7) Optional: LLM repair + evaluation (writes to output_dir/repair)
if [[ "${RUN_LLM_REPAIR}" == "1" ]]; then
  # For WorkBench repair we prefer the enriched tree (includes all retrieved tool results).
  REPAIR_TREE_JSON="${ENRICHED_TREE_JSON}"
  if [[ ! -f "${REPAIR_TREE_JSON}" ]]; then
    REPAIR_TREE_JSON="${TREE_JSON}"
  fi
  if [[ ! -f "${REPAIR_TREE_JSON}" ]]; then
    echo "Missing tree JSON: ${REPAIR_TREE_JSON} (run compose_match + twin_track first, or point TASK/MODEL/SPLIT to an existing run dir)." >&2
    exit 1
  fi
  if [[ ! -f "${OPTIMIZED_JSONL}" ]]; then
    echo "Missing optimized JSONL: ${OPTIMIZED_JSONL} (run twin_track first, or point TASK/MODEL/SPLIT to an existing run dir)." >&2
    exit 1
  fi
  if [[ ! -f "${COST_TXT}" ]]; then
    echo "Missing base cost.txt: ${COST_TXT} (needed to compute combined cost)." >&2
    exit 1
  fi

  mkdir -p "${REPAIR_DIR}"
  mkdir -p "${REPAIR_RESULTS_DIR}"
  printf "" >"${REPAIR_RESULTS_TXT}"

  REPAIR_START="$(seconds_now)"
  cat >"${REPAIR_COST_TXT}" <<EOF
run_start=$(timestamp)
task=${TASK}
split=${SPLIT}
model=${REPAIR_MODEL}
model_slug=${REPAIR_MODEL_SLUG}
base_output_dir=${OUTPUT_DIR}
repair_tree_json=${REPAIR_TREE_JSON}
base_cost_txt=${COST_TXT}
base_results_txt=${RESULTS_TXT}
EOF

  # Copy init_template token usage line into repair cost.txt so money_memplan can compute combined cost.
  INIT_USAGE_LINE="$(grep -F "LLM token usage (init_template):" "${COST_TXT}" | tail -n 1 || true)"
  if [[ -n "${INIT_USAGE_LINE}" ]]; then
    printf "%s\n" "${INIT_USAGE_LINE}" >>"${REPAIR_COST_TXT}"
  else
    INIT_TOKENS="$(python -m task_helper.sum_init_tokens "${INIT_TEMPLATES}" || true)"
    if [[ -n "${INIT_TOKENS}" ]]; then
      printf "LLM token usage (init_template): %s\n" "${INIT_TOKENS}" >>"${REPAIR_COST_TXT}"
    fi
  fi

  # Run repair over optimized JSONL -> repaired JSONL (does not modify baseline optimized output).
  CMD=(
    python -m planner.llm_repair
    --task "${TASK}"
    --tree "${REPAIR_TREE_JSON}"
    --input "${OPTIMIZED_JSONL}"
    --model "${REPAIR_MODEL}"
    --semantic-threshold "${TOLERANCE_RATE}"
    --out "${REPAIR_JSONL}"
  )
  if [[ "${REPAIR_WORKERS}" != "0" ]]; then
    CMD+=(--workers "${REPAIR_WORKERS}")
  fi
  if [[ "${REPAIR_MAX_FAILED}" != "0" ]]; then
    CMD+=(--max-failed "${REPAIR_MAX_FAILED}")
  fi
  if [[ "${REPAIR_ONLY_INCORRECT}" == "1" ]]; then
    CMD+=(--workbench-only-incorrect)
  fi
  if [[ "${REPAIR_NO_LLM}" == "1" ]]; then
    CMD+=(--no-llm)
  fi
  run_stage_repair llm_repair "${CMD[@]}"

  # Capture token usage line from llm_repair output (if any).
  REPAIR_USAGE_LINE="$(grep -F "LLM token usage (llm_repair):" "${REPAIR_DIR}/llm_repair.log" | tail -n 1 || true)"
  if [[ -n "${REPAIR_USAGE_LINE}" ]]; then
    printf "%s\n" "${REPAIR_USAGE_LINE}" >>"${REPAIR_COST_TXT}"
  fi

  # Convert repaired output to WorkBench prediction CSV(s).
  CMD=(python task_helper/work/evaluation/convert_optimized_to_predictions.py --optimized "${REPAIR_JSONL}" --out-dir "${REPAIR_RESULTS_DIR}")
  run_stage_repair convert "${CMD[@]}"

  # Evaluation output (keep official script stdout; append trimmed pass-rate JSON without constraint descriptions).
  {
    echo "=============================="
    echo "WorkBench evaluation (repair): ${REPAIR_DIR}"
    echo "model=${REPAIR_MODEL} split=${SPLIT} mode=${MODE}"
    echo "base_run_dir=${OUTPUT_DIR}"
    echo "=============================="
  } >>"${REPAIR_RESULTS_TXT}"

  echo "========== WorkBench official-style metrics ==========" >>"${REPAIR_RESULTS_TXT}"
  run_stage_repair eval python task_helper/work/evaluation/calculate_all_metrics.py --predictions_dir "${REPAIR_RESULTS_DIR}"

  echo "========== Constraint pass rates (local/global) ==========" >>"${REPAIR_RESULTS_TXT}"
  python task_helper/work/evaluation/calculate_pass_rates_dir.py \
    --predictions_dir "${REPAIR_RESULTS_DIR}" \
    --ground_truth_dir "artifacts/input/work/dataset/queries_and_answers" \
    --json_out "${REPAIR_PASS_RATES_JSON}" >/dev/null

  REPAIR_END="$(seconds_now)"
  REPAIR_ELAPSED="$((REPAIR_END - REPAIR_START))"
  printf "run_end=%s\n" "$(timestamp)" >>"${REPAIR_COST_TXT}"
  printf "pipeline_elapsed_s_repair=%s\n" "${REPAIR_ELAPSED}" >>"${REPAIR_COST_TXT}"

  BASE_PASS_RATES_JSON="${RESULTS_DIR}/pass_rates.json"
  python -m task_helper.work.evaluation.report_repair_run \
    --base-cost-txt "${COST_TXT}" \
    --repair-cost-txt "${REPAIR_COST_TXT}" \
    --repair-elapsed-s "${REPAIR_ELAPSED}" \
    --repair-model "${REPAIR_MODEL}" \
    --base-pass-rates-json "${BASE_PASS_RATES_JSON}" \
    --repair-pass-rates-json "${REPAIR_PASS_RATES_JSON}" \
    --repair-results-txt "${REPAIR_RESULTS_TXT}" >/dev/null
fi

if [[ "${RUN_BASELINE_PIPELINE}" == "1" ]]; then
  PIPELINE_END="$(seconds_now)"
  PIPELINE_ELAPSED_TOTAL="$((PIPELINE_END - PIPELINE_START))"
  log_cost "run_end=$(timestamp)"
  log_cost "pipeline_elapsed_s_total=${PIPELINE_ELAPSED_TOTAL}"
  PIPELINE_ELAPSED_NO_EVAL="$((PIPELINE_ELAPSED_TOTAL - EVAL_ELAPSED_S_TOTAL))"
  if [[ "${PIPELINE_ELAPSED_NO_EVAL}" -lt 0 ]]; then
    PIPELINE_ELAPSED_NO_EVAL="0"
  fi
  log_cost "pipeline_elapsed_s_no_eval=${PIPELINE_ELAPSED_NO_EVAL}"

  PIPELINE_TOTAL_TOKENS="${INIT_TOTAL_TOKENS:-0}"
  log_cost "pipeline_total_tokens=${PIPELINE_TOTAL_TOKENS}"
  log_cost "outputs_dir=${OUTPUT_DIR}"
  log_cost "summary: time_s_total=${PIPELINE_ELAPSED_TOTAL} tokens_total=${PIPELINE_TOTAL_TOKENS:-0} tool_calls_total=${TOOL_CALLS_TOTAL_CALLS:-0}"

  (cd "${REPO_ROOT}" && python -m task_helper.money_memplan "${COST_TXT}" >/dev/null 2>&1) || log_cost "llm_money_error=1"
fi

echo "Done. Outputs: ${OUTPUT_DIR}"
