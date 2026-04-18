#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# MemPlan end-to-end runner (TravelPlanner default)
#
# Runs: init_template -> view_select -> compose_match -> twin_track (MCTS) -> llm_repair -> eval
#
# Outputs (per run) are written under:
#   artifacts/output/<task>/<model_slug>_<split>/
#
# Output files:
#   - init_templates_<split>.jsonl
#   - match_<split>.json
#   - tree_<split>.json
#   - optimized_<split>.jsonl        (MCTS output; optionally overwritten by llm_repair)
#   - cost.txt                       (timings + token usage + LLM cost)
#   - results.txt                    (official evaluation output, when enabled)
#
# Edit the variables in the "Config" section below.
# -----------------------------------------------------------------------------

# ----------------------------- Config (edit me) ------------------------------
TASK="${TASK:-travel}"
SPLIT="${SPLIT:-validation}"               # train | validation | test
MODEL="${MODEL:-deepseek-chat}"               # OpenAI model name

# Optional: override per-task planner config JSON (planner.json).
# If unset, the pipeline uses `artifacts/input/<task>/planner.json`.
CONFIG="${CONFIG:-}"

TOLERANCE_RATE="${TOLERANCE_RATE:-0.8}"    # semantic tolerance for MCTS / repair
MCTS_ITERATIONS="${MCTS_ITERATIONS:-150}"
MCTS_UCT_C="${MCTS_UCT_C:-1.4}"

# Max candidates per tool call in ComposeMatch (0 = unlimited)
MAX_TOOL_CANDIDATES="${MAX_TOOL_CANDIDATES:-5}"

# Max candidates per OR-slot included in the LLM repair prompt (0 is NOT supported; keep >= 1)
REPAIR_MAX_SLOT_CANDIDATES="${REPAIR_MAX_SLOT_CANDIDATES:-50}"
# Tree file used by llm_repair (defaults to the tree produced in this run).
REPAIR_TREE="${REPAIR_TREE:-}"

# Workers:
#   - init_template expects a concrete worker count
#   - view_select / compose_match / twin_track: 0 = all CPUs
#   - llm_repair expects a concrete worker count
CPUS="${CPUS:-0}"                          # 0 = all available CPUs

# Optional behaviour toggles
RUN_INIT_TEMPLATE="${RUN_INIT_TEMPLATE:-1}"  # 1 = run init_template (LLM); 0 = reuse existing init_templates_<split>.jsonl
RUN_REPAIR="${RUN_REPAIR:-1}"              # 1 = run llm_repair
KEEP_PRE_REPAIR_OPTIMIZED="${KEEP_PRE_REPAIR_OPTIMIZED:-1}"  # 1 = keep optimized_<split>.raw.jsonl
RUN_EVAL="${RUN_EVAL:-1}"                  # 1 = run TravelPlanner eval (skipped for non-travel or test split)

# ViewSelect strictness:
#   - 1: fail fast if any template has uncovered required edges
#   - 0: write match_<split>.json and continue (ComposeMatch may still fail later)
VIEW_SELECT_STRICT="${VIEW_SELECT_STRICT:-0}"

# -----------------------------------------------------------------------------

MODEL_SLUG="$(echo "${MODEL}" | tr -cd '[:alnum:]' | tr '[:upper:]' '[:lower:]')"
if [[ -z "${MODEL_SLUG}" ]]; then
  MODEL_SLUG="model"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_DIR="${REPO_ROOT}/artifacts/output/${TASK}/${MODEL_SLUG}_${SPLIT}"
mkdir -p "${OUTPUT_DIR}"

INIT_TEMPLATES="${OUTPUT_DIR}/init_templates_${SPLIT}.jsonl"
MATCH_JSON="${OUTPUT_DIR}/match_${SPLIT}.json"
TREE_JSON="${OUTPUT_DIR}/tree_${SPLIT}.json"
OPTIMIZED_JSONL="${OUTPUT_DIR}/optimized_${SPLIT}.jsonl"

COST_TXT="${OUTPUT_DIR}/cost.txt"
RESULTS_TXT="${OUTPUT_DIR}/results.txt"

resolve_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    echo "${path}"
  else
    echo "${REPO_ROOT}/${path}"
  fi
}

CONFIG_ARGS=()
if [[ -n "${CONFIG}" ]]; then
  CONFIG_ARGS=(--config "${CONFIG}")
fi

if [[ -z "${REPAIR_TREE}" ]]; then
  REPAIR_TREE="${TREE_JSON}"
fi
REPAIR_TREE_PATH="$(resolve_path "${REPAIR_TREE}")"

python_cpu_count() {
  python - <<'PY'
import os
import re

try:
    affinity = os.sched_getaffinity(0)
except Exception:
    affinity = None
if affinity:
    print(len(affinity))
    raise SystemExit

for key in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
    raw = os.environ.get(key)
    if not raw:
        continue
    match = re.match(r"^\\s*(\\d+)", str(raw))
    if match:
        value = int(match.group(1))
        if value > 0:
            print(value)
            raise SystemExit

raw = os.environ.get("SLURM_JOB_CPUS_PER_NODE")
if raw:
    match = re.match(r"^\\s*(\\d+)", str(raw))
    if match:
        value = int(match.group(1))
        if value > 0:
            print(value)
            raise SystemExit

print(os.cpu_count() or 1)
PY
}

if [[ "${CPUS}" == "0" ]]; then
  CPU_COUNT="$(python_cpu_count)"
else
  CPU_COUNT="${CPUS}"
fi

log_cost() {
  printf "%s\n" "$*" >>"${COST_TXT}"
}

seconds_now() {
  date +%s
}

iso_now() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

sum_init_tokens() {
  python - "${INIT_TEMPLATES}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
cache_hit = cache_miss = output = total = calls = 0
if path.exists():
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        notes = obj.get("notes") if isinstance(obj, dict) else None
        llm = notes.get("llm") if isinstance(notes, dict) else None
        if isinstance(llm, dict):
            calls += 1
            hit = int(llm.get("prompt_cache_hit_tokens") or 0)
            miss = llm.get("prompt_cache_miss_tokens")
            if miss is None:
                prompt_tokens = int(llm.get("prompt_tokens") or 0)
                miss = max(0, prompt_tokens - hit) if hit else prompt_tokens
            miss = int(miss or 0)
            cache_hit += hit
            cache_miss += miss
            completion_tokens = int(llm.get("completion_tokens") or 0)
            output += completion_tokens
            total_tokens = llm.get("total_tokens")
            if total_tokens is None:
                total_tokens = (hit + miss) + completion_tokens
            total += int(total_tokens or 0)
print(f"calls={calls} prompt_cache_hit={cache_hit} prompt_cache_miss={cache_miss} output={output} total={total}")
PY
}

sum_tool_calls() {
  python - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
calls = int(data.get("total_calls") or 0)
cost = float(data.get("total_cost") or 0.0)
by_tool = data.get("by_tool") if isinstance(data.get("by_tool"), dict) else {}
print(f"calls={calls} cost={cost} by_tool={json.dumps(by_tool, ensure_ascii=False)}")
PY
}

run_stage() {
  local name="$1"
  shift

  local start end elapsed
  start="$(seconds_now)"
  echo "[$(iso_now)] START ${name}" | tee -a "${COST_TXT}"

  # Run command and stream output. Capture full output in a temp file for parsing.
  local tmp
  tmp="$(mktemp)"
  local -a tee_args
  tee_args=("${tmp}")
  if [[ "${name}" == "eval" ]]; then
    tee_args+=("${RESULTS_TXT}")
  fi
  set +e
  (cd "${REPO_ROOT}" && "$@") 2>&1 | tee "${tee_args[@]}"
  local status="${PIPESTATUS[0]}"
  set -e

  end="$(seconds_now)"
  elapsed="$((end - start))"
  echo "[$(iso_now)] END ${name} status=${status} elapsed_s=${elapsed}" | tee -a "${COST_TXT}"

  if [[ "${name}" == "llm_repair" ]]; then
    local line
    line="$(grep -E "LLM token usage \\(llm_repair\\):" "${tmp}" | tail -n 1 || true)"
    if [[ -n "${line}" ]]; then
      log_cost "${line}"
      LLM_REPAIR_TOKENS_LINE="${line}"
    fi
  fi

  rm -f "${tmp}"
  if [[ "${status}" != "0" ]]; then
    exit "${status}"
  fi
}

PIPELINE_START="$(seconds_now)"
cat >"${COST_TXT}" <<EOF
run_start=$(iso_now)
task=${TASK}
split=${SPLIT}
model=${MODEL}
model_slug=${MODEL_SLUG}
cpus=${CPUS} (effective=${CPU_COUNT})
tolerance_rate=${TOLERANCE_RATE}
mcts_iterations=${MCTS_ITERATIONS}
mcts_uct_c=${MCTS_UCT_C}
max_tool_candidates=${MAX_TOOL_CANDIDATES}
repair_max_slot_candidates=${REPAIR_MAX_SLOT_CANDIDATES}
repair_tree=${REPAIR_TREE_PATH}
run_repair=${RUN_REPAIR}
keep_pre_repair_optimized=${KEEP_PRE_REPAIR_OPTIMIZED}
run_eval=${RUN_EVAL}
run_init_template=${RUN_INIT_TEMPLATE}
EOF

# 1) InitTemplate (LLM)
if [[ "${RUN_INIT_TEMPLATE}" == "1" ]]; then
  INIT_TEMPLATE_CMD=(
    python -m planner.init_template
    --task "${TASK}"
    --split "${SPLIT}"
    --all
    --plan-fields org,dest,days,date,query
    --model "${MODEL}"
    --workers "${CPU_COUNT}"
    --out "${INIT_TEMPLATES}"
  )
  run_stage init_template "${INIT_TEMPLATE_CMD[@]}"
else
  if [[ ! -f "${INIT_TEMPLATES}" ]]; then
    echo "Missing init templates file: ${INIT_TEMPLATES} (set RUN_INIT_TEMPLATE=1 to generate it)." | tee -a "${COST_TXT}"
    exit 1
  fi
  echo "Skipping init_template (RUN_INIT_TEMPLATE=${RUN_INIT_TEMPLATE}); using ${INIT_TEMPLATES}" | tee -a "${COST_TXT}"
fi

INIT_TOKENS="$(sum_init_tokens)"
log_cost "LLM token usage (init_template): ${INIT_TOKENS}"
INIT_TOTAL_TOKENS="$(echo "${INIT_TOKENS}" | sed -n 's/.*total=\([0-9][0-9]*\).*/\1/p')"
if [[ -n "${INIT_TOTAL_TOKENS}" ]]; then
  log_cost "init_template_total_tokens=${INIT_TOTAL_TOKENS}"
fi

# 2) ViewSelect
VIEW_SELECT_CMD=(
  python -m planner.view_select
  --task "${TASK}"
  --split "${SPLIT}"
  --templates "${INIT_TEMPLATES}"
  --out "${MATCH_JSON}"
  --workers "${CPUS}"
)
if [[ -n "${CONFIG}" ]]; then
  VIEW_SELECT_CMD+=(--config "${CONFIG}")
fi
if [[ "${VIEW_SELECT_STRICT}" == "1" ]]; then
  VIEW_SELECT_CMD+=(--strict)
fi
run_stage view_select "${VIEW_SELECT_CMD[@]}"

# 3) ComposeMatch
COMPOSE_MATCH_CMD=(
  python -m planner.compose_match
  --task "${TASK}"
  --split "${SPLIT}"
  --templates "${INIT_TEMPLATES}"
  --match "${MATCH_JSON}"
  --out "${TREE_JSON}"
  --max-tool-candidates "${MAX_TOOL_CANDIDATES}"
  --workers "${CPUS}"
)
if [[ -n "${CONFIG}" ]]; then
  COMPOSE_MATCH_CMD+=(--config "${CONFIG}")
fi
run_stage compose_match "${COMPOSE_MATCH_CMD[@]}"

TOOL_CALLS_JSON="${OUTPUT_DIR}/tool_calls_${SPLIT}.json"
if [[ -f "${TOOL_CALLS_JSON}" ]]; then
  TOOL_CALLS_SUMMARY="$(sum_tool_calls "${TOOL_CALLS_JSON}")"
  log_cost "Tool call usage (compose_match): ${TOOL_CALLS_SUMMARY}"
  TOOL_CALLS_TOTAL_CALLS="$(echo "${TOOL_CALLS_SUMMARY}" | sed -n 's/.*calls=\([^ ]*\).*/\1/p')"
  if [[ -n "${TOOL_CALLS_TOTAL_CALLS}" ]]; then
    log_cost "tool_calls_total_calls=${TOOL_CALLS_TOTAL_CALLS}"
  fi
  TOOL_CALLS_TOTAL_COST="$(echo "${TOOL_CALLS_SUMMARY}" | sed -n 's/.*cost=\([^ ]*\).*/\1/p')"
  if [[ -n "${TOOL_CALLS_TOTAL_COST}" ]]; then
    log_cost "tool_calls_total_cost=${TOOL_CALLS_TOTAL_COST}"
  fi
fi

# 4) Twin-Track (MCTS)
MCTS_CMD=(
  python -m planner.twin_track
  --task "${TASK}"
  --tree "${TREE_JSON}"
  --out "${OPTIMIZED_JSONL}"
  --workers "${CPUS}"
  --iterations "${MCTS_ITERATIONS}"
  --uct-c "${MCTS_UCT_C}"
  --semantic-tolerance "${TOLERANCE_RATE}"
)
if [[ -n "${CONFIG}" ]]; then
  MCTS_CMD+=(--config "${CONFIG}")
fi
run_stage mcts "${MCTS_CMD[@]}"

# 5) Optional: LLM repair (in-place overwrite of optimized_<split>.jsonl)
if [[ "${RUN_REPAIR}" == "1" ]]; then
  if [[ "${KEEP_PRE_REPAIR_OPTIMIZED}" == "1" ]]; then
    cp -f "${OPTIMIZED_JSONL}" "${OUTPUT_DIR}/optimized_${SPLIT}.raw.jsonl"
  fi
  if [[ ! -f "${REPAIR_TREE_PATH}" ]]; then
    echo "Skipping llm_repair: REPAIR_TREE not found: ${REPAIR_TREE_PATH}" | tee -a "${COST_TXT}"
  else
  LLM_REPAIR_CMD=(
    python -m planner.llm_repair
    --task "${TASK}"
    --tree "${REPAIR_TREE_PATH}"
    --input "${OPTIMIZED_JSONL}"
    --out "${OPTIMIZED_JSONL}"
    --model "${MODEL}"
    --semantic-threshold "${TOLERANCE_RATE}"
    --max-slot-candidates "${REPAIR_MAX_SLOT_CANDIDATES}"
    --workers "${CPU_COUNT}"
  )
  if [[ -n "${CONFIG}" ]]; then
    LLM_REPAIR_CMD+=(--config "${CONFIG}")
  fi
  run_stage llm_repair "${LLM_REPAIR_CMD[@]}"
  fi
fi

# 6) Evaluation (TravelPlanner only)
EVAL_ELAPSED=0
if [[ "${RUN_EVAL}" == "1" ]]; then
  if [[ "${TASK}" != "travel" ]]; then
    echo "Skipping evaluation: task=${TASK} (only travel is supported in this runner)." | tee -a "${COST_TXT}"
  elif [[ "${SPLIT}" == "test" ]]; then
    # Official hidden-label evaluation for TEST is hosted on the public leaderboard Space.
    EVAL_START="$(seconds_now)"
    run_stage eval \
      python -m task_helper.travel.runners.eval_leaderboard --split test --eval-mode two-stage --submission "${OPTIMIZED_JSONL}"
    EVAL_END="$(seconds_now)"
    EVAL_ELAPSED="$((EVAL_END - EVAL_START))"
  else
    # Stream output to terminal and write full report.
    EVAL_START="$(seconds_now)"
    run_stage eval \
      python -m task_helper.travel.runners.eval_bridge --set-type "${SPLIT}" --submission "${OPTIMIZED_JSONL}"
    EVAL_END="$(seconds_now)"
    EVAL_ELAPSED="$((EVAL_END - EVAL_START))"
  fi
fi

PIPELINE_END="$(seconds_now)"
PIPELINE_ELAPSED_TOTAL="$((PIPELINE_END - PIPELINE_START))"
PIPELINE_ELAPSED_NO_EVAL="$((PIPELINE_ELAPSED_TOTAL - EVAL_ELAPSED))"
if [[ "${PIPELINE_ELAPSED_NO_EVAL}" -lt 0 ]]; then
  PIPELINE_ELAPSED_NO_EVAL=0
fi
log_cost "run_end=$(iso_now)"
log_cost "pipeline_elapsed_s_total=${PIPELINE_ELAPSED_TOTAL}"
log_cost "pipeline_elapsed_s_no_eval=${PIPELINE_ELAPSED_NO_EVAL}"
log_cost "eval_elapsed_s=${EVAL_ELAPSED}"

# Best-effort whole-pipeline token total (init_template + llm_repair).
PIPELINE_TOTAL_TOKENS=""
if [[ -n "${INIT_TOTAL_TOKENS:-}" ]]; then
  PIPELINE_TOTAL_TOKENS="${INIT_TOTAL_TOKENS}"
fi
if [[ -n "${LLM_REPAIR_TOKENS_LINE:-}" ]]; then
  REPAIR_TOTAL="$(echo "${LLM_REPAIR_TOKENS_LINE}" | sed -n 's/.*total=\([0-9][0-9]*\).*/\1/p')"
  if [[ -n "${REPAIR_TOTAL}" ]]; then
    if [[ -n "${PIPELINE_TOTAL_TOKENS}" ]]; then
      PIPELINE_TOTAL_TOKENS="$((PIPELINE_TOTAL_TOKENS + REPAIR_TOTAL))"
    else
      PIPELINE_TOTAL_TOKENS="${REPAIR_TOTAL}"
    fi
    log_cost "llm_repair_total_tokens=${REPAIR_TOTAL}"
  fi
fi
if [[ -n "${PIPELINE_TOTAL_TOKENS}" ]]; then
  log_cost "pipeline_total_tokens=${PIPELINE_TOTAL_TOKENS}"
fi

log_cost "outputs_dir=${OUTPUT_DIR}"
log_cost "summary: time_s_no_eval=${PIPELINE_ELAPSED_NO_EVAL} tokens_total=${PIPELINE_TOTAL_TOKENS:-0} tool_calls_cost=${TOOL_CALLS_TOTAL_COST:-0}"

# 7) LLM $ cost (best-effort; appends to cost.txt)
(cd "${REPO_ROOT}" && python -m task_helper.money_memplan "${COST_TXT}" >/dev/null 2>&1) || log_cost "llm_money_error=1"

echo "Done. Outputs: ${OUTPUT_DIR}"
echo "Summary: time_s_no_eval=${PIPELINE_ELAPSED_NO_EVAL} tokens_total=${PIPELINE_TOTAL_TOKENS:-0} tool_calls_cost=${TOOL_CALLS_TOTAL_COST:-0}"
