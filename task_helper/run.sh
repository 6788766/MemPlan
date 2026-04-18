#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Task runner dispatcher
#
# This script dispatches to `task_helper/<task>/run.sh` while allowing you to set
# common pipeline variables here (or via CLI) to override task-specific defaults.
#
# Usage:
#   bash task_helper/run.sh --task work --set MODEL=gpt-5-mini --set SPLIT=validation
#   bash task_helper/run.sh --task work --set RUN_LLM_REPAIR=1 --set REPAIR_MODEL=gpt-5-mini
#
# Notes:
# - Task-specific runners still define their own defaults via ${VAR:-default}.
# - Any values you export here (or via `--set`) override those defaults.
# - Unknown variables are simply passed through as environment variables.
# -----------------------------------------------------------------------------

TASK="${TASK:-work}"

# Global override variables (edit or set via `--set KEY=VALUE`):
#   Common: TASK, SPLIT, MODEL, CPUS, TOLERANCE_RATE, MCTS_ITERATIONS, MCTS_UCT_C
#   Work: MODE, MULTI_WORKERS, RUN_INIT_TEMPLATE, RUN_VIEW_SELECT, RUN_COMPOSE_MATCH,
#         RUN_TWIN_TRACK, RUN_CONVERT, RUN_EVAL, RUN_LLM_REPAIR, REPAIR_MODEL,
#         REPAIR_WORKERS, REPAIR_MAX_FAILED, REPAIR_NO_LLM
#   Travel: CONFIG, MAX_TOOL_CANDIDATES, RUN_INIT_TEMPLATE, RUN_REPAIR, RUN_EVAL,
#          KEEP_PRE_REPAIR_OPTIMIZED, VIEW_SELECT_STRICT, REPAIR_MAX_SLOT_CANDIDATES, REPAIR_TREE

print_help() {
  cat <<'EOF'
task_helper/run.sh

Dispatches to a task-specific runner: task_helper/<task>/run.sh

Options:
  --task <name>            Set TASK (e.g. work, travel)
  --set KEY=VALUE          Export an env var override (repeatable)
  --help                   Show this help
  --                       Stop parsing flags; pass remaining args to the runner

Examples:
  bash task_helper/run.sh --task work --set MODEL=gpt-5-mini --set SPLIT=validation
  bash task_helper/run.sh --task work --set RUN_LLM_REPAIR=1 --set REPAIR_MODEL=gpt-5-mini
  bash task_helper/run.sh --task travel --set MODEL=deepseek-chat --set RUN_REPAIR=1
EOF
}

passthrough=()
while (( "$#" )); do
  case "$1" in
    --help|-h)
      print_help
      exit 0
      ;;
    --task)
      shift
      if (( "$#" == 0 )); then
        echo "Missing value for --task" >&2
        exit 2
      fi
      TASK="$1"
      shift
      ;;
    --set)
      shift
      if (( "$#" == 0 )); then
        echo "Missing value for --set (expected KEY=VALUE)" >&2
        exit 2
      fi
      kv="$1"
      shift
      if [[ "${kv}" != *=* ]]; then
        echo "Invalid --set '${kv}' (expected KEY=VALUE)" >&2
        exit 2
      fi
      key="${kv%%=*}"
      value="${kv#*=}"
      if [[ ! "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
        echo "Invalid env var name '${key}'" >&2
        exit 2
      fi
      export "${key}=${value}"
      ;;
    --)
      shift
      passthrough+=("$@")
      break
      ;;
    *)
      passthrough+=("$1")
      shift
      ;;
  esac
done

# Default overrides (only when TASK=work, and only if not already provided).
if [[ "${TASK}" == "work" ]]; then
  : "${MODEL:=gpt-5.2}"
  : "${SPLIT:=validation}"
  : "${RUN_INIT_TEMPLATE:=0}"
  : "${RUN_VIEW_SELECT:=0}"
  : "${RUN_COMPOSE_MATCH:=0}"
  : "${RUN_TWIN_TRACK:=0}"
  : "${RUN_CONVERT:=0}"
  : "${RUN_EVAL:=0}"
  : "${RUN_LLM_REPAIR:=1}"
  : "${REPAIR_MODEL:=gpt-5.2}"
  export MODEL SPLIT RUN_INIT_TEMPLATE RUN_VIEW_SELECT RUN_COMPOSE_MATCH RUN_TWIN_TRACK RUN_CONVERT RUN_EVAL RUN_LLM_REPAIR REPAIR_MODEL
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/${TASK}/run.sh"

if [[ -f "${RUNNER}" ]]; then
  export TASK
  if (( ${#passthrough[@]} )); then
    exec bash "${RUNNER}" "${passthrough[@]}"
  else
    exec bash "${RUNNER}"
  fi
fi

echo "No task runner found for TASK='${TASK}'." >&2
echo "Expected: ${RUNNER}" >&2
echo "Available task runners:" >&2

shopt -s nullglob
RUNNERS=("${SCRIPT_DIR}"/*/run.sh)
if (( ${#RUNNERS[@]} == 0 )); then
  echo "  (none)" >&2
else
  for path in "${RUNNERS[@]}"; do
    echo "  - $(basename "$(dirname "${path}")")" >&2
  done
fi
exit 1
