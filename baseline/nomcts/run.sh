#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-travel}"                     # travel | work
SPLIT="${SPLIT:-validation}"
MODEL="${MODEL:-gpt-5.2}"
WORKERS="${WORKERS:-0}"
SEED="${SEED:-0}"
LIMIT="${LIMIT:-0}"
MAX_ROUNDS="${MAX_ROUNDS:-10}"            # work only
TOLERANCE_RATE="${TOLERANCE_RATE:-0.8}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_REPAIR="${RUN_REPAIR:-1}"
REPAIR_MODEL="${REPAIR_MODEL:-${MODEL}}"
REPAIR_WORKERS="${REPAIR_WORKERS:-${WORKERS}}"
REPAIR_MAX_SLOT_CANDIDATES="${REPAIR_MAX_SLOT_CANDIDATES:-50}"   # travel only
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-}"
PYTHON="${PYTHON:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
  fi
fi
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "Could not find a Python interpreter. Set PYTHON=python3 or your project environment." >&2
  exit 1
fi

COMMON_ARGS=(
  --split "${SPLIT}"
  --model "${MODEL}"
  --workers "${WORKERS}"
  --seed "${SEED}"
  --limit "${LIMIT}"
)

if [[ "${RUN_EVAL}" != "1" ]]; then
  COMMON_ARGS+=(--skip-eval)
fi
if [[ "${RUN_REPAIR}" == "1" ]]; then
  COMMON_ARGS+=(--run-repair)
fi
if [[ -n "${SOURCE_RUN_DIR}" ]]; then
  COMMON_ARGS+=(--source-run-dir "${SOURCE_RUN_DIR}")
fi

case "${TASK}" in
  travel)
    exec "${PYTHON}" -m baseline.nomcts.travel \
      "${COMMON_ARGS[@]}" \
      --semantic-threshold "${TOLERANCE_RATE}" \
      --repair-model "${REPAIR_MODEL}" \
      --repair-workers "${REPAIR_WORKERS}" \
      --repair-max-slot-candidates "${REPAIR_MAX_SLOT_CANDIDATES}"
    ;;
  work)
    exec "${PYTHON}" -m baseline.nomcts.work \
      "${COMMON_ARGS[@]}" \
      --max-rounds "${MAX_ROUNDS}" \
      --semantic-threshold "${TOLERANCE_RATE}" \
      --repair-model "${REPAIR_MODEL}" \
      --repair-workers "${REPAIR_WORKERS}"
    ;;
  *)
    echo "Unsupported TASK=${TASK}. Expected travel or work." >&2
    exit 1
    ;;
esac
