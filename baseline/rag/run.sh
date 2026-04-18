#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-travel}"                     # travel | work
SPLIT="${SPLIT:-validation}"
MODEL="${MODEL:-}"
WORKERS="${WORKERS:-8}"
TOP_K="${TOP_K:-}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-1200}"
LIMIT="${LIMIT:-0}"
TOOL_SELECTION="${TOOL_SELECTION:-all}"   # work only: all | domains
RUN_EVAL="${RUN_EVAL:-1}"
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

case "${TASK}" in
  travel)
    if [[ -z "${MODEL}" ]]; then
      MODEL="gpt-5-mini"
    fi
    if [[ -z "${TOP_K}" ]]; then
      TOP_K="3"
    fi
    cmd=(
      "${PYTHON}" -m baseline.rag.travel
      --split "${SPLIT}"
      --model "${MODEL}"
      --workers "${WORKERS}"
      --top-k "${TOP_K}"
      --max-context-tokens "${MAX_CONTEXT_TOKENS}"
      --limit "${LIMIT}"
    )
    if [[ "${RUN_EVAL}" != "1" ]]; then
      cmd+=(--skip-eval)
    fi
    exec "${cmd[@]}"
    ;;
  work)
    if [[ -z "${MODEL}" ]]; then
      MODEL="gpt-5.2"
    fi
    if [[ -z "${TOP_K}" ]]; then
      TOP_K="4"
    fi
    cmd=(
      "${PYTHON}" -m baseline.rag.work
      --split "${SPLIT}"
      --model "${MODEL}"
      --workers "${WORKERS}"
      --tool-selection "${TOOL_SELECTION}"
      --top-k "${TOP_K}"
      --max-context-tokens "${MAX_CONTEXT_TOKENS}"
      --limit "${LIMIT}"
    )
    if [[ "${RUN_EVAL}" != "1" ]]; then
      cmd+=(--skip-eval)
    fi
    exec "${cmd[@]}"
    ;;
  *)
    echo "Unsupported TASK=${TASK}. Expected travel or work." >&2
    exit 1
    ;;
esac
