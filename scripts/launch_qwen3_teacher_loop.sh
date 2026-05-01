#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
LOOP_ID="${RAYUELA_QWEN3_LOOP_ID:-qwen3-teacher-loop-$(date +%Y%m%d%H%M%S)}"
SESSION="${RAYUELA_QWEN3_LOOP_SESSION:-rayuela-qwen3-teacher-loop}"
START_OFFSET="${RAYUELA_QWEN3_START_OFFSET:-3}"
LOOP_DIR="${ROOT_DIR}/outputs/reconstruction/analysis/agentic_loops/${LOOP_ID}"
SEED_CASES="${ROOT_DIR}/outputs/reconstruction/runs/phase4-qwen3-instruct-3cases-20260501a/prompt_baseline_cases.json"
EXTRA_CASES="${RAYUELA_QWEN3_EXTRA_CASES:-}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required to launch the autonomous teacher loop" >&2
  exit 1
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  echo "Attach with: tmux attach -t ${SESSION}" >&2
  exit 2
fi

mkdir -p "${LOOP_DIR}"

COMMAND=(
  "${PYTHON_BIN}"
  "src/reconstruction_agentic_loop.py"
  "--loop-id" "${LOOP_ID}"
  "--forever"
  "--start-offset" "${START_OFFSET}"
  "--include-teacher-cases-path" "${SEED_CASES}"
  "--sleep-seconds" "30"
)

if [[ -n "${EXTRA_CASES}" ]]; then
  IFS=':' read -r -a EXTRA_CASE_PATHS <<< "${EXTRA_CASES}"
  for extra_path in "${EXTRA_CASE_PATHS[@]}"; do
    COMMAND+=("--include-teacher-cases-path" "${extra_path}")
  done
fi

tmux new-session -d -s "${SESSION}" -c "${ROOT_DIR}" \
  "$(printf "%q " "${COMMAND[@]}") >> \"${LOOP_DIR}/loop.console.log\" 2>&1"

cat <<EOF
Started ${SESSION}
Loop ID: ${LOOP_ID}
Loop dir: ${LOOP_DIR}
Start offset: ${START_OFFSET}
Attach: tmux attach -t ${SESSION}
Status: tail -f ${LOOP_DIR}/events.jsonl
Stop: touch ${LOOP_DIR}/STOP
EOF
