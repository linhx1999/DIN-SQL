#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Explicit run configuration; edit these values for each experiment.
export OPENAI_API_BASE="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="EMPTY"  # Local endpoint; do not store real API keys here.
export MODEL_NAME="Ministral-3-14B-Reasoning-2512"
# export MODEL_NAME="Qwen2.5-Coder-7B-Instruct"
export MAX_TOKENS="16384"

DATA_ROOT="${SCRIPT_DIR}/../SAGE-SQL/datasets/bird/dev"
MODEL_TAG="${MODEL_NAME//\//_}"
RUN_NAME="dinsql_bird_dev_${MODEL_TAG}_run_1"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/${RUN_NAME}"
WORKERS="8"
PYTHON="python"

exec "${PYTHON}" "${SCRIPT_DIR}/run_bird_dev.py" \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_tokens "${MAX_TOKENS}" \
    --workers "${WORKERS}" \
    "$@"
