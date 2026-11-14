#!/usr/bin/env bash

export OPENAI_API_BASE="http://127.0.0.1:8003/v1"
export OPENAI_API_KEY="EMPTY"
export MODEL_NAME="Qwen2.5-Coder-7B-Instruct"
export MAX_TOKENS="16384"
TEMPERATURE="0"
DATA_ROOT="../SAGE-SQL/datasets/spider"
RUN_ID="run_1"
OUTPUT_DIR="outputs/dinsql_spider_dev_${MODEL_NAME}_${RUN_ID}"
WORKERS="8"

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python run_bird_dev.py \
    --dataset spider \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --workers "$WORKERS" \
    "$@"
