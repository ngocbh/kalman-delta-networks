#!/bin/bash
set -euo pipefail

: "${WANDB_API_KEY:?set WANDB_API_KEY}"
: "${TRAIN_DATA_RAW:?set TRAIN_DATA_RAW to the Parquet directory}"
: "${TOKENIZER_PATH:?set TOKENIZER_PATH to the tokenizer directory}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT to the output directory}"

MODEL="${MODEL:-iso_kdn_1.3B}"
NAME="${NAME:-${MODEL}_100B}"
DEVICES_PER_NODE="${DEVICES_PER_NODE:-8}"
KDN_PYTHON="${KDN_PYTHON:-python}"

cd "$(dirname "$0")/.."

exec "$KDN_PYTHON" pretrain.py \
  --model_name "$MODEL" \
  --train_config tsz128x4k_100B \
  --exp_name "$NAME" \
  --output_root "$OUTPUT_ROOT" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --use_stream_tok \
  --train_data_dir_raw "$TRAIN_DATA_RAW" \
  --micro_batch_size 4 \
  --nnodes 2 \
  --devices_per_node "$DEVICES_PER_NODE" \
  "$@"
