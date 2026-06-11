#!/usr/bin/env bash
# K-fold cross-validation: trains K models (one per held-out fold) from one config.
# Folds are derived deterministically from the seed -> reproducible, no split file.
#
#   bash scripts/run_kfold.sh configs/lora_b_plain.yaml 5
#   bash scripts/run_kfold.sh configs/lora_b_plain.yaml 5 batch_size=128 head_lr=1.4e-3
set -euo pipefail
CONFIG="$1"; K="$2"; shift 2
NAME="$(basename "$CONFIG" .yaml)"
for ((f=0; f<K; f++)); do
  echo "===== fold $f / $K ====="
  uv run python scripts/train.py --config "$CONFIG" \
    --override experiment_name="kfold_${NAME}_f${f}" n_folds="$K" fold="$f" "$@"
done
echo "done. ensemble with: uv run python scripts/ensemble.py --glob 'kfold_${NAME}_f*'"
