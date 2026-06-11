#!/usr/bin/env bash
# Overnight HP sweep: one deliberate run per line in RUNS (edit freely).
# A failed run does NOT abort the sweep. Each run saves history.csv + config.json.
# Compare afterwards with: uv run python scripts/summarize.py --prefix sweep_
#
#   bash scripts/sweep.sh configs/lora_b_plain.yaml batch_size=128 num_workers=16
#   (epochs are taken from the config = same horizon as the final k-fold)
CONFIG="${1:-configs/lora_b_plain.yaml}"
BASE="$(basename "$CONFIG" .yaml)"
EXTRA=("${@:2}")   # fixed overrides applied to every run (speed/infra)

# Overrides that VARY, one run per line. config.json records the full HPs.
RUNS=(
  # --- LR grid (primary lever) ---
  "backbone_lr=1e-4 head_lr=5e-4"
  "backbone_lr=1e-4 head_lr=1e-3"
  "backbone_lr=2e-4 head_lr=1e-3"
  "backbone_lr=2e-4 head_lr=2e-3"
  "backbone_lr=4e-4 head_lr=1e-3"
  "backbone_lr=4e-4 head_lr=2e-3"
  "backbone_lr=8e-4 head_lr=2e-3"
  # --- weight decay (at a central LR) ---
  "backbone_lr=2e-4 head_lr=1e-3 weight_decay=1e-2"
  "backbone_lr=2e-4 head_lr=1e-3 weight_decay=1e-5"
  # --- LoRA capacity / targets ---
  "backbone_lr=2e-4 head_lr=1e-3 lora_r=32 lora_alpha=64"
  "backbone_lr=2e-4 head_lr=1e-3 lora_r=8 lora_alpha=16"
  "backbone_lr=2e-4 head_lr=1e-3 lora_targets=qkv,proj,fc1,fc2"
)

i=0
for ov in "${RUNS[@]}"; do
  name="$(printf 'sweep_%s_%02d' "$BASE" "$i")"
  echo "===== $name | $ov ====="
  uv run scripts/train.py --config "$CONFIG" --override \
    experiment_name="$name" $ov "${EXTRA[@]}" || echo "FAILED: $name"
  i=$((i + 1))
done
echo "done. compare: uv run python scripts/summarize.py --prefix sweep_"
