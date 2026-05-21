#!/usr/bin/env bash
# Experiment: LOM codebook size scaling
#
# Sweeps num_options over [4, 8, 16, 32, 64, 128, 512, 1024, 2048, 4096,
# 8192, 16384, 32768] with 3 seeds each.
# All other hyperparameters are fixed (see config.yaml).
#
# Deployment overrides (environment variables):
#   NLE_DATA_DIR   path to nle_data/    (default: nle_data)
#   DATASET        top10 | full         (default: top10)
#   BATCH_SIZE     samples per step     (default: 32)
#   NUM_WORKERS    DataLoader workers   (default: 4)
set -euo pipefail

NLE_DATA_DIR=${NLE_DATA_DIR:-nle_data}
DATASET=${DATASET:-top10}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-4}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

CONFIG=experiments/lom_scale/config.yaml

BASE_CKPT=$(python -c "
import yaml, os
d = yaml.safe_load(open('$CONFIG'))
print(os.path.expandvars(d['train']['ckpt_dir']))
")

BASE=(
  --config "$CONFIG"
  --data.nle_data_dir "$NLE_DATA_DIR"
  --data.dataset      "$DATASET"
  --train.batch_size  "$BATCH_SIZE"
  --data.num_workers  "$NUM_WORKERS"
)

SIZES=(4 8 16 32 64 128 512 1024 2048 4096 8192 16384 32768)
SEEDS=(42 43 44)

for size in "${SIZES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "=== LOM num_options=${size}  seed=${seed} ==="
    python -m scripts.pretrain lom "${BASE[@]}" \
      --model.num_options "$size" \
      --train.seed        "$seed" \
      --train.ckpt_dir    "$BASE_CKPT/options${size}_seed${seed}" \
      --wandb.group       lom_scale
  done
done
