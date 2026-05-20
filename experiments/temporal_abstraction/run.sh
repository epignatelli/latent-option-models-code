#!/usr/bin/env bash
# Experiment: temporal abstraction
#
# Tests whether a latent option spanning k>1 steps (LOM) produces better
# representations than a single-step latent action code (LAM, baseline).
#
# Conditions:
#   LAM  — horizon=1,   num_options=98   (option VQ = one code per atomic action)
#   LOM  — horizon=128, num_options=256  (option VQ spans 128 steps)
#
# Both conditions share the same unified architecture (OptionEncoder + ActionEncoder
# + shared FrameDecoder) and train in a single stage.
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

CONFIG=experiments/temporal_abstraction/config.yaml

# Read base checkpoint dir from config (expands $USER and other env vars)
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

# ---------------------------------------------------------------------------
echo "=== LAM (baseline): single encoder, horizon=1, codebook=98 ==="
python -m scripts.pretrain "${BASE[@]}" \
  --model.model_type lam --data.horizon 1 --model.num_options 98 \
  --train.ckpt_dir "$BASE_CKPT/lam" \
  --wandb.group benchmark_lam

# ---------------------------------------------------------------------------
echo "=== LOM (proposed): two encoders + two dynamics, horizon=128, codebook=256 ==="
python -m scripts.pretrain "${BASE[@]}" \
  --model.model_type lom \
  --train.ckpt_dir "$BASE_CKPT/lom" \
  --wandb.group benchmark_lom
