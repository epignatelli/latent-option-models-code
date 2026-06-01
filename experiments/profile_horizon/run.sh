#!/usr/bin/env bash
# Experiment: memory and throughput vs horizon length.
#
# Sweeps horizon over [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192] for all
# 4 encoder architectures (LOM only — LAM uses horizon=1 by definition).
# Runs once per context length listed in CTX_LENGTHS.
# Produces one JSON per (encoder, context_len) pair in --out-dir.
# Results feed into scripts/plot_pareto.py (horizon.png).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash experiments/profile_horizon/run.sh [--out-dir DIR] [--no-compile]
#   CUDA_VISIBLE_DEVICES=0 bash experiments/profile_horizon/run.sh --out-dir profiling_results/ --no-compile
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

OUT_DIR="profiling_results"
EXTRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --out-dir) ;;
    --no-compile) EXTRA_ARGS+=("--no-compile") ;;
  esac
done
i=0; args=("$@")
while [ $i -lt ${#args[@]} ]; do
  if [ "${args[$i]}" = "--out-dir" ]; then
    i=$((i+1)); OUT_DIR="${args[$i]}"
  fi
  i=$((i+1))
done

mkdir -p "${OUT_DIR}"

ENCODERS=(reconstruction latent latent-medium latent-params)
CTX_LENGTHS=(4 16)

echo "===== profile_horizon — ${#ENCODERS[@]} encoders × ${#CTX_LENGTHS[@]} context lengths (LOM only) ====="
echo "  out-dir: ${OUT_DIR}"

for ctx in "${CTX_LENGTHS[@]}"; do
  for encoder in "${ENCODERS[@]}"; do
    out="${OUT_DIR}/horizon_lom_${encoder}_ctx${ctx}.json"
    echo "  encoder=${encoder}  ctx=${ctx}  -> ${out}"
    python -m scripts.profile_memory \
      --horizon-sweep \
      --encoder      "${encoder}" \
      --context-len  "${ctx}" \
      --json-out     "${out}" \
      "${EXTRA_ARGS[@]}"
  done
done

echo "===== profile_horizon complete ====="
