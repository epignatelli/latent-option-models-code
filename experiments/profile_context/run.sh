#!/usr/bin/env bash
# Experiment: memory and throughput vs context length.
#
# Sweeps context_len over [4, 8, 16, 32, 64, 128, 256] for all 4 encoder
# architectures × LAM and LOM. Produces one JSON per (method, encoder) pair
# in --out-dir. Results feed into scripts/plot_pareto.py (context_length.png).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash experiments/profile_context/run.sh [--out-dir DIR] [--no-compile]
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
# handle --out-dir VAL
i=0; args=("$@")
while [ $i -lt ${#args[@]} ]; do
  if [ "${args[$i]}" = "--out-dir" ]; then
    i=$((i+1)); OUT_DIR="${args[$i]}"
  fi
  i=$((i+1))
done

mkdir -p "${OUT_DIR}"

ENCODERS=(reconstruction latent latent-medium latent-params)
METHODS=(lam lom)

echo "===== profile_context — ${#ENCODERS[@]} encoders × ${#METHODS[@]} methods ====="
echo "  out-dir: ${OUT_DIR}"

for encoder in "${ENCODERS[@]}"; do
  for method in "${METHODS[@]}"; do
    out="${OUT_DIR}/pareto_${method}_${encoder}.json"
    echo "  encoder=${encoder}  method=${method}  -> ${out}"
    python -m scripts.profile_memory \
      --pareto \
      --method   "${method}" \
      --encoder  "${encoder}" \
      --json-out "${out}" \
      "${EXTRA_ARGS[@]}"
  done
done

echo "===== profile_context complete ====="
