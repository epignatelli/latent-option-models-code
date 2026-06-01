#!/usr/bin/env bash
# Experiment: memory and throughput vs context length.
#
# Sweeps context_len over [4, 8, 16, 32, 64, 128, 256] for all 4 encoder
# architectures × LAM and LOM. Produces one JSON per (method, encoder) pair
# in OUT_DIR. Results feed into scripts/plot_pareto.py (context_length.png).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash experiments/profile_context/run.sh [--out-dir DIR] [--no-compile]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${ROOT}/profiling_results"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)    OUT_DIR="$2"; shift 2 ;;
    --no-compile) EXTRA_ARGS+=("--no-compile"); shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
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
      "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
  done
done

echo "===== profile_context complete ====="
