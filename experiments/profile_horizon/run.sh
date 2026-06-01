#!/usr/bin/env bash
# Experiment: memory and throughput vs horizon length.
#
# Sweeps horizon over [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192] for all
# 4 encoder architectures (LOM only — LAM uses horizon=1 by definition).
# Runs once per context length listed in CTX_LENGTHS.
# Produces one JSON per (encoder, context_len) pair in OUT_DIR.
# Results feed into scripts/plot_pareto.py (horizon.png).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash experiments/profile_horizon/run.sh [--out-dir DIR] [--no-compile]
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
      "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
  done
done

echo "===== profile_horizon complete ====="
