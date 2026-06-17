#!/usr/bin/env bash
# Experiment: does temporal abstraction improve latent option quality?
#
# Runs LAM (baseline, horizon=1) and LOM (proposed, horizon=128) across
# multiple seeds (see config.yaml).
# If 2+ GPUs are available, fills one slot per GPU, waits when all are busy.
#
# Usage:
#   bash experiments/benchmark/run.sh [--lam-only] [--force]
#
#   --lam-only  run only the LAM condition (horizon=1)
#   --force     re-run jobs even if a 'done' sentinel exists
set -euo pipefail

FORCE=0
LAM_ONLY=0
for _arg in "$@"; do
  [ "$_arg" = "--force" ]    && FORCE=1
  [ "$_arg" = "--lam-only" ] && LAM_ONLY=1
done

export TORCHINDUCTOR_FX_GRAPH_CACHE=1  # reuse compiled kernels across seeds
export OMP_NUM_THREADS=1               # prevent libgomp thread explosion on GPU training
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CFG="${SCRIPT_DIR}/config.yaml"
cd "${ROOT}"

_cfg() { python3 -c "
import yaml, os
c = yaml.safe_load(open('${CFG}'))
v = c
for k in '$1'.split('.'):
    v = v[k]
expand = lambda s: os.path.expandvars(str(s))
print(' '.join(expand(str(x)) for x in v) if isinstance(v, list) else expand(v))
"; }

_done() {
  [ "${FORCE}" = "1" ] && return 1
  [ -f "$1/done" ] || return 1
  echo "    skipping — $1/done sentinel exists"
}

# ---------------------------------------------------------------------------
# GPU parallelism
# ---------------------------------------------------------------------------
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
else
  NUM_DETECTED=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)
  GPU_IDS=()
  for i in $(seq 0 $((NUM_DETECTED - 1))); do GPU_IDS+=("$i"); done
fi
NUM_GPUS=${#GPU_IDS[@]}
echo "Using ${NUM_GPUS} GPU(s): ${GPU_IDS[*]}"

_GPU_SLOT=0
_PIDS=()

_COMPILE_STAGGER=600  # 10 min — stagger backward-graph compilations so they don't overlap

_launch() {
  if [ "${NUM_GPUS}" -ge 2 ]; then
    if [ ${#_PIDS[@]} -ge "${NUM_GPUS}" ]; then
      wait -n
      local alive=()
      for pid in "${_PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null && alive+=("$pid")
      done
      _PIDS=("${alive[@]+"${alive[@]}"}")
    fi
    setsid env CUDA_VISIBLE_DEVICES=${GPU_IDS[${_GPU_SLOT}]} "$@" &
    _PIDS+=($!)
    _GPU_SLOT=$(( (_GPU_SLOT + 1) % NUM_GPUS ))
    sleep "${_COMPILE_STAGGER}" & wait $!
  else
    "$@"
  fi
}

_flush() {
  if [ ${#_PIDS[@]} -gt 0 ]; then
    wait "${_PIDS[@]}"
    _PIDS=()
  fi
}

# ---------------------------------------------------------------------------
# SIGINT/SIGTERM handler — kill all background jobs then exit
# ---------------------------------------------------------------------------
_cleanup() {
  echo ""
  echo "Caught signal — killing all background jobs..."
  for pid in "${_PIDS[@]}"; do
    kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  done
  exit 1
}
trap _cleanup SIGINT SIGTERM

# ---------------------------------------------------------------------------
CKPT_ROOT=$(_cfg train.ckpt_dir)
read -ra SEEDS        <<< "$(_cfg sweep.seeds)"
LAM_BATCH=$(_cfg sweep.lam_batch_size)
LOM_BATCH=$(_cfg sweep.lom_batch_size)

# ---------------------------------------------------------------------------
echo "===== benchmark — LAM vs LOM, ${#SEEDS[@]} seeds ====="

for seed in "${SEEDS[@]}"; do
  echo "  === seed=${seed} ==="

  CKPT_LAM="${CKPT_ROOT}/lam_seed${seed}"
  mkdir -p "${CKPT_LAM}"
  echo "  LAM  horizon=1  num_options=100  batch=${LAM_BATCH}"
  if ! _done "${CKPT_LAM}"; then
    _launch bash -c "python3 -m scripts.pretrain --method lam --signal latent \
      --config              '${CFG}' \
      --train.seed          '${seed}' \
      --train.ckpt_dir      '${CKPT_LAM}' \
      --train.batch_size    '${LAM_BATCH}' \
      2>&1 | tee '${CKPT_LAM}/train.log' \
      && touch '${CKPT_LAM}/done'"
  fi

  if [ "${LAM_ONLY}" = "0" ]; then
    CKPT_LOM="${CKPT_ROOT}/lom_seed${seed}"
    mkdir -p "${CKPT_LOM}"
    echo "  LOM  horizon=128  stride=4  num_options=256  batch=${LOM_BATCH}"
    if ! _done "${CKPT_LOM}"; then
      _launch bash -c "python3 -m scripts.pretrain --method lom --signal latent \
        --config              '${CFG}' \
        --train.seed          '${seed}' \
        --train.ckpt_dir      '${CKPT_LOM}' \
        --train.batch_size    '${LOM_BATCH}' \
        2>&1 | tee '${CKPT_LOM}/train.log' \
        && touch '${CKPT_LOM}/done'"
    fi
  fi
done

_flush
echo "===== benchmark complete ====="
