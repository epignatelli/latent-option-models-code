#!/usr/bin/env bash
# Experiment: LOM codebook size scaling
#
# Sweeps num_options over a wide range with multiple seeds (see config.yaml).
# All other hyperparameters are held fixed.
# If 2+ GPUs are available, fills one slot per GPU, waits when all are busy.
#
# Usage:
#   bash run.sh [--force]
#
#   --force   re-run jobs even if a 'done' sentinel exists
set -euo pipefail

FORCE=0
for _arg in "$@"; do [ "$_arg" = "--force" ] && FORCE=1; done

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

_COMPILE_STAGGER=30  # seconds between launches to stagger torch.compile RAM spikes

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
    CUDA_VISIBLE_DEVICES=${GPU_IDS[${_GPU_SLOT}]} "$@" &
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
  if [ ${#_PIDS[@]} -gt 0 ]; then
    kill "${_PIDS[@]}" 2>/dev/null || true
  fi
  exit 1
}
trap _cleanup SIGINT SIGTERM

# ---------------------------------------------------------------------------
CKPT_ROOT=$(_cfg train.ckpt_dir)
read -ra NUM_OPTIONS_LIST <<< "$(_cfg sweep.num_options_list)"
read -ra SEEDS <<< "$(_cfg sweep.seeds)"

# ---------------------------------------------------------------------------
echo "===== LOM codebook size scaling — ${#NUM_OPTIONS_LIST[@]} sizes × ${#SEEDS[@]} seeds ====="

for size in "${NUM_OPTIONS_LIST[@]}"; do
  for seed in "${SEEDS[@]}"; do
    CKPT_DIR="${CKPT_ROOT}/options${size}_seed${seed}"
    echo "  num_options=${size}  seed=${seed}  ckpt=${CKPT_DIR}"
    if ! _done "${CKPT_DIR}"; then
      _launch python3 -m scripts.pretrain lom \
        --config              "${CFG}" \
        --model.num_options   "${size}" \
        --train.seed          "${seed}" \
        --train.ckpt_dir      "${CKPT_DIR}" \
      && touch "${CKPT_DIR}/done"
    fi
  done
done

_flush
echo "===== lom_scale complete ====="
