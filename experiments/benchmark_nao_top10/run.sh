#!/usr/bin/env bash
# Benchmark: LAM vs LOM on NAO-TOP10, 3 seeds.
# Fills one GPU slot per job; waits when all GPUs are busy.
#
# Usage:
#   bash experiments/benchmark_nao_top10/run.sh [--force]
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
_COMPILE_STAGGER=30

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

_cleanup() {
  echo ""
  echo "Caught signal — killing all background jobs..."
  [ ${#_PIDS[@]} -gt 0 ] && kill "${_PIDS[@]}" 2>/dev/null || true
  exit 1
}
trap _cleanup SIGINT SIGTERM

# ---------------------------------------------------------------------------
CKPT_ROOT=$(_cfg train.ckpt_dir)
read -ra SEEDS <<< "$(_cfg sweep.seeds)"

echo "===== benchmark_nao_top10 — LAM vs LOM, ${#SEEDS[@]} seeds ====="

for seed in "${SEEDS[@]}"; do
  echo "  === seed=${seed} ==="

  CKPT_LAM="${CKPT_ROOT}/lam_seed${seed}"
  echo "  LAM  horizon=1  num_options=98"
  if ! _done "${CKPT_LAM}"; then
    _launch bash -c "python3 -m scripts.pretrain lam \
      --config             ${CFG} \
      --data.horizon       1 \
      --model.num_options  98 \
      --train.seed         ${seed} \
      --train.ckpt_dir     ${CKPT_LAM} \
      && touch ${CKPT_LAM}/done"
  fi

  CKPT_LOM="${CKPT_ROOT}/lom_seed${seed}"
  echo "  LOM  horizon=128  num_options=256"
  if ! _done "${CKPT_LOM}"; then
    _launch bash -c "python3 -m scripts.pretrain lom \
      --config             ${CFG} \
      --train.seed         ${seed} \
      --train.ckpt_dir     ${CKPT_LOM} \
      && touch ${CKPT_LOM}/done"
  fi
done

_flush
echo "===== benchmark_nao_top10 complete ====="
