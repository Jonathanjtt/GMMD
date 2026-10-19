#!/usr/bin/env bash
# exp4 stage 2: prefix cache -> three methods -> evaluation, sharded over local GPUs.
# Every step is resumable per 64-image chunk; rerunning skips finished chunks.
#
#   N_CHUNKS=3  TAG=pilot run/exp4_generate.sh      # 192-image pilot
#   N_CHUNKS=16 TAG=n1024 run/exp4_generate.sh      # 1,024 images per method
#   N_CHUNKS=79 TAG=n5056 run/exp4_generate.sh      # 5k extension (set generation.n_images too)
#   E2E=1 ...                                        # also time uncached x_T -> x_0, 64 images per method
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p run/logs
export PYTHONUNBUFFERED=1
GPUS=(${GPUS:-0 1})
NS=${#GPUS[@]}
N_CHUNKS=${N_CHUNKS:-3}
TAG=${TAG:-pilot}
M="python -m gmmd.experiments.interval_replacement"

wait_all() {  # wait for the given pids; fail if any failed
  local rc=0
  for p in "$@"; do wait "$p" || rc=1; done
  return $rc
}

sharded() {   # sharded <logname> <args...>
  local name=$1; shift
  local pids=()
  for k in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[$k]} $M "$@" --n-chunks "$N_CHUNKS" --shard "$k" --n-shards "$NS" \
        >> "run/logs/exp4_${name}_shard$k.log" 2>&1 &
    pids+=($!)
  done
  wait_all "${pids[@]}"
}

sharded prefix prefix
for method in teacher coarse gmmd; do
  sharded "$method" method --method "$method"
done
if [[ "${E2E:-0}" == 1 ]]; then
  # uncached x_T -> x_0, one method per GPU at a time, same batch size as the cached runs
  methods=(teacher coarse gmmd)
  for ((i = 0; i < ${#methods[@]}; i += NS)); do
    pids=()
    for ((k = 0; k < NS && i + k < ${#methods[@]}; k++)); do
      m=${methods[$((i + k))]}
      [[ -f outputs/exp4_interval_replacement/generation/e2e/$m.json ]] && continue
      CUDA_VISIBLE_DEVICES=${GPUS[$k]} $M e2e --method "$m" >> "run/logs/exp4_e2e_$m.log" 2>&1 &
      pids+=($!)
    done
    (( ${#pids[@]} )) && wait_all "${pids[@]}"
  done
fi
CUDA_VISIBLE_DEVICES=${GPUS[0]} $M evaluate --n-chunks "$N_CHUNKS" --tag "$TAG" >> "run/logs/exp4_evaluate_$TAG.log" 2>&1
$M table --tag "$TAG"
