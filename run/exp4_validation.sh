#!/usr/bin/env bash
# exp4 stage 1: interval validation on local GPUs (resumable per shard), then the frozen selection.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p run/logs
export PYTHONUNBUFFERED=1
GPUS=(${GPUS:-0 1})
N=${#GPUS[@]}
pids=()
for k in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPUS[$k]} python -m gmmd.experiments.interval_validation \
      --config exp4_interval_replacement --shard "$k" --n-shards "$N" \
      >> "run/logs/exp4_validation_shard$k.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
(( rc == 0 )) || { echo "a validation shard failed" >&2; exit 1; }
python -m gmmd.experiments.interval_validation --config exp4_interval_replacement --summarize
