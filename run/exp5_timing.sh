#!/usr/bin/env bash
# exp5 uncached end-to-end timing on GPU 0 (run when no other exp5 job uses GPU 0; another user's
# light job stays resident and is noted in every record).  Same precision on both sides for every
# speed comparison: student vs teacher at fp32 AND at TF32; batch 16 for all methods (the teacher at
# batch 64 would take ~80 min), plus batch 64 for the 10-evaluation methods.
#
#   tmux new-session -d -s exp5-timing 'bash run/exp5_timing.sh'
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${GPU:-0}
M="python -m gmmd.experiments.ibw_amortized"
OUT=outputs/exp5_ibw_amortized_10step_k8
status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/RUNLOG"; }
TAG=${TAG:-main}
{
  for p in tf32 fp32; do
    for n in 16 64; do
      $M timing --method student:$TAG --precision $p --n $n
      $M timing --method coarse:ancestral:10 --precision $p --n $n
    done
    $M timing --method teacher --precision $p --n 16
  done
} >> run/logs/exp5_timing.log 2>&1 && status "timing done ($TAG)" || status "timing FAILED"
