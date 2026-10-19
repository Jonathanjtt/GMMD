#!/usr/bin/env bash
# exp5 pilot, part 1: anchor trajectories (all splits), P0 step-size check, and the fp32 teacher
# reference rollouts for every validation (trajectory, interval) pair.  Resumable: rerun to finish.
#
#   tmux new-session -d -s exp5-gpu 'bash run/exp5_pilot_p0.sh'
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p run/logs
export PYTHONUNBUFFERED=1
M="python -m gmmd.experiments.ibw_amortized"
OUT=outputs/exp5_ibw_amortized_10step_k8
status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/RUNLOG"; }

gpu0() {
  CUDA_VISIBLE_DEVICES=0 $M anchors --split val || return 1
  CUDA_VISIBLE_DEVICES=0 $M refs --split val --trajs 0 1 --intervals 1 4 7 || return 1
  status "P0 refs done"
  CUDA_VISIBLE_DEVICES=0 $M refs --split val || return 1
  status "val refs done"
}

gpu1() {
  until [[ $(ls $OUT/anchors/val/traj*.json 2>/dev/null | wc -l) -ge 4 ]]; do sleep 30; done
  for c in 0p2 0p5 1p0; do
    CUDA_VISIBLE_DEVICES=1 $M fit --split val --variant p0_c$c --trajs 0 1 --intervals 1 4 7 || return 1
  done
  status "P0 fits done"
  CUDA_VISIBLE_DEVICES=1 $M anchors --split test || return 1
  CUDA_VISIBLE_DEVICES=1 $M anchors --split train || return 1
  status "test + train anchors done"
}

status "pilot P0 chain started"
gpu0 > run/logs/exp5_p0_gpu0.log 2>&1 &
p0=$!
gpu1 > run/logs/exp5_p0_gpu1.log 2>&1 &
p1=$!
wait $p0 || status "GPU0 chain FAILED (run/logs/exp5_p0_gpu0.log)"
wait $p1 || status "GPU1 chain FAILED (run/logs/exp5_p0_gpu1.log)"
status "pilot P0 chain finished"
