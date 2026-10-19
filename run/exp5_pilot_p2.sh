#!/usr/bin/env bash
# exp5 pilot, part 3 (P2), after the selection is frozen:
#   val frozen targets -> training fits on pilot trajectories 0-7 (both GPUs) ->
#   pilot student on GPU 0 (3,000 steps) + its stage A/B/C on val + 192-image rollouts and FID,
#   while GPU 1 starts the main training fits (trajectories 8-63; the IBW config is frozen).
#
#   tmux new-session -d -s exp5-p2 'bash run/exp5_pilot_p2.sh'
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
M="python -m gmmd.experiments.ibw_amortized"
OUT=outputs/exp5_ibw_amortized_10step_k8
mkdir -p run/logs
status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/RUNLOG"; }
[[ -f $OUT/selection.json ]] || { status "P2: no selection.json"; exit 1; }
PILOT=(0 1 2 3 4 5 6 7)
MAIN=($(seq 8 63))

status "P2 started"
CUDA_VISIBLE_DEVICES=0 $M freeze --split val >> run/logs/exp5_p2_gpu0.log 2>&1 || { status "P2 freeze FAILED"; exit 1; }
CUDA_VISIBLE_DEVICES=0 $M fit --split train --variant frozen --trajs "${PILOT[@]}" --shard 0 --n-shards 2 >> run/logs/exp5_p2_gpu0.log 2>&1 &
a=$!
CUDA_VISIBLE_DEVICES=1 $M fit --split train --variant frozen --trajs "${PILOT[@]}" --shard 1 --n-shards 2 >> run/logs/exp5_p2_gpu1.log 2>&1 &
b=$!
wait $a && wait $b || { status "P2 pilot fits FAILED"; exit 1; }
status "P2 pilot training fits done"

# GPU 1: main training fits, in the background of the pilot
( CUDA_VISIBLE_DEVICES=1 $M fit --split train --variant frozen --trajs "${MAIN[@]}" && status "main train fits (GPU1 pass) done" ) >> run/logs/exp5_main_fits_gpu1.log 2>&1 &

# GPU 0: pilot student, then its evaluation
{
  CUDA_VISIBLE_DEVICES=0 $M train --tag pilot --steps 3000 --train-trajs "${PILOT[@]}" &&
  CUDA_VISIBLE_DEVICES=0 $M evalstudent --tag pilot --split val &&
  CUDA_VISIBLE_DEVICES=0 $M generate --method student:pilot --n 192 &&
  CUDA_VISIBLE_DEVICES=0 $M generate --method coarse:reverse_diffusion:10 --n 192 &&
  CUDA_VISIBLE_DEVICES=0 $M generate --method coarse:ancestral:10 --n 192 &&
  CUDA_VISIBLE_DEVICES=0 $M fid --tag pilot --n 192 --methods student:pilot coarse:reverse_diffusion:10 coarse:ancestral:10 &&
  CUDA_VISIBLE_DEVICES=0 $M shift --method student:pilot
} >> run/logs/exp5_p2_gpu0.log 2>&1 && status "P2 pilot student evaluated" || status "P2 pilot student chain FAILED (run/logs/exp5_p2_gpu0.log)"
