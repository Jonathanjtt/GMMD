#!/usr/bin/env bash
# exp5 rollout-shift refinement (run only if the shift diagnostic shows substantial degradation):
#   student-rollout states at the interval boundaries (split "refine", x_T seeds 3.3M+) ->
#   offline IBW fits there with the FROZEN configuration (intervals 1-9; interval 0 starts at x_T,
#   which is the teacher's own prior) -> fine-tune the main student on train + refine targets ->
#   the same held-out test evaluation, 1,024 images, shift.  No online fitting is introduced.
#
#   tmux new-session -d -s exp5-refine 'bash run/exp5_refine.sh'
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
M="python -m gmmd.experiments.ibw_amortized"
OUT=outputs/exp5_ibw_amortized_10step_k8
STEPS=${STEPS:-4000}
status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/RUNLOG"; }
N_REF=$(python -c "import tomllib;print(tomllib.load(open('configs/exp5_ibw_amortized_10step_k8.toml','rb'))['splits']['n_refine'])")

status "refinement started (main student -> refine anchors)"
CUDA_VISIBLE_DEVICES=0 $M refine-anchors --tag main >> run/logs/exp5_refine_gpu0.log 2>&1 || { status "refine anchors FAILED"; exit 1; }
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=$g $M fit --split refine --variant frozen --intervals 1 2 3 4 5 6 7 8 9 >> run/logs/exp5_refine_gpu$g.log 2>&1 &
done
wait
[[ $(ls $OUT/fits/refine/frozen/traj*_j*.json 2>/dev/null | wc -l) -ge $((N_REF * 9)) ]] || { status "refine fits incomplete"; exit 1; }
status "refine fits done"
{
  CUDA_VISIBLE_DEVICES=0 $M train --tag refined --init-from $OUT/student/main/ckpt_best.pt --extra refine:frozen \
      --steps "$STEPS" --lr-scale 0.5 &&
  CUDA_VISIBLE_DEVICES=0 $M evalstudent --tag refined --split test &&
  CUDA_VISIBLE_DEVICES=0 $M generate --method student:refined --n 1024 &&
  CUDA_VISIBLE_DEVICES=0 $M shift --method student:refined
} >> run/logs/exp5_refine_gpu0.log 2>&1 && status "refined student evaluated" || status "refinement chain FAILED"
