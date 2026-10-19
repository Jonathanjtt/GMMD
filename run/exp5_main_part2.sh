#!/usr/bin/env bash
# exp5 main run, GPU 0 part 2 (replaces the tail of run/exp5_main.sh's GPU0 chain after the main
# student's images, first FID and shift; reordered after the pilot showed that the IBW targets carry
# a mean bias, so the endpoint-NLL ablation -- which isolates that bias -- comes before the long
# fp32 test references).  Resumable; all data stages are claim-shared with GPU 1's chain.
#
#   GPU0: refinement (student-rollout anchors -> frozen IBW fits -> fine-tune) -> ablation endpoints
#         (fp32, E = 4) -> ablation student -> 1,024 images -> FID (n = 1,024) -> test refs (shared)
#         -> stage A/B/C on test for main, refined, ablation, zero-training control -> timing
#
#   tmux new-session -d -s exp5-main0b 'bash run/exp5_main_part2.sh'
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${GPU:-0}
M="python -m gmmd.experiments.ibw_amortized"
OUT=outputs/exp5_ibw_amortized_10step_k8
LOG=run/logs/exp5_main_gpu0b.log
status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/RUNLOG"; }
cfgval() { python -c "import tomllib;print(tomllib.load(open('configs/exp5_ibw_amortized_10step_k8.toml','rb'))$1)"; }
N_TRAIN=$(cfgval "['splits']['n_train']"); N_TEST=$(cfgval "['splits']['n_test']"); N_REF=$(cfgval "['splits']['n_refine']")
E=${N_ENDPOINTS:-4}
count() { ls $1 2>/dev/null | wc -l; }
all_test_ready() { [[ $(count "$OUT/fits/test/frozen/traj*_j*.json") -ge $((N_TEST * 10)) &&
                      $(count "$OUT/fits/test/frozen_r1/traj*_j*.json") -ge $((N_TEST * 10)) &&
                      $(count "$OUT/refs/test/traj*_j*.json") -ge $((N_TEST * 10)) ]]; }
all_endpoints() { [[ $(count "$OUT/endpoints/train/traj*_j*.json") -ge $((N_TRAIN * 10)) ]]; }
COARSE=("coarse:reverse_diffusion:10" "coarse:ancestral:10")

refinement() {
  $M refine-anchors --tag main || return 1
  $M fit --split refine --variant frozen --intervals 1 2 3 4 5 6 7 8 9 || return 1
  [[ $(count "$OUT/fits/refine/frozen/traj*_j*.json") -ge $((N_REF * 9)) ]] || { status "refine fits incomplete"; return 1; }
  $M train --tag refined --init-from $OUT/student/main/ckpt_best.pt --extra refine:frozen --steps ${REFINE_STEPS:-4000} --lr-scale 0.5 || return 1
  $M generate --method student:refined --n 1024 || return 1
  $M shift --method student:refined || return 1
  status "refined student trained, 1,024 images, shift"
}

ablation() {
  until all_endpoints; do $M endpoints --split train --n-endpoints "$E" || return 1; sleep 30; done
  status "ablation endpoints done (E = $E)"
  $M train --tag ablation_endpoints --data endpoints || return 1
  $M generate --method student:ablation_endpoints --n 1024 || return 1
  status "ablation student trained, 1,024 images"
}

{
  status "main part 2 (GPU0) started"
  refinement || status "refinement FAILED ($LOG)"
  ablation || status "ablation FAILED ($LOG)"
  $M fid --tag n1024_students --n 1024 --methods student:main student:refined student:ablation_endpoints student:init_main "${COARSE[@]}" \
    && status "FID n = 1,024 (students + 10-step coarse)"
  $M refs --split test || status "test refs (GPU0 share) FAILED"
  until all_test_ready; do sleep 60; done
  for t in main refined ablation_endpoints init_main; do
    $M evalstudent --tag $t --split test || status "evalstudent $t FAILED"
  done
  status "stage A/B/C on test: main, refined, ablation, init"
  TAG=main bash run/exp5_timing.sh
} >> "$LOG" 2>&1
status "main part 2 (GPU0) finished"
