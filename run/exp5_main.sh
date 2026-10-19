#!/usr/bin/env bash
# exp5 main run, after the P2 pilot gate.  Resumable: every data stage is shared by both GPUs
# through work claims (fits / refs / endpoints per (anchor, interval), generation per 64-image
# chunk, evaluation per pair), so rerunning the script picks up exactly what is missing.
#
# GPU 1 is shared with another user's jobs (~3x slower since 2026-10-19 15:40), so GPU 0 runs the
# latency-critical chain and GPU 1 the claimable background work:
#   GPU0: train fits -> test fits (+ replicate) -> main student -> 1,024 student + 10-step coarse
#         images -> first FID -> rollout shift -> test refs (shared) -> stage A/B/C on test ->
#         ablation student (once its endpoints exist) -> its evaluation and images
#   GPU1: train fits -> test fits -> test refs (fp32) -> ablation endpoints (fp32, 4 per anchor =
#         the IBW targets' 800 score evaluations) -> coarse ancestral 20/50 -> IBW chain (64)
#   then: FID / P-R for every method at n = 1,024 against exp4's teacher images (same x_T)
# Refinement (run/exp5_refine.sh) and timing (run/exp5_timing.sh) are separate decisions.
#
#   tmux new-session -d -s exp5-main 'bash run/exp5_main.sh'
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
M="python -m gmmd.experiments.ibw_amortized"
OUT=outputs/exp5_ibw_amortized_10step_k8
mkdir -p run/logs
status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/RUNLOG"; }
cfgval() { python -c "import tomllib;print(tomllib.load(open('configs/exp5_ibw_amortized_10step_k8.toml','rb'))$1)"; }
N_TRAIN=$(cfgval "['splits']['n_train']")
N_TEST=$(cfgval "['splits']['n_test']")
E=${N_ENDPOINTS:-4}
count() { ls $1 2>/dev/null | wc -l; }
all_train_fits() { [[ $(count "$OUT/fits/train/frozen/traj*_j*.json") -ge $((N_TRAIN * 10)) ]]; }
all_test_ready() { [[ $(count "$OUT/fits/test/frozen/traj*_j*.json") -ge $((N_TEST * 10)) &&
                      $(count "$OUT/fits/test/frozen_r1/traj*_j*.json") -ge $((N_TEST * 10)) &&
                      $(count "$OUT/refs/test/traj*_j*.json") -ge $((N_TEST * 10)) ]]; }
all_endpoints() { [[ $(count "$OUT/endpoints/train/traj*_j*.json") -ge $((N_TRAIN * 10)) ]]; }
FIRST=("coarse:reverse_diffusion:10" "coarse:ancestral:10")
MORE=("coarse:ancestral:20" "coarse:ancestral:50")

shared_fits() {
  $M fit --split train --variant frozen &&
  $M fit --split test --variant frozen &&
  $M fit --split test --variant frozen_r1
}

gpu0() {
  export CUDA_VISIBLE_DEVICES=0
  shared_fits || return 1
  until all_train_fits; do sleep 60; done
  status "all train fits done"
  $M train --tag main || return 1
  status "main student trained"
  $M generate --method student:main --n 1024 || return 1
  for m in "${FIRST[@]}"; do $M generate --method "$m" --n 1024 || return 1; done
  $M fid --tag n1024_first --n 1024 --methods student:main "${FIRST[@]}" && status "first FID (student + 10-step coarse, n = 1,024)"
  $M shift --method student:main && status "rollout shift (main) done"
  $M refs --split test || return 1
  until all_test_ready; do sleep 60; done
  $M evalstudent --tag main --split test || return 1
  status "main student: stage A/B/C on test done"
  until all_endpoints; do $M endpoints --split train --n-endpoints "$E" || return 1; sleep 60; done
  $M train --tag ablation_endpoints --data endpoints || return 1
  status "ablation student trained"
  $M generate --method student:ablation_endpoints --n 1024 || return 1
  $M evalstudent --tag ablation_endpoints --split test || return 1
  status "ablation evaluated"
}

gpu1() {
  export CUDA_VISIBLE_DEVICES=1
  shared_fits || return 1
  $M refs --split test || return 1
  until all_test_ready; do sleep 60; done          # (GPU0 finishes its claimed refs)
  status "test refs + fits ready"
  $M endpoints --split train --n-endpoints "$E" || return 1
  status "ablation endpoints done (E = $E)"
  for m in "${MORE[@]}"; do $M generate --method "$m" --n 1024 || return 1; done
  status "coarse ancestral 20/50 generated"
  $M generate --method ibwchain --n 64 && status "IBW chain (64 images) generated"
}

# ONLY=gpu0 / ONLY=gpu1 starts one chain (e.g. when the other GPU has no memory to spare yet; the
# other chain can be started later with the other value).  The final FID runs after a full run.
ONLY=${ONLY:-both}
status "main run started (chains: $ONLY; E = $E endpoints per anchor)"
pids=()
if [[ $ONLY != gpu1 ]]; then gpu0 >> run/logs/exp5_main_gpu0.log 2>&1 & pids+=($!); fi
if [[ $ONLY != gpu0 ]]; then gpu1 >> run/logs/exp5_main_gpu1.log 2>&1 & pids+=($!); fi
for p in "${pids[@]}"; do wait $p || status "main chain FAILED (pid $p; run/logs/exp5_main_gpu*.log)"; done
[[ $ONLY == both ]] || { status "main chain(s) $ONLY finished"; exit 0; }
CUDA_VISIBLE_DEVICES=0 $M fid --tag n1024 --n 1024 --methods student:main student:ablation_endpoints "${FIRST[@]}" "${MORE[@]}" \
  >> run/logs/exp5_main_fid.log 2>&1 && status "FID n1024 done" || status "FID n1024 FAILED"
