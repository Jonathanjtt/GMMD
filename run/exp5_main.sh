#!/usr/bin/env bash
# exp5 main run, after the P2 pilot gate.  Resumable; both GPUs share every data stage through
# work claims (outputs/.../locks), so rerunning the script picks up exactly what is missing.
#
#   GPU0: train fits -> test fits -> [all train fits] main student -> [test refs+fits] stage A/B/C on
#         test -> 1,024 student images -> rollout shift -> uncached timing
#   GPU1: train fits -> test fits -> test refs (fp32) -> ablation endpoints (fp32) -> coarse
#         baselines (1,024 each) -> ablation student -> its evaluation and images
#   then: FID / P-R for every method at n = 1,024 against exp4's teacher images (same x_T)
#
#   tmux new-session -d -s exp5-main 'bash run/exp5_main.sh'
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
M="python -m gmmd.experiments.ibw_amortized"
OUT=outputs/exp5_ibw_amortized_10step_k8
mkdir -p run/logs
status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/RUNLOG"; }
N_TRAIN=$(python -c "import tomllib;print(tomllib.load(open('configs/exp5_ibw_amortized_10step_k8.toml','rb'))['splits']['n_train'])")
N_TEST=$(python -c "import tomllib;print(tomllib.load(open('configs/exp5_ibw_amortized_10step_k8.toml','rb'))['splits']['n_test'])")
count() { ls $1 2>/dev/null | wc -l; }
all_train_fits() { [[ $(count "$OUT/fits/train/frozen/traj*_j*.json") -ge $((N_TRAIN * 10)) ]]; }
all_test_ready() { [[ $(count "$OUT/fits/test/frozen/traj*_j*.json") -ge $((N_TEST * 10)) && $(count "$OUT/refs/test/traj*_j*.json") -ge $((N_TEST * 10)) ]]; }
all_endpoints() { [[ $(count "$OUT/endpoints/train/traj*_j*.json") -ge $((N_TRAIN * 10)) ]]; }
COARSE=("coarse:reverse_diffusion:10" "coarse:ancestral:10" "coarse:ancestral:20" "coarse:ancestral:50" "coarse:ancestral:100")

gpu0() {
  export CUDA_VISIBLE_DEVICES=0
  $M fit --split train --variant frozen || return 1
  $M fit --split test --variant frozen || return 1
  until all_train_fits; do sleep 60; done
  $M train --tag main || return 1
  status "main student trained"
  until all_test_ready; do sleep 60; done
  $M evalstudent --tag main --split test || return 1
  $M generate --method student:main --n 1024 || return 1
  $M shift --method student:main || return 1
  status "main student evaluated (stage A/B/C test, 1,024 images, shift)"
  $M timing --method student:main --precision tf32 --n 64 && $M timing --method student:main --precision fp32 --n 64
  $M timing --method coarse:ancestral:10 --precision fp32 --n 64
  $M timing --method teacher --precision fp32 --n 64
  status "timing done"
}

gpu1() {
  export CUDA_VISIBLE_DEVICES=1
  $M fit --split train --variant frozen || return 1
  $M fit --split test --variant frozen || return 1
  $M refs --split test || return 1
  status "test refs done"
  $M endpoints --split train --n-endpoints 8 || return 1
  status "ablation endpoints done"
  for m in "${COARSE[@]}"; do $M generate --method "$m" --n 1024 || return 1; done
  status "coarse baselines generated"
  until all_endpoints; do sleep 60; done
  $M train --tag ablation_endpoints --data endpoints || return 1
  until all_test_ready; do sleep 60; done
  $M evalstudent --tag ablation_endpoints --split test || return 1
  $M generate --method student:ablation_endpoints --n 1024 || return 1
  status "ablation evaluated"
}

status "main run started"
gpu0 >> run/logs/exp5_main_gpu0.log 2>&1 &
a=$!
gpu1 >> run/logs/exp5_main_gpu1.log 2>&1 &
b=$!
wait $a || status "main GPU0 chain FAILED (run/logs/exp5_main_gpu0.log)"
wait $b || status "main GPU1 chain FAILED (run/logs/exp5_main_gpu1.log)"
CUDA_VISIBLE_DEVICES=0 $M fid --tag n1024 --n 1024 --methods student:main student:ablation_endpoints "${COARSE[@]}" \
  >> run/logs/exp5_main_fid.log 2>&1 && status "FID n1024 done" || status "FID n1024 FAILED"
