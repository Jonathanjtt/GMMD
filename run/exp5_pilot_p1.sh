#!/usr/bin/env bash
# exp5 pilot, part 2 (P1): every teacher-free initialisation variant on all 4 val trajectories x
# 10 intervals, stage-A evaluation at every snapshot, then the pre-registered selection.
# GPU 1 starts immediately; GPU 0 joins after the val reference rollouts (run/exp5_pilot_p0.sh).
# Variants are claimed with atomic lock directories, so no fit runs twice.  Resumable.
#
#   ITERS="--set ibw.max_iterations=400 ibw.step_c=0.5" tmux new-session -d -s exp5-p1 'bash run/exp5_pilot_p1.sh'
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
M="python -m gmmd.experiments.ibw_amortized"
OUT=outputs/exp5_ibw_amortized_10step_k8
LOCK=$OUT/locks
mkdir -p "$LOCK" run/logs
SET=${SET:-}
VARIANTS=(${VARIANTS:-prior_free principal_r3 principal_r1 principal_r3_vgap random_far})
status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/RUNLOG"; }

claim_and_run() {   # claim_and_run <gpu> <lockname> <args...>
  local gpu=$1 name=$2; shift 2
  mkdir "$LOCK/$name" 2>/dev/null || return 0
  CUDA_VISIBLE_DEVICES=$gpu $M "$@" $SET || { rmdir "$LOCK/$name"; return 1; }
  touch "$LOCK/$name/done"
}

worker() {          # worker <gpu> <variants in claim order...>
  local gpu=$1; shift
  for v in "$@"; do claim_and_run "$gpu" "fit_val_$v" fit --split val --variant "$v" || return 1; done
  # stage-A evaluation needs every val reference; wait for them
  until grep -q "val refs done" "$OUT/RUNLOG" 2>/dev/null; do sleep 60; done
  for v in "$@"; do
    # the variant's fits may still be running on the other GPU: wait for them (or for their failure)
    until [[ -f "$LOCK/fit_val_$v/done" || ! -d "$LOCK/fit_val_$v" ]]; do sleep 60; done
    claim_and_run "$gpu" "evalfits_val_$v" evalfits --split val --variant "$v" || return 1
  done
}

status "P1 started: variants ${VARIANTS[*]} ${SET}"
worker 1 "${VARIANTS[@]}" > run/logs/exp5_p1_gpu1.log 2>&1 &
p1=$!
( until grep -q "val refs done" "$OUT/RUNLOG" 2>/dev/null; do sleep 60; done
  rev=(); for ((i=${#VARIANTS[@]}-1; i>=0; i--)); do rev+=("${VARIANTS[$i]}"); done
  worker 0 "${rev[@]}" ) > run/logs/exp5_p1_gpu0.log 2>&1 &
p0=$!
wait $p1 || status "P1 GPU1 worker FAILED (run/logs/exp5_p1_gpu1.log)"
wait $p0 || status "P1 GPU0 worker FAILED (run/logs/exp5_p1_gpu0.log)"
for v in "${VARIANTS[@]}"; do
  [[ -f "$LOCK/fit_val_$v/done" && -f "$LOCK/evalfits_val_$v/done" ]] || { status "P1 incomplete: $v"; exit 1; }
done
$M select $SET >> run/logs/exp5_p1_select.log 2>&1 && status "P1 selection frozen: $(python -c "import json;s=json.load(open('$OUT/selection.json'));print(s['variant'], s['iterations'], 'iterations, step_c', s['step_c'])")" \
  || status "P1 selection FAILED (run/logs/exp5_p1_select.log)"
