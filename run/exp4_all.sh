#!/usr/bin/env bash
# exp4 end to end, unattended and resumable:
#   validation -> frozen selection -> smoke check -> 192-image pilot -> pilot gate -> 1,024-image run
#   (+ uncached end-to-end timing) -> evaluation -> results table.
# Progress lines go to outputs/exp4_interval_replacement/STATUS; it stops (and says why) on any failure.
#
#   tmux new-session -d -s exp4 'bash run/exp4_all.sh 2>&1 | tee -a run/logs/exp4_all.log'
#   tmux attach -t exp4          # watch;  Ctrl-b d to detach again
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export GPUS=${GPUS:-"0 1"}
OUT=outputs/exp4_interval_replacement
M="python -m gmmd.experiments.interval_replacement"
mkdir -p run/logs "$OUT"

status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/STATUS"; }
die() { status "STOPPED: $*"; exit 1; }
wait_pids() { for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 60; done; done; }

status "chain started (pid $$, GPUs $GPUS)"

# 1. jobs launched before this chain (WAIT_PIDS) keep their progress; wait for them, then resume
#    whatever is missing (both steps skip finished work)
wait_pids ${WAIT_PIDS:-}
run/exp4_validation.sh >> run/logs/exp4_validation.log 2>&1 \
  || die "validation failed (see run/logs/exp4_validation*.log)"
[[ -f $OUT/validation/selection.json ]] || die "no selection.json after validation"
status "selection frozen: $(python -c "import json;s=json.load(open('$OUT/validation/selection.json'));print(s['interval']['name'], s['interval']['i_t'], '->', s['interval']['i_s'], '|', s['gmmd']['n_iterations'], 'IBW iterations')")"

# 2. plumbing smoke test (8 images, placeholder config)
CUDA_VISIBLE_DEVICES=${GPUS%% *} run/exp4_smoke.sh >> run/logs/smoke_gen.log 2>&1 \
  || die "smoke test failed (see run/logs/smoke_gen.log)"
status "smoke test passed"

# 3. pilot: 192 images per method, then the gate
status "pilot started"
N_CHUNKS=3 TAG=pilot run/exp4_generate.sh >> run/logs/exp4_pilot.log 2>&1 \
  || die "pilot generation failed (see run/logs/exp4_*shard*.log, run/logs/exp4_evaluate_pilot.log)"
$M check --n-chunks 3 --tag pilot >> run/logs/exp4_check_pilot.log 2>&1 \
  || die "pilot check failed (see run/logs/exp4_check_pilot.log)"
status "pilot passed: $OUT/results_table_pilot.md"

# 4. full run: 1,024 images per method (pilot chunks are reused), uncached timing, evaluation
status "1,024-image run started"
N_CHUNKS=16 TAG=n1024 E2E=1 run/exp4_generate.sh >> run/logs/exp4_n1024.log 2>&1 \
  || die "1,024-image run failed (see run/logs/exp4_*shard*.log, run/logs/exp4_e2e_*.log)"
$M check --n-chunks 16 --tag n1024 >> run/logs/exp4_check_n1024.log 2>&1 \
  || die "final check failed (see run/logs/exp4_check_n1024.log)"
status "DONE: $OUT/metrics_n1024.json, $OUT/results_table_n1024.md"
