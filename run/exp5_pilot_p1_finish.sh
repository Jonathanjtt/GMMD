#!/usr/bin/env bash
# Completes P1 after the evalfits race of 2026-10-19 (principal_r3_vgap was evaluated while its
# last 3 intervals were still being fitted on the other GPU; fixed in exp5_pilot_p1.sh).
# Evaluates only the missing pairs (evalfits skips evaluated ones), then runs the selection.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
M="python -m gmmd.experiments.ibw_amortized"
OUT=outputs/exp5_ibw_amortized_10step_k8
status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/RUNLOG"; }
GPU=${GPU:-0}
mkdir "$OUT/locks/evalfits_val_principal_r3_vgap" 2>/dev/null
CUDA_VISIBLE_DEVICES=$GPU $M evalfits --split val --variant principal_r3_vgap >> run/logs/exp5_p1_gpu0.log 2>&1 \
  && touch "$OUT/locks/evalfits_val_principal_r3_vgap/done" || { status "P1 finish: evalfits FAILED"; exit 1; }
$M select >> run/logs/exp5_p1_select.log 2>&1 && status "P1 selection frozen: $(python -c "import json;s=json.load(open('$OUT/selection.json'));print(s['variant'], s['iterations'], 'iterations, step_c', s['step_c'])")" \
  || status "P1 selection FAILED (run/logs/exp5_p1_select.log)"
