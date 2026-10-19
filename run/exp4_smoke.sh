#!/usr/bin/env bash
# exp4 smoke test: one 8-image chunk through prefix, all methods, evaluation and the gate check,
# with a placeholder selection (5 IBW iterations) in outputs/_smoke_gen.  Checks plumbing only.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p outputs/_smoke_gen/validation
[[ -f outputs/_smoke_gen/validation/selection.json ]] || cat > outputs/_smoke_gen/validation/selection.json <<'J'
{"interval": {"name": "short", "t": 0.5998, "s": 0.5798, "i_t": 800, "i_s": 840, "teacher_steps": 40},
 "gmmd": {"n_components": 1, "n_grad_samples": 4, "step_size": 0.1, "n_iterations": 5, "init": "prior_free", "init_spread": 1.0}}
J
S=(--set 'output.dir="outputs/_smoke_gen"' generation.chunk=8 fid.n_boot=20)
M="python -m gmmd.experiments.interval_replacement"
$M prefix --n-chunks 1 "${S[@]}"
for m in teacher coarse gmmd; do $M method --method $m --n-chunks 1 "${S[@]}"; done
[[ -f outputs/_smoke_gen/metrics_smoke.json ]] || $M evaluate --n-chunks 1 --tag smoke "${S[@]}"
$M check --n-chunks 1 --tag smoke "${S[@]}"
