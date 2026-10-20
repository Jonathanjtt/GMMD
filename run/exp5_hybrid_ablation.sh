#!/usr/bin/env bash
# exp5 post-hoc diagnostic: WHERE along the chain do the IBW targets cost FID relative to the same
# network trained on teacher endpoints?  Hybrid samplers (gmmd/exp5/oracle.py::hybrid_sample) run the
# endpoint-NLL ablation at the intervals where mask[j] == 1 and the main IBW student elsewhere, with
# the same prior and the same per-image draw generators as the standalone sampler -- so the two pure
# masks reproduce student:ablation_endpoints and student:main exactly and every difference is due
# to which network makes which transition.  Blocks: high noise j0-j2, mid j3-j5, low j6-j8, terminal j9.
#   swap-in  (main + one ablation block):  1110000000 0001110000 0000001110 0000000001
#   swap-out (ablation + one main block):  0001111111 1110001111 1111110001 1111111110
#
#   bash run/exp5_hybrid_ablation.sh        (resumable: generation skips finished chunks)
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${GPU:-0}
M="python -m gmmd.experiments.ibw_amortized"
OUT=outputs/exp5_ibw_amortized_10step_k8
status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/RUNLOG"; }
MASKS=(1110000000 0001110000 0000001110 0000000001 0001111111 1110001111 1111110001 1111111110)
METHODS=()
for m in "${MASKS[@]}"; do
  $M generate --method hybrid:ablation_endpoints:main:$m --n 1024 || { status "hybrid $m FAILED"; exit 1; }
  METHODS+=("hybrid:ablation_endpoints:main:$m")
done
$M fid --tag n1024_hybrid_ablation --n 1024 --methods student:main student:ablation_endpoints "${METHODS[@]}" \
  && status "diag: hybrids ablation/main by interval block (fid_n1024_hybrid_ablation.json)" \
  || status "hybrid FID FAILED"
