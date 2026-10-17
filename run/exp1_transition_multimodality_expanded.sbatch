#!/usr/bin/env bash
#SBATCH --job-name=gmmd-exp1-expand
#SBATCH --account=torch_pr_468_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/jt4994/GMMD/run/logs/%x-%j.out
#SBATCH --error=/scratch/jt4994/GMMD/run/logs/%x-%j.err

set -euo pipefail

REPO=/scratch/jt4994/GMMD
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "$REPO/.venv"
cd "$REPO"

export PYTHONUNBUFFERED=1
export TORCH_HOME="$REPO/.torch-cache"
export TMPDIR="$REPO/.runtime/tmp/${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$REPO/.runtime/cache/${SLURM_JOB_ID}"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$TORCH_HOME" "$REPO/.runtime" "$REPO/run/logs"

(
  flock -x 9
  python -m gmmd.download \
    --checkpoint ve/celebahq_256_ncsnpp_continuous \
    --celebahq validation
  python -c 'from gmmd.features import load_dinov2; load_dinov2("cpu"); print("DINOv2 cache ready")'
) 9>"$REPO/.runtime/assets.lock"

python -m gmmd.experiments.transition_multimodality_sweep \
  --config exp1_transition_multimodality_expanded
