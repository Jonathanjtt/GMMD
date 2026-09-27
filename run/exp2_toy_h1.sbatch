#!/usr/bin/env bash
#SBATCH --job-name=gmmd-exp2
#SBATCH --account=torch_pr_468_general
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/jt4994/GMMD/run/logs/%x-%j.out
#SBATCH --error=/scratch/jt4994/GMMD/run/logs/%x-%j.err

set -euo pipefail

REPO=/scratch/jt4994/GMMD
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "$REPO/.venv"
cd "$REPO"

export PYTHONUNBUFFERED=1
export TMPDIR="$REPO/.runtime/tmp/${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$REPO/.runtime/cache/${SLURM_JOB_ID}"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"

python -m gmmd.experiments.toy_h1 --config exp2_toy_h1
