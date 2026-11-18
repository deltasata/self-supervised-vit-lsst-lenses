#!/bin/bash -l
#
# 1-GPU job (AMD MI300A nodes). Submit with:
#   tools/submit.sh --name <name> jobs/<name>.sbatch -- <script.py> [args]
#
#SBATCH -o logs/slurm/%x_%j.out
#SBATCH -e logs/slurm/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --constraint=apu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=110000
#SBATCH --time=11:58:00
#SBATCH --mail-type=none

# abort if more than 15 of your jobs are queued
CURRENT=$(squeue -u "$USER" -h -t PD,R,CG,CF 2>/dev/null | wc -l)
if [ "$CURRENT" -gt 15 ]; then
    echo "ABORT: $CURRENT jobs in queue at start (limit 15). Cancelling self."
    scancel "$SLURM_JOB_ID"
    exit 1
fi

module purge
module load rocm/7.2
module load python-waterboa/2025.06

source "${LAGN_VENV:?set LAGN_VENV to your Python venv}/bin/activate"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# keep every cache inside the working dir
W="${LAGN_WORKDIR:-$PWD}"
export HF_HOME=$W/cache/huggingface HF_HUB_OFFLINE=1 TORCH_HOME=$W/cache/torch
export XDG_CACHE_HOME=$W/cache/xdg MPLCONFIGDIR=$W/cache/matplotlib TRITON_CACHE_DIR=$W/cache/triton
export MIOPEN_USER_DB_PATH=$W/cache/miopen/$SLURM_JOB_ID MIOPEN_CUSTOM_CACHE_DIR=$W/cache/miopen/$SLURM_JOB_ID
export TMPDIR=$W/tmp/$SLURM_JOB_ID
mkdir -p "$MIOPEN_USER_DB_PATH" "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT

echo "job $SLURM_JOB_ID on $(hostname) | git $(git rev-parse --short HEAD 2>/dev/null)"

srun python -u "$@"
