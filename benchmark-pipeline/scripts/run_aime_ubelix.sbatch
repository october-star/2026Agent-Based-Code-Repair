#!/bin/bash
#SBATCH --job-name=aime-qwen
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --account=gratis
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair}"
PIPELINE_DIR="${PIPELINE_DIR:-${REPO_ROOT}/benchmark-pipeline}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-ubelix}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

module load Anaconda3
source "${VENV_DIR}/bin/activate"

export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export PYTHONUNBUFFERED=1

exec > >(tee -a "${LOG_DIR}/ubelix-aime-${SLURM_JOB_ID}.log") 2>&1

echo "Started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Repo root: ${REPO_ROOT}"
echo "Venv: ${VENV_DIR}"
echo "HF cache: ${HF_HOME}"

cd "${REPO_ROOT}"
srun bash "${PIPELINE_DIR}/scripts/run_aime.sh"

echo "Finished at: $(date)"
