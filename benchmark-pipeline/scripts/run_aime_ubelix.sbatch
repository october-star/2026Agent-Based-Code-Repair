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
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Notes for UBELIX accounts/QOS:
# - gratis/free-tier GPU jobs may need:   #SBATCH --qos=job_gpu_preemptable
# - paygo jobs need:                      #SBATCH --wckey=<your-project>
# - invest GPU jobs usually use:          #SBATCH --partition=gpu-invest
#                                         #SBATCH --qos=job_gpu_<investor>

set -euo pipefail

echo "Started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PIPELINE_DIR}/.." && pwd)"

module load Anaconda3

VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-ubelix}"
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export PYTHONUNBUFFERED=1

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtual environment at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${REPO_ROOT}/requirements.txt"

echo "Python: $(which python)"
python --version

cd "${REPO_ROOT}"
srun bash "${PIPELINE_DIR}/scripts/run_aime.sh"

echo "Finished at: $(date)"
