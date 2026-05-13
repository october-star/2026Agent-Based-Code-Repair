#!/bin/bash
#SBATCH --job-name=proofbench-lean
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

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PIPELINE_DIR="${PIPELINE_DIR:-${REPO_ROOT}/proofbench-pipeline}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-ubelix}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/hf-cache}"

export PROOFBENCH_RUN_MODE="${PROOFBENCH_RUN_MODE:-pilot}"
export PROOFBENCH_PREPARE_DATA="${PROOFBENCH_PREPARE_DATA:-1}"
export PROOFBENCH_NUM_SAMPLES="${PROOFBENCH_NUM_SAMPLES:-10}"
export PROOFBENCH_HF_DATASET="${PROOFBENCH_HF_DATASET:-wenjiema02/ProofBench}"
export PROOFBENCH_SPLIT="${PROOFBENCH_SPLIT:-train}"
export PROOFBENCH_MODEL_CONFIG="${PROOFBENCH_MODEL_CONFIG:-${PIPELINE_DIR}/configs/model_ubelix.yaml}"
export PROOFBENCH_EXPERIMENT_CONFIG="${PROOFBENCH_EXPERIMENT_CONFIG:-${PIPELINE_DIR}/configs/experiment.yaml}"
export PROOFBENCH_LEAN_CONFIG="${PROOFBENCH_LEAN_CONFIG:-${PIPELINE_DIR}/configs/lean.yaml}"

mkdir -p "${LOG_DIR}" "${HF_HOME}"

module load Anaconda3

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${PIPELINE_DIR}/requirements.txt"

export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export PYTHONUNBUFFERED=1

exec > >(tee -a "${LOG_DIR}/ubelix-proofbench-${SLURM_JOB_ID}.log") 2>&1

echo "Started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Repo root: ${REPO_ROOT}"
echo "Pipeline dir: ${PIPELINE_DIR}"
echo "Venv dir: ${VENV_DIR}"
echo "HF cache: ${HF_HOME}"
echo "Run mode: ${PROOFBENCH_RUN_MODE}"
echo "Num samples: ${PROOFBENCH_NUM_SAMPLES}"
echo "Model config: ${PROOFBENCH_MODEL_CONFIG}"

cd "${REPO_ROOT}"
srun bash "${PIPELINE_DIR}/scripts/run_proofbench.sh"

echo "Finished at: $(date)"
