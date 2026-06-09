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
PIPELINE_DIR="${PIPELINE_DIR:-${REPO_ROOT}/formal-benchmark}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-ubelix}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/hf-cache}"
LEAN_BIN_DIR="${LEAN_BIN_DIR:-$HOME/.elan/bin}"

export PROOFBENCH_RUN_MODE="${PROOFBENCH_RUN_MODE:-pilot}"
export PROOFBENCH_PREPARE_DATA="${PROOFBENCH_PREPARE_DATA:-0}"
export PROOFBENCH_NUM_SAMPLES="${PROOFBENCH_NUM_SAMPLES:-10}"
export PROOFBENCH_DATA_INPUT="${PROOFBENCH_DATA_INPUT:-}"
export PROOFBENCH_MODEL_CONFIG="${PROOFBENCH_MODEL_CONFIG:-${PIPELINE_DIR}/configs/model_ubelix.yaml}"
if [[ "${PROOFBENCH_RUN_MODE}" == "full" ]]; then
  export PROOFBENCH_EXPERIMENT_CONFIG="${PROOFBENCH_EXPERIMENT_CONFIG:-${PIPELINE_DIR}/configs/experiments/minif2f_full.yaml}"
else
  export PROOFBENCH_EXPERIMENT_CONFIG="${PROOFBENCH_EXPERIMENT_CONFIG:-${PIPELINE_DIR}/configs/experiments/minif2f_pilot.yaml}"
fi
export PROOFBENCH_LEAN_CONFIG="${PROOFBENCH_LEAN_CONFIG:-${PIPELINE_DIR}/configs/lean.yaml}"
export PROOFBENCH_LEAN_PROJECT_DIR="${PROOFBENCH_LEAN_PROJECT_DIR:-${PIPELINE_DIR}/lean_project}"

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

if [[ -d "${LEAN_BIN_DIR}" ]]; then
  export PATH="${LEAN_BIN_DIR}:$PATH"
fi

exec > >(tee -a "${LOG_DIR}/ubelix-proofbench-${SLURM_JOB_ID}.log") 2>&1

echo "Started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Repo root: ${REPO_ROOT}"
echo "Pipeline dir: ${PIPELINE_DIR}"
echo "Venv dir: ${VENV_DIR}"
echo "HF cache: ${HF_HOME}"
echo "Lean bin dir: ${LEAN_BIN_DIR}"
echo "Run mode: ${PROOFBENCH_RUN_MODE}"
echo "Num samples: ${PROOFBENCH_NUM_SAMPLES}"
echo "Model config: ${PROOFBENCH_MODEL_CONFIG}"
echo "Lean project dir: ${PROOFBENCH_LEAN_PROJECT_DIR}"

if ! command -v lake >/dev/null 2>&1; then
  echo "Error: 'lake' was not found in PATH."
  echo "Expected Lean 4 via elan, usually at: ${LEAN_BIN_DIR}/lake"
  echo "Install Lean 4 on Ubelix first, for example on a login node:"
  echo "  curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y"
  exit 1
fi

echo "Using lake: $(command -v lake)"
lake --version || true

if [[ ! -f "${PROOFBENCH_LEAN_PROJECT_DIR}/lakefile.toml" ]]; then
  echo "Error: Lean project not found at ${PROOFBENCH_LEAN_PROJECT_DIR}"
  exit 1
fi

cd "${PROOFBENCH_LEAN_PROJECT_DIR}"
MATHLIB_DIR=".lake/packages/mathlib"
SMOKE_TARGET=".lake/build/lib/lean/ProofbenchLean.olean"

if [[ ! -d "${MATHLIB_DIR}" ]]; then
  echo "Mathlib cache missing. Initializing dependencies with lake update..."
  lake update
else
  echo "Mathlib cache found at ${PROOFBENCH_LEAN_PROJECT_DIR}/${MATHLIB_DIR}. Skipping lake update."
fi

if [[ ! -f "${SMOKE_TARGET}" ]]; then
  echo "Smoke target missing. Building Lean project smoke target..."
  lake build ProofbenchLean
else
  echo "Smoke target found at ${PROOFBENCH_LEAN_PROJECT_DIR}/${SMOKE_TARGET}. Skipping lake build."
fi

cd "${REPO_ROOT}"
srun bash "${PIPELINE_DIR}/scripts/run_proofbench.sh"

echo "Finished at: $(date)"
