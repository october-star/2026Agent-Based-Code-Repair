#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PIPELINE_DIR="${PIPELINE_DIR:-${REPO_ROOT}/proofbench-pipeline}"
RUN_MODE="${PROOFBENCH_RUN_MODE:-pilot}"
PREPARE_DATA="${PROOFBENCH_PREPARE_DATA:-1}"
DATASET_NAME="${PROOFBENCH_HF_DATASET:-wenjiema02/ProofBench}"
DATASET_SPLIT="${PROOFBENCH_SPLIT:-train}"
NUM_SAMPLES="${PROOFBENCH_NUM_SAMPLES:-10}"

export PROOFBENCH_EXPERIMENT_CONFIG="${PROOFBENCH_EXPERIMENT_CONFIG:-${PIPELINE_DIR}/configs/experiment.yaml}"
export PROOFBENCH_MODEL_CONFIG="${PROOFBENCH_MODEL_CONFIG:-${PIPELINE_DIR}/configs/model.yaml}"
export PROOFBENCH_LEAN_CONFIG="${PROOFBENCH_LEAN_CONFIG:-${PIPELINE_DIR}/configs/lean.yaml}"

cd "${REPO_ROOT}"

if [[ "${PREPARE_DATA}" == "1" ]]; then
  python3 "${PIPELINE_DIR}/scripts/prepare_data.py" \
    --hf-dataset "${DATASET_NAME}" \
    --split "${DATASET_SPLIT}" \
    --num-samples "${NUM_SAMPLES}"
fi

if [[ "${RUN_MODE}" == "full" ]]; then
  python3 "${PIPELINE_DIR}/scripts/run_full.py"
else
  python3 "${PIPELINE_DIR}/scripts/run_10_samples.py"
fi
