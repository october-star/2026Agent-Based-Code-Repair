#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PIPELINE_DIR="${PIPELINE_DIR:-${REPO_ROOT}/proofbench-pipeline}"
RUN_MODE="${PROOFBENCH_RUN_MODE:-pilot}"
PREPARE_DATA="${PROOFBENCH_PREPARE_DATA:-0}"
DATA_INPUT="${PROOFBENCH_DATA_INPUT:-}"
NUM_SAMPLES="${PROOFBENCH_NUM_SAMPLES:-10}"

if [[ "${RUN_MODE}" == "full" ]]; then
  DEFAULT_EXPERIMENT_CONFIG="${PIPELINE_DIR}/configs/experiments/minif2f_full.yaml"
  DEFAULT_OUTPUT_DATA="${PIPELINE_DIR}/data/minif2f/test.jsonl"
else
  DEFAULT_EXPERIMENT_CONFIG="${PIPELINE_DIR}/configs/experiments/minif2f_pilot.yaml"
  DEFAULT_OUTPUT_DATA="${PIPELINE_DIR}/data/minif2f/pilot_10.jsonl"
fi

export PROOFBENCH_EXPERIMENT_CONFIG="${PROOFBENCH_EXPERIMENT_CONFIG:-${DEFAULT_EXPERIMENT_CONFIG}}"
export PROOFBENCH_MODEL_CONFIG="${PROOFBENCH_MODEL_CONFIG:-${PIPELINE_DIR}/configs/model.yaml}"
export PROOFBENCH_LEAN_CONFIG="${PROOFBENCH_LEAN_CONFIG:-${PIPELINE_DIR}/configs/lean.yaml}"

cd "${REPO_ROOT}"

if [[ "${PREPARE_DATA}" == "1" ]]; then
  if [[ -z "${DATA_INPUT}" ]]; then
    echo "Error: PROOFBENCH_DATA_INPUT must be set when PROOFBENCH_PREPARE_DATA=1."
    exit 1
  fi
  python3 "${PIPELINE_DIR}/scripts/prepare_minif2f_data.py" \
    --input "${DATA_INPUT}" \
    --output "${DEFAULT_OUTPUT_DATA}" \
    --num-samples "${NUM_SAMPLES}"
fi

if [[ "${RUN_MODE}" == "full" ]]; then
  python3 "${PIPELINE_DIR}/scripts/run_full.py"
else
  python3 "${PIPELINE_DIR}/scripts/run_10_samples.py"
fi
