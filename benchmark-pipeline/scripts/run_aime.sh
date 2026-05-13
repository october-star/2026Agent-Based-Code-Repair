#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export RUN_ID="${RUN_ID:-$(date +"%Y%m%d-%H%M%S")}"
echo "RUN_ID=${RUN_ID}"

python "$PIPELINE_DIR/benchmarks/aime2025/run_infer.py"
python "$PIPELINE_DIR/benchmarks/aime2025/eval.py"
