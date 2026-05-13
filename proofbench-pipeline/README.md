# ProofBench Lean Evaluation

This directory contains the project scaffold for:

**Evaluating Multi-Stage LLM-to-Lean Proof Generation on ProofBench**

Current implementation status:
- experiment configuration files
- prompt templates
- data preparation script
- baseline pipeline skeleton
- Lean runner
- result writer and summary utilities

## Directory Overview

```text
proofbench-pipeline/
├── configs/
├── data/
├── prompts/
├── src/
├── scripts/
├── outputs/
└── reports/
```

## Recommended First Run

1. The default data source is the official Hugging Face dataset:

[`wenjiema02/ProofBench`](https://huggingface.co/datasets/wenjiema02/ProofBench)

The pilot uses the `train` split by default. You can optionally override this
with a local JSONL file via `--input`.

2. Build the 10-sample pilot file:

```bash
cd /Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair
python3 proofbench-pipeline/scripts/prepare_data.py
```

Optional local override:

```bash
python3 proofbench-pipeline/scripts/prepare_data.py \
  --input proofbench-pipeline/data/raw/proofbench.jsonl
```

3. Install the pipeline dependencies for local `transformers` inference:

```bash
pip install -r proofbench-pipeline/requirements.txt
```

The default model configuration now uses local inference:
- provider: `local`
- model: `Qwen/Qwen2.5-1.5B-Instruct`
- device: `cpu` (chosen intentionally to avoid common Apple MPS memory crashes)
- max input tokens: `1024`
- max new tokens: `256`

4. If the model is gated or you want authenticated downloads, set your Hugging Face token:

```bash
export HF_TOKEN=your_huggingface_token
```

5. Run the pilot experiment with the default local backend:

```bash
python3 proofbench-pipeline/scripts/run_10_samples.py
```

Outputs are written to:
- `proofbench-pipeline/outputs/results.csv`
- `proofbench-pipeline/outputs/generations/`
- `proofbench-pipeline/outputs/lean_files/`
- `proofbench-pipeline/outputs/logs/`

## Current Methods

- `reference_to_lean`
- `cot_then_lean`
- `direct_lean`
- `mixed_cot_lean`

The model backend now defaults to local `transformers` inference.
The default configuration lives in [configs/model.yaml](/Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair/proofbench-pipeline/configs/model.yaml:1).

If you want to switch providers later, the pipeline still supports:
- `local`
- `hf_inference`
- `openai`
- `openai_compatible`
- `dry_run`

If you later want to retry Apple GPU inference, update
[configs/model.yaml](/Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair/proofbench-pipeline/configs/model.yaml:1)
carefully and expect tighter limits on sequence length.

## Ubelix

For Ubelix, the repository now includes:
- [run_proofbench.sh](/Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair/proofbench-pipeline/scripts/run_proofbench.sh:1)
- [run_proofbench_ubelix.sbatch](/Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair/proofbench-pipeline/scripts/run_proofbench_ubelix.sbatch:1)
- [model_ubelix.yaml](/Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair/proofbench-pipeline/configs/model_ubelix.yaml:1)

Submit the 10-sample pilot with:

```bash
cd /path/to/2026Agent-Based-Code-Repair
sbatch proofbench-pipeline/scripts/run_proofbench_ubelix.sbatch
```

Useful overrides:

```bash
sbatch --export=REPO_ROOT=$(pwd),PROOFBENCH_NUM_SAMPLES=10 proofbench-pipeline/scripts/run_proofbench_ubelix.sbatch
```

Run a larger/full dataset pass by switching the mode and sample count:

```bash
sbatch --export=REPO_ROOT=$(pwd),PROOFBENCH_RUN_MODE=full,PROOFBENCH_NUM_SAMPLES=0 proofbench-pipeline/scripts/run_proofbench_ubelix.sbatch
```
