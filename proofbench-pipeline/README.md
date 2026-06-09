# miniF2F Lean Evaluation

This directory now serves as the project scaffold for the new proposal's
miniF2F-only evaluation workflow:

**CoT-to-Lean, Direct Lean, and Interleaved Lean generation on miniF2F-test,
with and without a repair loop.**

Current implementation status:
- proposal-aligned `miniF2F-test` experiment configs
- strategy aliases for `cot_to_lean`, `direct_lean`, and `interleaved`
- explicit `agent_mode` support (`no_agent` vs `repair_loop`)
- Lean runner and Lean4 repair loop
- result writer and summary utilities
- metric scaffolding for `BEq`, `GTED`, and repair statistics

## Directory Overview

```text
proofbench-pipeline/
├── configs/
├── data/
├── references/
├── prompts/
├── src/
├── scripts/
├── outputs/
└── reports/
```

## Proposal-Aligned Layout

- Dataset config: [configs/datasets/minif2f_test.yaml](/Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair/proofbench-pipeline/configs/datasets/minif2f_test.yaml:1)
- Strategy configs:
  - [reference_formalization.yaml](/Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair/proofbench-pipeline/configs/strategies/reference_formalization.yaml:1)
  - [cot_to_lean.yaml](/Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair/proofbench-pipeline/configs/strategies/cot_to_lean.yaml:1)
  - [direct_lean.yaml](/Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair/proofbench-pipeline/configs/strategies/direct_lean.yaml:1)
  - [interleaved.yaml](/Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair/proofbench-pipeline/configs/strategies/interleaved.yaml:1)
- Agent configs:
  - [no_agent.yaml](/Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair/proofbench-pipeline/configs/agents/no_agent.yaml:1)
  - [repair_loop.yaml](/Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair/proofbench-pipeline/configs/agents/repair_loop.yaml:1)
- Experiment configs:
  - [minif2f_reference_formalization.yaml](/Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair/proofbench-pipeline/configs/experiments/minif2f_reference_formalization.yaml:1)
  - [minif2f_pilot.yaml](/Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair/proofbench-pipeline/configs/experiments/minif2f_pilot.yaml:1)
  - [minif2f_pilot_no_agent.yaml](/Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair/proofbench-pipeline/configs/experiments/minif2f_pilot_no_agent.yaml:1)
  - [minif2f_full.yaml](/Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair/proofbench-pipeline/configs/experiments/minif2f_full.yaml:1)
  - [minif2f_full_no_agent.yaml](/Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair/proofbench-pipeline/configs/experiments/minif2f_full_no_agent.yaml:1)

## Recommended First Run

1. Prepare a local miniF2F JSONL file in your workspace.

2. Build the 10-sample pilot file:

```bash
cd /Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair
python3 proofbench-pipeline/scripts/prepare_minif2f_data.py \
  --input path/to/minif2f_test.jsonl \
  --output proofbench-pipeline/data/minif2f/pilot_10.jsonl
```

3. Install the pipeline dependencies for local `transformers` inference:

```bash
pip install -r proofbench-pipeline/requirements.txt
```

The default model configuration now uses local inference:
- provider: `local`
- model: `Qwen/Qwen2.5-Coder-7B-Instruct`
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
- `proofbench-pipeline/outputs/minif2f/pilot/repair_loop/results.csv`
- `proofbench-pipeline/outputs/minif2f/pilot/repair_loop/generations/`
- `proofbench-pipeline/outputs/minif2f/pilot/repair_loop/lean_files/`
- `proofbench-pipeline/outputs/minif2f/pilot/repair_loop/logs/`

## Lean Project

The pipeline now includes a minimal Lean 4 + Mathlib project in:

- [lean_project](/Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair/proofbench-pipeline/lean_project/ProofbenchLean.lean:1)

The verifier configuration points to this project via
[lean.yaml](/Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair/proofbench-pipeline/configs/lean.yaml:1).

For a first local setup, initialize the Lean dependencies once:

```bash
cd /Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair/proofbench-pipeline/lean_project
lake update
lake build ProofbenchLean
```

## Current Strategies

- `cot_to_lean`
- `direct_lean`
- `interleaved`

These proposal-level strategy names are normalized internally to the legacy
pipeline implementations, so old prompt files and generation logic still work
while the experiment interface matches the new proposal.

## Agent Modes

- `no_agent`
  - single-pass generation only
- `repair_loop`
  - extraction repair
  - Lean4 repair after compiler feedback

The summary output now also records `agent_mode` and `avg_repairs`.

## Reference Formalization Stage

The proposal's Step 1 now has a dedicated entrypoint:

```bash
python3 proofbench-pipeline/scripts/run_reference_formalization.py
```

This uses `reference_formalization -> reference_to_lean` internally and writes
results under `outputs/minif2f/reference_formalization/`.

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

Lean verification requires Lean 4 / `lake` to be available on the cluster node.
The sbatch script now looks for `lake` in the normal `PATH` and also in
`$HOME/.elan/bin`. If Lean is not installed yet, install `elan` first on a
login node, for example:

```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
```

The Ubelix job now also initializes Mathlib automatically with `lake update`
and runs a small `lake build ProofbenchLean` smoke build before launching the
benchmark.

By default this is cache-aware:
- if `.lake/packages/mathlib` already exists, `lake update` is skipped
- if `.lake/build/lib/lean/ProofbenchLean.olean` already exists, the smoke build is skipped

Useful overrides:

```bash
sbatch --export=REPO_ROOT=$(pwd),PROOFBENCH_NUM_SAMPLES=10,PROOFBENCH_DATA_INPUT=/path/to/minif2f_test.jsonl,PROOFBENCH_PREPARE_DATA=1 proofbench-pipeline/scripts/run_proofbench_ubelix.sbatch
```

Run a larger/full dataset pass by switching the mode and sample count:

```bash
sbatch --export=REPO_ROOT=$(pwd),PROOFBENCH_RUN_MODE=full,PROOFBENCH_NUM_SAMPLES=0,PROOFBENCH_DATA_INPUT=/path/to/minif2f_test.jsonl,PROOFBENCH_PREPARE_DATA=1 proofbench-pipeline/scripts/run_proofbench_ubelix.sbatch
```
