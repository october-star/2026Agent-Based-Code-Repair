# ProofBench

A systematic evaluation framework for LLM-based formal proof generation strategies in Lean 4.

## Overview

ProofBench compares three proof generation paradigms across two agent modes on miniF2F-test, and evaluates with six metrics (Pass@K, Compile%, BEq, GTED, refine@K, Avg. repairs).

```
Dataset (miniF2F)
       │
       ├── Step 1: Reference formalization  ──→  ref Lean proof (ground truth)
       │
       └── Step 2: Generation (×2 models)
              ├── S1: CoT → Lean    ┐
              ├── S2: Direct Lean   ├── × {No agent, Agent (repair loop)}
              └── S3: Interleaved   ┘
                         │
                  Lean 4 executor
                         │
              ┌──────────┴──────────┐
           Pass@K               BEq / GTED
           Compile%             refine@K / Avg.repairs
                         │
                  Analysis & comparison
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API keys
export OPENAI_API_KEY=sk-...
export DEEPSEEK_API_KEY=...

# 3. Run 10-sample pilot (mock Lean — no Lean 4 install needed)
python -m experiments.run --config config.yaml --model gpt4o --pilot

# 4. Compare results against published baselines
python -m analysis.compare --results results/pilot_*/summary.json
```

## Using real Lean 4

Edit `config.yaml`:
```yaml
lean:
  mock_mode: false
  project_dir: /path/to/your/mathlib-project   # where `lake exe repl` works
  timeout: 60
```

Then re-run the experiment.

## Project structure

```
proofbench/
├── config.yaml                  Main configuration
├── requirements.txt
├── src/
│   ├── data/
│   │   ├── schemas.py           Problem, LeanResult, ProofAttempt dataclasses
│   │   └── loader.py            miniF2F HuggingFace loader
│   ├── formalization/
│   │   └── ref_pipeline.py      Step 1: NL → Lean ref proof
│   ├── generation/
│   │   ├── prompts.py           All prompt templates (S1/S2/S3/REF/REPAIR)
│   │   ├── base.py              LLMClient + output parsers
│   │   ├── strategy1.py         CoT → Lean
│   │   ├── strategy2.py         Direct Lean
│   │   └── strategy3.py         Interleaved CoT + Lean
│   ├── agent/
│   │   └── repair_loop.py       Iterative proof repair agent
│   ├── lean/
│   │   └── executor.py          Lean 4 REPL interface + mock mode
│   └── evaluation/
│       ├── pass_at_k.py         Pass@K, compile rate
│       ├── beq_gted.py          BEq (semantic equiv.) + GTED (structural sim.)
│       └── agent_metrics.py     refine@K, avg_repairs, error type analysis
├── experiments/
│   └── run.py                   Full experiment runner (CLI)
├── analysis/
│   └── compare.py               Build comparison table vs. published baselines
└── results/                     JSON output (gitignored)
```

## Experimental conditions

| Condition | Strategy | Agent | Note |
|-----------|----------|-------|------|
| C1 | S1: CoT → Lean | No | compare vs DSP-V1.5-RL CoT |
| C2 | S2: Direct Lean | No | compare vs DSP-V1.5-RL non-CoT |
| C3 | S3: Interleaved | No | compare vs Lean-STaR |
| C4 | S1 + Agent | Yes | agent benefit for CoT strategy |
| C5 | S2 + Agent | Yes | compare vs APOLLO |
| C6 | S3 + Agent | Yes | agent benefit for interleaved |

## Citation

```bibtex
@misc{proofbench2026,
  title  = {ProofBench: A Systematic Evaluation Framework for LLM-Based Formal Proof Generation},
  year   = {2026},
  note   = {https://github.com/wenjiema02/proofbench}
}
```
