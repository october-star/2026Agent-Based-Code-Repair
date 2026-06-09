# Legacy Benchmark

This directory preserves the earlier ProofBench-style pipeline so you can keep
the old workflow separate from the new proposal-driven miniF2F workflow in
[formal-benchmark](/Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair/formal-benchmark/README.md:1).

## Purpose

- dataset default: `wenjiema02/ProofBench`
- legacy methods:
  - `reference_to_lean`
  - `cot_then_lean`
  - `direct_lean`
  - `mixed_cot_lean`
- default output root:
  - `legacy-benchmark/outputs/`

## Main Entry Points

- pilot run:
  [run_10_samples.py](/Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair/legacy-benchmark/scripts/run_10_samples.py:1)
- full run:
  [run_full.py](/Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair/legacy-benchmark/scripts/run_full.py:1)
- data preparation:
  [prepare_data.py](/Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair/legacy-benchmark/scripts/prepare_data.py:1)
- cluster run:
  [run_proofbench_ubelix.sbatch](/Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair/legacy-benchmark/scripts/run_proofbench_ubelix.sbatch:1)

## Default Config

The old workflow still defaults to:
- [experiment.yaml](/Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair/legacy-benchmark/configs/experiment.yaml:1)
- [model.yaml](/Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair/legacy-benchmark/configs/model.yaml:1)
- [lean.yaml](/Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair/legacy-benchmark/configs/lean.yaml:1)

## Quick Start

```bash
cd /Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair
python3 legacy-benchmark/scripts/prepare_data.py
python3 legacy-benchmark/scripts/run_10_samples.py
```

## Relation To The New Workflow

- use [legacy-benchmark](/Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair/legacy-benchmark/README.md:1)
  when you want the earlier ProofBench-style pipeline
- use [formal-benchmark](/Users/octobercity/Desktop/project/2026Agent-Based-Code-Repair/formal-benchmark/README.md:1)
  when you want the new proposal-aligned `miniF2F-test` workflow
