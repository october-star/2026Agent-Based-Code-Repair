"""
ProofBench experiment runner.

Runs the full 3-strategy × 2-agent-mode pipeline for one model
and saves JSON results.

Usage:
    python -m experiments.run --config config.yaml --model gpt4o --pilot
    python -m experiments.run --config config.yaml --model deepseek --full
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import yaml

# Make sure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.schemas import Problem, Strategy
from src.generation.base import LLMClient
from src.generation.strategy1 import CoTStrategy
from src.generation.strategy2 import DirectStrategy
from src.generation.strategy3 import InterleavedStrategy
from src.lean.executor import LeanExecutor
from src.agent.repair_loop import RepairLoop
from src.formalization.ref_pipeline import RefPipeline
from src.evaluation.pass_at_k import compute_pass_at_k, compute_compile_rate
from src.evaluation.beq_gted import compute_beq, compute_gted
from src.evaluation.agent_metrics import (
    compute_refine_at_k,
    compute_avg_repairs,
    analyze_error_types,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-20s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("proofbench.runner")


# ─────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Expand env vars in api_key fields
    for m in cfg.get("models", {}).values():
        key = m.get("api_key", "")
        if key.startswith("${") and key.endswith("}"):
            env_var = key[2:-1]
            m["api_key"] = os.environ.get(env_var, "")
    return cfg


# ─────────────────────────────────────────────────────────────
# Core runner
# ─────────────────────────────────────────────────────────────

def run_condition(
    problems: List[Problem],
    strategy,
    agent: bool,
    executor: LeanExecutor,
    K: int,
    max_iter: int,
) -> List[Dict]:
    """
    Run one (strategy, agent_mode) condition over all problems.
    Returns list of per-problem result dicts.
    """
    repair_loop = RepairLoop(executor, max_iter=max_iter)
    results = []

    for prob in problems:
        attempts = []

        for sample_i in range(K):
            if agent:
                out = repair_loop.run(prob, strategy)
                attempts.append({
                    "sample": sample_i,
                    "success": out["success"],
                    "proof": out["proof"],
                    "error_msg": (
                        out["history"][-1]["error"] if out["history"] else None
                    ),
                    "iterations": out["iterations"],
                    "history": out["history"],
                })
            else:
                proof, cot = strategy.generate(prob)
                lean_result = executor.verify(proof) if proof else None
                attempts.append({
                    "sample": sample_i,
                    "success": lean_result.success if lean_result else False,
                    "proof": proof,
                    "error_msg": lean_result.error_msg if lean_result else "no proof",
                    "iterations": 1,
                    "history": [],
                })

        # Per-problem BEq / GTED against reference Lean proof
        beq = gted = 0.0
        best = next((a["proof"] for a in attempts if a["success"]), None)
        if best and prob.ref_lean_proof:
            beq = compute_beq(best, prob.ref_lean_proof, prob.formal_statement, executor)
            gted = compute_gted(best, prob.ref_lean_proof)

        results.append({
            "problem_id": prob.id,
            "attempts": attempts,
            "beq": beq,
            "gted": gted,
            "success": any(a["success"] for a in attempts),
        })

        n_ok = sum(1 for a in attempts if a["success"])
        logger.info(
            f"  [{prob.id}] {n_ok}/{K} passed | "
            f"BEq={beq:.2f} GTED={gted:.2f}"
        )

    return results


def aggregate_metrics(results: List[Dict], agent: bool, K: int) -> Dict:
    k_vals = [k for k in [1, 32] if k <= K]
    metrics: Dict = {
        **compute_pass_at_k(results, k_values=k_vals),
        "compile_rate": compute_compile_rate(results),
        "beq_mean": sum(r["beq"] for r in results) / max(len(results), 1),
        "gted_mean": sum(r["gted"] for r in results) / max(len(results), 1),
    }
    if agent:
        metrics["refine_at_5"]  = compute_refine_at_k(
            [{"success": r["success"],
              "iterations": min(
                  (a["iterations"] for a in r["attempts"] if a["success"]),
                  default=99
              )} for r in results],
            k=5,
        )
        metrics["avg_repairs"]  = compute_avg_repairs(
            [{"success": r["success"],
              "iterations": min(
                  (a["iterations"] for a in r["attempts"] if a["success"]),
                  default=99
              )} for r in results]
        )
        metrics["error_types"]  = analyze_error_types(results)
    return metrics


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ProofBench experiment runner")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default="gpt4o",
                        help="Key in config.models (e.g. gpt4o, deepseek_prover)")
    parser.add_argument("--pilot", action="store_true",
                        help="Run on pilot_n samples only")
    parser.add_argument("--full",  action="store_true",
                        help="Run on full miniF2F-test")
    parser.add_argument("--skip-ref", action="store_true",
                        help="Skip Step 1 reference formalization")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── Dataset ─────────────────────────────────────────────
    from src.data.loader import load_from_jsonl, load_pilot, load_minif2f
    local_jsonl = cfg["dataset"].get("local_jsonl")
    if args.pilot:
        if local_jsonl:
            problems = load_from_jsonl(local_jsonl)[:cfg["dataset"]["pilot_n"]]
        else:
            problems = load_pilot(n=cfg["dataset"]["pilot_n"])
        run_tag = "pilot"
    else:
        if local_jsonl:
            problems = load_from_jsonl(local_jsonl)
        else:
            problems = load_minif2f(split=cfg["dataset"]["split"])
        run_tag = "full"
    logger.info(f"Loaded {len(problems)} problems ({run_tag})")

    # ── Executor ─────────────────────────────────────────────
    lean_cfg = cfg["lean"]
    executor = LeanExecutor(
        timeout=lean_cfg["timeout"],
        lean_bin=lean_cfg["lean_bin"],
        project_dir=lean_cfg.get("project_dir"),
        mock_mode=lean_cfg["mock_mode"],
        mock_pass_rate=lean_cfg.get("mock_pass_rate", 0.30),
    )

    # ── LLM client ───────────────────────────────────────────
    model_cfg = cfg["models"][args.model]
    client = LLMClient(
        model=model_cfg["name"],
        api_key=model_cfg["api_key"],
        base_url=model_cfg.get("base_url"),
        temperature=cfg["inference"]["temperature"],
        max_tokens=cfg["inference"]["max_tokens"],
        backend=model_cfg.get("backend", "openai"),
        local_files_only=model_cfg.get("local_files_only", False),
    )

    # ── Step 1: Reference formalization ──────────────────────
    if not args.skip_ref:
        logger.info("=== Step 1: Reference formalization ===")
        ref = RefPipeline(
            client=client,
            executor=executor,
            max_retries=3,
            cache_path=cfg["paths"]["formalized_dir"] + "ref_cache.json",
        )
        ref_stats = ref.formalize_all(problems)
        logger.info(f"Reference formalization: {ref_stats}")

    # ── Step 2: Strategy experiments ─────────────────────────
    strategies = {
        "S1_CoT":        CoTStrategy(client),
        "S2_Direct":     DirectStrategy(client),
        "S3_Interleaved": InterleavedStrategy(client),
    }

    K         = cfg["inference"]["K"]
    max_iter  = cfg["agent"]["max_iter"]

    out_dir = Path(cfg["paths"]["results_dir"]) / f"{run_tag}_{args.model}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    summary_rows = []

    for strat_name, strategy in strategies.items():
        for agent_mode in [False, True]:
            label = f"{strat_name}_{'agent' if agent_mode else 'noagent'}"
            logger.info(f"=== Condition: {label} ===")

            results = run_condition(
                problems=problems,
                strategy=strategy,
                agent=agent_mode,
                executor=executor,
                K=K,
                max_iter=max_iter,
            )

            metrics = aggregate_metrics(results, agent=agent_mode, K=K)
            logger.info(f"  Metrics: {metrics}")

            all_results[label] = {"problems": results, "metrics": metrics}

            # Save per-condition results
            cond_path = out_dir / f"{label}.json"
            with open(cond_path, "w") as f:
                json.dump({"condition": label, "metrics": metrics,
                           "problems": results}, f, indent=2, default=str)

            summary_rows.append({"condition": label, "model": args.model, **metrics})

    # ── Summary table ─────────────────────────────────────────
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2, default=str)

    logger.info(f"\n{'='*50}")
    logger.info(f"Results saved to: {out_dir}")
    logger.info(f"{'='*50}")

    # Print quick summary
    print("\n── Summary ──────────────────────────────────────")
    print(f"{'Condition':<35} {'Pass@1':>7} {'Pass@32':>8} {'Compile%':>9} {'BEq':>6} {'GTED':>6}")
    print("-" * 75)
    for row in summary_rows:
        print(
            f"{row['condition']:<35} "
            f"{row.get('pass_at_1', 0):.3f}   "
            f"{row.get('pass_at_32', 0):.3f}    "
            f"{row.get('compile_rate', 0):.3f}     "
            f"{row.get('beq_mean', 0):.3f}  "
            f"{row.get('gted_mean', 0):.3f}"
        )


if __name__ == "__main__":
    main()
