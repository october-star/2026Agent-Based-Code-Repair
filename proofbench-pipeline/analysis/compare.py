"""
Analysis module: load results, build comparison table, generate plots.

Usage:
    python -m analysis.compare --results results/pilot_gpt4o_*/summary.json
"""
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)

# Published baselines (miniF2F-test, Pass@32)
PUBLISHED_BASELINES = [
    {"system": "GPT-4o",              "strategy": "Direct",        "agent": False, "pass_at_32": 0.230, "source": "LLM Eval 2026"},
    {"system": "DSP-V1.5-RL",         "strategy": "non-CoT",       "agent": False, "pass_at_32": 0.500, "source": "Xin 2024"},
    {"system": "DSP-V1.5-RL",         "strategy": "CoT",           "agent": False, "pass_at_32": 0.516, "source": "Xin 2024"},
    {"system": "Goedel-Prover-SFT",   "strategy": "Direct",        "agent": False, "pass_at_32": 0.576, "source": "Lin 2025"},
    {"system": "APOLLO + Goedel",      "strategy": "Direct+repair", "agent": True,  "pass_at_32": 0.656, "source": "Ospanov NeurIPS 2025"},
    {"system": "DSP-V2-7B",           "strategy": "non-CoT",       "agent": False, "pass_at_32": 0.734, "source": "Ren 2025"},
    {"system": "DSP-V2-7B",           "strategy": "CoT",           "agent": False, "pass_at_32": 0.750, "source": "Ren 2025"},
    {"system": "DSP-V2-671B",         "strategy": "CoT",           "agent": False, "pass_at_32": 0.824, "source": "Ren 2025 (SOTA)"},
    {"system": "Delta Prover",         "strategy": "Reflect+repair","agent": True,  "pass_at_32": 0.959, "source": "Zhou 2025 (SOTA-agent)"},
]


def load_results(paths: List[str]) -> List[Dict]:
    rows = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, list):
            rows.extend(data)
        elif isinstance(data, dict):
            rows.append(data)
    return rows


def print_comparison_table(our_rows: List[Dict]):
    """Print a formatted comparison table to stdout."""
    header = (
        f"{'System / Condition':<38} {'Strategy':<18} {'Agent':<6} "
        f"{'Pass@1':>7} {'Pass@32':>8} {'Compile%':>9} "
        f"{'BEq':>6} {'GTED':>6} {'Avg.Rep':>8}"
    )
    sep = "─" * len(header)

    print(sep)
    print(header)
    print(sep)

    # Published baselines
    print("  Published baselines")
    for b in PUBLISHED_BASELINES:
        print(
            f"  {b['system']:<36} {b['strategy']:<18} {'✓' if b['agent'] else '✗':<6} "
            f"{'—':>7} {b['pass_at_32']:>8.1%} {'—':>9} {'—':>6} {'—':>6} {'—':>8}"
        )

    print(sep)

    # Our results
    if our_rows:
        print("  ProofBench results (ours)")
        for row in our_rows:
            cond = row.get("condition", "?")
            model = row.get("model", "")
            p1   = row.get("pass_at_1", 0.0)
            p32  = row.get("pass_at_32", 0.0)
            comp = row.get("compile_rate", 0.0)
            beq  = row.get("beq_mean", 0.0)
            gted = row.get("gted_mean", 0.0)
            ar   = row.get("avg_repairs", None)
            agent_sym = "✓" if "agent" in cond else "✗"
            ar_str = f"{ar:.2f}" if ar is not None else "—"

            print(
                f"  [{model}] {cond:<32} {'—':<18} {agent_sym:<6} "
                f"{p1:>7.1%} {p32:>8.1%} {comp:>9.1%} "
                f"{beq:>6.3f} {gted:>6.3f} {ar_str:>8}"
            )

    print(sep)


def save_csv(our_rows: List[Dict], out_path: str):
    """Save comparison table as CSV."""
    import csv
    fieldnames = [
        "system", "condition", "model", "agent",
        "pass_at_1", "pass_at_32", "compile_rate",
        "beq_mean", "gted_mean", "avg_repairs", "source",
    ]
    rows = []
    for b in PUBLISHED_BASELINES:
        rows.append({
            "system": b["system"], "condition": b["strategy"],
            "model": "—", "agent": b["agent"],
            "pass_at_1": "—", "pass_at_32": b["pass_at_32"],
            "compile_rate": "—", "beq_mean": "—", "gted_mean": "—",
            "avg_repairs": "—", "source": b["source"],
        })
    for r in our_rows:
        rows.append({
            "system": "ProofBench", "condition": r.get("condition", ""),
            "model": r.get("model", ""),
            "agent": "agent" in r.get("condition", ""),
            "pass_at_1":     r.get("pass_at_1", ""),
            "pass_at_32":    r.get("pass_at_32", ""),
            "compile_rate":  r.get("compile_rate", ""),
            "beq_mean":      r.get("beq_mean", ""),
            "gted_mean":     r.get("gted_mean", ""),
            "avg_repairs":   r.get("avg_repairs", ""),
            "source": "ours",
        })
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"CSV saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", default=[],
                        help="Paths to summary JSON files")
    parser.add_argument("--csv", default="results/comparison.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    our_rows = load_results(args.results) if args.results else []
    print_comparison_table(our_rows)

    if our_rows:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        save_csv(our_rows, args.csv)


if __name__ == "__main__":
    main()
