import json
from collections import defaultdict
from pathlib import Path

from extract import extract_answer

BASE_DIR = Path(__file__).resolve().parents[2]

def load_results(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def evaluate(records):
    correct = 0
    total = len(records)
    parse_fail = 0

    by_type = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in records:
        pred, fail = extract_answer(r["model_output"])
        gold = int(r["gold"])

        if fail:
            parse_fail += 1
            continue

        if pred == gold:
            correct += 1

        for t in r.get("problem_type", []):
            by_type[t]["total"] += 1
            if pred == gold:
                by_type[t]["correct"] += 1

    summary = {
        "accuracy": correct / total,
        "parse_failure_rate": parse_fail / total,
        "total": total,
    }

    for t in by_type:
        summary[f"type_{t}"] = (
            by_type[t]["correct"] / by_type[t]["total"]
            if by_type[t]["total"] > 0 else 0
        )

    return summary

def main():
    path = BASE_DIR / "benchmarks" / "aime2025" / "results" / "raw_generations.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Result file not found: {path}. Run run_infer.py successfully before eval.py."
        )
    records = load_results(path)
    summary = evaluate(records)

    print(summary)

if __name__ == "__main__":
    main()
