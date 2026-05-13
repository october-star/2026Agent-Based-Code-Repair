import json
from collections import defaultdict
from pathlib import Path
import os
import yaml

from extract import extract_answer
from utils import read_latest_run_id, save_json

BASE_DIR = Path(__file__).resolve().parents[2]

def load_results(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_config():
    config_path = BASE_DIR / "configs" / "base.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def evaluate(records):
    correct = 0
    total = len(records)
    parse_fail = 0

    by_type = defaultdict(lambda: {"correct": 0, "total": 0})

    if total == 0:
        return {
            "accuracy": 0.0,
            "parse_failure_rate": 0.0,
            "parsed_rate": 0.0,
            "total": 0,
            "correct": 0,
            "parse_failures": 0,
        }

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
        "parsed_rate": (total - parse_fail) / total,
        "total": total,
        "correct": correct,
        "parse_failures": parse_fail,
    }

    if records:
        summary["model_name"] = records[0].get("model_name")
        summary["model_slug"] = records[0].get("model_slug")
        summary["run_id"] = records[0].get("run_id")
        summary["provider"] = records[0].get("provider", "local")
        summary["dataset_name"] = records[0].get("dataset_name")
        summary["dataset_split"] = records[0].get("dataset_split")
        summary["max_examples"] = records[0].get("max_examples")
        summary["tool_use"] = records[0].get("tool_use", "disabled")

    for t in by_type:
        summary[f"type_{t}"] = (
            by_type[t]["correct"] / by_type[t]["total"]
            if by_type[t]["total"] > 0 else 0
        )

    return summary

def main():
    config = load_config()
    output_root = BASE_DIR / config["output"]["dir"]
    run_id = os.getenv("RUN_ID") or read_latest_run_id(str(output_root))
    if not run_id:
        raise RuntimeError("No RUN_ID found for evaluation. Run run_infer.py first.")

    output_dir = output_root / run_id
    path = output_dir / config["output"]["file"]
    if not path.exists():
        raise FileNotFoundError(
            f"Result file not found: {path}. Run run_infer.py successfully before eval.py."
        )
    records = load_results(path)
    summary = evaluate(records)
    summary_path = output_dir / config["output"].get("summary_file", "summary.json")
    save_json(str(summary_path), summary)

    print(summary)

if __name__ == "__main__":
    main()
