import csv
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[2]
    results_path = repo_root / "proofbench-pipeline" / "outputs" / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    by_method = {}
    with open(results_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"]
            stats = by_method.setdefault(method, {"total": 0, "success": 0})
            stats["total"] += 1
            if row["lean_success"] == "True":
                stats["success"] += 1

    for method, stats in sorted(by_method.items()):
        accuracy = stats["success"] / stats["total"] if stats["total"] else 0.0
        print(f"{method}: total={stats['total']} pass@1={accuracy:.3f}")


if __name__ == "__main__":
    main()
