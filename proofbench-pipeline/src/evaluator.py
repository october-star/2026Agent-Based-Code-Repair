import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


RESULT_COLUMNS = [
    "sample_id",
    "method",
    "problem",
    "reference_solution",
    "raw_generation",
    "lean_code",
    "lean_success",
    "error_type",
    "stderr",
    "uses_sorry",
    "runtime_sec",
    "extraction_success",
]


def write_results_csv(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for record in records:
            row = {column: record.get(column, "") for column in RESULT_COLUMNS}
            writer.writerow(row)


def write_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def summarize_by_method(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["method"]].append(record)

    summary = []
    for method, items in grouped.items():
        total = len(items)
        if total == 0:
            continue
        success_count = sum(1 for item in items if item.get("lean_success"))
        no_sorry_count = sum(
            1
            for item in items
            if not item.get("uses_sorry") and item.get("error_type") != "sorry_used"
        )
        extraction_success = sum(1 for item in items if item.get("extraction_success"))
        main_error = Counter(item.get("error_type", "unknown_error") for item in items).most_common(1)[0][0]
        summary.append(
            {
                "method": method,
                "pass_at_1": success_count / total,
                "compile_rate": success_count / total,
                "no_sorry_rate": no_sorry_count / total,
                "extraction_success_rate": extraction_success / total,
                "main_error": main_error,
            }
        )
    return summary
