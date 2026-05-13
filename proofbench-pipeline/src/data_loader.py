from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class ProofBenchSample:
    sample_id: str
    problem: str
    reference_solution: str = ""
    source: str = ""
    generator: str = ""
    model_solution: str = ""
    expert_rating: int | None = None
    marking_scheme: str = ""
    generation_prompt: str = ""
    contest: str = ""
    contest_year: str = ""

# normalize the raw sample dict to ProofBenchSample
def _normalize_sample(raw, index):
    metadata = raw.get("metadata") or {}
    sample_id = raw.get("sample_id") or raw.get("problem_id") or raw.get("id") or f"sample-{index}"
    problem = raw.get("problem") or raw.get("question") or raw.get("prompt") or ""
    reference_solution = (
        raw.get("reference_solution")
        or raw.get("reference")
        or raw.get("solution")
        or ""
    )
    source = raw.get("source") or raw.get("competition") or metadata.get("contest") or ""
    return ProofBenchSample(
        sample_id=sample_id,
        problem=problem.strip(),
        reference_solution=reference_solution.strip(),
        source=source.strip(),
        generator=(raw.get("generator") or "").strip(),
        model_solution=(raw.get("model_solution") or "").strip(),
        expert_rating=raw.get("expert_rating"),
        marking_scheme=(raw.get("marking_scheme") or "").strip(),
        generation_prompt=(raw.get("generation_prompt") or "").strip(),
        contest=(metadata.get("contest") or "").strip(),
        contest_year=str(metadata.get("contest_year") or "").strip(),
    )


def load_jsonl(path):
    path = Path(path)
    samples = []
    with open(path, encoding="utf-8") as f:
        for index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            samples.append(_normalize_sample(json.loads(line), index))
    return samples


def save_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")

# Load dataset from Hugging Face Hub
def load_hf_dataset(dataset_name, split):
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)
    samples = []
    for index, raw in enumerate(dataset):
        samples.append(_normalize_sample(dict(raw), index))
    return samples

# Convert list of ProofBenchSample to list of dict for saving
def build_sample_records(samples):
    records = []
    for sample in samples:
        records.append(
            {
                "sample_id": sample.sample_id,
                "problem": sample.problem,
                "reference_solution": sample.reference_solution,
                "source": sample.source,
                "generator": sample.generator,
                "model_solution": sample.model_solution,
                "expert_rating": sample.expert_rating,
                "marking_scheme": sample.marking_scheme,
                "generation_prompt": sample.generation_prompt,
                "contest": sample.contest,
                "contest_year": sample.contest_year,
            }
        )
    return records
