import json
from pathlib import Path

import yaml

from code_extractor import extract_lean_code
from data_loader import load_jsonl
from evaluator import summarize_by_method, write_jsonl, write_results_csv
from lean_runner import run_lean
from llm_client import LLMClient
from prompt_builder import build_prompt


class ProofBenchPipeline:
    def __init__(self, base_dir, experiment_path, model_path, lean_path):
        self.base_dir = Path(base_dir)
        self.experiment = self._load_yaml(experiment_path)["experiment"]
        self.model = self._load_yaml(model_path)["model"]
        self.lean = self._load_yaml(lean_path)["lean"]
        self.prompts_dir = self.base_dir / "prompts"
        self.outputs_dir = self.base_dir / self.experiment["outputs_dir"]
        self.client = LLMClient(self.model)

    def _load_yaml(self, path):
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_samples(self, sample_limit=None):
        dataset_path = self.base_dir / self.experiment["dataset_path"]
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {dataset_path}. "
                "Run proofbench-pipeline/scripts/prepare_data.py first."
            )
        samples = load_jsonl(dataset_path)
        if sample_limit is None:
            sample_limit = self.experiment.get("sample_limit")
        return samples[:sample_limit]

    def _build_generation(self, method, sample):
        natural_language_proof = ""
        intermediate_generation = ""

        # COT methods
        # 1. generate the natural language proof with a COT prompt
        # 2. feed the natural language proof to a second prompt to generate the Lean code
        if method == "cot_then_lean":
            cot_prompt = build_prompt("cot_generation", sample, self.prompts_dir)
            natural_language_proof = self.client.generate(cot_prompt)
            prompt = build_prompt(
                "cot_then_lean",
                sample,
                self.prompts_dir,
                natural_language_proof=natural_language_proof,
            )
            raw_generation = self.client.generate(prompt)
            intermediate_generation = natural_language_proof

        # Direct generation methods    
        else:
            prompt = build_prompt(method, sample, self.prompts_dir)
            raw_generation = self.client.generate(prompt)

        return {
            "raw_generation": raw_generation,
            "intermediate_generation": intermediate_generation,
        }

    # 1. For each sample and method, generate the Lean code 
    # 2. Run the generated Lean code and record the results
    # 3. Save the generation and evalution results to json files and a summary csv file.
    def run(self, sample_limit=None, methods=None):
        methods = methods or self.experiment["methods"]
        samples = self._load_samples(sample_limit=sample_limit)
        records = []

        generations_dir = self.outputs_dir / "generations"
        lean_files_dir = self.outputs_dir / "lean_files"
        logs_dir = self.outputs_dir / "logs"
        generations_dir.mkdir(parents=True, exist_ok=True)
        lean_files_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        for sample in samples:
            for method in methods:
                generation = self._build_generation(method, sample)
                extraction = extract_lean_code(generation["raw_generation"])
                lean_filename = f"{sample.sample_id}__{method}.lean"
                lean_path = lean_files_dir / lean_filename
                if extraction["extraction_success"]:
                    lean_result = run_lean(
                        extraction["lean_code"],
                        timeout=self.lean.get("timeout", 30),
                        command=self.lean.get("command"),
                        project_dir=self.lean.get("project_dir", ""),
                        output_path=lean_path,
                    )
                else:
                    lean_path.parent.mkdir(parents=True, exist_ok=True)
                    lean_path.write_text(extraction["lean_code"], encoding="utf-8")
                    lean_result = {
                        "success": False,
                        "stdout": "",
                        "stderr": "Lean extraction failed before verification.",
                        "contains_sorry": False,
                        "contains_admit": False,
                        "error_type": "extraction_failed",
                        "runtime_sec": 0.0,
                    }

                record = {
                    "sample_id": sample.sample_id,
                    "method": method,
                    "problem": sample.problem,
                    "reference_solution": sample.reference_solution,
                    "raw_generation": generation["raw_generation"],
                    "intermediate_generation": generation["intermediate_generation"],
                    "lean_code": extraction["lean_code"],
                    "lean_success": lean_result["success"],
                    "error_type": lean_result["error_type"],
                    "stderr": lean_result["stderr"],
                    "uses_sorry": lean_result["contains_sorry"] or lean_result["contains_admit"],
                    "runtime_sec": lean_result["runtime_sec"],
                    "extraction_success": extraction["extraction_success"],
                }
                records.append(record)

                generation_path = generations_dir / f"{sample.sample_id}__{method}.json"
                with open(generation_path, "w", encoding="utf-8") as f:
                    json.dump(record, f, ensure_ascii=False, indent=2)
                    f.write("\n")

        write_results_csv(records, self.outputs_dir / "results.csv")
        write_jsonl(records, logs_dir / "results.jsonl")

        summary = summarize_by_method(records)
        with open(logs_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")

        return {
            "records": records,
            "summary": summary,
        }
