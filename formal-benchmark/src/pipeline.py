import json
from pathlib import Path
import re

import yaml

from code_extractor import extract_lean_code
from data_loader import load_jsonl
from evaluator import summarize_by_method, write_jsonl, write_results_csv
from lean_runner import run_lean
from llm_client import LLMClient
from prompt_builder import build_prompt
from strategies import normalize_strategy_name


class ProofBenchPipeline:
    def __init__(self, base_dir, experiment_path, model_path, lean_path):
        self.base_dir = Path(base_dir)
        self.experiment = self._load_yaml(experiment_path)["experiment"]
        self.model = self._load_yaml(model_path)["model"]
        self.lean = self._load_yaml(lean_path)["lean"]
        self.prompts_dir = self.base_dir / "prompts"
        self.outputs_dir = self.base_dir / self.experiment["outputs_dir"]
        self.client = LLMClient(self.model)
        self.agent_mode = (self.experiment.get("agent_mode") or "repair_loop").strip().lower()

    def _load_yaml(self, path):
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_samples(self, sample_limit=None):
        dataset_path = self.base_dir / self.experiment["dataset_path"]
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {dataset_path}. "
                "Prepare the configured dataset file first, for example with "
                "formal-benchmark/scripts/prepare_minif2f_data.py."
            )
        samples = load_jsonl(dataset_path)
        if sample_limit is None:
            sample_limit = self.experiment.get("sample_limit")
        return samples[:sample_limit]

    def _resolve_project_dir(self):
        project_dir = (self.lean.get("project_dir") or "").strip()
        if not project_dir:
            return ""
        project_path = Path(project_dir)
        if not project_path.is_absolute():
            project_path = self.base_dir / project_path
        return str(project_path)

    def _extract_final_answer_text(self, generation):
        intermediate = (generation.get("intermediate_generation") or "").strip()
        if intermediate:
            return intermediate

        for key in ("lean4_repair_generation", "repair_generation", "raw_generation"):
            text = (generation.get(key) or "").strip()
            if not text:
                continue
            text = re.sub(r"```lean\s*.*?```", "", text, flags=re.IGNORECASE | re.DOTALL)
            text = re.sub(r"```\s*.*?```", "", text, flags=re.DOTALL)
            text = re.sub(r"```lean\s*.*\Z", "", text, flags=re.IGNORECASE | re.DOTALL)
            text = re.sub(r"```\s*.*\Z", "", text, flags=re.DOTALL)
            text = text.strip()
            if text:
                return text

        return ""

    def _repair_loop_enabled(self):
        return self.agent_mode not in {"no_agent", "off", "none", "disabled"}

    def _build_generation(self, method, sample):
        natural_language_proof = ""
        intermediate_generation = ""
        repair_generation = ""
        lean4_repair_generation = ""

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
            "repair_generation": repair_generation,
            "lean4_repair_generation": lean4_repair_generation,
        }

    def _repair_extraction(self, sample, generation):
        raw_generation = generation.get("repair_generation") or generation["raw_generation"]
        try:
            repair_prompt = build_prompt(
                "format_to_lean",
                sample,
                self.prompts_dir,
                raw_generation=raw_generation,
            )
        except TypeError:
            # Backward compatibility for environments still using the older
            # prompt_builder.build_prompt signature without raw_generation.
            repair_prompt = build_prompt("format_to_lean", sample, self.prompts_dir)
            repair_prompt = repair_prompt.replace("{{raw_generation}}", raw_generation)
        repair_generation = self.client.generate(repair_prompt)
        return {
            "raw_generation": generation["raw_generation"],
            "intermediate_generation": generation["intermediate_generation"],
            "repair_generation": repair_generation,
            "lean4_repair_generation": generation.get("lean4_repair_generation", ""),
        }

    def _needs_lean4_repair(self, lean_result, lean_code):
        error_type = lean_result.get("error_type", "")
        lower_code = (lean_code or "").lower()
        if error_type in {
            "lean3_style_or_import",
            "missing_import",
            "syntax_error",
            "unknown_identifier",
            "placeholder_proof",
            "sorry_used",
        }:
            return True
        return (
            "import data." in lower_code
            or "import tactic" in lower_code
            or "#rinteractive" in lower_code
        )

    def _repair_lean4(self, sample, generation, lean_code, lean_result):
        repair_prompt = build_prompt(
            "lean4_repair",
            sample,
            self.prompts_dir,
            lean_code=lean_code,
            stdout=lean_result.get("stdout", ""),
            stderr=lean_result.get("stderr", ""),
        )
        lean4_repair_generation = self.client.generate(repair_prompt)
        return {
            "raw_generation": generation["raw_generation"],
            "intermediate_generation": generation["intermediate_generation"],
            "repair_generation": generation.get("repair_generation", ""),
            "lean4_repair_generation": lean4_repair_generation,
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
        project_dir = self._resolve_project_dir()

        for sample in samples:
            for strategy_name in methods:
                backend_method = normalize_strategy_name(strategy_name)
                repair_iterations = 0
                generation = self._build_generation(backend_method, sample)
                extraction = extract_lean_code(generation["raw_generation"])
                if self._repair_loop_enabled() and (
                    extraction.get("possibly_truncated") or not extraction["extraction_success"]
                ):
                    generation = self._repair_extraction(sample, generation)
                    extraction = extract_lean_code(generation["repair_generation"])
                    repair_iterations += 1
                lean_filename = f"{sample.sample_id}__{strategy_name}.lean"
                lean_path = lean_files_dir / lean_filename
                if extraction["extraction_success"]:
                    lean_result = run_lean(
                        extraction["lean_code"],
                        timeout=self.lean.get("timeout", 30),
                        command=self.lean.get("command"),
                        project_dir=project_dir,
                        output_path=lean_path,
                    )
                    if self._repair_loop_enabled() and not lean_result["success"] and self._needs_lean4_repair(
                        lean_result, extraction["lean_code"]
                    ):
                        generation = self._repair_lean4(
                            sample, generation, extraction["lean_code"], lean_result
                        )
                        repair_iterations += 1
                        repaired_extraction = extract_lean_code(
                            generation["lean4_repair_generation"]
                        )
                        if repaired_extraction["extraction_success"]:
                            extraction = repaired_extraction
                            lean_result = run_lean(
                                extraction["lean_code"],
                                timeout=self.lean.get("timeout", 30),
                                command=self.lean.get("command"),
                                project_dir=project_dir,
                                output_path=lean_path,
                            )
                else:
                    lean_path.parent.mkdir(parents=True, exist_ok=True)
                    lean_path.write_text(extraction["lean_code"], encoding="utf-8")
                    lean_result = {
                        "success": False,
                        "returncode": None,
                        "stdout": "",
                        "stderr": "Lean extraction failed before verification.",
                        "contains_sorry": False,
                        "contains_admit": False,
                        "error_type": "extraction_failed",
                        "runtime_sec": 0.0,
                    }

                record = {
                    "sample_id": sample.sample_id,
                    "method": strategy_name,
                    "strategy_name": strategy_name,
                    "backend_method": backend_method,
                    "agent_mode": self.agent_mode,
                    "repair_iterations": repair_iterations,
                    "problem": sample.problem,
                    "reference_solution": sample.reference_solution,
                    "final_answer_text": self._extract_final_answer_text(generation),
                    "raw_generation": generation["raw_generation"],
                    "intermediate_generation": generation["intermediate_generation"],
                    "repair_generation": generation.get("repair_generation", ""),
                    "lean4_repair_generation": generation.get("lean4_repair_generation", ""),
                    "lean_code": extraction["lean_code"],
                    "lean_success": lean_result["success"],
                    "returncode": lean_result.get("returncode"),
                    "error_type": lean_result["error_type"],
                    "stdout": lean_result.get("stdout", ""),
                    "stderr": lean_result["stderr"],
                    "uses_sorry": lean_result["contains_sorry"] or lean_result["contains_admit"],
                    "runtime_sec": lean_result["runtime_sec"],
                    "extraction_success": extraction["extraction_success"],
                }
                records.append(record)

                generation_path = generations_dir / f"{sample.sample_id}__{strategy_name}.json"
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
