from pathlib import Path


TEMPLATE_FILES = {
    "reference_to_lean": "reference_to_lean.txt",
    "cot_generation": "cot_generation.txt",
    "cot_then_lean": "cot_to_lean.txt",
    "direct_lean": "direct_lean.txt",
    "mixed_cot_lean": "mixed_cot_lean.txt",
    "format_to_lean": "format_to_lean.txt",
    "lean4_repair": "lean4_repair.txt",
}


def load_template(prompts_dir, method):
    filename = TEMPLATE_FILES[method]
    path = Path(prompts_dir) / filename
    with open(path, encoding="utf-8") as f:
        return f.read()


def build_prompt(
    method,
    sample,
    prompts_dir,
    natural_language_proof="",
    raw_generation="",
    lean_code="",
    stdout="",
    stderr="",
):
    template = load_template(prompts_dir, method)
    return (
        template.replace("{{problem}}", sample.problem)
        .replace("{{reference_solution}}", sample.reference_solution or "")
        .replace("{{natural_language_proof}}", natural_language_proof or "")
        .replace("{{raw_generation}}", raw_generation or "")
        .replace("{{lean_code}}", lean_code or "")
        .replace("{{stdout}}", stdout or "")
        .replace("{{stderr}}", stderr or "")
    )
