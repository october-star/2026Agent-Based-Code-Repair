from pathlib import Path


TEMPLATE_FILES = {
    "reference_to_lean": "reference_to_lean.txt",
    "cot_generation": "cot_generation.txt",
    "cot_then_lean": "cot_to_lean.txt",
    "direct_lean": "direct_lean.txt",
    "mixed_cot_lean": "mixed_cot_lean.txt",
}


def load_template(prompts_dir, method):
    filename = TEMPLATE_FILES[method]
    path = Path(prompts_dir) / filename
    with open(path, encoding="utf-8") as f:
        return f.read()


def build_prompt(method, sample, prompts_dir, natural_language_proof=""):
    template = load_template(prompts_dir, method)
    return (
        template.replace("{{problem}}", sample.problem)
        .replace("{{reference_solution}}", sample.reference_solution or "")
        .replace("{{natural_language_proof}}", natural_language_proof or "")
    )
