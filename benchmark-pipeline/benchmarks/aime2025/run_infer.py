from pathlib import Path
import yaml
from tqdm import tqdm
from transformers import pipeline
from dataset import load_aime
from prompts import build_prompt
from utils import save_jsonl

BASE_DIR = Path(__file__).resolve().parents[2]


def load_config():
    config_path = BASE_DIR / "configs" / "base.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def init_model(config):
    generator = pipeline(
        task="text-generation",
        model=config["model"]["name"],
        device_map="auto",
    )
    return generator


def generate(generator, prompt, config):
    outputs = generator(
        prompt,
        max_new_tokens=config["model"]["max_new_tokens"],
        do_sample=config["model"]["do_sample"],
        temperature=config["model"]["temperature"],
        return_full_text=False,
    )
    return outputs[0]["generated_text"]


def main():
    config = load_config()
    ds = load_aime()
    generator = init_model(config)

    output_path = BASE_DIR / config["output"]["dir"] / config["output"]["file"]

    for i, item in enumerate(tqdm(ds)):
        problem = item["problem"]
        gold = item["answer"]
        prompt = build_prompt(problem)

        output = generate(generator, prompt, config)

        record = {
            "idx": i,
            "problem": problem,
            "gold": gold,
            "problem_type": item.get("problem_type", []),
            "model_output": output,
            "model_name": config["model"]["name"],
        }
        save_jsonl(str(output_path), record)


if __name__ == "__main__":
    main()
