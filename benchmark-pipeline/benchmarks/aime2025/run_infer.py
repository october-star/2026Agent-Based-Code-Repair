from pathlib import Path
import os

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset import load_aime
from prompts import build_prompt
from utils import get_run_id, reset_file, save_jsonl, slugify_model_name, write_latest_run_id

BASE_DIR = Path(__file__).resolve().parents[2]


def load_config():
    config_path = BASE_DIR / "configs" / "base.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def init_hf_inference_client(config):
    try:
        from huggingface_hub import InferenceClient
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for provider=hf_inference."
        ) from exc

    token_env = config["model"].get("auth_token_env", "HF_TOKEN")
    token = os.getenv(token_env)
    timeout = config["model"].get("timeout", 120)
    provider_name = config["model"].get("provider_name", "auto")
    client = InferenceClient(token=token, timeout=timeout, provider=provider_name)
    return {
        "provider": "hf_inference",
        "provider_name": provider_name,
        "client": client,
    }


def get_device_and_dtype(config):
    requested_device = config["model"].get("device", "auto")

    if requested_device == "cpu":
        return torch.device("cpu"), torch.float32
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda"), torch.float16
    if requested_device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps"), torch.float16
    if requested_device != "auto":
        raise ValueError(f"Unsupported device setting: {requested_device}")

    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


def init_model(config):
    provider = config["model"].get("provider", "local")
    if provider == "hf_inference":
        return init_hf_inference_client(config)
    if provider != "local":
        raise ValueError(f"Unsupported provider setting: {provider}")

    device, dtype = get_device_and_dtype(config)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    return {
        "provider": "local",
        "device": device,
        "tokenizer": tokenizer,
        "model": model,
    }


def generate(runtime, prompt, config):
    if runtime["provider"] == "hf_inference":
        generation_kwargs = {
            "model": config["model"]["name"],
            "max_new_tokens": config["model"]["max_new_tokens"],
            "do_sample": config["model"]["do_sample"],
            "return_full_text": False,
        }
        if config["model"]["do_sample"]:
            generation_kwargs["temperature"] = config["model"]["temperature"]

        try:
            return runtime["client"].text_generation(prompt, **generation_kwargs).strip()
        except Exception as exc:
            message = str(exc)
            if (
                "Supported task: conversational" not in message
                and "StopIteration" not in exc.__class__.__name__
            ):
                raise

        chat_kwargs = {
            "model": config["model"]["name"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": config["model"]["max_new_tokens"],
        }
        if config["model"]["do_sample"]:
            chat_kwargs["temperature"] = config["model"]["temperature"]

        response = runtime["client"].chat_completion(**chat_kwargs)
        return response.choices[0].message.content.strip()

    tokenizer = runtime["tokenizer"]
    model = runtime["model"]
    device = runtime["device"]

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config["model"].get("max_input_tokens"),
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": config["model"]["max_new_tokens"],
        "do_sample": config["model"]["do_sample"],
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if config["model"]["do_sample"]:
        generation_kwargs["temperature"] = config["model"]["temperature"]

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def log_result(item, output, index, config):
    if not config["output"].get("print_logs", False):
        return ""

    from extract import extract_answer

    parsed_answer, parse_failed = extract_answer(output)
    lines = [
        f"[sample {index}] problem_type={item.get('problem_type', [])}",
        f"[sample {index}] gold={item['answer']}",
        f"[sample {index}] parsed_answer={parsed_answer} parse_failed={parse_failed}",
        f"[sample {index}] model_output={output}",
        "-" * 80,
    ]
    log_text = "\n".join(lines)
    tqdm.write(log_text)
    return log_text


def append_log(path, text):
    if not text:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")


def main():
    config = load_config()
    ds = load_aime(
        name=config["dataset"]["name"],
        split=config["dataset"].get("split", "train"),
        max_examples=config["dataset"].get("max_examples"),
    )
    runtime = init_model(config)

    run_id = get_run_id()
    if not run_id:
        raise RuntimeError("RUN_ID is not set. Use scripts/run_aime.sh to start a timed run.")

    model_slug = slugify_model_name(config["model"]["name"])
    output_root = BASE_DIR / config["output"]["dir"]
    output_dir = output_root / run_id
    output_path = output_dir / config["output"]["file"]
    log_path = output_dir / config["output"].get("log_file", "run.log")
    if config["output"].get("overwrite", True):
        reset_file(str(output_path))
        reset_file(str(log_path))
    write_latest_run_id(str(output_root), run_id)

    for i, item in enumerate(tqdm(ds)):
        problem = item["problem"]
        gold = item["answer"]
        prompt = build_prompt(problem)

        output = generate(runtime, prompt, config)
        log_text = log_result(item, output, i, config)
        append_log(str(log_path), log_text)

        record = {
            "idx": i,
            "problem": problem,
            "gold": gold,
            "problem_type": item.get("problem_type", []),
            "model_output": output,
            "model_name": config["model"]["name"],
            "model_slug": model_slug,
            "run_id": run_id,
            "provider": config["model"].get("provider", "local"),
            "dataset_name": config["dataset"]["name"],
            "dataset_split": config["dataset"].get("split", "train"),
            "max_examples": config["dataset"].get("max_examples"),
            "tool_use": "disabled",
        }
        save_jsonl(str(output_path), record)


if __name__ == "__main__":
    main()
