import json
import os
import re

def save_jsonl(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def reset_file(path):
    if os.path.exists(path):
        os.remove(path)


def slugify_model_name(name):
    slug = name.strip()
    slug = slug.replace("/", "__")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", slug)
    return slug or "unknown-model"


def get_run_id():
    return os.getenv("RUN_ID")


def write_latest_run_id(output_root, run_id):
    os.makedirs(output_root, exist_ok=True)
    latest_path = os.path.join(output_root, "latest_run.txt")
    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(run_id)
        f.write("\n")


def read_latest_run_id(output_root):
    latest_path = os.path.join(output_root, "latest_run.txt")
    if not os.path.exists(latest_path):
        return None

    with open(latest_path, encoding="utf-8") as f:
        return f.read().strip() or None
