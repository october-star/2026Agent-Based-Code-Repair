import argparse
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "proofbench-pipeline" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import build_sample_records, load_hf_dataset, load_jsonl, save_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare the 10-sample ProofBench pilot file.")
    parser.add_argument(
        "--input",
        default="",
        help="Optional path to a local ProofBench JSONL file. If omitted, use the Hugging Face dataset.",
    )
    parser.add_argument(
        "--output",
        default="proofbench-pipeline/data/samples_10.jsonl",
        help="Path to write the pilot sample file.",
    )
    parser.add_argument("--hf-dataset", default="wenjiema02/ProofBench")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = REPO_ROOT / args.input if args.input else None
    output_path = REPO_ROOT / args.output

    if input_path and input_path.exists():
        samples = load_jsonl(input_path)
        source_label = str(input_path)
    else:
        samples = load_hf_dataset(args.hf_dataset, args.split)
        source_label = f"{args.hf_dataset}:{args.split}"

    random.Random(args.seed).shuffle(samples)
    if args.num_samples <= 0:
        pilot_samples = samples
    else:
        pilot_samples = samples[: args.num_samples]
    save_jsonl(build_sample_records(pilot_samples), output_path)

    print(f"Loaded {len(samples)} samples from {source_label}")
    print(f"Wrote {len(pilot_samples)} samples to {output_path}")


if __name__ == "__main__":
    main()
