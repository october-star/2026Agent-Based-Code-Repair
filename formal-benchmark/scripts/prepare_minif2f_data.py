import argparse
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "formal-benchmark" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import build_sample_records, load_jsonl, save_jsonl


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a local miniF2F JSONL file for pilot or full experiments."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a local miniF2F-style JSONL file.",
    )
    parser.add_argument(
        "--output",
        default="formal-benchmark/data/minif2f/pilot_10.jsonl",
        help="Path to write the prepared miniF2F sample file.",
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = REPO_ROOT / args.input
    output_path = REPO_ROOT / args.output
    samples = load_jsonl(input_path)
    random.Random(args.seed).shuffle(samples)
    selected = samples if args.num_samples <= 0 else samples[: args.num_samples]
    save_jsonl(build_sample_records(selected), output_path)
    print(f"Loaded {len(samples)} local miniF2F samples from {input_path}")
    print(f"Wrote {len(selected)} samples to {output_path}")


if __name__ == "__main__":
    main()
