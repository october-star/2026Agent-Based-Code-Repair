from pathlib import Path
import os
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "proofbench-pipeline" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline import ProofBenchPipeline


def main():
    base_dir = REPO_ROOT / "proofbench-pipeline"
    experiment_path = Path(
        os.getenv("PROOFBENCH_EXPERIMENT_CONFIG", base_dir / "configs" / "experiment.yaml")
    )
    model_path = Path(os.getenv("PROOFBENCH_MODEL_CONFIG", base_dir / "configs" / "model.yaml"))
    lean_path = Path(os.getenv("PROOFBENCH_LEAN_CONFIG", base_dir / "configs" / "lean.yaml"))
    pipeline = ProofBenchPipeline(
        base_dir=base_dir,
        experiment_path=experiment_path,
        model_path=model_path,
        lean_path=lean_path,
    )
    result = pipeline.run(sample_limit=10)
    for row in result["summary"]:
        print(row)


if __name__ == "__main__":
    main()
