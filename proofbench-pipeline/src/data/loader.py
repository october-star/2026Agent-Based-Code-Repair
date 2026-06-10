"""
miniF2F dataset loader.
Loads problems from HuggingFace and wraps them into Problem objects.
"""
import logging
from typing import List, Optional
from src.data.schemas import Problem

logger = logging.getLogger(__name__)

# Known HuggingFace dataset names for miniF2F Lean 4
_MINIF2F_SOURCES = [
    ("cat-searcher/minif2f-lean4",       "test"),
    ("hoskinson-center/minif2f-lean4",   "test"),
    ("mathlib4/miniF2F",                 "test"),
]


def _normalise_item(item: dict, idx: int) -> Problem:
    """Map a raw HuggingFace row to a Problem."""
    pid = (
        item.get("id")
        or item.get("name")
        or item.get("problem_name")
        or f"prob_{idx:04d}"
    )
    informal = (
        item.get("informal_stmt")
        or item.get("informal_statement")
        or item.get("problem")
        or ""
    )
    formal = (
        item.get("formal_statement")
        or item.get("formal_stmt")
        or item.get("lean_code")
        or ""
    )
    ref = (
        item.get("informal_proof")
        or item.get("reference_solution")
        or item.get("solution")
        or ""
    )
    return Problem(
        id=str(pid),
        informal_statement=informal,
        formal_statement=formal,
        reference_solution=ref,
        metadata={"source": "miniF2F", "split": "test", "index": idx},
    )


def load_minif2f(split: str = "test",
                 max_samples: Optional[int] = None) -> List[Problem]:
    """
    Load miniF2F problems from HuggingFace.
    Tries multiple known dataset names in order.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run `pip install datasets` to load miniF2F.")

    last_error = None
    for repo, default_split in _MINIF2F_SOURCES:
        try:
            logger.info(f"Trying {repo} …")
            ds = load_dataset(repo, split=split, trust_remote_code=True)
            logger.info(f"Loaded {len(ds)} problems from {repo}")
            problems = []
            for i, item in enumerate(ds):
                if max_samples and i >= max_samples:
                    break
                problems.append(_normalise_item(dict(item), i))
            return problems
        except Exception as e:
            logger.warning(f"{repo}: {e}")
            last_error = e

    raise RuntimeError(
        f"Could not load miniF2F from any known source. Last error: {last_error}"
    )


def load_pilot(n: int = 10) -> List[Problem]:
    """Return n evenly-spaced pilot samples from miniF2F-test."""
    all_problems = load_minif2f()
    step = max(1, len(all_problems) // n)
    return all_problems[::step][:n]


def load_from_jsonl(path: str) -> List[Problem]:
    """Load problems from a local JSONL file (for offline use)."""
    import json
    problems = []
    with open(path) as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            problems.append(_normalise_item(item, i))
    return problems
