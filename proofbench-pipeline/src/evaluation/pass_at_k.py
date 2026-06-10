"""
Pass@K and compile-rate metrics.

Uses the standard unbiased estimator from Chen et al. (2021):
  pass@k = 1 - C(n-c, k) / C(n, k)
"""
import math
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def _comb(n: int, k: int) -> int:
    """Exact binomial coefficient (no floating-point issues for small k)."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased pass@k estimator.

    Parameters
    ----------
    n : total number of generated samples for this problem
    c : number of correct (verified) samples
    k : the k in pass@k
    """
    if n == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - _comb(n - c, k) / _comb(n, k)


def compute_pass_at_k(
    problem_results: List[Dict],
    k_values: List[int] = None,
) -> Dict[str, float]:
    """
    Compute Pass@K across a list of per-problem result dicts.

    Each dict must contain an 'attempts' list where each attempt has 'success' (bool).

    Returns
    -------
    {'pass_at_1': float, 'pass_at_32': float, ...}
    """
    if k_values is None:
        k_values = [1, 32]

    per_problem: List[Dict[str, float]] = []

    for res in problem_results:
        attempts = res.get("attempts", [])
        n = len(attempts)
        c = sum(1 for a in attempts if a.get("success", False))
        per_problem.append({"n": n, "c": c})

    metrics: Dict[str, float] = {}
    for k in k_values:
        scores = [
            pass_at_k(p["n"], p["c"], min(k, p["n"]))
            for p in per_problem
            if p["n"] > 0
        ]
        metrics[f"pass_at_{k}"] = sum(scores) / len(scores) if scores else 0.0

    return metrics


def compute_compile_rate(problem_results: List[Dict]) -> float:
    """
    Fraction of all generated proofs that compile (no parse/syntax errors).
    A proof "compiles" if it either passes or fails with a semantic/tactic error
    (as opposed to a syntax/parse error that means the code is malformed).
    """
    total = compiled = 0
    syntax_keywords = {"parse", "syntax", "expected token", "unexpected token"}

    for res in problem_results:
        for attempt in res.get("attempts", []):
            total += 1
            if attempt.get("success"):
                compiled += 1
                continue
            err = (attempt.get("error_msg") or "").lower()
            if not any(kw in err for kw in syntax_keywords):
                compiled += 1  # logic/tactic error → code at least compiled

    return compiled / total if total else 0.0
