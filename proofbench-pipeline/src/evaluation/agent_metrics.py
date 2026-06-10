"""Agent metrics: refine@K, avg_repairs, error-type distribution."""
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def compute_refine_at_k(results: List[Dict], k: int) -> float:
    """Fraction of problems solved within ≤k agent iterations."""
    if not results:
        return 0.0
    solved = sum(
        1 for r in results
        if r.get("success") and r.get("iterations", float("inf")) <= k
    )
    return solved / len(results)


def compute_avg_repairs(results: List[Dict]) -> float:
    """Mean repair iterations for problems that were eventually solved."""
    solved = [r for r in results if r.get("success")]
    if not solved:
        return 0.0
    return sum(max(0, r.get("iterations", 1) - 1) for r in solved) / len(solved)


_SYNTAX_KWS    = {"parse", "syntax", "expected token", "unexpected token", "invalid"}
_TYPE_KWS      = {"type mismatch", "application type mismatch"}
_TACTIC_KWS    = {"tactic", "failed", "unknown tactic", "unknown identifier"}
_TIMEOUT_KWS   = {"timeout"}


def analyze_error_types(results: List[Dict]) -> Dict[str, float]:
    """
    Classify Lean 4 errors across all failed attempts into four buckets.
    Returns fractions (sum = 1.0) or all-zeros if no errors.
    """
    counts = {"syntax": 0, "type_mismatch": 0, "tactic_failed": 0,
              "timeout": 0, "other": 0}
    total = 0

    for res in results:
        for item in res.get("history", []):
            if item.get("success"):
                continue
            err = (item.get("error") or "").lower()
            total += 1
            if any(k in err for k in _TIMEOUT_KWS):
                counts["timeout"] += 1
            elif any(k in err for k in _SYNTAX_KWS):
                counts["syntax"] += 1
            elif any(k in err for k in _TYPE_KWS):
                counts["type_mismatch"] += 1
            elif any(k in err for k in _TACTIC_KWS):
                counts["tactic_failed"] += 1
            else:
                counts["other"] += 1

    if total == 0:
        return {k: 0.0 for k in counts}
    return {k: v / total for k, v in counts.items()}
