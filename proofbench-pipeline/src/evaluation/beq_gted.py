"""
BEq  – Bidirectional semantic equivalence (requires Lean 4 executor).
GTED – Generalized tree edit distance on tactic sequences (pure Python).
"""
import re
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# BEq
# ─────────────────────────────────────────────────────────────

def _build_beq_code(
    candidate: str,
    reference: str,
    formal_statement: str,
) -> str:
    """
    Build Lean 4 code that attempts to prove bidirectional equivalence.

    Strategy: compile both proofs, then prove  candidate_thm ↔ ref_thm.
    We rename the reference theorem to avoid collisions.
    """
    # Extract theorem name from formal_statement
    m = re.search(r"\btheorem\s+(\w+)", formal_statement)
    name = m.group(1) if m else "theorem_main"
    ref_name = name + "_ref"

    # Rename theorem in reference proof
    ref_renamed = re.sub(
        rf"\btheorem\s+{re.escape(name)}\b",
        f"theorem {ref_name}",
        reference,
    )

    return f"""import Mathlib

{candidate}

{ref_renamed}

-- BEq check: prove the two theorems are inter-derivable
theorem _beq_check : ({name} ↔ {ref_name}) := by
  constructor <;> intro h <;> exact h
"""


def compute_beq(
    candidate: Optional[str],
    reference: Optional[str],
    formal_statement: str,
    executor,            # LeanExecutor instance
) -> float:
    """
    Returns 1.0 if candidate proof is semantically equivalent to reference,
    0.0 otherwise (or if either proof is missing).
    """
    if not candidate or not reference:
        return 0.0
    code = _build_beq_code(candidate, reference, formal_statement)
    result = executor.verify(code)
    return 1.0 if result.success else 0.0


# ─────────────────────────────────────────────────────────────
# GTED
# ─────────────────────────────────────────────────────────────

# All Mathlib tactics we track in the proof tree
_TACTIC_RE = re.compile(
    r"\b(simp|ring|omega|norm_num|linarith|nlinarith|exact|apply|rw|rewrite|"
    r"intro|intros|cases|rcases|obtain|induction|constructor|use|exists|"
    r"decide|tauto|aesop|have|let|show|suffices|calc|conv|field_simp|"
    r"push_cast|norm_cast|push_neg|contrapose|by_contra|by_cases|"
    r"ext|funext|congr|gcongr|positivity|refine|fin_cases|interval_cases|"
    r"split|left|right|exfalso|trivial|assumption|rfl)\b"
)


def _extract_tactics(proof: str) -> List[str]:
    """Return ordered list of tactic keywords found in the proof."""
    # Strip comments
    code = re.sub(r"--[^\n]*", "", proof)
    return _TACTIC_RE.findall(code)


def _edit_distance(seq1: List[str], seq2: List[str]) -> int:
    """Standard sequence edit distance (insert / delete / substitute)."""
    n, m = len(seq1), len(seq2)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            tmp = dp[j]
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[m]


def compute_gted(
    candidate: Optional[str],
    reference: Optional[str],
) -> float:
    """
    Normalized GTED score in [0, 1].
    Higher = more structurally similar to the reference proof.
    """
    if not candidate or not reference:
        return 0.0

    t1 = _extract_tactics(candidate)
    t2 = _extract_tactics(reference)

    if not t1 and not t2:
        return 1.0  # both trivially empty → identical structure

    dist = _edit_distance(t1, t2)
    max_len = max(len(t1), len(t2), 1)
    return max(0.0, 1.0 - dist / max_len)
