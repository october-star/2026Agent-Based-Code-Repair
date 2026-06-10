"""
Prompt templates for all ProofBench generation strategies.
"""

# ─────────────────────────────────────────────────────────────
# Shared system preamble
# ─────────────────────────────────────────────────────────────

_BASE = (
    "You are an expert mathematician and Lean 4 formal proof specialist. "
    "All proofs must be syntactically correct, complete, and verifiable by the "
    "Lean 4 compiler with Mathlib. Never use `sorry`."
)

# ─────────────────────────────────────────────────────────────
# Strategy 1 – CoT → Lean
# ─────────────────────────────────────────────────────────────

S1_SYSTEM = _BASE + (
    "\nApproach: reason informally first, then translate into Lean 4. "
    "Wrap your reasoning in <reasoning>…</reasoning> and the final proof in <proof>…</proof>."
)

S1_USER = """\
Problem:
{informal_statement}

Complete this Lean 4 theorem:
{formal_statement}

First reason through the proof in natural language, then write the Lean 4 proof.

<reasoning>
[Your mathematical reasoning here]
</reasoning>

<proof>
[Complete Lean 4 proof — starting from the theorem declaration]
</proof>"""

# ─────────────────────────────────────────────────────────────
# Strategy 2 – Direct Lean (no natural language)
# ─────────────────────────────────────────────────────────────

S2_SYSTEM = _BASE + (
    "\nApproach: output ONLY a complete Lean 4 proof. No explanations. "
    "Wrap the proof in <proof>…</proof>."
)

S2_USER = """\
Problem:
{informal_statement}

Complete this Lean 4 theorem:
{formal_statement}

Output ONLY the complete Lean 4 proof inside <proof> tags. No explanations.

<proof>
[Complete Lean 4 proof]
</proof>"""

# ─────────────────────────────────────────────────────────────
# Strategy 3 – Interleaved CoT + Lean
# ─────────────────────────────────────────────────────────────

S3_SYSTEM = _BASE + (
    "\nApproach: alternate between short reasoning steps and Lean 4 fragments, "
    "then give the final complete proof. "
    "Use <step_N_reasoning>…</step_N_reasoning> and <step_N_code>…</step_N_code> tags, "
    "then <proof>…</proof> for the final complete proof."
)

S3_USER = """\
Problem:
{informal_statement}

Complete this Lean 4 theorem:
{formal_statement}

Alternate between reasoning steps and code fragments, then give the complete proof.

<step_1_reasoning>
[First proof idea]
</step_1_reasoning>
<step_1_code>
[Corresponding Lean 4 fragment]
</step_1_code>

... (continue as needed) ...

<proof>
[Final complete Lean 4 proof]
</proof>"""

# ─────────────────────────────────────────────────────────────
# Reference formalization (Step 1)
# ─────────────────────────────────────────────────────────────

REF_SYSTEM = (
    "You are an expert in Lean 4 formal mathematics. "
    "Translate the given natural-language proof into a verified Lean 4 proof. "
    "Use Mathlib tactics. Never use `sorry`. "
    "Wrap the proof in <proof>…</proof>."
)

REF_USER = """\
Formalize the following proof into Lean 4.

Problem:
{informal_statement}

Reference solution (natural language):
{reference_solution}

Lean 4 theorem declaration to complete:
{formal_statement}

<proof>
[Complete Lean 4 proof]
</proof>"""

# ─────────────────────────────────────────────────────────────
# Agent repair prompt
# ─────────────────────────────────────────────────────────────

REPAIR_SYSTEM = _BASE + (
    "\nYou are fixing a Lean 4 proof that failed to compile. "
    "Study the compiler error carefully and output a corrected proof in <proof>…</proof>."
)

REPAIR_USER = """\
The following Lean 4 proof failed to compile.

[Previous proof — attempt {current_iter}/{max_iter}]
{previous_attempt}

[Lean 4 compiler error]
{error_message}
(error at line {error_line})

[Repair history]
{repair_history}

[Original problem]
{informal_statement}

[Lean 4 theorem declaration]
{formal_statement}

Fix the proof. Pay close attention to line {error_line}.

<proof>
[Corrected complete Lean 4 proof]
</proof>"""
