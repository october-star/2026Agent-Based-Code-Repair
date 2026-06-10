"""Strategy 1 – Chain-of-Thought → Lean."""
from typing import Optional, Tuple
from src.data.schemas import Problem, Strategy
from src.generation.base import LLMClient, extract_proof, extract_cot
from src.generation.prompts import S1_SYSTEM, S1_USER, REPAIR_SYSTEM, REPAIR_USER


class CoTStrategy:
    name = Strategy.S1_COT

    def __init__(self, client: LLMClient):
        self.client = client

    def generate(self, problem: Problem) -> Tuple[Optional[str], Optional[str]]:
        """Returns (lean_proof, cot_text)."""
        user = S1_USER.format(
            informal_statement=problem.informal_statement,
            formal_statement=problem.formal_statement,
        )
        output = self.client.generate(S1_SYSTEM, user)
        return extract_proof(output), extract_cot(output)

    def repair(self, problem: Problem, history: list) -> Tuple[Optional[str], Optional[str]]:
        """Generate a repair attempt using compiler feedback."""
        last = history[-1]
        repair_hist = "\n".join(
            f"Attempt {h['iteration']}: {h['error']}" for h in history[:-1]
        ) or "(first repair attempt)"

        user = REPAIR_USER.format(
            previous_attempt=last.get("proof") or "",
            error_message=last.get("error") or "unknown error",
            error_line=last.get("error_line") or "?",
            current_iter=len(history),
            max_iter=5,
            repair_history=repair_hist,
            informal_statement=problem.informal_statement,
            formal_statement=problem.formal_statement,
        )
        output = self.client.generate(REPAIR_SYSTEM, user, temperature=0.5)
        return extract_proof(output), extract_cot(output)
