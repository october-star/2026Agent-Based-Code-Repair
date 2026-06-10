"""
Agent proof repair loop.

Wraps any generation strategy in an iterative compiler-feedback loop.
"""
import logging
from typing import Dict, Optional

from src.lean.executor import LeanExecutor
from src.data.schemas import Problem

logger = logging.getLogger(__name__)


class RepairLoop:
    """
    Iterative proof repair agent.

    On each failed attempt:
      1. Record the compiler error in `history`.
      2. Call strategy.repair(problem, history) to get a new proof.
      3. Re-verify. Repeat up to max_iter times.
    """

    def __init__(self, executor: LeanExecutor, max_iter: int = 5):
        self.executor = executor
        self.max_iter = max_iter

    def run(self, problem: Problem, strategy) -> Dict:
        """
        Execute the repair loop for one problem.

        Returns
        -------
        dict with keys:
            success       : bool
            proof         : Optional[str]  — the final (possibly failing) proof
            cot           : Optional[str]
            iterations    : int            — attempts used (1 = no repair needed)
            history       : list           — per-attempt records
            compile_time  : float
        """
        history = []

        # ── iteration 1: standard generation ────────────────
        proof, cot = strategy.generate(problem)

        for iteration in range(1, self.max_iter + 1):

            # handle case where LLM returned no parseable proof
            if proof is None:
                logger.warning(
                    f"[{problem.id}] iteration {iteration}: no proof extracted"
                )
                history.append({
                    "iteration": iteration,
                    "proof": None,
                    "error": "No proof extracted from model output",
                    "error_line": None,
                    "success": False,
                })
                if iteration < self.max_iter:
                    proof, cot = strategy.repair(problem, history)
                continue

            # ── verify ──────────────────────────────────────
            result = self.executor.verify(proof)

            if result.success:
                logger.info(
                    f"[{problem.id}] verified on iteration {iteration}"
                )
                return {
                    "success": True,
                    "proof": proof,
                    "cot": cot,
                    "iterations": iteration,
                    "history": history,
                    "compile_time": result.compile_time,
                }

            history.append({
                "iteration": iteration,
                "proof": proof,
                "error": result.error_msg,
                "error_line": result.error_line,
                "success": False,
            })
            logger.debug(
                f"[{problem.id}] iteration {iteration} failed: {result.error_msg}"
            )

            if iteration < self.max_iter:
                proof, cot = strategy.repair(problem, history)

        # ── all iterations exhausted ─────────────────────────
        logger.info(f"[{problem.id}] failed after {self.max_iter} iterations")
        return {
            "success": False,
            "proof": proof,
            "cot": cot,
            "iterations": self.max_iter,
            "history": history,
            "compile_time": 0.0,
        }
