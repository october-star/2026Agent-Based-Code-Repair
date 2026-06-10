"""
Step 1 – Reference formalization pipeline.
Converts natural-language proofs into verified Lean 4 proofs.
"""
import json
import logging
from pathlib import Path
from typing import List, Optional

from src.data.schemas import Problem
from src.generation.base import LLMClient, extract_proof
from src.generation.prompts import REF_SYSTEM, REF_USER
from src.lean.executor import LeanExecutor

logger = logging.getLogger(__name__)


class RefPipeline:
    """
    Formalizes each reference natural-language proof into Lean 4.
    Result is stored back in problem.ref_lean_proof.
    """

    def __init__(
        self,
        client: LLMClient,
        executor: LeanExecutor,
        max_retries: int = 3,
        cache_path: Optional[str] = None,
    ):
        self.client = client
        self.executor = executor
        self.max_retries = max_retries
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict = {}
        if self.cache_path and self.cache_path.exists():
            with open(self.cache_path) as f:
                self._cache = json.load(f)
            logger.info(f"Loaded {len(self._cache)} cached ref proofs")

    def formalize(self, problem: Problem) -> bool:
        """
        Attempt to formalize problem.reference_solution into Lean 4.
        Sets problem.ref_lean_proof if successful.
        Returns True on success.
        """
        # Check cache first
        if problem.id in self._cache:
            problem.ref_lean_proof = self._cache[problem.id]
            logger.debug(f"[{problem.id}] loaded ref proof from cache")
            return True

        user_prompt = REF_USER.format(
            informal_statement=problem.informal_statement,
            reference_solution=problem.reference_solution,
            formal_statement=problem.formal_statement,
        )

        for attempt in range(1, self.max_retries + 1):
            output = self.client.generate(REF_SYSTEM, user_prompt)
            proof = extract_proof(output)

            if not proof:
                logger.warning(f"[{problem.id}] ref attempt {attempt}: no proof extracted")
                continue

            result = self.executor.verify(proof)
            if result.success:
                problem.ref_lean_proof = proof
                self._cache[problem.id] = proof
                self._save_cache()
                logger.info(f"[{problem.id}] ref proof verified on attempt {attempt}")
                return True

            logger.debug(
                f"[{problem.id}] ref attempt {attempt} failed: {result.error_msg}"
            )

        logger.warning(f"[{problem.id}] reference formalization failed after {self.max_retries} attempts")
        return False

    def formalize_all(self, problems: List[Problem]) -> dict:
        """
        Formalize all problems. Returns summary stats.
        """
        success = failed = 0
        for i, prob in enumerate(problems, 1):
            logger.info(f"Formalizing {i}/{len(problems)}: {prob.id}")
            if self.formalize(prob):
                success += 1
            else:
                failed += 1

        rate = success / len(problems) if problems else 0.0
        stats = {"success": success, "failed": failed, "rate": rate}
        logger.info(f"Ref formalization complete: {stats}")
        return stats

    def _save_cache(self):
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self._cache, f, indent=2)
