"""
ProofBench data schemas.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum


class Strategy(str, Enum):
    S1_COT       = "S1_CoT"
    S2_DIRECT    = "S2_Direct"
    S3_INTERLEAVED = "S3_Interleaved"


class ModelName(str, Enum):
    GPT4O        = "gpt-4o"
    DSP_V15      = "deepseek-prover-v1.5"


@dataclass
class Problem:
    id: str
    informal_statement: str       # Natural-language problem
    formal_statement: str         # Lean 4 theorem declaration (header only)
    reference_solution: str       # Natural-language reference proof
    metadata: Dict[str, Any] = field(default_factory=dict)
    # populated after Step 1
    ref_lean_proof: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LeanResult:
    success: bool
    error_msg: Optional[str]  = None
    error_line: Optional[int] = None
    compile_time: float       = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProofAttempt:
    sample_idx:  int
    lean_code:   Optional[str]
    cot_text:    Optional[str]
    lean_result: LeanResult
    iterations:  int = 1        # > 1 only in agent mode

    @property
    def success(self) -> bool:
        return self.lean_result.success

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class ProblemResult:
    problem_id: str
    strategy:   Strategy
    agent:      bool
    model:      str
    attempts:   List[ProofAttempt] = field(default_factory=list)
    # per-problem aggregate metrics (filled by evaluation module)
    beq:        float = 0.0
    gted:       float = 0.0

    @property
    def n_correct(self) -> int:
        return sum(1 for a in self.attempts if a.success)

    @property
    def best_proof(self) -> Optional[str]:
        for a in self.attempts:
            if a.success:
                return a.lean_code
        return None

    def to_dict(self) -> dict:
        return {
            'problem_id': self.problem_id,
            'strategy':   self.strategy.value,
            'agent':      self.agent,
            'model':      self.model,
            'n_attempts': len(self.attempts),
            'n_correct':  self.n_correct,
            'beq':        self.beq,
            'gted':       self.gted,
            'attempts':   [a.to_dict() for a in self.attempts],
        }
