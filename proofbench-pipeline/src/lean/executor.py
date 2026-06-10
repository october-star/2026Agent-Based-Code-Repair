"""
Lean 4 executor.

Real mode  : calls `lake exe repl` via subprocess (requires Lean 4 + Mathlib).
Mock mode  : simulates results for pipeline development/testing.
"""
import json
import logging
import random
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LeanResult:
    success: bool
    error_msg: Optional[str] = None
    error_line: Optional[int] = None
    compile_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "error_msg": self.error_msg,
            "error_line": self.error_line,
            "compile_time": self.compile_time,
        }


class LeanExecutor:
    """
    Verifies Lean 4 proofs using the leanprover-community/repl tool.

    Setup (real mode):
        cd your_mathlib_project
        lake exe repl          # should start an interactive REPL
    """

    def __init__(
        self,
        timeout: int = 60,
        lean_bin: str = "lean",
        project_dir: Optional[str] = None,
        mock_mode: bool = False,
        mock_pass_rate: float = 0.30,
    ):
        self.timeout = timeout
        self.lean_bin = lean_bin
        self.project_dir = project_dir
        self.mock_mode = mock_mode
        self.mock_pass_rate = mock_pass_rate

    # ── public API ──────────────────────────────────────────

    def verify(self, lean_code: str) -> LeanResult:
        """Verify lean_code and return a LeanResult."""
        if not lean_code or not lean_code.strip():
            return LeanResult(success=False, error_msg="Empty proof")

        if self.mock_mode:
            return self._mock_verify(lean_code)

        start = time.time()
        try:
            return self._repl_verify(lean_code, start)
        except subprocess.TimeoutExpired:
            return LeanResult(
                success=False,
                error_msg=f"Timeout after {self.timeout}s",
                compile_time=self.timeout,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "Lean 4 / lake not found. "
                "Install Lean 4 (https://leanprover.github.io/lean4/doc/quickstart.html) "
                "or set mock_mode=True in config.yaml."
            )

    # ── real REPL verification ───────────────────────────────

    def _repl_verify(self, lean_code: str, start: float) -> LeanResult:
        payload = json.dumps({"cmd": lean_code, "env": 0}) + "\n"
        cwd = self.project_dir

        proc = subprocess.run(
            ["lake", "exe", "repl"],
            input=payload,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            cwd=cwd,
        )
        elapsed = time.time() - start

        # Try to parse JSON response
        stdout = proc.stdout.strip()
        if not stdout:
            # No output at all – treat stderr as error
            err = proc.stderr.strip() or "No output from Lean REPL"
            return LeanResult(success=False, error_msg=err, compile_time=elapsed)

        try:
            response = json.loads(stdout)
        except json.JSONDecodeError:
            # Non-JSON output: fall back to stderr check
            if proc.returncode != 0:
                return LeanResult(
                    success=False,
                    error_msg=proc.stderr.strip(),
                    compile_time=elapsed,
                )
            return LeanResult(success=True, compile_time=elapsed)

        messages = response.get("messages", [])

        # Filter out errors and sorry-warnings
        errors = [
            m for m in messages
            if m.get("severity") == "error"
        ]
        sorries = [
            m for m in messages
            if m.get("severity") == "warning"
            and "sorry" in m.get("data", "").lower()
        ]

        bad = errors + sorries
        if bad:
            first = bad[0]
            pos = first.get("pos", {})
            return LeanResult(
                success=False,
                error_msg=first.get("data", "unknown error"),
                error_line=pos.get("line"),
                compile_time=elapsed,
            )

        return LeanResult(success=True, compile_time=elapsed)

    # ── mock verification ────────────────────────────────────

    _MOCK_ERRORS = [
        ("unknown tactic 'simp'", 6),
        ("type mismatch\n  expected: Nat\n  got: Int", 9),
        ("failed to synthesize instance\n  Decidable (n = 0)", 4),
        ("application type mismatch", 7),
        ("unknown identifier 'h'", 5),
        ("tactic 'exact' failed", 11),
    ]

    def _mock_verify(self, lean_code: str) -> LeanResult:
        """Simulate verification for pipeline testing."""
        time.sleep(random.uniform(0.05, 0.3))  # realistic latency
        if random.random() < self.mock_pass_rate:
            return LeanResult(success=True, compile_time=random.uniform(1.0, 8.0))
        msg, line = random.choice(self._MOCK_ERRORS)
        return LeanResult(
            success=False,
            error_msg=msg,
            error_line=line,
            compile_time=random.uniform(0.3, 3.0),
        )
