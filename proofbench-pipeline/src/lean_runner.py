import subprocess
import tempfile
import time
from pathlib import Path

from error_classifier import classify_error

# Runs Lean code and captures the output, error messages, and runtime
def run_lean(lean_code, timeout=30, command=None, project_dir="", output_path=None):
    command = command or ["lake", "env", "lean"]
    contains_sorry = "sorry" in lean_code.lower()
    contains_admit = "admit" in lean_code.lower()

    start = time.perf_counter()

    try:
        if output_path:
            file_path = Path(output_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(lean_code, encoding="utf-8")
            working_dir = project_dir or file_path.parent
            result = subprocess.run(
                [*command, str(file_path)],
                cwd=str(working_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = Path(tmpdir) / "proof.lean"
                file_path.write_text(lean_code, encoding="utf-8")
                working_dir = project_dir or tmpdir
                result = subprocess.run(
                    [*command, str(file_path)],
                    cwd=str(working_dir),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
    except subprocess.TimeoutExpired as exc:
        runtime = time.perf_counter() - start
        return {
            "success": False,
            "stdout": exc.stdout or "",
            "stderr": (exc.stderr or "") + "\nTimeout expired",
            "contains_sorry": contains_sorry,
            "contains_admit": contains_admit,
            "error_type": "timeout",
            "runtime_sec": runtime,
        }
    except FileNotFoundError as exc:
        runtime = time.perf_counter() - start
        stderr = f"Lean command not found: {exc}"
        return {
            "success": False,
            "stdout": "",
            "stderr": stderr,
            "contains_sorry": contains_sorry,
            "contains_admit": contains_admit,
            "error_type": "lean_unavailable",
            "runtime_sec": runtime,
        }

    runtime = time.perf_counter() - start
    error_type = classify_error(result.stdout, result.stderr, lean_code, result.returncode)
    return {
        "success": result.returncode == 0 and not contains_sorry and not contains_admit,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "contains_sorry": contains_sorry,
        "contains_admit": contains_admit,
        "error_type": error_type,
        "runtime_sec": runtime,
    }
