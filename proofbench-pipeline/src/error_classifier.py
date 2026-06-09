def classify_error(stdout, stderr, lean_code="", returncode=None):
    lower_stdout = (stdout or "").lower()
    lower_stderr = (stderr or "").lower()
    combined_output = "\n".join(part for part in [lower_stdout, lower_stderr] if part)
    lower_code = (lean_code or "").lower()

    # Lean code compiles successfully without using "sorry" or "admit" => success
    if returncode == 0 and "sorry" not in lower_code and "admit" not in lower_code:
        return "success"
    # Check for specific error patterns in stderr
    if "lean command not found" in combined_output or "no such file or directory: 'lake'" in combined_output:
        return "lean_unavailable"
    # Check for common error patterns in stderr
    if "extraction failed" in combined_output:
        return "extraction_failed"
    if (
        "import data." in lower_code
        or "import tactic" in lower_code
        or "#rinteractive" in lower_code
        or "unknown module prefix 'data'" in combined_output
        or "unknown module prefix 'algebra'" in combined_output
        or "unknown module prefix 'tactic'" in combined_output
    ):
        return "lean3_style_or_import"
    if (
        "unknown module prefix" in combined_output
        or "object file" in combined_output
        or "unknown package" in combined_output
        or "imports are disabled" in combined_output
    ):
        return "missing_import"
    if "your implementation here" in lower_code:
        return "placeholder_proof"
    if "# welcome to the lean theorem prover!" in lower_code:
        return "placeholder_proof"
    if "inductive seq" in lower_code and "interesting_pairs" in lower_code:
        return "statement_mismatch"
    # Check for specific error patterns in stderr
    if "unknown identifier" in combined_output or "unknown constant" in combined_output:
        return "unknown_identifier"
    if "type mismatch" in combined_output or "application type mismatch" in combined_output:
        return "type_error"
    if "unexpected token" in combined_output or "expected" in combined_output:
        return "syntax_error"
    if "theorem" in combined_output and "type mismatch" in combined_output:
        return "theorem_statement_error"
    # Lean code compiles but still contains placeholders => partial success
    if "sorry" in lower_code or "admit" in lower_code:
        return "sorry_used"
    # Check for timeout
    if "timeout" in combined_output:
        return "timeout"
    return "unknown_error"
