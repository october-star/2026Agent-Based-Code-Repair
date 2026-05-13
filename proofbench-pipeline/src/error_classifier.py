def classify_error(stdout, stderr, lean_code="", returncode=None):
    lower_stderr = (stderr or "").lower()
    lower_code = (lean_code or "").lower()

    # Lean code compiles successfully without using "sorry" or "admit" => success
    if returncode == 0 and "sorry" not in lower_code and "admit" not in lower_code:
        return "success"
    # Lean code compiles successfully but uses "sorry" or "admit" => partial success
    if "sorry" in lower_code or "admit" in lower_code:
        return "sorry_used"
    # Check for specific error patterns in stderr
    if "lean command not found" in lower_stderr or "no such file or directory: 'lake'" in lower_stderr:
        return "lean_unavailable"
    # Check for common error patterns in stderr
    if "extraction failed" in lower_stderr:
        return "extraction_failed"
    # Check for specific error patterns in stderr
    if "unknown identifier" in lower_stderr or "unknown constant" in lower_stderr:
        return "unknown_identifier"
    if "type mismatch" in lower_stderr or "application type mismatch" in lower_stderr:
        return "type_error"
    if "unexpected token" in lower_stderr or "expected" in lower_stderr:
        return "syntax_error"
    if "unknown module prefix" in lower_stderr or "object file" in lower_stderr:
        return "missing_import"
    if "theorem" in lower_stderr and "type mismatch" in lower_stderr:
        return "theorem_statement_error"
    # Check for timeout
    if "timeout" in lower_stderr:
        return "timeout"
    return "unknown_error"
