import re


LEAN_BLOCK_RE = re.compile(r"```lean\s*(.*?)```", re.IGNORECASE | re.DOTALL)
GENERIC_BLOCK_RE = re.compile(r"```\s*(.*?)```", re.DOTALL)
OPEN_LEAN_BLOCK_RE = re.compile(r"```lean\s*(.*)\Z", re.IGNORECASE | re.DOTALL)
OPEN_GENERIC_BLOCK_RE = re.compile(r"```\s*(.*)\Z", re.DOTALL)

LEAN_HINTS = (
    "import ",
    "open ",
    "namespace ",
    "theorem ",
    "lemma ",
    "def ",
    "inductive ",
    "structure ",
    "example ",
    ":=",
    "begin",
    "by",
)


def extract_lean_code(raw_generation):
    match = LEAN_BLOCK_RE.search(raw_generation)
    if match:
        return {
            "lean_code": match.group(1).strip(),
            "extraction_success": True,
            "possibly_truncated": False,
        }

    match = OPEN_LEAN_BLOCK_RE.search(raw_generation)
    if match:
        lean_code = match.group(1).strip()
        return {
            "lean_code": lean_code,
            "extraction_success": _looks_like_lean(lean_code),
            "possibly_truncated": True,
        }

    match = GENERIC_BLOCK_RE.search(raw_generation)
    if match:
        return {
            "lean_code": match.group(1).strip(),
            "extraction_success": True,
            "possibly_truncated": False,
        }

    match = OPEN_GENERIC_BLOCK_RE.search(raw_generation)
    if match:
        lean_code = match.group(1).strip()
        return {
            "lean_code": lean_code,
            "extraction_success": _looks_like_lean(lean_code),
            "possibly_truncated": True,
        }

    stripped = raw_generation.strip()
    return {
        "lean_code": stripped,
        "extraction_success": _looks_like_lean(stripped),
        "possibly_truncated": False,
    }


def _looks_like_lean(text):
    lower_text = text.lower()
    return any(hint in lower_text for hint in LEAN_HINTS)
