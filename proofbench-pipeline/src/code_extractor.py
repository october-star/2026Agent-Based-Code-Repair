import re


LEAN_BLOCK_RE = re.compile(r"```lean\s*(.*?)```", re.IGNORECASE | re.DOTALL)
GENERIC_BLOCK_RE = re.compile(r"```\s*(.*?)```", re.DOTALL)


def extract_lean_code(raw_generation):
    match = LEAN_BLOCK_RE.search(raw_generation)
    if match:
        return {
            "lean_code": match.group(1).strip(),
            "extraction_success": True,
        }

    match = GENERIC_BLOCK_RE.search(raw_generation)
    if match:
        return {
            "lean_code": match.group(1).strip(),
            "extraction_success": True,
        }

    stripped = raw_generation.strip()
    looks_like_lean = "theorem " in stripped or "by" in stripped
    return {
        "lean_code": stripped,
        "extraction_success": looks_like_lean,
    }
