import re

def extract_answer(text: str):
    for line in text.splitlines():
        stripped = line.strip()

        match = re.fullmatch(r"FINAL_ANSWER:\s*([0-9]{1,3})", stripped, re.IGNORECASE)
        if match:
            return int(match.group(1)), False

        match = re.fullmatch(r"Answer:\s*([0-9]{1,3})", stripped, re.IGNORECASE)
        if match:
            return int(match.group(1)), False

    return None, True
