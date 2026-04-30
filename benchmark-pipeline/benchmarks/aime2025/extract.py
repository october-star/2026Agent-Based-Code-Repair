import re

def extract_answer(text: str):
    match = re.search(r"FINAL_ANSWER:\s*([0-9]{1,3})", text)
    if match:
        return int(match.group(1)), False
    return None, True