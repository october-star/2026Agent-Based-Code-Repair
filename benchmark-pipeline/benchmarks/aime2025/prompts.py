def build_prompt(problem: str) -> str:
    return f"""
You are solving an AIME competition problem.

Solve the problem carefully, but keep the response concise. Do not use any
external tools, calculators, code execution, or web search.

You must end with exactly one final line in this format:
FINAL_ANSWER: <integer from 0 to 999>

Do not omit the FINAL_ANSWER line.
Do not end with any text after the FINAL_ANSWER line.
If you include reasoning, keep it very short so that you do not get cut off.

Problem:
{problem}
""".strip()
