def build_prompt(problem: str) -> str:
    return f"""
You are solving an AIME competition problem.

Solve the problem carefully. You may reason step by step, but do not use tools.

At the end, output exactly one line:
FINAL_ANSWER: <integer from 0 to 999>

Problem:
{problem}
"""