"""
Base LLM client (OpenAI-compatible) and output parsers.
Works for GPT-4o (OpenAI API) and DeepSeek-Prover-V1.5 (DeepSeek API).
"""
import re
import time
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Output parsers
# ─────────────────────────────────────────────────────────────

def extract_tag(text: str, tag: str) -> Optional[str]:
    """Extract content between <tag>…</tag>."""
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else None


def extract_proof(text: str) -> Optional[str]:
    """Extract Lean 4 proof from <proof> tags or a lean code fence."""
    proof = extract_tag(text, "proof")
    if proof:
        return proof
    # Fallback: lean code block
    m = re.search(r"```(?:lean4?|lean)\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def extract_cot(text: str) -> Optional[str]:
    """Extract CoT reasoning from <reasoning> tags."""
    return extract_tag(text, "reasoning")


def extract_interleaved(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract proof and concatenated reasoning steps from interleaved output."""
    proof = extract_proof(text)
    steps = re.findall(
        r"<step_\d+_reasoning>\s*(.*?)\s*</step_\d+_reasoning>",
        text, re.DOTALL
    )
    cot = "\n\n".join(steps) if steps else None
    return proof, cot


# ─────────────────────────────────────────────────────────────
# LLM Client
# ─────────────────────────────────────────────────────────────

class LLMClient:
    """
    Thin wrapper around OpenAI-compatible chat completions.
    Supports GPT-4o (api.openai.com) and DeepSeek (api.deepseek.com/v1).
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Run `pip install openai` to use LLMClient.")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        history: Optional[List[dict]] = None,
    ) -> str:
        """Generate a chat completion with exponential-backoff retry."""
        t = temperature if temperature is not None else self.temperature
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})

        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=t,
                    max_tokens=self.max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                if attempt == 2:
                    logger.error(f"LLM call failed after 3 attempts: {exc}")
                    raise
                wait = 2 ** attempt
                logger.warning(f"LLM call attempt {attempt+1} failed ({exc}), retrying in {wait}s")
                time.sleep(wait)
        return ""
