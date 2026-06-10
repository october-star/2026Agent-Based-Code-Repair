"""
Base LLM client and output parsers.

Supports OpenAI-compatible chat endpoints, local Transformers models, and a
deterministic mock backend for pipeline smoke tests.
"""
import json
import re
import time
import logging
import urllib.error
import urllib.request
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
    Thin wrapper around supported text-generation backends.

    backend="openai"       : OpenAI-compatible chat completions.
    backend="transformers" : Local HuggingFace Transformers causal LM.
    backend="mock_qwen"    : Deterministic Qwen-shaped output for smoke tests.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        backend: str = "openai",
        local_files_only: bool = False,
    ):
        self.model = model
        self.api_key = api_key or ""
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.backend = backend
        self.local_files_only = local_files_only
        self._client = None
        self._tokenizer = None
        self._model = None

        if self.backend == "openai":
            self._init_openai_client()
        elif self.backend == "transformers":
            self._init_transformers_model()
        elif self.backend == "mock_qwen":
            logger.info("Using mock Qwen backend for offline pipeline smoke tests.")
        else:
            raise ValueError(f"Unknown LLM backend: {self.backend}")

    def _init_openai_client(self) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning(
                "openai package is not installed; falling back to urllib for "
                "OpenAI-compatible HTTP calls."
            )
            return

        kwargs: dict = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)

    def _init_transformers_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Run `pip install torch transformers` to use backend=transformers."
            ) from exc

        logger.info("Loading local Transformers model: %s", self.model)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )

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

        if self.backend == "mock_qwen":
            return self._mock_generate(system_prompt, user_prompt)

        if self.backend == "transformers":
            return self._transformers_generate(messages, t)

        for attempt in range(3):
            try:
                if self._client is not None:
                    resp = self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=t,
                        max_tokens=self.max_tokens,
                    )
                    return resp.choices[0].message.content or ""
                return self._http_generate(messages, t)
            except Exception as exc:
                if _is_non_retryable_api_error(exc):
                    logger.error("LLM call failed with non-retryable API error: %s", exc)
                    raise
                if attempt == 2:
                    logger.error(f"LLM call failed after 3 attempts: {exc}")
                    raise
                wait = 2 ** attempt
                logger.warning(f"LLM call attempt {attempt+1} failed ({exc}), retrying in {wait}s")
                time.sleep(wait)
        return ""

    def _http_generate(self, messages: List[dict], temperature: float) -> str:
        if not self.base_url:
            raise ImportError(
                "Run `pip install openai`, or set base_url for urllib fallback."
            )

        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
        return data["choices"][0]["message"]["content"] or ""

    def _transformers_generate(self, messages: List[dict], temperature: float) -> str:
        import torch

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer([prompt], return_tensors="pt").to(self._model.device)
        do_sample = temperature > 0
        max_new_tokens = min(self.max_tokens, 1024)
        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-5) if do_sample else None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = generated[:, inputs.input_ids.shape[-1]:]
        return self._tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

    def _mock_generate(self, system_prompt: str, user_prompt: str) -> str:
        proof = _mock_lean_proof(user_prompt)
        if "alternate between" in system_prompt.lower():
            return (
                "<step_1_reasoning>\nUse a compact Lean proof shape for the "
                "pipeline smoke test.\n</step_1_reasoning>\n"
                f"<proof>\n{proof}\n</proof>"
            )
        if "reason informally" in system_prompt.lower():
            return (
                "<reasoning>\nUse a direct theorem proof skeleton so the "
                "ProofBench pipeline can exercise parsing and verification.\n"
                "</reasoning>\n"
                f"<proof>\n{proof}\n</proof>"
            )
        return f"<proof>\n{proof}\n</proof>"


def _mock_lean_proof(user_prompt: str) -> str:
    theorem = _extract_formal_statement(user_prompt)
    if theorem:
        if ":=" in theorem:
            return theorem
        return theorem.rstrip() + " := by\n  trivial"
    return "theorem qwen_smoke : True := by\n  trivial"


def _extract_formal_statement(text: str) -> Optional[str]:
    marker = "Complete this Lean 4 theorem:"
    idx = text.find(marker)
    if idx < 0:
        marker = "Lean 4 theorem declaration to complete:"
        idx = text.find(marker)
    if idx < 0:
        marker = "[Lean 4 theorem declaration]"
        idx = text.find(marker)
    if idx < 0:
        return None

    tail = text[idx + len(marker):].strip()
    lines = []
    for line in tail.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("<") or stripped.startswith("["):
            break
        lines.append(line.rstrip())
    return "\n".join(lines).strip() or None


def _is_non_retryable_api_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "insufficient balance",
            "insufficient_quota",
            "payment required",
            "http 402",
        )
    )
