import os


def _dry_run_response(prompt):
    lower_prompt = prompt.lower()
    if "convert the proof sketch into lean code" in lower_prompt:
        return """```lean
theorem proofbench_placeholder_from_sketch : True := by
  trivial
```"""

    if "write a lean proof directly" in lower_prompt:
        return """```lean
theorem proofbench_placeholder_direct : True := by
  trivial
```"""

    if "write a lean proof that matches the statement faithfully" in lower_prompt:
        return """```lean
theorem proofbench_placeholder_reference : True := by
  trivial
```"""

    if "final answer must be" in lower_prompt and "valid lean code in a fenced lean block" in lower_prompt:
        return """```lean
theorem proofbench_placeholder_mixed : True := by
  trivial
```"""

    if "proof sketch" in lower_prompt or "natural-language proof strategy" in lower_prompt:
        return "Use the reference theorem, reduce the goal, and finish with a direct tactic proof."

    return """```lean
theorem proofbench_placeholder : True := by
  trivial
```"""


class LLMClient:
    def __init__(self, model_config):
        self.model_config = model_config
        self._local_runtime = None

    def generate(self, prompt):
        if self.model_config.get("dry_run", True):
            return _dry_run_response(prompt)

        provider = self.model_config.get("provider", "").lower()
        if provider == "local":
            return self._generate_local(prompt)
        if provider == "hf_inference":
            return self._generate_hf(prompt)

        if provider not in {"openai", "openai_compatible"}:
            raise RuntimeError(
                f"Unsupported provider '{provider}'. Set provider to 'dry_run', 'local', "
                "'hf_inference', or implement the backend."
            )

        return self._generate_openai(prompt)

    def _generate_openai(self, prompt):
        from openai import OpenAI

        api_key = os.getenv(self.model_config.get("api_key_env", "OPENAI_API_KEY"))
        client_kwargs = {"api_key": api_key}
        base_url = self.model_config.get("base_url", "").strip()
        if base_url:
            client_kwargs["base_url"] = base_url

        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=self.model_config["model_name"],
            messages=[{"role": "user", "content": prompt}],
            temperature=self.model_config.get("temperature", 0.2),
            max_tokens=self.model_config.get("max_tokens", 2048),
        )
        return response.choices[0].message.content or ""

    def _generate_hf(self, prompt):
        from huggingface_hub import InferenceClient

        token = os.getenv(self.model_config.get("api_key_env", "HF_TOKEN"))
        client_kwargs = {"token": token}
        provider_name = self.model_config.get("provider_name", "").strip()
        if provider_name and provider_name.lower() != "auto":
            client_kwargs["provider"] = provider_name

        client = InferenceClient(**client_kwargs)
        model_name = self.model_config["model_name"]
        max_new_tokens = self.model_config.get("max_tokens", 2048)
        temperature = self.model_config.get("temperature", 0.2)
        do_sample = self.model_config.get("do_sample", temperature > 0)

        generation_kwargs = {
            "model": model_name,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
        }

        try:
            response = client.text_generation(prompt, **generation_kwargs)
            return response.strip()
        except StopIteration:
            pass
        except ValueError as exc:
            message = str(exc).lower()
            if "conversational" not in message and "chat" not in message:
                raise

        completion = client.chat_completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        return (completion.choices[0].message.content or "").strip()

    def _generate_local(self, prompt):
        runtime = self._get_local_runtime()
        tokenizer = runtime["tokenizer"]
        model = runtime["model"]
        device = runtime["device"]
        torch = runtime["torch"]

        if self.model_config.get("use_chat_template", True) and hasattr(tokenizer, "apply_chat_template"):
            rendered_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered_prompt = prompt

        max_input_tokens = self.model_config.get("max_input_tokens")
        tokenized = tokenizer(
            rendered_prompt,
            return_tensors="pt",
            truncation=bool(max_input_tokens),
            max_length=max_input_tokens,
        )
        tokenized = {name: tensor.to(device) for name, tensor in tokenized.items()}

        max_new_tokens = self.model_config.get("max_tokens", 2048)
        temperature = self.model_config.get("temperature", 0.2)
        do_sample = self.model_config.get("do_sample", temperature > 0)
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id

        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": pad_token_id,
        }
        if tokenizer.eos_token_id is not None:
            generate_kwargs["eos_token_id"] = tokenizer.eos_token_id
        if do_sample:
            generate_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = model.generate(**tokenized, **generate_kwargs)

        prompt_length = tokenized["input_ids"].shape[1]
        generated_tokens = outputs[0][prompt_length:]
        return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def _get_local_runtime(self):
        if self._local_runtime is not None:
            return self._local_runtime

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = self.model_config["model_name"]
        token = os.getenv(self.model_config.get("api_key_env", "HF_TOKEN")) or None
        trust_remote_code = self.model_config.get("trust_remote_code", False)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            token=token,
        )

        device = self._resolve_local_device(torch)
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "token": token,
        }
        torch_dtype = self._resolve_torch_dtype(torch, device)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.to(device)
        model.eval()

        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        self._local_runtime = {
            "tokenizer": tokenizer,
            "model": model,
            "device": device,
            "torch": torch,
        }
        return self._local_runtime

    def _resolve_local_device(self, torch):
        configured = self.model_config.get("device", "auto").lower()
        if configured != "auto":
            return configured
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_torch_dtype(self, torch, device):
        configured = self.model_config.get("torch_dtype", "auto").lower()
        if configured == "auto":
            if device in {"cuda", "mps"}:
                return torch.float16
            return None
        if configured in {"float16", "fp16"}:
            return torch.float16
        if configured in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if configured in {"float32", "fp32"}:
            return torch.float32
        return None
