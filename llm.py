"""Small shared interface for local and OpenAI-compatible language models."""

from __future__ import annotations

import json
import os
import random
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def load_config(path: str | Path, section: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("llm_config.yaml requires PyYAML; install pyyaml first") from exc
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if section not in payload or not isinstance(payload[section], dict):
        raise ValueError(f"missing LLM configuration section: {section}")
    return dict(payload[section])


def parse_json(text: str) -> Any:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as original_error:
        start = candidate.find("{")
        if start < 0:
            raise original_error
        depth, quoted, escaped = 0, False, False
        for index, char in enumerate(candidate[start:], start=start):
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
                continue
            if char == '"':
                quoted = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(candidate[start:index + 1])
        raise original_error


@dataclass
class LLM:
    backend: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2048
    request_timeout: int = 120
    retry_delays: list[int] | tuple[int, ...] = (2, 30, 60, 120, 240, 300)

    @classmethod
    def from_config(cls, path: str | Path, section: str, **overrides: Any) -> "LLM":
        values = load_config(path, section)
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)

    def generate(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        if self.backend in {"openai", "openai_compatible"}:
            return self._openai(messages, json_mode=json_mode)
        if self.backend == "local":
            return self._local(messages)
        raise ValueError("LLM backend must be 'openai_compatible' or 'local'")

    def _openai(self, messages: list[dict[str, str]], *, json_mode: bool) -> str:
        if not self.base_url or not self.api_key_env:
            raise ValueError("openai backend requires base_url and api_key_env")
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(f"environment variable {self.api_key_env} is not set")
        endpoint = self.base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt, delay in enumerate((*self.retry_delays, None), start=1):
            try:
                with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                break
            except Exception as exc:
                last_error = exc
                if delay is None:
                    raise TimeoutError(
                        f"LLM API request failed after {attempt} attempts; retry budget exhausted"
                    ) from exc
                jitter = random.uniform(0, min(30, delay))
                print(
                    f"LLM API attempt {attempt} failed ({type(exc).__name__}); "
                    f"retrying in {delay + jitter:.1f}s",
                    flush=True,
                )
                time.sleep(delay + jitter)
        content = result["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(item.get("text", "") for item in content if isinstance(item, dict))
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM response does not contain text content")
        return content

    def _local(self, messages: list[dict[str, str]]) -> str:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("local LLM generation requires PyTorch and Transformers") from exc
        tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(self.model, trust_remote_code=True, torch_dtype="auto", device_map="auto")
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=self.temperature > 0, temperature=self.temperature if self.temperature > 0 else None)
        return tokenizer.decode(output[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
