"""Unified model backends for local Hugging Face and OpenAI-compatible models."""

from __future__ import annotations

import json
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from llm import LLM, load_config

from .dataset import serialized_prompt
from .protocol import EvalSample, Generation


@dataclass(frozen=True)
class ModelSpec:
    """Backend-neutral model settings used by the EvalTS runner."""

    backend: str
    model: str = ""
    input_mode: str = "auto"
    base_url: str | None = None
    api_key_env: str | None = None
    max_tokens: int = 512
    batch_size: int = 1
    device: str | None = None
    output_path: str | None = None
    options: dict[str, Any] | None = None

    @classmethod
    def from_file(cls, path: Path) -> "ModelSpec":
        values = load_config(path, "evaluation_model")
        generation = values.pop("generation", {})
        if generation:
            if not isinstance(generation, dict):
                raise ValueError("evaluation_model.generation must be a mapping")
            values.setdefault("max_tokens", generation.get("max_new_tokens"))
        known = set(cls.__dataclass_fields__) - {"options"}
        options = values.pop("options", {})
        if not isinstance(options, dict):
            raise ValueError("evaluation_model.options must be a mapping")
        options.update({key: value for key, value in values.items() if key not in known})
        values = {key: value for key, value in values.items() if key in known and value is not None}
        return cls(**values, options=options or None)


class ModelBackend(Protocol):
    def generate(self, samples: list[EvalSample]) -> list[Generation]: ...


class OpenAICompatibleBackend:
    def __init__(self, spec: ModelSpec) -> None:
        if not spec.base_url or not spec.api_key_env:
            raise ValueError("openai_compatible backend requires base_url and api_key_env")
        self.llm = LLM(
            backend="openai_compatible",
            base_url=spec.base_url,
            api_key_env=spec.api_key_env,
            model=spec.model,
            max_tokens=spec.max_tokens,
        )

    def generate(self, samples: list[EvalSample]) -> list[Generation]:
        return [
            Generation(sample.id, self.llm.generate([{"role": "user", "content": serialized_prompt(sample)}]))
            for sample in samples
        ]


class HuggingFaceBackend:
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    def generate(self, samples: list[EvalSample]) -> list[Generation]:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("local EvalTS evaluation requires PyTorch and Transformers") from exc
        model_path = Path(self.spec.model)
        if not model_path.exists():
            raise FileNotFoundError(f"local model not found: {model_path}")
        if self.spec.input_mode not in {"auto", "serialized", "timeseries"}:
            raise ValueError("input_mode must be 'auto', 'serialized', or 'timeseries'")
        device = torch.device(self.spec.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        serialized = self.spec.input_mode == "serialized"
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=not serialized, torch_dtype="auto"
        ).to(device)
        model.eval()
        model_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        supports_processor = "AutoProcessor" in model_config.get("auto_map", {})
        use_timeseries = self.spec.input_mode == "timeseries" or (
            self.spec.input_mode == "auto" and supports_processor
        )
        if use_timeseries and not supports_processor:
            raise RuntimeError(f"{model_path} does not provide a native time-series processor")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True) if use_timeseries else None
        tokenizer = None if use_timeseries else AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=not serialized
        )
        if tokenizer is not None:
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        outputs: list[Generation] = []
        with torch.inference_mode():
            if processor is not None:
                for sample in samples:
                    inputs = processor(text=sample.question, timeseries=sample.timeseries, return_tensors="pt")
                    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
                    generated = model.generate(**inputs, max_new_tokens=self.spec.max_tokens, do_sample=False)
                    outputs.append(Generation(sample.id, processor.decode(generated[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)))
            else:
                for start in range(0, len(samples), self.spec.batch_size):
                    batch = samples[start:start + self.spec.batch_size]
                    prompts = [tokenizer.apply_chat_template(
                        [{"role": "user", "content": serialized_prompt(sample)}], tokenize=False, add_generation_prompt=True
                    ) for sample in batch]
                    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
                    generated = model.generate(**inputs, max_new_tokens=self.spec.max_tokens, do_sample=False)
                    prompt_length = inputs["input_ids"].shape[-1]
                    outputs.extend(
                        Generation(sample.id, tokenizer.decode(item[prompt_length:], skip_special_tokens=True))
                        for sample, item in zip(batch, generated)
                    )
        return outputs


class ExternalJSONLBackend:
    """Score outputs written by an external model without importing its runtime."""

    def __init__(self, spec: ModelSpec) -> None:
        if not spec.output_path:
            raise ValueError("external_jsonl backend requires output_path")
        self.output_path = Path(spec.output_path)

    def generate(self, samples: list[EvalSample]) -> list[Generation]:
        rows = [json.loads(line) for line in self.output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        requested = {sample.id for sample in samples}
        generations: list[Generation] = []
        for row in rows:
            if set(row) != {"id", "output"}:
                raise ValueError("external output rows must contain exactly id and output")
            if str(row["id"]) in requested:
                generations.append(Generation(str(row["id"]), row["output"]))
        return generations


def build_backend(spec: ModelSpec) -> ModelBackend:
    if spec.backend == "openai_compatible":
        return OpenAICompatibleBackend(spec)
    if spec.backend == "huggingface":
        return HuggingFaceBackend(spec)
    if spec.backend == "external_jsonl":
        return ExternalJSONLBackend(spec)
    if ":" in spec.backend:
        module_name, factory_name = spec.backend.split(":", maxsplit=1)
        factory: Callable[[ModelSpec], ModelBackend] = getattr(importlib.import_module(module_name), factory_name)
        backend = factory(spec)
        if not hasattr(backend, "generate"):
            raise TypeError("custom backend factory must return an object with generate(samples)")
        return backend
    raise ValueError("backend must be a built-in name or 'package.module:factory'")
