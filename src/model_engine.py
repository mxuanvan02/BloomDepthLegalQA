"""
Model Engine — unified interface for LLM inference on GPU.

Supports:
    - vLLM (primary): continuous batching, high GPU utilization
    - HuggingFace Transformers (fallback): single-request
    - OpenAI-compatible API (external models)

Usage:
    engine = create_engine("Qwen/Qwen3-8B-Instruct-AWQ", backend="vllm")
    outputs = engine.generate_batch(["prompt1", "prompt2", ...])
    single  = engine.generate("single prompt")
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("bloom_depth.model_engine")


class ModelEngine:
    """Unified LLM inference engine with batched generation."""

    def __init__(
        self,
        model_name: str,
        backend: str = "vllm",
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        gpu_memory_utilization: float = 0.85,
        quantization: str | None = "awq",
        tensor_parallel_size: int = 1,
    ) -> None:
        self.model_name = model_name
        self.backend = backend
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._model = None
        self._tokenizer = None
        self._sampling_params = None

        if backend == "vllm":
            self._init_vllm(gpu_memory_utilization, quantization, tensor_parallel_size)
        elif backend == "transformers":
            self._init_transformers(quantization)
        elif backend == "api":
            self._init_api()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _init_vllm(self, gpu_mem: float, quantization: str | None, tp: int) -> None:
        from vllm import LLM, SamplingParams

        quant = quantization if "AWQ" in self.model_name.upper() or "GPTQ" in self.model_name.upper() else None

        logger.info("Loading vLLM: %s (quant=%s, gpu_mem=%.0f%%)", self.model_name, quant, gpu_mem * 100)
        self._model = LLM(
            model=self.model_name,
            quantization=quant,
            gpu_memory_utilization=gpu_mem,
            tensor_parallel_size=tp,
            trust_remote_code=True,
            max_model_len=4096,
            enforce_eager=True,  # Skip CUDA graph for L4 compatibility
        )
        self._sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        logger.info("vLLM ready: %s", self.model_name)

    def _init_transformers(self, quantization: str | None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
        if quantization == "4bit":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            kwargs["torch_dtype"] = torch.bfloat16

        logger.info("Loading HF Transformers: %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        logger.info("HF ready: %s", self.model_name)

    def _init_api(self) -> None:
        from openai import OpenAI

        self._api_client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
            base_url=os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"),
        )
        logger.info("API client ready: %s → %s", self.model_name, self._api_client.base_url)

    # ── Core inference ──

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate single response."""
        results = self.generate_batch([prompt], **kwargs)
        return results[0]

    def generate_batch(self, prompts: list[str], **kwargs) -> list[str]:
        """Generate responses for a batch of prompts (high GPU utilization)."""
        if not prompts:
            return []

        if self.backend == "vllm":
            return self._generate_vllm(prompts, **kwargs)
        elif self.backend == "transformers":
            return self._generate_hf(prompts, **kwargs)
        elif self.backend == "api":
            return self._generate_api(prompts, **kwargs)
        return [""] * len(prompts)

    def _generate_vllm(self, prompts: list[str], **kwargs) -> list[str]:
        from vllm import SamplingParams

        temperature = kwargs.get("temperature", self.temperature)
        params = SamplingParams(
            max_tokens=kwargs.get("max_tokens", self.max_new_tokens),
            temperature=temperature,
            top_p=kwargs.get("top_p", self.top_p),
        )
        # vLLM handles continuous batching internally — one call = full GPU
        outputs = self._model.generate(prompts, params)
        return [o.outputs[0].text for o in outputs]

    def _generate_hf(self, prompts: list[str], **kwargs) -> list[str]:
        import torch

        results = []
        # HF doesn't batch well for causal LM — process in small chunks
        chunk_size = 4
        for i in range(0, len(prompts), chunk_size):
            batch = prompts[i:i + chunk_size]
            inputs = self._tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=3072,
            ).to(self._model.device)

            with torch.inference_mode():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get("max_tokens", self.max_new_tokens),
                    temperature=kwargs.get("temperature", self.temperature),
                    top_p=kwargs.get("top_p", self.top_p),
                    do_sample=True,
                )
            for j, output in enumerate(outputs):
                new_tokens = output[inputs["input_ids"].shape[1]:]
                results.append(self._tokenizer.decode(new_tokens, skip_special_tokens=True))

        return results

    def _generate_api(self, prompts: list[str], **kwargs) -> list[str]:
        results = []
        for prompt in prompts:
            try:
                resp = self._api_client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=kwargs.get("max_tokens", self.max_new_tokens),
                    temperature=kwargs.get("temperature", self.temperature),
                )
                results.append(resp.choices[0].text)
            except Exception as e:
                logger.error("API error: %s", e)
                results.append("")
        return results

    def unload(self) -> None:
        """Free GPU memory (for sequential model loading on L4)."""
        import gc
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("Unloaded: %s", self.model_name)

    def __repr__(self) -> str:
        return f"ModelEngine({self.model_name}, backend={self.backend})"


# ── Factory ──

def create_engine(
    model_name: str,
    backend: str = "vllm",
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    gpu_memory_utilization: float = 0.85,
    **kwargs,
) -> ModelEngine:
    """Create a ModelEngine instance."""
    return ModelEngine(
        model_name=model_name,
        backend=backend,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        gpu_memory_utilization=gpu_memory_utilization,
        **kwargs,
    )
