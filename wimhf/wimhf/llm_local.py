"""Local LLM utilities for WIMHF."""

import gc
import os

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# FlashInfer's top-k/top-p sampler JIT-compiles CUDA code at runtime and
# requires nvcc. Default to vLLM's native sampler unless the user opts in.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import torch
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from requests.exceptions import HTTPError
from tqdm.auto import tqdm  # noqa: F401  # kept for compatibility with upstream expectations
from vllm import LLM, SamplingParams

torch.set_float32_matmul_precision("high")

_LOCAL_ENGINES: dict[str, LLM] = {}

LOCAL_SAMPLING_KEYS = {
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
    "stop",
    "skip_special_tokens",
    "spaces_between_special_tokens",
}


@lru_cache(maxsize=256)
def hf_model_exists(repo_id: str) -> bool:
    """Return True if a Hugging Face repo is accessible to the current user."""
    try:
        HfApi().model_info(repo_id, timeout=3)
        return True
    except RepositoryNotFoundError:
        return False
    except HTTPError as exc:
        return exc.response is not None and exc.response.status_code in {401, 403}


def is_local_model(model: str) -> bool:
    """Check whether a model should be served locally via vLLM."""
    return model in _LOCAL_ENGINES or hf_model_exists(model)


def _sleep_all_except(active_model: Optional[str] = None) -> None:
    """Put every cached vLLM engine *except* ``active_model`` to sleep."""
    for name, engine in _LOCAL_ENGINES.items():
        if name == active_model:
            continue
        if engine.llm_engine.is_sleeping():
            continue
        print(f"Sleeping {name} to free GPU memory...")
        engine.llm_engine.reset_prefix_cache()
        engine.sleep(level=2)  # level 2 clears cache and weights entirely


def sleep_all_vllm_engines() -> None:
    """Sleep cached vLLM engines and clear local CUDA allocator caches."""
    _sleep_all_except(active_model=None)
    gc.collect()
    if not torch.cuda.is_available():
        return
    for device_idx in range(torch.cuda.device_count()):
        with torch.cuda.device(device_idx):
            torch.cuda.empty_cache()


def strip_thinking(text: str) -> str:
    """Remove Qwen-style thinking blocks from a generated completion."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1].strip()
    if "<think>" in text:
        return text.replace("<think>", "").strip()
    return text.strip()


def pop_local_generation_kwargs(kwargs: Dict[str, Any]) -> Tuple[dict, dict, dict]:
    """Extract local vLLM kwargs from a generic completion kwargs dict."""
    tokenizer_kwargs = dict(kwargs.pop("tokenizer_kwargs", {}) or {})
    llm_sampling_kwargs = dict(kwargs.pop("llm_sampling_kwargs", {}) or {})
    llm_engine_kwargs = dict(kwargs.pop("llm_engine_kwargs", {}) or {})

    for key in list(kwargs.keys()):
        if key in LOCAL_SAMPLING_KEYS:
            llm_sampling_kwargs[key] = kwargs.pop(key)

    return tokenizer_kwargs, llm_sampling_kwargs, llm_engine_kwargs


def _prepare_cuda_memory_for_vllm(gpu_memory_utilization: float) -> float:
    if not torch.cuda.is_available():
        return gpu_memory_utilization

    free_fractions = []
    for device_idx in range(torch.cuda.device_count()):
        with torch.cuda.device(device_idx):
            gc.collect()
            torch.cuda.empty_cache()
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_fractions.append(free_bytes / total_bytes)

    if not free_fractions:
        return gpu_memory_utilization

    min_free_fraction = min(free_fractions)
    max_safe_utilization = max(0.05, min_free_fraction - 0.02)
    if gpu_memory_utilization > max_safe_utilization:
        capped = min(gpu_memory_utilization, max_safe_utilization)
        print(
            "[vLLM] capping gpu_memory_utilization "
            f"from {gpu_memory_utilization:.3f} to {capped:.3f} "
            f"(least-free GPU has {min_free_fraction:.3f} free)."
        )
        return capped

    return gpu_memory_utilization


def get_vllm_engine(model: str, **kwargs) -> LLM:
    """
    Return a vLLM engine for ``model``.

    If the engine is cached, wake it up and sleep the rest.
    Otherwise, first sleep every engine to free GPU memory, then load.
    """
    engine = _LOCAL_ENGINES.get(model)

    if engine is None:
        _sleep_all_except(active_model=None)

        print(f"Loading {model} in vLLM...")
        t0 = time.time()
        gpu_memory_utilization = float(kwargs.pop("gpu_memory_utilization", 0.85))
        gpu_memory_utilization = _prepare_cuda_memory_for_vllm(gpu_memory_utilization)
        if torch.cuda.is_available() and "tensor_parallel_size" not in kwargs:
            kwargs["tensor_parallel_size"] = torch.cuda.device_count()
        engine = LLM(
            model=model,
            enable_sleep_mode=True,
            gpu_memory_utilization=gpu_memory_utilization,
            **kwargs,
        )
        _LOCAL_ENGINES[model] = engine
        dtype = "unknown"
        try:
            llm_engine = getattr(engine, "llm_engine", None)
            if llm_engine is not None and hasattr(llm_engine, "get_model_config"):
                dtype = getattr(llm_engine.get_model_config(), "dtype", "unknown")
            elif llm_engine is not None and hasattr(llm_engine, "model_config"):
                dtype = getattr(llm_engine.model_config, "dtype", "unknown")
            elif llm_engine is not None and hasattr(llm_engine, "vllm_config"):
                model_config = getattr(llm_engine.vllm_config, "model_config", None)
                dtype = getattr(model_config, "dtype", "unknown")
        except Exception:
            pass
        print(f"Loaded {model} with dtype: {dtype} (took {time.time()-t0:.1f}s)")
    else:
        _sleep_all_except(active_model=model)
        if engine.llm_engine.is_sleeping():
            print(
                f"Engine found for {model} but model is sleeping, waking up..."
            )
            print(
                "[WARNING]: vLLM wake-up can occasionally produce degraded outputs; "
                "restart the engine if you observe issues."
            )
            engine.wake_up()
            engine.llm_engine.reset_prefix_cache()

    return engine


def get_local_completions(
    prompts: List[str],
    model: str = "Qwen/Qwen3-0.6B",
    max_tokens: int = 128,
    show_progress: bool = True,
    tokenizer_kwargs: Optional[dict] = None,
    llm_sampling_kwargs: Optional[dict] = None,
    llm_engine_kwargs: Optional[dict] = None,
    strip_thinking_blocks: bool = True,
) -> List[str]:
    """Generate completions using vLLM with ``engine.generate``."""
    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}
    if llm_sampling_kwargs is None:
        llm_sampling_kwargs = {}
    if llm_engine_kwargs is None:
        llm_engine_kwargs = {}

    tokenizer_kwargs = dict(tokenizer_kwargs)
    engine = get_vllm_engine(model, **llm_engine_kwargs)
    tokenizer = engine.get_tokenizer()

    if getattr(tokenizer, "chat_template", None) is not None:
        messages_lists = [[{"role": "user", "content": p}] for p in prompts]
        enable_thinking = tokenizer_kwargs.pop(
            "enable_thinking", False
        )  # default False to avoid surprises
        prompts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                **tokenizer_kwargs,
            )
            for messages in messages_lists
        ]

    sampling_params = SamplingParams(max_tokens=max_tokens, **llm_sampling_kwargs)
    outputs = engine.generate(prompts, sampling_params=sampling_params, use_tqdm=show_progress)

    completions = [str(out.outputs[0].text) for out in outputs]
    if strip_thinking_blocks:
        return [strip_thinking(text) for text in completions]
    return completions
