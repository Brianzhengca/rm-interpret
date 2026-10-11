"""Higher-level LLM tasks used by WIMHF."""

from __future__ import annotations

import concurrent.futures
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from tqdm.auto import tqdm

from .llm_api import get_completion
from .llm_local import (
    get_local_completions,
    is_local_model,
    pop_local_generation_kwargs,
    strip_thinking,
)

PROMPTS_DIR = Path(__file__).parent / "prompts"


@lru_cache(maxsize=None)
def _load_prompt_template(prompt_name: str) -> str:
    return (PROMPTS_DIR / prompt_name).read_text()


def abbreviate_concept(
    concept: str,
    model: str = "gpt-5-mini",
    completion_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    prompt = _load_prompt_template("abbreviate-concept.txt").format(concept=concept)
    for _ in range(3):
        kwargs: Dict[str, Any] = dict(completion_kwargs or {})
        if is_local_model(model):
            tokenizer_kwargs, llm_sampling_kwargs, llm_engine_kwargs = pop_local_generation_kwargs(kwargs)
            response = get_local_completions(
                [prompt],
                model=model,
                max_tokens=int(kwargs.pop("max_tokens", 1024)),
                show_progress=False,
                tokenizer_kwargs=tokenizer_kwargs,
                llm_sampling_kwargs=llm_sampling_kwargs,
                llm_engine_kwargs=llm_engine_kwargs,
            )[0].strip()
        else:
            if "gpt-5" in model:
                kwargs["reasoning_effort"] = kwargs.get("reasoning_effort", "low")
            response = get_completion(prompt, model=model, **kwargs).strip()
        response = strip_thinking(response).strip().strip('"')
        if response:
            return response
    return concept


def parallel_apply(texts: List[str], func: Callable[[str], str], n_workers: int = 20, desc: str = "Processing") -> List[str]:
    """Apply ``func`` to ``texts`` in parallel, preserving order."""
    results: Dict[int, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(func, text): idx for idx, text in enumerate(texts)}
        iterator = concurrent.futures.as_completed(futures)
        iterator = tqdm(iterator, total=len(texts), desc=desc)
        for future in iterator:
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                print(f"[parallel_apply] error on item {idx}: {exc}")
                results[idx] = ""
    return [results.get(i, "") for i in range(len(texts))]
