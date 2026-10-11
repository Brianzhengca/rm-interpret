"""Text annotation utilities used throughout WIMHF."""

from __future__ import annotations

import concurrent.futures
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from .llm_api import get_completion
from .llm_local import (
    get_local_completions,
    is_local_model,
    pop_local_generation_kwargs,
    strip_thinking,
)
from .utils import get_artifact_subdir, load_prompt, truncate_text

ANNOTATION_CACHE_DIR = get_artifact_subdir("annotation_cache", env_override="ANNOT_CACHE_DIR")
DEFAULT_N_WORKERS = 30


# -----------------------------------------------------------------------------
# Cache helpers
# -----------------------------------------------------------------------------

def get_annotation_cache(cache_path: Optional[Path]) -> Dict[str, Any]:
    if cache_path and cache_path.exists():
        try:
            with cache_path.open("r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"[annotate] failed to parse cache at {cache_path}; starting fresh")
    return {}


def save_annotation_cache(cache_path: Optional[Path], cache: Dict[str, Any]) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(cache, f)
    tmp_path.replace(cache_path)


def generate_cache_key(concept: str, text: str) -> str:
    return f"{concept}|||{text[:100]}...{text[-100:]}"


def _extract_discrete_label(response_text: str, labels: Tuple[str, ...]) -> Optional[str]:
    text = strip_thinking(response_text).strip().lower()
    text = text.strip(' \t\r\n"\'`.,;:!?()[]{}')
    if text in labels:
        return text

    marker = re.search(
        rf"(?:final answer|answer|output)\s*[:\-]\s*['\"]?\b({'|'.join(labels)})\b",
        text,
        flags=re.I,
    )
    if marker:
        return marker.group(1).lower()

    for line in reversed([line.strip() for line in text.splitlines() if line.strip()]):
        cleaned = line.strip(' \t\r\n"\'`.,;:!?()[]{}').lower()
        if cleaned in labels:
            return cleaned
    return None


def _format_annotation_prompt(
    prompt_data: Dict[str, str],
    prompt_name: str,
    max_words_per_example: Optional[int] = None,
) -> str:
    if max_words_per_example:
        prompt_data = dict(prompt_data)
        for key in ("text", "text_a", "text_b"):
            if key in prompt_data:
                prompt_data[key] = truncate_text(prompt_data[key], max_words=max_words_per_example)

    prompt_template = load_prompt(prompt_name)
    return prompt_template.format(**prompt_data)


def _parse_annotation_response(
    completion: str,
    prompt_name: str,
    response_parsing_function: Optional[Callable[[str], Any]] = None,
) -> Optional[Any]:
    response_text = strip_thinking(completion).strip().lower()

    if response_parsing_function:
        parsed = response_parsing_function(response_text)
        if parsed is not None:
            return parsed
        label = _extract_discrete_label(response_text, ("a", "b", "tie"))
        if label is not None:
            return response_parsing_function(label)
        return None

    if prompt_name in {"pairwise-annotate-singleconcept", "annotate-pairwise"}:
        label = _extract_discrete_label(response_text, ("a", "b", "tie"))
        if label == "a":
            return 1
        if label == "b":
            return -1
        if label == "tie":
            return 0
        return None

    label = _extract_discrete_label(response_text, ("yes", "no"))
    if label == "yes":
        return 1
    if label == "no":
        return 0
    return None


def _local_annotation_kwargs(
    *,
    max_tokens: Optional[int],
    temperature: Optional[float],
    tokenizer_kwargs: Optional[Dict[str, Any]],
    llm_sampling_kwargs: Optional[Dict[str, Any]],
    llm_engine_kwargs: Optional[Dict[str, Any]],
    completion_kwargs: Dict[str, Any],
) -> Tuple[int, Dict[str, Any], Dict[str, Any], Dict[str, Any], int]:
    local_kwargs = dict(completion_kwargs)
    local_batch_size = int(local_kwargs.pop("batch_size", local_kwargs.pop("local_batch_size", 256)))
    local_tokenizer_kwargs, local_sampling_kwargs, local_engine_kwargs = pop_local_generation_kwargs(
        local_kwargs
    )
    local_tokenizer_kwargs.update(tokenizer_kwargs or {})
    local_sampling_kwargs.update(llm_sampling_kwargs or {})
    local_engine_kwargs.update(llm_engine_kwargs or {})
    if temperature is not None:
        local_sampling_kwargs["temperature"] = temperature

    thinking_enabled = bool(local_tokenizer_kwargs.get("enable_thinking"))
    max_tokens_local = max_tokens
    if max_tokens_local in (None, 1):
        max_tokens_local = 2048 if thinking_enabled else 8

    return (
        int(max_tokens_local),
        local_tokenizer_kwargs,
        local_sampling_kwargs,
        local_engine_kwargs,
        max(1, local_batch_size),
    )


# -----------------------------------------------------------------------------
# Single-concept annotation
# -----------------------------------------------------------------------------

def annotate_single_text(
    prompt_data: Dict[str, str],
    prompt_name: str = "pairwise-annotate-singleconcept",
    model: str = "gpt-4.1-mini",
    reasoning_effort: Optional[str] = "low",
    max_words_per_example: Optional[int] = None,
    temperature: Optional[float] = None,
    max_retries: int = 3,
    timeout: float = 5.0,
    max_tokens: int = 1,
    response_parsing_function: Optional[Callable[[str], Any]] = None,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    llm_sampling_kwargs: Optional[Dict[str, Any]] = None,
    llm_engine_kwargs: Optional[Dict[str, Any]] = None,
    **completion_kwargs: Any,
) -> Tuple[Optional[Any], float]:
    """
    Annotate a single text using a prompt template and return (parsed_output, api_time).
    """
    prompt = _format_annotation_prompt(prompt_data, prompt_name, max_words_per_example)

    total_api_time = 0.0
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            if is_local_model(model):
                max_tokens_local, local_tokenizer_kwargs, local_sampling_kwargs, local_engine_kwargs, _ = (
                    _local_annotation_kwargs(
                        max_tokens=max_tokens,
                        temperature=temperature,
                        tokenizer_kwargs=tokenizer_kwargs,
                        llm_sampling_kwargs=llm_sampling_kwargs,
                        llm_engine_kwargs=llm_engine_kwargs,
                        completion_kwargs=completion_kwargs,
                    )
                )
                completion = get_local_completions(
                    [prompt],
                    model=model,
                    max_tokens=max_tokens_local,
                    show_progress=False,
                    tokenizer_kwargs=local_tokenizer_kwargs,
                    llm_sampling_kwargs=local_sampling_kwargs,
                    llm_engine_kwargs=local_engine_kwargs,
                )[0]
            elif "gpt-5" in model:
                completion = get_completion(
                    prompt=prompt,
                    model=model,
                    max_completion_tokens=max_tokens if max_tokens not in (None, 1) else 2048,
                    timeout=timeout if timeout not in (None, 5.0) else 20.0,
                    reasoning_effort=reasoning_effort,
                )
            else:
                max_tokens_local = max_tokens
                timeout_local = timeout
                if model.startswith("o"):
                    max_tokens_local = 2048 if max_tokens in (None, 1) else max_tokens
                    timeout_local = 20.0 if timeout in (None, 5.0) else timeout
                completion_kwargs = dict(
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens_local,
                    timeout=timeout_local,
                )
                if temperature is not None:
                    completion_kwargs["temperature"] = temperature
                completion = get_completion(**completion_kwargs)
            total_api_time += time.time() - start_time
            parsed = _parse_annotation_response(
                completion,
                prompt_name=prompt_name,
                response_parsing_function=response_parsing_function,
            )
            if parsed is not None:
                return parsed, total_api_time
        except Exception as exc:
            if attempt == max_retries - 1:
                print(f"[annotate] failure after {max_retries} attempts: {exc}")
                return None, total_api_time
    return None, total_api_time


def _batch_annotate_local(
    tasks: List[Tuple[str, str]],
    cache: Dict[str, Any],
    cache_path: Optional[Path],
    results: Dict[str, Dict[str, Any]],
    progress_desc: str,
    show_progress: bool,
    **kwargs: Any,
) -> None:
    kwargs = dict(kwargs)
    model = kwargs.pop("model", "gpt-4.1-mini")
    prompt_name = kwargs.pop("prompt_name", "pairwise-annotate-singleconcept")
    response_parsing_function = kwargs.pop("response_parsing_function", None)
    max_words_per_example = kwargs.pop("max_words_per_example", None)
    temperature = kwargs.pop("temperature", None)
    max_retries = int(kwargs.pop("max_retries", 3))
    kwargs.pop("timeout", None)
    kwargs.pop("reasoning_effort", None)
    max_tokens = kwargs.pop("max_tokens", 1)
    tokenizer_kwargs = kwargs.pop("tokenizer_kwargs", None)
    llm_sampling_kwargs = kwargs.pop("llm_sampling_kwargs", None)
    llm_engine_kwargs = kwargs.pop("llm_engine_kwargs", None)

    (
        max_tokens_local,
        local_tokenizer_kwargs,
        local_sampling_kwargs,
        local_engine_kwargs,
        local_batch_size,
    ) = _local_annotation_kwargs(
        max_tokens=max_tokens,
        temperature=temperature,
        tokenizer_kwargs=tokenizer_kwargs,
        llm_sampling_kwargs=llm_sampling_kwargs,
        llm_engine_kwargs=llm_engine_kwargs,
        completion_kwargs=kwargs,
    )

    formatted_tasks = [
        (
            text,
            concept,
            _format_annotation_prompt(
                {"concept": concept, "text": text},
                prompt_name=prompt_name,
                max_words_per_example=max_words_per_example,
            ),
        )
        for text, concept in tasks
    ]

    progress = tqdm(
        total=len(formatted_tasks),
        desc=progress_desc,
        disable=not show_progress,
        miniters=1,
        mininterval=0,
    )
    for start in range(0, len(formatted_tasks), local_batch_size):
        chunk = formatted_tasks[start : start + local_batch_size]
        remaining = chunk
        parsed_count = 0
        for attempt in range(max_retries):
            if not remaining:
                break
            prompts = [prompt for _, _, prompt in remaining]
            try:
                completions = get_local_completions(
                    prompts,
                    model=model,
                    max_tokens=max_tokens_local,
                    show_progress=False,
                    tokenizer_kwargs=local_tokenizer_kwargs,
                    llm_sampling_kwargs=local_sampling_kwargs,
                    llm_engine_kwargs=local_engine_kwargs,
                )
            except Exception as exc:
                if attempt == max_retries - 1:
                    print(f"[annotate] batch failure after {max_retries} attempts: {exc}")
                continue

            retry_items = []
            for (text, concept, prompt), completion in zip(remaining, completions):
                annotation = _parse_annotation_response(
                    completion,
                    prompt_name=prompt_name,
                    response_parsing_function=response_parsing_function,
                )
                if annotation is None:
                    retry_items.append((text, concept, prompt))
                    continue
                results.setdefault(concept, {})[text] = annotation
                if cache_path:
                    cache[generate_cache_key(concept, text)] = annotation
                parsed_count += 1
                progress.update(1)
            remaining = retry_items
        if cache_path:
            save_annotation_cache(cache_path, cache)
        if parsed_count < len(chunk):
            progress.update(len(chunk) - parsed_count)
    progress.close()


def _parallel_annotate(
    tasks: List[Tuple[str, str]],
    n_workers: int,
    cache: Dict[str, Any],
    cache_path: Optional[Path],
    results: Dict[str, Dict[str, Any]],
    progress_desc: str,
    show_progress: bool,
    **kwargs: Any,
) -> None:
    retry_tasks: List[Tuple[str, str]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                annotate_single_text,
                prompt_data={"concept": concept, "text": text},
                **kwargs,
            ): (text, concept)
            for text, concept in tasks
        }

        iterator = concurrent.futures.as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(futures), desc=progress_desc)

        for future in iterator:
            text, concept = futures[future]
            try:
                annotation, _ = future.result()
                if annotation is not None:
                    results.setdefault(concept, {})[text] = annotation
                    if cache_path:
                        cache[generate_cache_key(concept, text)] = annotation
                        if len(cache) % max(1, min(n_workers, 100)) == 0:
                            save_annotation_cache(cache_path, cache)
            except Exception as exc:
                print(f"[annotate] task failed ({concept}): {exc}")
                retry_tasks.append((text, concept))

    for text, concept in retry_tasks:
        annotation, _ = annotate_single_text(
            prompt_data={"concept": concept, "text": text}, **kwargs
        )
        if annotation is not None:
            results.setdefault(concept, {})[text] = annotation
            if cache_path:
                cache[generate_cache_key(concept, text)] = annotation


def annotate(
    tasks: List[Tuple[str, str]],
    cache_path: Optional[Path] = None,
    n_workers: int = DEFAULT_N_WORKERS,
    progress_desc: str = "Annotating",
    show_progress: bool = True,
    **kwargs: Any,
) -> Dict[str, Dict[str, Any]]:
    """Annotate a list of ``(text, concept)`` tasks, optionally using a JSON cache."""
    cache = get_annotation_cache(cache_path)
    results: Dict[str, Dict[str, Any]] = {}
    uncached_tasks: List[Tuple[str, str]] = []

    for text, concept in tasks:
        key = generate_cache_key(concept, text)
        if key in cache:
            results.setdefault(concept, {})[text] = cache[key]
        else:
            uncached_tasks.append((text, concept))

    print(
        f"[annotate] cache hits: {len(tasks) - len(uncached_tasks)} / {len(tasks)}; "
        f"querying {len(uncached_tasks)} items"
    )

    if uncached_tasks:
        model = kwargs.get("model", "gpt-4.1-mini")
        if is_local_model(model):
            _batch_annotate_local(
                tasks=uncached_tasks,
                cache=cache,
                cache_path=cache_path,
                results=results,
                progress_desc=progress_desc,
                show_progress=show_progress,
                **kwargs,
            )
        else:
            _parallel_annotate(
                tasks=uncached_tasks,
                n_workers=n_workers,
                cache=cache,
                cache_path=cache_path,
                results=results,
                progress_desc=progress_desc,
                show_progress=show_progress,
                **kwargs,
            )

    save_annotation_cache(cache_path, cache)
    return results


def annotate_texts_with_concepts(
    texts: List[str],
    concepts: List[str],
    cache_dir: Optional[Path] = None,
    progress_desc: str = "Annotating",
    show_progress: bool = True,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Annotate each ``text`` with every concept, returning concept-indexed arrays."""
    cache_path = (cache_dir / "annotations.json") if cache_dir else None
    tasks = [(text, concept) for text in texts for concept in concepts]

    raw_results = annotate(
        tasks=tasks,
        cache_path=cache_path,
        progress_desc=progress_desc,
        show_progress=show_progress,
        **kwargs,
    )

    concept_arrays: Dict[str, np.ndarray] = {}
    for concept in concepts:
        concept_arrays[concept] = np.array(
            [raw_results.get(concept, {}).get(text, 0) for text in texts]
        )
    return concept_arrays


# -----------------------------------------------------------------------------
# Multiconcept annotation helpers
# -----------------------------------------------------------------------------

_PAIRWISE_MAPPING = {"a": 1, "b": -1, "tie": 0}


def _normalize_label(label: str) -> Optional[int]:
    label = label.strip().strip(' \t\r\n"\'`.,;:!?()[]{}').lower()
    return _PAIRWISE_MAPPING.get(label)


def _parse_concept_annotations(raw_output: Any, n_items: int) -> Optional[List[int]]:
    """Parse multi-concept outputs into a list of {-1, 0, 1} labels."""
    if isinstance(raw_output, (list, tuple)):
        parsed = [_normalize_label(x) for x in raw_output]
        if None not in parsed and len(parsed) >= n_items:
            return parsed[:n_items]
        return None

    text = str(raw_output).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            parsed = [_normalize_label(x) for x in obj]
            if None not in parsed and len(parsed) >= n_items:
                return parsed[:n_items]
    except Exception:
        pass

    match = re.search(r"\[([^\]]+)\]", text, flags=re.S)
    if match:
        parts = re.split(r"[,\n]+", match.group(1))
        parsed = [_normalize_label(x) for x in parts]
        if None not in parsed and len(parsed) >= n_items:
            return parsed[:n_items]

    return None


def multiconcept_annotation(
    concepts_to_annotate: List[str],
    text: str,
    prompt_name: str = "pairwise-annotate-multiconcept",
    **kwargs: Any,
) -> Optional[List[int]]:
    """Run a multiconcept annotation call for ``text`` and return parsed labels."""
    n = len(concepts_to_annotate)
    order = list(range(n))
    random.shuffle(order)

    shuffled_concepts = [concepts_to_annotate[i] for i in order]
    concepts_block = "\n".join(f"- {concept}" for concept in shuffled_concepts)

    parsed, _ = annotate_single_text(
        prompt_data={"concepts_block": concepts_block, "text": text},
        prompt_name=prompt_name,
        response_parsing_function=lambda output: _parse_concept_annotations(output, n),
        **kwargs,
    )
    if parsed is None:
        return None

    restored = [None] * n
    for position, original_idx in enumerate(order):
        restored[original_idx] = parsed[position]
    return restored


def annotate_texts_with_concepts_multiconcept(
    texts: List[str],
    concepts: List[str],
    cache_dir: Optional[Path] = None,
    n_workers: int = DEFAULT_N_WORKERS,
    progress_desc: str = "Annotating (multiconcept)",
    show_progress: bool = True,
    prompt_name: str = "pairwise-annotate-multiconcept",
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """
    Annotate each text with all concepts using a single LLM call per text (when uncached).
    """
    cache_path = (cache_dir / "annotations_multiconcept.json") if cache_dir else None
    cache = get_annotation_cache(cache_path)

    concept_results = {concept: np.zeros(len(texts)) for concept in concepts}
    tasks: List[int] = []

    # Determine which texts require annotation (any missing concepts)
    for idx, text in enumerate(texts):
        missing = []
        for concept in concepts:
            key = generate_cache_key(concept, text)
            cached = cache.get(key)
            if cached is not None:
                concept_results[concept][idx] = cached
            else:
                missing.append(concept)
        if missing:
            tasks.append(idx)

    print(
        f"[annotate-multiconcept] cache hits: {len(texts) - len(tasks)} / {len(texts)};"
        f" querying {len(tasks)} texts"
    )

    if tasks:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    multiconcept_annotation,
                    [concepts[i] for i in range(len(concepts))],
                    texts[idx],
                    prompt_name=prompt_name,
                    **kwargs,
                ): idx
                for idx in tasks
            }

            iterator = concurrent.futures.as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(futures), desc=progress_desc)

            for future in iterator:
                idx = futures[future]
                parsed = future.result()
                if parsed is None:
                    continue
                for concept, value in zip(concepts, parsed):
                    concept_results[concept][idx] = value
                    cache[generate_cache_key(concept, texts[idx])] = value

    save_annotation_cache(cache_path, cache)
    return concept_results
