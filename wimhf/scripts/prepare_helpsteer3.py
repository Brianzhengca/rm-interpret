#!/usr/bin/env python3
"""Convert nvidia/HelpSteer3 preference data into WIMHF JSONL format."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from tqdm.auto import tqdm


ROLE_LABELS = {
    "user": "Human",
    "assistant": "Assistant",
    "system": "System",
}


DEFAULT_CONFIG = {
    "dataset": {
        "name": "HelpSteer3",
        "path": "data/helpsteer3_preference_wimhf.jsonl",
        "split_columns": ["conversation_id"],
        "train_split_size": 0.8,
        "split_random_seed": 42,
        "random_seed": 42,
        "max_words_prompt": 256,
        "max_words_response": 256,
    },
    "embedding": {
        "model": "Qwen/Qwen3-Embedding-8B",
        "n_workers": 2,
        "provider": "local",
        "batch_size": 1,
        "text_mode": "instructed_prompt_response",
    },
    "sae": {
        "M": 32,
        "K": 4,
        "prefix_lengths": [8, 32],
        "batch_size": 256,
        "n_epochs": 50,
        "min_epochs": 10,
        "patience": 3,
        "learning_rate": 0.0005,
        "dead_neuron_threshold_steps": 64,
    },
    "interpretation": {
        "interpreter_model": "Qwen/Qwen3-32B",
        "annotator_model": "Qwen/Qwen3-32B",
        "abbreviator_model": "Qwen/Qwen3-32B",
        "n_candidates": 5,
        "interpret_n_examples": 10,
        "interpret_sampling_kwargs": {
            "percentile_bin": [95, 100],
            "nonzero_only": True,
            "swap_negative_response_pairs": True,
        },
        "scoring_n_examples": 300,
        "scoring_sampling_kwargs": {
            "high_percentile": [50, 100],
            "low_percentile": [0, 50],
            "nonzero_only": True,
        },
        "min_correlation": 0.0,
        "p_value_threshold": 0.0015625,
        "n_workers_interpretation": 1,
        "n_workers_annotation": 1,
        "timeout": 120.0,
        "max_interpretation_tokens": 2048,
        "completion_kwargs": {
            "max_tokens": 2048,
            "tokenizer_kwargs": {"enable_thinking": False},
            "llm_sampling_kwargs": {
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20
            },
            "llm_engine_kwargs": {
                "gpu_memory_utilization": 0.7,
                "max_model_len": 16384,
                "max_num_seqs": 1,
                "disable_custom_all_reduce": True,
            }
        },
    },
    "selection": {
        "controls": ["length_delta"],
        "lasso_top_k": [5, 10],
        "max_samples_lasso": None,
        "classification": True,
        "standardize": True,
    },
    "runtime": {
        "checkpoint_dir": "outputs/helpsteer3/checkpoints",
        "cache_dir": "outputs/helpsteer3/emb_cache",
        "retrain_sae": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare nvidia/HelpSteer3 Preference for WIMHF."
    )
    parser.add_argument(
        "--output",
        default="data/helpsteer3_preference_wimhf.jsonl",
        help="Path to write the converted WIMHF JSONL dataset.",
    )
    parser.add_argument(
        "--config-output",
        default="configs/helpsteer3.json",
        help="Path to write a matching WIMHF config.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        help="HelpSteer3 splits to include before WIMHF re-splits by conversation_id.",
    )
    parser.add_argument(
        "--min-abs-preference",
        type=int,
        default=1,
        help="Keep examples with abs(overall_preference) at least this value. Ties are always dropped.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Optional domain filter, e.g. --domains general stem code multilingual.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Optional language filter, e.g. --languages english python.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap after filtering, useful for smoke tests.",
    )
    parser.add_argument(
        "--keep-individual-preferences",
        action="store_true",
        help="Retain HelpSteer3 individual annotator preference records in the JSONL.",
    )
    parser.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-8B",
        help="Embedding model to write into the generated WIMHF config.",
    )
    parser.add_argument(
        "--embedding-provider",
        default="local",
        choices=["openai", "local"],
        help="Embedding provider to write into the generated WIMHF config.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=1,
        help="Embedding batch size to write into the generated WIMHF config.",
    )
    parser.add_argument(
        "--embedding-text-mode",
        default="instructed_prompt_response",
        choices=["response", "prompt_response", "instructed_prompt_response"],
        help="How WIMHF should format text before embedding.",
    )
    return parser.parse_args()


def normalize_filter(values: Iterable[str] | None) -> set[str] | None:
    if values is None:
        return None
    return {value.strip().lower() for value in values}


def serialize_context(context: list[dict[str, Any]]) -> str:
    turns: list[str] = []
    for message in context or []:
        role = str(message.get("role", "")).strip().lower()
        speaker = ROLE_LABELS.get(role, role.title() if role else "Message")
        content = str(message.get("content", "")).strip()
        if content:
            turns.append(f"{speaker}: {content}")
    return "\n\n".join(turns)


def conversation_id(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]


def convert_row(
    row: dict[str, Any],
    source_split: str,
    source_index: int,
    keep_individual_preferences: bool,
) -> dict[str, Any] | None:
    preference = int(row["overall_preference"])
    if preference == 0:
        return None

    prompt = serialize_context(row["context"])
    response1 = str(row["response1"]).strip()
    response2 = str(row["response2"]).strip()
    if not prompt or not response1 or not response2:
        return None

    # HelpSteer3 uses negative scores when response1 is preferred and positive
    # scores when response2 is preferred. WIMHF expects label=1 when A wins.
    record = {
        "prompt": prompt,
        "response_A": response1,
        "response_B": response2,
        "label": 1 if preference < 0 else 0,
        "overall_preference": preference,
        "preference_strength": abs(preference),
        "domain": row["domain"],
        "language": row["language"],
        "conversation_id": conversation_id(prompt),
        "source_split": source_split,
        "source_index": source_index,
    }
    if keep_individual_preferences:
        record["individual_preference"] = row["individual_preference"]
    return record


def write_config(
    path: Path,
    dataset_path: str,
    embedding_model: str,
    embedding_provider: str,
    embedding_batch_size: int,
    embedding_text_mode: str,
) -> None:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    config["dataset"]["path"] = dataset_path
    config["embedding"]["model"] = embedding_model
    config["embedding"]["provider"] = embedding_provider
    config["embedding"]["batch_size"] = embedding_batch_size
    config["embedding"]["text_mode"] = embedding_text_mode
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    domains = normalize_filter(args.domains)
    languages = normalize_filter(args.languages)

    kept = 0
    skipped_ties_or_weak = 0
    skipped_filters = 0
    skipped_invalid = 0
    label_counts = {0: 0, 1: 0}

    with output_path.open("w") as f:
        for split in args.splits:
            ds = load_dataset("nvidia/HelpSteer3", "preference", split=split)
            for idx, row in enumerate(tqdm(ds, desc=f"Converting {split}")):
                preference = int(row["overall_preference"])
                if preference == 0 or abs(preference) < args.min_abs_preference:
                    skipped_ties_or_weak += 1
                    continue

                domain = str(row["domain"]).strip().lower()
                language = str(row["language"]).strip().lower()
                if domains is not None and domain not in domains:
                    skipped_filters += 1
                    continue
                if languages is not None and language not in languages:
                    skipped_filters += 1
                    continue

                record = convert_row(
                    row,
                    source_split=split,
                    source_index=idx,
                    keep_individual_preferences=args.keep_individual_preferences,
                )
                if record is None:
                    skipped_invalid += 1
                    continue

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                label_counts[record["label"]] += 1
                kept += 1

                if args.max_rows is not None and kept >= args.max_rows:
                    break

            if args.max_rows is not None and kept >= args.max_rows:
                break

    write_config(
        Path(args.config_output),
        args.output,
        args.embedding_model,
        args.embedding_provider,
        args.embedding_batch_size,
        args.embedding_text_mode,
    )

    print(f"Wrote {kept} WIMHF rows to {output_path}")
    print(f"Wrote WIMHF config to {args.config_output}")
    print(f"Labels: A preferred={label_counts[1]}, B preferred={label_counts[0]}")
    print(
        "Skipped: "
        f"ties/weak={skipped_ties_or_weak}, filters={skipped_filters}, invalid={skipped_invalid}"
    )


if __name__ == "__main__":
    main()
