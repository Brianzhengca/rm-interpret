#!/usr/bin/env python3
"""Convert lmarena-ai/arena-human-preference-55k into WIMHF JSONL format."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm.auto import tqdm


DEFAULT_CONFIG = {
    "dataset": {
        "name": "ArenaHumanPreference55K",
        "path": "data/arena_human_preference_55k_wimhf.jsonl",
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
        "completion_kwargs": {},
        "interpret_completion_kwargs": {
            "max_tokens": 2048,
            "batch_size": 16,
            "tokenizer_kwargs": {"enable_thinking": False},
            "llm_sampling_kwargs": {
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
            },
            "llm_engine_kwargs": {
                "gpu_memory_utilization": 0.7,
                "max_model_len": 16384,
                "max_num_seqs": 16,
                "disable_custom_all_reduce": True,
            },
        },
        "annotation_completion_kwargs": {
            "max_tokens": 8,
            "batch_size": 64,
            "tokenizer_kwargs": {"enable_thinking": False},
            "llm_sampling_kwargs": {"temperature": 0.0},
            "llm_engine_kwargs": {
                "gpu_memory_utilization": 0.7,
                "max_model_len": 16384,
                "max_num_seqs": 16,
                "disable_custom_all_reduce": True,
            },
        },
        "abbreviation_completion_kwargs": {
            "max_tokens": 64,
            "tokenizer_kwargs": {"enable_thinking": False},
            "llm_sampling_kwargs": {"temperature": 0.0},
            "llm_engine_kwargs": {
                "gpu_memory_utilization": 0.7,
                "max_model_len": 16384,
                "max_num_seqs": 16,
                "disable_custom_all_reduce": True,
            },
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
        "checkpoint_dir": "outputs/arena_human_preference_55k/checkpoints",
        "cache_dir": "outputs/arena_human_preference_55k/emb_cache",
        "retrain_sae": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare lmarena-ai/arena-human-preference-55k for WIMHF."
    )
    parser.add_argument(
        "--output",
        default="data/arena_human_preference_55k_wimhf.jsonl",
        help="Path to write the converted WIMHF JSONL dataset.",
    )
    parser.add_argument(
        "--config-output",
        default="configs/arena_human_preference_55k.json",
        help="Path to write a matching WIMHF config.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap after filtering, useful for smoke tests.",
    )
    parser.add_argument(
        "--keep-ties",
        action="store_true",
        help="Keep tie rows with label -1. By default WIMHF binary preference rows only are emitted.",
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


def parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = [value]
        items = parsed if isinstance(parsed, list) else [parsed]
    else:
        items = [value]
    return [str(item).strip() for item in items if str(item).strip()]


def serialize_prompt(turns: list[str]) -> str:
    return "\n\n".join(f"Human: {turn}" for turn in turns)


def serialize_response(turns: list[str]) -> str:
    if len(turns) == 1:
        return turns[0]
    return "\n\n".join(f"Turn {idx}: {turn}" for idx, turn in enumerate(turns, start=1))


def conversation_id(prompt_turns: list[str]) -> str:
    canonical = json.dumps(prompt_turns, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


def winner_label(row: dict[str, Any], keep_ties: bool) -> int | None:
    if int(row["winner_model_a"]) == 1:
        return 1
    if int(row["winner_model_b"]) == 1:
        return 0
    if int(row["winner_tie"]) == 1 and keep_ties:
        return -1
    return None


def convert_row(row: dict[str, Any], source_index: int, keep_ties: bool) -> dict[str, Any] | None:
    label = winner_label(row, keep_ties=keep_ties)
    if label is None:
        return None

    prompt_turns = parse_json_list(row["prompt"])
    response_a_turns = parse_json_list(row["response_a"])
    response_b_turns = parse_json_list(row["response_b"])
    if not prompt_turns or not response_a_turns or not response_b_turns:
        return None

    return {
        "prompt": serialize_prompt(prompt_turns),
        "response_A": serialize_response(response_a_turns),
        "response_B": serialize_response(response_b_turns),
        "label": label,
        "conversation_id": conversation_id(prompt_turns),
        "arena_id": int(row["id"]),
        "model_A": row["model_a"],
        "model_B": row["model_b"],
        "winner_model_a": int(row["winner_model_a"]),
        "winner_model_b": int(row["winner_model_b"]),
        "winner_tie": int(row["winner_tie"]),
        "num_turns": len(prompt_turns),
        "source_dataset": "lmarena-ai/arena-human-preference-55k",
        "source_split": "train",
        "source_index": source_index,
    }


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

    label_counts = {0: 0, 1: 0, -1: 0}
    kept = 0
    skipped_ties = 0
    skipped_invalid = 0

    ds = load_dataset("lmarena-ai/arena-human-preference-55k", split="train")
    with output_path.open("w") as f:
        for idx, row in enumerate(tqdm(ds, desc="Converting arena-human-preference-55k")):
            record = convert_row(row, source_index=idx, keep_ties=args.keep_ties)
            if record is None:
                if int(row["winner_tie"]) == 1 and not args.keep_ties:
                    skipped_ties += 1
                else:
                    skipped_invalid += 1
                continue

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            label_counts[record["label"]] += 1
            kept += 1

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
    print(f"Labels: A preferred={label_counts[1]}, B preferred={label_counts[0]}, ties={label_counts[-1]}")
    print(f"Skipped: ties={skipped_ties}, invalid={skipped_invalid}")


if __name__ == "__main__":
    main()
