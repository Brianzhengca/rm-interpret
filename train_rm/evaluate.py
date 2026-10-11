#!/usr/bin/env python3
"""Evaluate a reward model on Arena Human Preference JSONL data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_PATH = str(Path(__file__).resolve().parent / "outputs" / "qwen3_rm")
DATA_PATH = str(Path(__file__).resolve().parents[1] / "wimhf" / "data" / "arena_human_preference_55k_wimhf.jsonl")
MAX_LENGTH = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument(
        "--dataset-cache-dir",
        default=None,
        help="Optional tokenized DatasetDict cache from train.py. When set, --data-path is ignored.",
    )
    parser.add_argument("--split", default="validation", help="Dataset cache split to evaluate.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH, help="Skip pairs longer than this; use <=0 for no limit.")
    parser.add_argument("--output-path", default=None)
    return parser.parse_args()


def to_messages(prompt: str, response: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": str(prompt).strip()},
        {"role": "assistant", "content": str(response).strip()},
    ]


def encode(tokenizer, prompt: str, response: str) -> list[int]:
    return tokenizer.apply_chat_template(to_messages(prompt, response), return_dict=True)["input_ids"]


def load_examples(path: str, tokenizer, max_length: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    examples = []
    stats = {"total": 0, "ties": 0, "too_long": 0}
    length_limit = max_length if max_length > 0 else None

    with open(path) as f:
        for data_index, line in enumerate(tqdm(f, desc="Tokenizing")):
            stats["total"] += 1
            row = json.loads(line)
            if row["label"] not in (0, 1):
                stats["ties"] += 1
                continue

            if row["label"] == 1:
                chosen = row["response_A"]
                rejected = row["response_B"]
                chosen_position = "A"
                rejected_position = "B"
            else:
                chosen = row["response_B"]
                rejected = row["response_A"]
                chosen_position = "B"
                rejected_position = "A"

            chosen_ids = encode(tokenizer, row["prompt"], chosen)
            rejected_ids = encode(tokenizer, row["prompt"], rejected)

            if length_limit is not None and (len(chosen_ids) > length_limit or len(rejected_ids) > length_limit):
                stats["too_long"] += 1
                continue

            examples.append(
                {
                    "eval_index": len(examples),
                    "data_index": data_index,
                    "source_index": row.get("source_index"),
                    "arena_id": row.get("arena_id"),
                    "conversation_id": row.get("conversation_id"),
                    "model_A": row.get("model_A"),
                    "model_B": row.get("model_B"),
                    "label": int(row["label"]),
                    "chosen_position": chosen_position,
                    "rejected_position": rejected_position,
                    "chosen_ids": chosen_ids,
                    "rejected_ids": rejected_ids,
                }
            )

    stats["kept"] = len(examples)
    return examples, stats


def load_cached_split_examples(cache_dir: str, split_name: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    split = load_from_disk(cache_dir)
    if split_name not in split:
        raise ValueError(f"Split {split_name!r} not found in {cache_dir}. Available splits: {list(split)}")

    dataset = split[split_name]
    examples = []
    for eval_index, row in enumerate(tqdm(dataset, desc=f"Loading {split_name} split")):
        examples.append(
            {
                "eval_index": eval_index,
                "data_index": row.get("source_index"),
                "source_index": row.get("source_index"),
                "arena_id": row.get("arena_id"),
                "conversation_id": row.get("conversation_id"),
                "model_A": row.get("model_A"),
                "model_B": row.get("model_B"),
                "label": int(row["label"]),
                "chosen_position": row["chosen_position"],
                "rejected_position": row["rejected_position"],
                "chosen_ids": row["chosen_ids"],
                "rejected_ids": row["rejected_ids"],
                "prompt": row.get("prompt"),
                "response_A": row.get("response_A"),
                "response_B": row.get("response_B"),
            }
        )

    return examples, {"total": len(dataset), "kept": len(examples)}


def collate_examples(examples: list[dict[str, Any]], pad_token_id: int) -> dict[str, torch.Tensor]:
    chosen_ids = [torch.tensor(example["chosen_ids"], dtype=torch.long) for example in examples]
    rejected_ids = [torch.tensor(example["rejected_ids"], dtype=torch.long) for example in examples]
    input_ids = chosen_ids + rejected_ids
    attention_mask = [torch.ones_like(ids) for ids in input_ids]
    return {
        "eval_index": torch.tensor([example["eval_index"] for example in examples], dtype=torch.long),
        "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id),
        "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0),
    }


@torch.inference_mode()
def evaluate(model, dataloader: DataLoader, device: torch.device, examples: list[dict[str, Any]], output_path: Path) -> dict[str, float]:
    total = 0
    correct = 0
    loss_sum = 0.0
    margin_sum = 0.0
    chosen_sum = 0.0
    rejected_sum = 0.0
    min_reward = float("inf")
    max_reward = float("-inf")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for batch in tqdm(dataloader, desc="Evaluating"):
            eval_indices = batch.pop("eval_index").tolist()
            batch = {key: value.to(device) for key, value in batch.items()}
            rewards = model(**batch, use_cache=False).logits.squeeze(-1).float()
            rewards_chosen, rewards_rejected = torch.chunk(rewards, chunks=2)
            margins = rewards_chosen - rewards_rejected

            batch_size = margins.numel()
            total += batch_size
            correct += (margins > 0).sum().item()
            loss_sum += (-F.logsigmoid(margins)).sum().item()
            margin_sum += margins.sum().item()
            chosen_sum += rewards_chosen.sum().item()
            rejected_sum += rewards_rejected.sum().item()
            min_reward = min(min_reward, rewards.min().item())
            max_reward = max(max_reward, rewards.max().item())

            for eval_index, chosen_reward, rejected_reward, margin in zip(
                eval_indices,
                rewards_chosen.cpu().tolist(),
                rewards_rejected.cpu().tolist(),
                margins.cpu().tolist(),
            ):
                example = examples[eval_index]
                predicted_position = example["chosen_position"] if margin > 0 else example["rejected_position"]
                f.write(
                    json.dumps(
                        {
                            "eval_index": int(eval_index),
                            "data_index": example["data_index"],
                            "source_index": example["source_index"],
                            "arena_id": example["arena_id"],
                            "conversation_id": example["conversation_id"],
                            "model_A": example["model_A"],
                            "model_B": example["model_B"],
                            "label": example["label"],
                            "chosen_position": example["chosen_position"],
                            "rejected_position": example["rejected_position"],
                            "predicted_position": predicted_position,
                            "chosen_reward": float(chosen_reward),
                            "rejected_reward": float(rejected_reward),
                            "margin": float(margin),
                            "correct": bool(margin > 0),
                        }
                    )
                    + "\n"
                )

    return {
        "num_examples": total,
        "accuracy": correct / total,
        "loss": loss_sum / total,
        "mean_margin": margin_sum / total,
        "mean_chosen_reward": chosen_sum / total,
        "mean_rejected_reward": rejected_sum / total,
        "min_reward": min_reward,
        "max_reward": max_reward,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.dataset_cache_dir:
        examples, data_stats = load_cached_split_examples(args.dataset_cache_dir, args.split)
        data_path_for_log = f"{args.dataset_cache_dir}:{args.split}"
    else:
        examples, data_stats = load_examples(args.data_path, tokenizer, args.max_length)
        data_path_for_log = args.data_path
    if not examples:
        raise RuntimeError("No evaluable examples found.")
    output_path = Path(args.output_path) if args.output_path else Path(args.model_path) / "eval_results.jsonl"

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    dataloader = DataLoader(
        examples,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_examples(batch, tokenizer.pad_token_id),
    )
    metrics = evaluate(model, dataloader, device, examples, output_path)

    print(
        json.dumps(
            {
                "model_path": str(Path(args.model_path)),
                "data_path": str(data_path_for_log),
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "output_path": str(output_path),
                "data": data_stats,
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
