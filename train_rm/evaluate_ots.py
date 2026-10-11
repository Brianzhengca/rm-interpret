#!/usr/bin/env python3
"""Evaluate an off-the-shelf reward model on Arena JSONL data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODEL_NAME = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
DATA_PATH = str(Path(__file__).resolve().parents[1] / "wimhf" / "data" / "arena_human_preference_55k_wimhf.jsonl")
OUTPUT_PATH = Path(__file__).resolve().parent / "outputs" / "skywork_reward_v2_llama3_1_8b" / "eval_results.jsonl"
MAX_LENGTH = 16384


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--output-path", default=str(OUTPUT_PATH))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH, help="Skip pairs longer than this; use <=0 for no limit.")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional cap on evaluable examples for smoke tests.")
    parser.add_argument(
        "--keep-leading-bos",
        action="store_true",
        help=(
            "Keep a leading BOS token emitted by the chat template. "
            "By default it is stripped to preserve the existing Skywork evaluation path."
        ),
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional Transformers attention implementation, e.g. flash_attention_2, sdpa, or eager.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.max_examples is not None and args.max_examples <= 0:
        parser.error("--max-examples must be positive when provided.")
    return args


def to_messages(prompt: str, response: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": str(prompt).strip()},
        {"role": "assistant", "content": str(response).strip()},
    ]


def encode(tokenizer, prompt: str, response: str, keep_leading_bos: bool) -> list[int]:
    if keep_leading_bos:
        return tokenizer.apply_chat_template(
            to_messages(prompt, response),
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )

    formatted = tokenizer.apply_chat_template(
        to_messages(prompt, response),
        tokenize=False,
        add_generation_prompt=False,
    )
    if not keep_leading_bos and tokenizer.bos_token is not None and formatted.startswith(tokenizer.bos_token):
        formatted = formatted[len(tokenizer.bos_token) :]
    return tokenizer(formatted)["input_ids"]


def load_examples(
    path: str,
    tokenizer,
    max_length: int,
    max_examples: int | None,
    keep_leading_bos: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    from tqdm.auto import tqdm

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

            chosen_ids = encode(tokenizer, row["prompt"], chosen, keep_leading_bos)
            rejected_ids = encode(tokenizer, row["prompt"], rejected, keep_leading_bos)

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

            if max_examples is not None and len(examples) >= max_examples:
                break

    stats["kept"] = len(examples)
    if max_examples is not None:
        stats["max_examples"] = max_examples
    return examples, stats


def collate_examples(examples: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    import torch
    from torch.nn.utils.rnn import pad_sequence

    chosen_ids = [torch.tensor(example["chosen_ids"], dtype=torch.long) for example in examples]
    rejected_ids = [torch.tensor(example["rejected_ids"], dtype=torch.long) for example in examples]
    input_ids = chosen_ids + rejected_ids
    attention_mask = [torch.ones_like(ids) for ids in input_ids]
    return {
        "eval_index": torch.tensor([example["eval_index"] for example in examples], dtype=torch.long),
        "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id),
        "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0),
    }


def evaluate(model, dataloader: Any, device: Any, examples: list[dict[str, Any]], output_path: Path) -> dict[str, float]:
    import torch
    import torch.nn.functional as F
    from tqdm.auto import tqdm

    total = 0
    correct = 0
    loss_sum = 0.0
    margin_sum = 0.0
    chosen_sum = 0.0
    rejected_sum = 0.0
    min_reward = float("inf")
    max_reward = float("-inf")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_name(f"{output_path.name}.tmp")
    with torch.inference_mode(), tmp_output_path.open("w") as f:
        for batch in tqdm(dataloader, desc="Evaluating"):
            eval_indices = batch.pop("eval_index").tolist()
            batch = {key: value.to(device) for key, value in batch.items()}
            rewards = model(**batch, use_cache=False).logits.squeeze(-1).float()
            if not torch.isfinite(rewards).all():
                raise RuntimeError(f"Reward model produced non-finite rewards: {rewards.detach().cpu().tolist()}")

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

    tmp_output_path.replace(output_path)

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

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples, data_stats = load_examples(
        args.data_path,
        tokenizer,
        args.max_length,
        args.max_examples,
        args.keep_leading_bos,
    )
    if not examples:
        raise RuntimeError("No evaluable examples found.")

    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "num_labels": 1,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, **model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.to(device)
    model.eval()

    dataloader = DataLoader(
        examples,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
        collate_fn=lambda batch: collate_examples(batch, tokenizer.pad_token_id),
    )
    output_path = Path(args.output_path)
    metrics = evaluate(model, dataloader, device, examples, output_path)

    print(
        json.dumps(
            {
                "model_name": args.model_name,
                "data_path": args.data_path,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "max_examples": args.max_examples,
                "keep_leading_bos": args.keep_leading_bos,
                "attn_implementation": args.attn_implementation,
                "output_path": str(output_path),
                "data": data_stats,
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
