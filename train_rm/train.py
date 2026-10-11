#!/usr/bin/env python3
"""Train a reward model on Arena Human Preference 55K, ignoring ties."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from accelerate import PartialState
from datasets import Dataset, DatasetDict, load_from_disk
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer


DATA_PATH = (
    str(Path(__file__).resolve().parents[1] / "wimhf" / "data" / "arena_human_preference_55k_wimhf.jsonl")
)
OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "outputs"
    / "qwen3_rm"
)
MODEL_NAME = "Qwen/Qwen3-8B"
MAX_LENGTH = 1024
DATASET_NUM_PROC = 8
TRAIN_SPLIT_SIZE = 0.8
SPLIT_RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--num-train-epochs", type=float, default=3)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--dataset-num-proc", type=int, default=DATASET_NUM_PROC)
    parser.add_argument("--dataset-cache-dir", default=None)
    parser.add_argument("--eval-predictions-path", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--format-mode", choices=("chat", "plain_human_assistant"), default="chat")
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def to_messages(prompt: str, response: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": str(prompt).strip()},
        {"role": "assistant", "content": str(response).strip()},
    ]


def load_dataset(path: str) -> Dataset:
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row["label"] not in (0, 1):
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

            rows.append(
                {
                    "chosen": to_messages(row["prompt"], chosen),
                    "rejected": to_messages(row["prompt"], rejected),
                    "prompt": row["prompt"],
                    "response_A": row["response_A"],
                    "response_B": row["response_B"],
                    "label": row["label"],
                    "chosen_position": chosen_position,
                    "rejected_position": rejected_position,
                    "conversation_id": row["conversation_id"],
                    "arena_id": row.get("arena_id"),
                    "model_A": row.get("model_A"),
                    "model_B": row.get("model_B"),
                    "source_index": row.get("source_index"),
                }
            )

    return Dataset.from_list(rows)


def default_dataset_cache_dir(data_path: str, model_name: str, max_length: int, format_mode: str = "chat") -> Path:
    data_name = Path(data_path).stem
    model_name = model_name.replace("/", "_")
    split_name = f"wimhf_cc_train{int(TRAIN_SPLIT_SIZE * 100)}_seed{SPLIT_RANDOM_SEED}_withmeta"
    format_suffix = "" if format_mode == "chat" else f"_{format_mode}"
    return Path(__file__).resolve().parent / "data_cache" / (
        f"{data_name}_{model_name}_{split_name}{format_suffix}_max{max_length}"
    )


def create_wimhf_split(dataset: Dataset) -> DatasetDict:
    conversation_to_rows: dict[str, list[int]] = {}
    for idx, conversation_id in enumerate(dataset["conversation_id"]):
        conversation_to_rows.setdefault(conversation_id, []).append(idx)

    components = list(conversation_to_rows.values())
    np.random.seed(SPLIT_RANDOM_SEED)
    np.random.shuffle(components)

    train_indices = []
    total_rows = 0
    target_train = len(dataset) * TRAIN_SPLIT_SIZE
    for component in components:
        if total_rows < target_train:
            train_indices.extend(component)
            total_rows += len(component)

    train_mask = [False] * len(dataset)
    for idx in train_indices:
        train_mask[idx] = True

    train_indices = [idx for idx, keep in enumerate(train_mask) if keep]
    val_indices = [idx for idx, keep in enumerate(train_mask) if not keep]
    print(
        "WIMHF connected-component split -> "
        f"train {len(train_indices)} ({len(train_indices) / len(dataset):.1%}), "
        f"validation {len(val_indices)} ({len(val_indices) / len(dataset):.1%})"
    )
    return DatasetDict({"train": dataset.select(train_indices), "validation": dataset.select(val_indices)})


def encode_plain_human_assistant(messages: list[dict[str, str]], tokenizer) -> list[int]:
    if len(messages) != 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        raise ValueError(f"Expected [user, assistant] messages, got {messages}")
    prompt = str(messages[0]["content"]).strip()
    response = str(messages[1]["content"]).strip()
    eos_token = tokenizer.eos_token or ""
    formatted = f"{prompt}\n\nAssistant: {response}{eos_token}"
    return tokenizer(formatted, add_special_tokens=False)["input_ids"]


def encode_messages(messages: list[dict[str, str]], tokenizer, format_mode: str) -> list[int]:
    if format_mode == "chat":
        return tokenizer.apply_chat_template(messages, return_dict=True)["input_ids"]
    if format_mode == "plain_human_assistant":
        return encode_plain_human_assistant(messages, tokenizer)
    raise ValueError(f"Unsupported format_mode={format_mode!r}")


def ensure_chat_template(tokenizer, model_name: str) -> None:
    if getattr(tokenizer, "chat_template", None):
        return

    local_template_path = Path(model_name) / "chat_template.jinja"
    if local_template_path.exists():
        tokenizer.chat_template = local_template_path.read_text()
        print(f"Loaded chat template from {local_template_path}")
        return

    try:
        from huggingface_hub import hf_hub_download

        template_path = hf_hub_download(repo_id=model_name, filename="chat_template.jinja")
    except Exception as exc:
        raise ValueError(
            f"Tokenizer for {model_name!r} has no chat_template and no chat_template.jinja "
            "could be loaded. Use --format-mode plain_human_assistant for base models without "
            "chat templates."
        ) from exc

    tokenizer.chat_template = Path(template_path).read_text()
    print(f"Loaded chat template from {template_path}")


def tokenize_and_filter(dataset: Dataset, tokenizer, max_length: int, num_proc: int, format_mode: str) -> Dataset:
    def tokenize(example):
        chosen_ids = encode_messages(example["chosen"], tokenizer, format_mode)
        rejected_ids = encode_messages(example["rejected"], tokenizer, format_mode)
        return {"chosen_ids": chosen_ids, "rejected_ids": rejected_ids}

    map_kwargs = {"num_proc": min(num_proc, len(dataset))} if num_proc > 1 else {}
    dataset = dataset.map(tokenize, remove_columns=["chosen", "rejected"], desc="Tokenizing dataset", **map_kwargs)
    return dataset.filter(
        lambda example: len(example["chosen_ids"]) <= max_length and len(example["rejected_ids"]) <= max_length,
        desc=f"Filtering dataset >{max_length} tokens",
        **map_kwargs,
    )


def load_or_prepare_dataset(args: argparse.Namespace, tokenizer) -> DatasetDict:
    cache_dir = Path(args.dataset_cache_dir) if args.dataset_cache_dir else default_dataset_cache_dir(
        args.data_path,
        args.model_name,
        args.max_length,
        args.format_mode,
    )
    ready_file = cache_dir / "_READY"

    with PartialState().main_process_first():
        if not ready_file.exists():
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            dataset = load_dataset(args.data_path)
            split = create_wimhf_split(dataset)
            split = DatasetDict(
                {
                    name: tokenize_and_filter(
                        dataset,
                        tokenizer,
                        args.max_length,
                        args.dataset_num_proc,
                        args.format_mode,
                    )
                    for name, dataset in split.items()
                }
            )
            split.save_to_disk(cache_dir)
            ready_file.touch()

    print(f"Loading tokenized dataset from {cache_dir}")
    return load_from_disk(cache_dir)


def zero_reward_head(model) -> None:
    if hasattr(model, "score"):
        torch.nn.init.zeros_(model.score.weight)
        if getattr(model.score, "bias", None) is not None:
            torch.nn.init.zeros_(model.score.bias)


def check_initial_rewards(model, tokenizer, dataset: Dataset) -> None:
    if not torch.cuda.is_available():
        return

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    model.to(f"cuda:{local_rank}")

    examples = dataset.select(range(min(2, len(dataset))))
    input_ids = []
    for example in examples:
        input_ids.append(example["chosen_ids"])
        input_ids.append(example["rejected_ids"])

    device = next(model.parameters()).device
    inputs = tokenizer.pad([{"input_ids": ids} for ids in input_ids], return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    model.eval()
    with torch.no_grad():
        rewards = model(**inputs).logits.float()
    model.train()

    if not torch.isfinite(rewards).all():
        raise RuntimeError(f"Initial reward head produced non-finite rewards: {rewards.flatten().tolist()}")

    if local_rank == 0:
        print(f"Initial rewards are finite: {rewards.flatten().tolist()}")


def collate_prediction_batch(examples: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    chosen_ids = [torch.tensor(example["chosen_ids"]) for example in examples]
    rejected_ids = [torch.tensor(example["rejected_ids"]) for example in examples]
    input_ids = chosen_ids + rejected_ids
    attention_mask = [torch.ones_like(ids) for ids in input_ids]
    return {
        "eval_index": torch.tensor([example["eval_index"] for example in examples], dtype=torch.long),
        "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id),
        "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0),
    }


def export_eval_predictions(trainer: RewardTrainer, dataset: Dataset, output_path: str, pad_token_id: int) -> None:
    output_path = Path(output_path)
    prediction_dataset = dataset.add_column("eval_index", list(range(len(dataset))))
    dataloader = DataLoader(
        prediction_dataset,
        batch_size=trainer.args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=lambda examples: collate_prediction_batch(examples, pad_token_id),
    )
    dataloader = trainer.accelerator.prepare(dataloader)

    model = trainer.model
    model.eval()
    gathered_indices = []
    gathered_chosen_rewards = []
    gathered_rejected_rewards = []

    for batch in dataloader:
        eval_index = batch.pop("eval_index").to(trainer.accelerator.device)
        batch = {key: value.to(trainer.accelerator.device) for key, value in batch.items()}
        with torch.no_grad():
            rewards = model(**batch, use_cache=False).logits.squeeze(-1).float()
        chosen_rewards, rejected_rewards = torch.chunk(rewards, chunks=2)

        gathered_indices.append(trainer.accelerator.gather_for_metrics(eval_index).detach().cpu())
        gathered_chosen_rewards.append(trainer.accelerator.gather_for_metrics(chosen_rewards).detach().cpu())
        gathered_rejected_rewards.append(trainer.accelerator.gather_for_metrics(rejected_rewards).detach().cpu())

    if not trainer.is_world_process_zero():
        return

    indices = torch.cat(gathered_indices).tolist()
    chosen_rewards = torch.cat(gathered_chosen_rewards).tolist()
    rejected_rewards = torch.cat(gathered_rejected_rewards).tolist()

    rows_by_index = {}
    for eval_index, chosen_reward, rejected_reward in zip(indices, chosen_rewards, rejected_rewards):
        example = dataset[int(eval_index)]
        margin = float(chosen_reward - rejected_reward)
        predicted_position = example["chosen_position"] if margin > 0 else example["rejected_position"]
        rows_by_index[int(eval_index)] = {
            "eval_index": int(eval_index),
            "source_index": example.get("source_index"),
            "arena_id": example.get("arena_id"),
            "conversation_id": example.get("conversation_id"),
            "model_A": example.get("model_A"),
            "model_B": example.get("model_B"),
            "label": int(example["label"]),
            "chosen_position": example["chosen_position"],
            "rejected_position": example["rejected_position"],
            "predicted_position": predicted_position,
            "chosen_reward": float(chosen_reward),
            "rejected_reward": float(rejected_reward),
            "margin": margin,
            "correct": bool(margin > 0),
            "prompt": example.get("prompt"),
            "response_A": example.get("response_A"),
            "response_B": example.get("response_B"),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for eval_index in sorted(rows_by_index):
            f.write(json.dumps(rows_by_index[eval_index], ensure_ascii=False) + "\n")
    print(f"Wrote eval predictions to {output_path}")


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.format_mode == "chat":
        ensure_chat_template(tokenizer, args.model_name)

    split = load_or_prepare_dataset(args, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    zero_reward_head(model)
    check_initial_rewards(model, tokenizer, split["train"])

    training_args = RewardConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        max_grad_norm=0.3,
        optim="adamw_torch",
        logging_steps=10,
        bf16=torch.cuda.is_available(),
        max_length=None,
        num_train_epochs=args.num_train_epochs,
        run_name=args.run_name,
        report_to=["wandb"] if args.wandb else [],
    )

    try:
        trainer = RewardTrainer(
            model=model,
            args=training_args,
            train_dataset=split["train"],
            eval_dataset=split["validation"],
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = RewardTrainer(
            model=model,
            args=training_args,
            train_dataset=split["train"],
            eval_dataset=split["validation"],
            tokenizer=tokenizer,
        )

    trainer.train()
    eval_metrics = trainer.evaluate()
    if trainer.is_world_process_zero():
        print(f"Final eval metrics: {eval_metrics}")
    eval_predictions_path = args.eval_predictions_path or str(Path(args.output_dir) / "eval_predictions.jsonl")
    export_eval_predictions(trainer, split["validation"], eval_predictions_path, tokenizer.pad_token_id)
    trainer.save_model(args.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
