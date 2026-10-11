#!/usr/bin/env python3
"""Convert Arena reward-model predictions into WIMHF JSONL format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_BASE_DATA = "data/arena_human_preference_55k_wimhf.jsonl"
DEFAULT_TEMPLATE_CONFIG = "configs/arena_human_preference_55k_qwen3_8b_rm.json"
DEFAULT_PREDICTIONS = (
    "../train_rm/outputs/"
    "skywork_reward_v2_llama3_1_8b/eval_results.jsonl"
)
DEFAULT_OUTPUT = "data/arena_human_preference_55k_skywork_reward_v2_llama3_1_8b_rm_wimhf.jsonl"
DEFAULT_CONFIG_OUTPUT = "configs/arena_human_preference_55k_skywork_reward_v2_llama3_1_8b_rm.json"
DEFAULT_DATASET_NAME = "ArenaHumanPreference55K_Skywork_Reward_V2_Llama3_1_8B_RM"
DEFAULT_OUTPUT_DIR = "outputs/arena_human_preference_55k_skywork_reward_v2_llama3_1_8b_rm"
DEFAULT_REWARD_MODEL_NAME = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join Arena WIMHF rows with reward-model predictions."
    )
    parser.add_argument("--base-data", default=DEFAULT_BASE_DATA)
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--template-config", default=DEFAULT_TEMPLATE_CONFIG)
    parser.add_argument("--config-output", default=DEFAULT_CONFIG_OUTPUT)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reward-model-name", default=DEFAULT_REWARD_MODEL_NAME)
    return parser.parse_args()


def load_jsonl_by_source_index(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            row = json.loads(line)
            source_index = row.get("source_index")
            if source_index is None:
                raise ValueError(f"{path}:{line_no} is missing source_index")
            source_index = int(source_index)
            if source_index in rows:
                raise ValueError(f"{path} has duplicate source_index={source_index}")
            rows[source_index] = row
    return rows


def reward_by_position(prediction: dict[str, Any]) -> dict[str, float]:
    chosen_position = prediction["chosen_position"]
    rejected_position = prediction["rejected_position"]
    rewards = {
        str(chosen_position): float(prediction["chosen_reward"]),
        str(rejected_position): float(prediction["rejected_reward"]),
    }
    if set(rewards) != {"A", "B"}:
        raise ValueError(
            "Expected chosen/rejected positions to cover A and B, got "
            f"{sorted(rewards)} for source_index={prediction.get('source_index')}"
        )
    return rewards


def prediction_label(prediction: dict[str, Any]) -> int:
    predicted_position = prediction.get("predicted_position")
    if predicted_position == "A":
        return 1
    if predicted_position == "B":
        return 0
    raise ValueError(
        f"Invalid predicted_position={predicted_position!r} "
        f"for source_index={prediction.get('source_index')}"
    )


def validate_prediction_matches_base(
    base_row: dict[str, Any],
    prediction: dict[str, Any],
) -> None:
    for key in ("arena_id", "conversation_id", "model_A", "model_B"):
        pred_value = prediction.get(key)
        if pred_value is not None and pred_value != base_row.get(key):
            raise ValueError(
                f"Prediction/source mismatch for source_index={prediction.get('source_index')}: "
                f"{key} prediction={pred_value!r}, base={base_row.get(key)!r}"
            )


def convert_row(
    base_row: dict[str, Any],
    prediction: dict[str, Any],
    prediction_file: Path,
    reward_model_name: str,
) -> dict[str, Any]:
    validate_prediction_matches_base(base_row, prediction)

    label = prediction_label(prediction)
    rewards = reward_by_position(prediction)
    reward_a = rewards["A"]
    reward_b = rewards["B"]

    row = dict(base_row)
    row["human_label"] = int(base_row["label"])
    row["human_winner_model_a"] = int(base_row.get("winner_model_a", 0))
    row["human_winner_model_b"] = int(base_row.get("winner_model_b", 0))
    row["human_winner_tie"] = int(base_row.get("winner_tie", 0))

    row["label"] = label
    row["winner_model_a"] = 1 if label == 1 else 0
    row["winner_model_b"] = 1 if label == 0 else 0
    row["winner_tie"] = 0

    row["rm_model_name"] = reward_model_name
    row["rm_prediction_file"] = str(prediction_file)
    row["rm_eval_index"] = prediction.get("eval_index")
    row["rm_data_index"] = prediction.get("data_index")
    row["rm_prediction_data_index"] = prediction.get("data_index")
    row["rm_source_index"] = prediction.get("source_index")
    row["rm_predicted_position"] = prediction["predicted_position"]
    row["rm_chosen_position"] = prediction["chosen_position"]
    row["rm_rejected_position"] = prediction["rejected_position"]
    row["rm_chosen_reward"] = float(prediction["chosen_reward"])
    row["rm_rejected_reward"] = float(prediction["rejected_reward"])
    row["rm_margin"] = float(prediction["margin"])
    row["rm_correct_against_human"] = bool(prediction["correct"])
    row["rm_reward_A"] = reward_a
    row["rm_reward_B"] = reward_b
    row["rm_margin_A_minus_B"] = reward_a - reward_b
    return row


def write_config(
    template_config_path: Path,
    config_output_path: Path,
    dataset_name: str,
    dataset_path: str,
    output_dir: str,
) -> None:
    with template_config_path.open() as f:
        config = json.load(f)

    config["dataset"]["name"] = dataset_name
    config["dataset"]["path"] = dataset_path
    config["runtime"]["checkpoint_dir"] = f"{output_dir}/checkpoints"
    config["runtime"]["cache_dir"] = f"{output_dir}/emb_cache"

    config_output_path.parent.mkdir(parents=True, exist_ok=True)
    config_output_path.write_text(json.dumps(config, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_data)
    prediction_path = Path(args.predictions)
    output_path = Path(args.output)
    config_output_path = Path(args.config_output)

    base_rows = load_jsonl_by_source_index(base_path)

    kept = 0
    label_counts = {0: 0, 1: 0}
    correct_against_human = 0
    missing_source_indices: list[int] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open() as pred_file, output_path.open("w") as out_file:
        for line in pred_file:
            prediction = json.loads(line)
            source_index = int(prediction["source_index"])
            base_row = base_rows.get(source_index)
            if base_row is None:
                missing_source_indices.append(source_index)
                continue

            row = convert_row(
                base_row=base_row,
                prediction=prediction,
                prediction_file=prediction_path,
                reward_model_name=args.reward_model_name,
            )
            out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            label_counts[int(row["label"])] += 1
            correct_against_human += int(row["label"] == row["human_label"])

    if missing_source_indices:
        raise RuntimeError(
            f"Missing {len(missing_source_indices)} source_index values from base data; "
            f"first few: {missing_source_indices[:10]}"
        )

    write_config(
        template_config_path=Path(args.template_config),
        config_output_path=config_output_path,
        dataset_name=args.dataset_name,
        dataset_path=args.output,
        output_dir=args.output_dir,
    )

    agreement = correct_against_human / kept if kept else 0.0
    print(f"Wrote {kept} WIMHF rows to {output_path}")
    print(f"Wrote WIMHF config to {config_output_path}")
    print(f"Labels: A preferred={label_counts[1]}, B preferred={label_counts[0]}")
    print(f"Agreement with human label: {agreement:.16f}")


if __name__ == "__main__":
    main()
