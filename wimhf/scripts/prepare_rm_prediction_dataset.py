#!/usr/bin/env python3
"""Convert reward-model prediction JSONL into a WIMHF preference dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SOURCE_DISAMBIGUATION_KEYS = ("conversation_id", "prompt", "response_A", "response_B", "label")
CONTENT_MATCH_KEYS = (
    "conversation_id",
    "prompt",
    "response_A",
    "response_B",
    "label",
    "model_A",
    "model_B",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join evaluate.py reward-model predictions back to a WIMHF-format "
            "base dataset by source_index, replacing label with the model's "
            "predicted A/B preference."
        )
    )
    parser.add_argument("--base-data", required=True, help="Original WIMHF JSONL dataset.")
    parser.add_argument("--predictions", required=True, help="evaluate.py prediction JSONL.")
    parser.add_argument("--output", required=True, help="Converted WIMHF JSONL output path.")
    parser.add_argument("--template-config", required=True, help="WIMHF config JSON to copy settings from.")
    parser.add_argument("--config-output", required=True, help="Generated WIMHF config path.")
    parser.add_argument("--dataset-name", required=True, help="Dataset name for generated WIMHF config.")
    parser.add_argument("--output-dir", required=True, help="WIMHF output directory for runtime cache/checkpoints.")
    parser.add_argument("--reward-model-name", required=True, help="Reward model/checkpoint identifier to store in metadata.")
    return parser.parse_args()


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], dict[int, list[int]], dict[tuple[Any, ...], list[int]]]:
    rows: list[dict[str, Any]] = []
    source_index_to_rows: dict[int, list[int]] = {}
    content_key_to_rows: dict[tuple[Any, ...], list[int]] = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            source_index = row.get("source_index")
            if source_index is not None:
                source_index = int(source_index)
                source_index_to_rows.setdefault(source_index, []).append(len(rows))
            content_key_to_rows.setdefault(content_match_key(row), []).append(len(rows))
            rows.append(row)
    return rows, source_index_to_rows, content_key_to_rows


def content_match_key(row: dict[str, Any]) -> tuple[Any, ...]:
    values: list[Any] = []
    for key in CONTENT_MATCH_KEYS:
        value = row.get(key)
        if key == "label" and value is not None:
            value = int(value)
        values.append(value)
    return tuple(values)


def candidate_matches_prediction(
    base_row: dict[str, Any],
    prediction: dict[str, Any],
    keys: tuple[str, ...] = SOURCE_DISAMBIGUATION_KEYS,
) -> bool:
    for key in keys:
        if key not in prediction:
            return False

        base_value = base_row.get(key)
        pred_value = prediction.get(key)
        if key == "label":
            if int(base_value) != int(pred_value):
                return False
        elif base_value != pred_value:
            return False
    return True


def find_base_row(
    rows: list[dict[str, Any]],
    source_index_to_rows: dict[int, list[int]],
    content_key_to_rows: dict[tuple[Any, ...], list[int]],
    prediction: dict[str, Any],
) -> dict[str, Any] | None:
    data_index = prediction.get("data_index")
    source_index_value = prediction.get("source_index")
    source_index = int(source_index_value) if source_index_value is not None else None

    if data_index is not None and source_index is not None:
        data_index = int(data_index)
        if 0 <= data_index < len(rows):
            row = rows[data_index]
            if int(row.get("source_index", -1)) == source_index:
                return row
            raise ValueError(
                f"data_index/source_index mismatch: data_index={data_index}, "
                f"prediction source_index={source_index}, "
                f"base source_index={row.get('source_index')}"
            )

    if source_index is not None:
        row_indices = source_index_to_rows.get(source_index, [])
        if len(row_indices) == 1:
            return rows[row_indices[0]]
        if len(row_indices) > 1:
            matching_indices = [
                row_idx
                for row_idx in row_indices
                if candidate_matches_prediction(rows[row_idx], prediction)
            ]
            if len(matching_indices) == 1:
                return rows[matching_indices[0]]
            if not matching_indices:
                missing_keys = [key for key in SOURCE_DISAMBIGUATION_KEYS if key not in prediction]
                missing_key_text = f"; prediction missing keys {missing_keys}" if missing_keys else ""
                raise ValueError(
                    f"source_index={source_index} is ambiguous and no candidate matched "
                    f"{SOURCE_DISAMBIGUATION_KEYS}{missing_key_text}; base rows {row_indices[:10]}"
                )
            raise ValueError(
                f"source_index={source_index} is ambiguous after matching "
                f"{SOURCE_DISAMBIGUATION_KEYS}; matched base rows {matching_indices[:10]}"
            )

    content_row_indices = content_key_to_rows.get(content_match_key(prediction), [])
    if len(content_row_indices) == 1:
        return rows[content_row_indices[0]]
    if len(content_row_indices) > 1:
        raise ValueError(
            "Prediction matched multiple base rows by content key "
            f"{CONTENT_MATCH_KEYS}; matched base rows {content_row_indices[:10]}"
        )
    return None


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


def reward_by_position(prediction: dict[str, Any]) -> dict[str, float]:
    rewards = {
        str(prediction["chosen_position"]): float(prediction["chosen_reward"]),
        str(prediction["rejected_position"]): float(prediction["rejected_reward"]),
    }
    if set(rewards) != {"A", "B"}:
        raise ValueError(
            "Expected chosen/rejected positions to cover A and B, got "
            f"{sorted(rewards)} for source_index={prediction.get('source_index')}"
        )
    return rewards


def validate_match(base_row: dict[str, Any], prediction: dict[str, Any]) -> None:
    source_index = prediction.get("source_index")
    pred_label = prediction.get("label")
    if pred_label is not None and int(pred_label) != int(base_row["label"]):
        raise ValueError(
            f"label mismatch for source_index={source_index}: "
            f"prediction={pred_label!r}, base={base_row['label']!r}"
        )

    for key in ("conversation_id", "arena_id", "model_A", "model_B"):
        pred_value = prediction.get(key)
        base_value = base_row.get(key)
        if pred_value is not None and base_value is not None and pred_value != base_value:
            raise ValueError(
                f"{key} mismatch for source_index={source_index}: "
                f"prediction={pred_value!r}, base={base_value!r}"
            )


def convert_row(
    base_row: dict[str, Any],
    prediction: dict[str, Any],
    prediction_path: Path,
    reward_model_name: str,
) -> dict[str, Any]:
    validate_match(base_row, prediction)

    label = prediction_label(prediction)
    rewards = reward_by_position(prediction)
    reward_a = rewards["A"]
    reward_b = rewards["B"]
    human_label = int(base_row["label"])

    row = dict(base_row)
    row["human_label"] = human_label
    if "overall_preference" in base_row:
        row["human_overall_preference"] = base_row["overall_preference"]
    if "preference_strength" in base_row:
        row["human_preference_strength"] = base_row["preference_strength"]
    if "winner_model_a" in base_row:
        row["human_winner_model_a"] = int(base_row["winner_model_a"])
    if "winner_model_b" in base_row:
        row["human_winner_model_b"] = int(base_row["winner_model_b"])
    if "winner_tie" in base_row:
        row["human_winner_tie"] = int(base_row["winner_tie"])

    row["label"] = label
    row["rm_model_name"] = reward_model_name
    row["rm_prediction_file"] = str(prediction_path)
    row["rm_eval_index"] = prediction.get("eval_index")
    row["rm_data_index"] = prediction.get("data_index")
    row["rm_source_index"] = prediction.get("source_index")
    row["rm_predicted_position"] = prediction["predicted_position"]
    row["rm_chosen_position"] = prediction["chosen_position"]
    row["rm_rejected_position"] = prediction["rejected_position"]
    row["rm_chosen_reward"] = float(prediction["chosen_reward"])
    row["rm_rejected_reward"] = float(prediction["rejected_reward"])
    row["rm_eval_margin_chosen_minus_rejected"] = float(prediction["margin"])
    row["rm_margin"] = float(prediction["margin"])
    row["rm_reward_A"] = reward_a
    row["rm_reward_B"] = reward_b
    row["rm_margin_A_minus_B"] = reward_a - reward_b
    row["rm_correct_against_human_label"] = bool(label == human_label)
    return row


def write_config(
    template_config_path: Path,
    config_output_path: Path,
    dataset_name: str,
    dataset_path: Path,
    output_dir: Path,
) -> None:
    with template_config_path.open() as f:
        config = json.load(f)

    config["dataset"]["name"] = dataset_name
    config["dataset"]["path"] = str(dataset_path.resolve())
    config["runtime"]["checkpoint_dir"] = str((output_dir / "checkpoints").resolve())
    config["runtime"]["cache_dir"] = str((output_dir / "emb_cache").resolve())

    config_output_path.parent.mkdir(parents=True, exist_ok=True)
    config_output_path.write_text(json.dumps(config, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_data)
    prediction_path = Path(args.predictions)
    output_path = Path(args.output)
    config_output_path = Path(args.config_output)
    output_dir = Path(args.output_dir)

    base_rows, source_index_to_rows, content_key_to_rows = load_jsonl(base_path)

    kept = 0
    label_counts = {0: 0, 1: 0}
    agreement_count = 0
    missing_predictions: list[dict[str, Any]] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open() as pred_file, output_path.open("w") as out_file:
        for line in pred_file:
            prediction = json.loads(line)
            base_row = find_base_row(base_rows, source_index_to_rows, content_key_to_rows, prediction)
            if base_row is None:
                missing_predictions.append(prediction)
                continue

            row = convert_row(
                base_row=base_row,
                prediction=prediction,
                prediction_path=prediction_path,
                reward_model_name=args.reward_model_name,
            )
            out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            label_counts[int(row["label"])] += 1
            agreement_count += int(row["rm_correct_against_human_label"])

    if missing_predictions:
        missing_source_indices = [prediction.get("source_index") for prediction in missing_predictions[:10]]
        missing_eval_indices = [prediction.get("eval_index") for prediction in missing_predictions[:10]]
        raise RuntimeError(
            f"Missing {len(missing_predictions)} predictions from base data; "
            f"first source_index values: {missing_source_indices}; "
            f"first eval_index values: {missing_eval_indices}"
        )

    write_config(
        template_config_path=Path(args.template_config),
        config_output_path=config_output_path,
        dataset_name=args.dataset_name,
        dataset_path=output_path,
        output_dir=output_dir,
    )

    agreement = agreement_count / kept if kept else 0.0
    print(f"Wrote {kept} WIMHF rows to {output_path}")
    print(f"Wrote WIMHF config to {config_output_path}")
    print(f"Labels: A preferred={label_counts[1]}, B preferred={label_counts[0]}")
    print(f"Agreement with original label: {agreement:.16f}")


if __name__ == "__main__":
    main()
