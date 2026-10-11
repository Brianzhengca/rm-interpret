#!/usr/bin/env python3
"""Compare human and RM coefficients on a frozen human WIMHF basis."""

from __future__ import annotations

import argparse
from typing import List, Optional


DEFAULT_RM_PREDICTIONS = (
    "../train_rm/outputs/qwen3_8b_rm_lmarena_epoch_sweep_35249460/"
    "*epoch/eval_predictions.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit human and reward-model coefficients on a frozen WIMHF basis "
            "learned from human preferences."
        )
    )
    parser.add_argument(
        "--basis-config",
        default="configs/arena_human_preference_55k.json",
        help="Human WIMHF config used to define dataset split and embedding settings.",
    )
    parser.add_argument(
        "--basis-output-dir",
        default="outputs/arena_human_preference_55k",
        help="Directory containing human feature_table.jsonl and SAE checkpoint.",
    )
    parser.add_argument(
        "--sae-checkpoint",
        default=None,
        help="Optional explicit frozen SAE checkpoint path.",
    )
    parser.add_argument(
        "--rm-predictions",
        nargs="+",
        default=[DEFAULT_RM_PREDICTIONS],
        help="One or more RM prediction JSONL paths or glob patterns.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/frozen_basis_lmarena_qwen3_epochs",
        help="Directory where frozen-basis comparison artifacts will be written.",
    )
    parser.add_argument(
        "--axis-scope",
        choices=["kept", "all"],
        default="kept",
        help="Use retained interpreted axes or all SAE neurons.",
    )
    parser.add_argument(
        "--fit-rows",
        choices=["intersection"],
        default="intersection",
        help="Validation row policy. Intersection fits human and RM on the same rows.",
    )
    parser.add_argument(
        "--join-key",
        default="source_index",
        help="Column/key used to align base validation rows with RM predictions.",
    )
    parser.add_argument(
        "--controls",
        default=None,
        help=(
            "Comma-separated control columns. Defaults to basis config selection.controls. "
            "Use an empty string for no controls."
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional prefix limit after validation/RM intersection for smoke tests.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use a small default row cap when --max-rows is omitted and reduce progress output.",
    )
    parser.add_argument(
        "--allow-embedding-compute",
        action="store_true",
        help="Compute missing embeddings instead of requiring the human basis embedding cache.",
    )
    parser.add_argument(
        "--logistic-c",
        type=float,
        default=1.0,
        help="Inverse regularization strength for logistic human/binary-RM fits.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=0.0,
        help="Ridge alpha for continuous RM fits. Default 0.0 uses ordinary least squares.",
    )
    return parser.parse_args()


def parse_controls(raw: Optional[str], default_controls: List[str]) -> List[str]:
    if raw is None:
        return list(default_controls)
    if raw.strip() == "":
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> None:
    args = parse_args()

    import numpy as np

    from wimhf.frozen_basis import (
        aligned_prediction_targets,
        coefficient_comparison_metrics,
        coefficient_vector,
        common_prediction_keys,
        compute_frozen_activations,
        contribution_mass,
        contribution_mass_to_dict,
        expand_prediction_paths,
        fit_coefficients,
        fit_result_to_dict,
        load_basis_feature_table,
        load_frozen_sae,
        load_frozen_config,
        load_rm_predictions,
        load_split_data,
        make_axis_comparison_table,
        normalize_config_paths,
        resolve_path,
        sanitize_name,
        subset_val_rows,
        unique_checkpoint_names,
        write_json,
    )
    from wimhf.utils import get_repo_root

    repo_root = get_repo_root()

    basis_config_path = resolve_path(args.basis_config, repo_root)
    config = normalize_config_paths(load_frozen_config(basis_config_path), repo_root)
    basis_output_dir = resolve_path(args.basis_output_dir, repo_root)
    output_dir = resolve_path(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_rows = args.max_rows
    if args.debug and max_rows is None:
        max_rows = 256

    controls = parse_controls(args.controls, config.selection.controls)
    prediction_paths = expand_prediction_paths(args.rm_predictions, repo_root)
    checkpoint_names = unique_checkpoint_names(prediction_paths)

    print(f"[frozen-basis] basis_config={basis_config_path}")
    print(f"[frozen-basis] basis_output_dir={basis_output_dir}")
    print(f"[frozen-basis] output_dir={output_dir}")
    print(f"[frozen-basis] RM prediction files={len(prediction_paths)}")

    feature_table = load_basis_feature_table(basis_output_dir, axis_scope=args.axis_scope, base_dir=repo_root)
    axes = [int(axis) for axis in feature_table["neuron_idx"].tolist()]
    feature_table.to_json(output_dir / "frozen_basis_feature_table.jsonl", orient="records", lines=True)

    predictions = {
        checkpoint_names[path]: load_rm_predictions(path, join_key=args.join_key)
        for path in prediction_paths
    }

    _, val_df = load_split_data(config)
    label_col = config.dataset.label_column
    val_binary = val_df[val_df[label_col].isin([0, 1])].reset_index(drop=True)
    common_keys = common_prediction_keys(val_binary, predictions, args.join_key)
    if not common_keys:
        raise RuntimeError(
            f"No validation rows overlapped all RM prediction files on join key {args.join_key!r}"
        )
    rows_df = subset_val_rows(val_binary, common_keys, args.join_key, max_rows=max_rows)
    if rows_df.empty:
        raise RuntimeError("No rows left after validation/RM alignment and --max-rows filtering.")

    print(
        "[frozen-basis] aligned rows="
        f"{len(rows_df)} (common validation/RM rows before max_rows={len(common_keys)})"
    )
    print(f"[frozen-basis] axes={len(axes)} axis_scope={args.axis_scope} controls={controls}")

    sae = load_frozen_sae(
        config,
        basis_output_dir=basis_output_dir,
        checkpoint_path=args.sae_checkpoint,
        base_dir=repo_root,
    )
    activations = compute_frozen_activations(
        rows_df,
        config,
        sae,
        allow_embedding_compute=args.allow_embedding_compute,
        show_progress=not args.debug,
    )

    y_human = rows_df[label_col].to_numpy(dtype=int)
    human_result = fit_coefficients(
        activations,
        y_human,
        axes,
        rows_df,
        controls,
        target_type="binary_logistic",
        logistic_c=args.logistic_c,
    )
    human_beta = coefficient_vector(human_result, axes)
    human_mass = contribution_mass(activations, human_beta, axes)

    human_payload = fit_result_to_dict(human_result)
    human_payload["fit_rows"] = args.fit_rows
    human_payload["join_key"] = args.join_key
    human_payload["basis_config"] = str(basis_config_path)
    human_payload["basis_output_dir"] = str(basis_output_dir)
    write_json(human_payload, output_dir / "human_coefficients.json")
    write_json(contribution_mass_to_dict(human_mass), output_dir / "contribution_mass_human.json")

    rm_coefficients = {}
    rm_masses = {}
    comparison_metrics = {
        "basis_config": str(basis_config_path),
        "basis_output_dir": str(basis_output_dir),
        "axis_scope": args.axis_scope,
        "axes": axes,
        "controls": controls,
        "fit_rows": args.fit_rows,
        "join_key": args.join_key,
        "n_common_rows_before_max_rows": len(common_keys),
        "n_fit_rows": len(rows_df),
        "checkpoints": {},
    }

    for path in prediction_paths:
        checkpoint = checkpoint_names[path]
        pred_df = predictions[checkpoint]
        y_rm, rm_target_kind = aligned_prediction_targets(pred_df, rows_df, args.join_key)
        rm_target_type = "continuous_regression" if rm_target_kind == "continuous_margin" else "binary_logistic"
        rm_result = fit_coefficients(
            activations,
            y_rm,
            axes,
            rows_df,
            controls,
            target_type=rm_target_type,
            logistic_c=args.logistic_c,
            ridge_alpha=args.ridge_alpha,
        )
        rm_beta = coefficient_vector(rm_result, axes)
        rm_mass = contribution_mass(activations, rm_beta, axes)

        rm_payload = fit_result_to_dict(rm_result)
        rm_payload["prediction_file"] = str(path)
        rm_payload["rm_target_kind"] = rm_target_kind
        rm_coefficients[checkpoint] = rm_payload

        human_p = np.array([human_mass.mass_by_neuron[int(axis)] for axis in axes], dtype=float)
        rm_p = np.array([rm_mass.mass_by_neuron[int(axis)] for axis in axes], dtype=float)
        mass_payload = contribution_mass_to_dict(rm_mass)
        mass_payload["prediction_file"] = str(path)
        mass_payload["rm_target_kind"] = rm_target_kind
        mass_payload["axis_amplification_by_neuron"] = {
            str(axis): float(rm_p[idx] - human_p[idx]) for idx, axis in enumerate(axes)
        }
        rm_masses[checkpoint] = mass_payload

        metrics = coefficient_comparison_metrics(human_beta, rm_beta, human_mass, rm_mass, axes)
        metrics["prediction_file"] = str(path)
        metrics["rm_target_kind"] = rm_target_kind
        metrics["rm_fit_target_type"] = rm_target_type
        metrics["rm_fit_metric_primary"] = (
            rm_result.metrics.get("pearson_r")
            if rm_target_type == "continuous_regression"
            else rm_result.metrics.get("roc_auc")
        )
        comparison_metrics["checkpoints"][checkpoint] = metrics

        axis_table = make_axis_comparison_table(
            feature_table,
            human_result,
            rm_result,
            human_mass,
            rm_mass,
        )
        safe_checkpoint = sanitize_name(checkpoint)
        axis_table.to_csv(output_dir / f"axis_comparison_{safe_checkpoint}.csv", index=False)
        axis_table.to_json(
            output_dir / f"axis_comparison_{safe_checkpoint}.jsonl",
            orient="records",
            lines=True,
        )
        if len(prediction_paths) == 1:
            axis_table.to_csv(output_dir / "axis_comparison.csv", index=False)
            axis_table.to_json(output_dir / "axis_comparison.jsonl", orient="records", lines=True)

    write_json(rm_coefficients, output_dir / "rm_coefficients_by_checkpoint.json")
    write_json(rm_masses, output_dir / "contribution_mass_rm.json")
    write_json(comparison_metrics, output_dir / "comparison_metrics.json")

    print(f"[frozen-basis] wrote artifacts to {output_dir}")


if __name__ == "__main__":
    main()
