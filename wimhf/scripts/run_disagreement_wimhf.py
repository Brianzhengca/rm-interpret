#!/usr/bin/env python3
"""Build and run native WIMHF for human/RM disagreement.

Response A is rewritten to mean "human preferred" and response B to mean
"human rejected", so WIMHF trains on:

    emb(human preferred) - emb(human rejected)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
WIMHF_DIR = SCRIPT.parents[1]
REPO_ROOT = SCRIPT.parents[2]
sys.path.insert(0, str(WIMHF_DIR))

DEFAULT_BASE_CONFIG = "wimhf/configs/arena_human_preference_55k.json"
DEFAULT_HUMAN_DATA = "wimhf/data/arena_human_preference_55k_wimhf.jsonl"
DEFAULT_RM_DATA = "wimhf/data/arena_human_preference_55k_qwen3_8b_rm_wimhf.jsonl"
DEFAULT_OUTPUT_DIR = "wimhf/outputs/disagreement_wimhf"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Native WIMHF for human/RM disagreement.")
    p.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    p.add_argument("--human-data", default=DEFAULT_HUMAN_DATA)
    p.add_argument("--rm-data", default=DEFAULT_RM_DATA)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--skip-interpretation", action="store_true")
    p.add_argument("--allow-embedding-compute", action="store_true")
    p.add_argument("--retrain-sae", action="store_true")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--examples-per-axis", type=int, default=5)
    p.add_argument("--cache-coverage-check", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def resolve(path: str | Path, *, output: bool = False) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    if path.parts[:1] == ("wimhf",):
        return (REPO_ROOT / path).resolve()
    candidate = (Path.cwd() / path).resolve()
    if output or candidate.exists():
        return candidate
    return (WIMHF_DIR / path).resolve()


def config_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else (WIMHF_DIR / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                row = json.loads(line)
                row["_line_no"] = line_no
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")


def sha(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def maybe_int(value: Any) -> Any:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return value


def source_id(row: dict[str, Any]) -> Any:
    return maybe_int(row.get("source_index", row.get("rm_source_index")))


def match_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    needed = ("prompt", "response_A", "response_B")
    if any(row.get(k) is None for k in needed):
        return None
    return (
        maybe_int(row.get("arena_id")),
        row.get("conversation_id"),
        sha(row["prompt"]),
        sha(row["response_A"]),
        sha(row["response_B"]),
    )


def rm_rewards(row: dict[str, Any]) -> tuple[float, float]:
    if row.get("rm_reward_A") is not None and row.get("rm_reward_B") is not None:
        return float(row["rm_reward_A"]), float(row["rm_reward_B"])
    rewards = {
        str(row["chosen_position"]): float(row["chosen_reward"]),
        str(row["rejected_position"]): float(row["rejected_reward"]),
    }
    if set(rewards) != {"A", "B"}:
        raise ValueError(f"RM rewards do not cover A/B for source_index={source_id(row)!r}")
    return rewards["A"], rewards["B"]


def index_human(rows: list[dict[str, Any]]) -> tuple[dict[Any, list[dict[str, Any]]], dict[Any, list[dict[str, Any]]]]:
    by_source: dict[Any, list[dict[str, Any]]] = {}
    by_hash: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        if source_id(row) is not None:
            by_source.setdefault(source_id(row), []).append(row)
        if match_key(row) is not None:
            by_hash.setdefault(match_key(row), []).append(row)
    return by_source, by_hash


def find_match(
    rm_row: dict[str, Any],
    by_source: dict[Any, list[dict[str, Any]]],
    by_hash: dict[Any, list[dict[str, Any]]],
    report: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    source_matches = by_source.get(source_id(rm_row), [])
    rm_key = match_key(rm_row)

    if len(source_matches) == 1:
        human = source_matches[0]
        if rm_key is None or match_key(human) == rm_key:
            report["matched_by_source_index"] += 1
            return human, "source_index"
        report["source_index_content_mismatch"] += 1
    elif len(source_matches) > 1:
        exact = [row for row in source_matches if rm_key is not None and match_key(row) == rm_key]
        if len(exact) == 1:
            report["matched_by_source_index_and_hash"] += 1
            return exact[0], "source_index+content_hash"
        report["ambiguous_source_index"] += 1

    hash_matches = by_hash.get(rm_key, []) if rm_key is not None else []
    if len(hash_matches) == 1:
        report["matched_by_content_hash"] += 1
        return hash_matches[0], "content_hash"
    if len(hash_matches) > 1:
        report["ambiguous_content_hash"] += 1
    return None, None


def orient(human: dict[str, Any], rm_row: dict[str, Any], method: str, row_id: int) -> dict[str, Any]:
    label = int(human["label"])
    reward_a, reward_b = rm_rewards(rm_row)
    prompt = human["prompt"]
    orig_a = human["response_A"]
    orig_b = human["response_B"]

    if label == 1:
        preferred, rejected = orig_a, orig_b
        preferred_pos, rejected_pos = "A", "B"
        preferred_reward, rejected_reward = reward_a, reward_b
    elif label == 0:
        preferred, rejected = orig_b, orig_a
        preferred_pos, rejected_pos = "B", "A"
        preferred_reward, rejected_reward = reward_b, reward_a
    else:
        raise ValueError(f"non-binary human label {label!r}")

    delta = float(preferred_reward - rejected_reward)
    return {
        "disagreement_row_id": row_id,
        "source_index": source_id(human),
        "rm_source_index": source_id(rm_row),
        "arena_id": maybe_int(human.get("arena_id", rm_row.get("arena_id"))),
        "conversation_id": human.get("conversation_id", rm_row.get("conversation_id")),
        "model_A": human.get("model_A", rm_row.get("model_A")),
        "model_B": human.get("model_B", rm_row.get("model_B")),
        "source_dataset": human.get("source_dataset", rm_row.get("source_dataset")),
        "source_split": human.get("source_split", rm_row.get("source_split")),
        "prompt": prompt,
        "response_A": preferred,
        "response_B": rejected,
        "human_preferred_response": preferred,
        "human_rejected_response": rejected,
        "original_response_A": orig_a,
        "original_response_B": orig_b,
        "human_label": label,
        "human_preferred_position": preferred_pos,
        "human_rejected_position": rejected_pos,
        "rm_reward_A": reward_a,
        "rm_reward_B": reward_b,
        "rm_reward_human_preferred": preferred_reward,
        "rm_reward_human_rejected": rejected_reward,
        "delta_r_rm": delta,
        "agree": delta > 0,
        "disagree": delta < 0,
        "wimhf_disagree_label": int(delta < 0),
        "disagreement_strength": max(0.0, -delta),
        "length_delta": len(preferred.split()) - len(rejected.split()),
        "match_method": method,
        "base_line_no": human.get("_line_no"),
        "rm_line_no": rm_row.get("_line_no"),
        "rm_prediction_file": rm_row.get("rm_prediction_file"),
        "rm_model_name": rm_row.get("rm_model_name"),
        "rm_eval_index": rm_row.get("rm_eval_index", rm_row.get("eval_index")),
        "rm_data_index": rm_row.get("rm_data_index", rm_row.get("data_index")),
        "rm_predicted_position": rm_row.get("rm_predicted_position", rm_row.get("predicted_position")),
        "rm_chosen_position": rm_row.get("rm_chosen_position", rm_row.get("chosen_position")),
        "rm_rejected_position": rm_row.get("rm_rejected_position", rm_row.get("rejected_position")),
        "prompt_hash": sha(prompt),
        "response_A_hash": sha(preferred),
        "response_B_hash": sha(rejected),
        "original_response_A_hash": sha(orig_a),
        "original_response_B_hash": sha(orig_b),
        "human_preferred_response_hash": sha(preferred),
        "human_rejected_response_hash": sha(rejected),
        "content_hash": sha(f"{prompt}\n<|A|>\n{preferred}\n<|B|>\n{rejected}"),
        "original_content_hash": sha(f"{prompt}\n<|A|>\n{orig_a}\n<|B|>\n{orig_b}"),
    }


def build_rows(human_rows: list[dict[str, Any]], rm_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_source, by_hash = index_human(human_rows)
    report = {
        "human_rows": len(human_rows),
        "rm_rows": len(rm_rows),
        "rm_schema_source": "converted_wimhf_rm" if rm_rows and "rm_reward_A" in rm_rows[0] else "raw_rm_predictions",
        "matched_by_source_index": 0,
        "matched_by_source_index_and_hash": 0,
        "matched_by_content_hash": 0,
        "source_index_content_mismatch": 0,
        "ambiguous_source_index": 0,
        "ambiguous_content_hash": 0,
        "unmatched_rm_rows": 0,
        "skipped_non_binary_human_label": 0,
        "reward_parse_errors": 0,
        "duplicate_human_source_indices": {str(k): len(v) for k, v in by_source.items() if len(v) > 1},
        "duplicate_human_content_key_count": sum(len(v) > 1 for v in by_hash.values()),
        "unmatched_examples": [],
    }

    out: list[dict[str, Any]] = []
    for rm_row in rm_rows:
        human, method = find_match(rm_row, by_source, by_hash, report)
        if human is None or method is None:
            report["unmatched_rm_rows"] += 1
            if len(report["unmatched_examples"]) < 20:
                report["unmatched_examples"].append(
                    {
                        "source_index": source_id(rm_row),
                        "arena_id": rm_row.get("arena_id"),
                        "conversation_id": rm_row.get("conversation_id"),
                        "rm_line_no": rm_row.get("_line_no"),
                    }
                )
            continue
        try:
            out.append(orient(human, rm_row, method, len(out)))
        except ValueError as exc:
            if "non-binary" in str(exc):
                report["skipped_non_binary_human_label"] += 1
            else:
                report["reward_parse_errors"] += 1
    report["matched_rows_before_max_rows"] = len(out)
    return out, report


def limit_rows(rows: list[dict[str, Any]], max_rows: int | None, seed: int) -> list[dict[str, Any]]:
    if not max_rows or len(rows) <= max_rows:
        return rows
    rng = random.Random(seed)
    agree = [r for r in rows if not r["disagree"]]
    disagree = [r for r in rows if r["disagree"]]
    n_disagree = max(1, round(max_rows * len(disagree) / len(rows)))
    n_disagree = min(n_disagree, len(disagree), max_rows - 1)
    selected = rng.sample(disagree, n_disagree) + rng.sample(agree, max_rows - n_disagree)
    selected.sort(key=lambda r: (r["source_index"] is None, r["source_index"], r["rm_line_no"]))
    for i, row in enumerate(selected):
        row["disagreement_row_id"] = i
    return selected


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    values = sorted(values)
    return {f"q{q:g}": values[round(q * (len(values) - 1))] for q in (0, .01, .05, .1, .5, .9, .95, .99, 1)}


def row_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    agree = sum(r["agree"] for r in rows)
    disagree = sum(r["disagree"] for r in rows)
    zero = n - agree - disagree
    return {
        "n_rows": n,
        "agree": agree,
        "disagree": disagree,
        "zero_margin": zero,
        "agreement_rate": agree / n if n else 0.0,
        "disagreement_rate": disagree / n if n else 0.0,
        "zero_margin_rate": zero / n if n else 0.0,
        "delta_r_rm_quantiles": quantiles([float(r["delta_r_rm"]) for r in rows]),
        "disagreement_strength_quantiles": quantiles([float(r["disagreement_strength"]) for r in rows]),
    }


def make_config(base: dict[str, Any], examples_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base))
    cfg["dataset"].update(
        {
            "name": "ArenaHumanPreference55K_Qwen3_8B_RM_Disagreement",
            "path": str(examples_path),
            "prompt_column": "prompt",
            "response_a_column": "response_A",
            "response_b_column": "response_B",
            "label_column": "wimhf_disagree_label",
        }
    )
    cfg.setdefault("selection", {})["classification"] = True
    cfg["runtime"].update(
        {
            "checkpoint_dir": str(output_dir / "checkpoints"),
            "cache_dir": str(output_dir / "emb_cache"),
            "retrain_sae": args.retrain_sae,
            "debug": args.debug,
        }
    )
    if args.debug:
        cfg["sae"]["n_epochs"] = min(cfg["sae"].get("n_epochs", 50), 3)
        cfg["sae"]["min_epochs"] = min(cfg["sae"].get("min_epochs", 10), 1)
        cfg["sae"]["patience"] = min(cfg["sae"].get("patience", 3), 1)
        interp = cfg.setdefault("interpretation", {})
        interp["interpret_n_examples"] = min(interp.get("interpret_n_examples", 5), 2)
        interp["scoring_n_examples"] = min(interp.get("scoring_n_examples", 300), 32)
    return cfg


def seed_cache(base_cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    dest = output_dir / "emb_cache"
    dest.mkdir(parents=True, exist_ok=True)
    source = base_cfg.get("runtime", {}).get("cache_dir")
    report = {"source_cache_dir": None, "dest_cache_dir": str(dest), "symlinked_chunks": 0, "skipped_existing_chunks": 0}
    if not source:
        return report
    source_dir = config_path(source)
    report["source_cache_dir"] = str(source_dir)
    if not source_dir.exists():
        return report
    for chunk in sorted(source_dir.glob("chunk_*.npy")):
        link = dest / chunk.name
        if link.exists() or link.is_symlink():
            report["skipped_existing_chunks"] += 1
            continue
        os.symlink(chunk, link)
        report["symlinked_chunks"] += 1
    return report


def embedding_text(row: dict[str, Any], emb_cfg: dict[str, Any], col: str) -> str:
    response = row[col]
    mode = emb_cfg.get("text_mode", "response")
    if mode == "response":
        return response
    if mode == "prompt_response":
        return f"{row['prompt']}\n\nAssistant: {response}"
    instruction = emb_cfg.get(
        "instruction",
        "Represent this user-assistant exchange for predicting which assistant response humans would prefer, "
        "focusing on helpfulness, correctness, harmlessness, relevance, and style.",
    )
    return f"{instruction}\n\nConversation:\n{row['prompt']}\n\nAssistant: {response}"


def cache_coverage(rows: list[dict[str, Any]], cfg: dict[str, Any], cache_dir: Path) -> dict[str, Any]:
    import numpy as np

    required = {embedding_text(row, cfg["embedding"], col) for row in rows for col in ("response_A", "response_B")}
    cached: set[str] = set()
    chunks = sorted(cache_dir.glob("chunk_*.npy"))
    for chunk in chunks:
        for text, _ in np.load(chunk, allow_pickle=True):
            cached.add(str(text))
    missing = list(required - cached)
    return {
        "cache_dir": str(cache_dir),
        "chunk_files": len(chunks),
        "required_texts": len(required),
        "cached_required_texts": len(required) - len(missing),
        "missing_required_texts": len(missing),
        "first_missing_text_prefixes": [m[:200].replace("\n", " ") for m in missing[:5]],
    }


def run_native(config_file: Path, skip_interpretation: bool, debug: bool) -> dict[str, Any]:
    if not skip_interpretation:
        from wimhf.quickstart import load_config, run_wimhf_pipeline

        result = run_wimhf_pipeline(load_config(str(config_file)))
        return {
            "stage": "full_wimhf",
            "feature_table": result["feature_table"],
            "activ_train": result["activations_train"],
            "activ_val": result["activations_val"],
            "train_df": result["train_prediction_df"].reset_index(drop=True),
            "val_df": result["val_prediction_df"].reset_index(drop=True),
            "wimhf_summary_metrics": result["summary_metrics"],
            "wimhf_auc_metrics": result["auc_metrics"],
        }

    import numpy as np
    import pandas as pd
    from wimhf.quickstart import get_embeddings, load_and_preprocess_dataframe, load_config, train_sae

    cfg = load_config(str(config_file))
    train, val, train_pred, val_pred, dedup_train, dedup_val = load_and_preprocess_dataframe(cfg.dataset, cfg.runtime)
    _, delta_train, delta_val, dedup_delta_train, dedup_delta_val = get_embeddings(
        train, val, dedup_train, dedup_val, cfg.dataset, cfg.embedding, cfg.runtime.cache_dir
    )
    sae = train_sae(dedup_delta_train, dedup_delta_val, cfg.sae, cfg.runtime)
    activ_train = sae.get_activations(delta_train, show_progress=not debug)
    activ_val = sae.get_activations(delta_val, show_progress=not debug)
    feature_table = pd.DataFrame(
        {
            "neuron_idx": list(range(activ_train.shape[1])),
            "interpretation": [None] * activ_train.shape[1],
            "abbreviated_interpretation": [None] * activ_train.shape[1],
            "kept": [True] * activ_train.shape[1],
            "prevalence": [float((activ_train[:, i] != 0).mean()) for i in range(activ_train.shape[1])],
            "correlation": [np.nan] * activ_train.shape[1],
            "p_value": [np.nan] * activ_train.shape[1],
        }
    )
    return {
        "stage": "skip_interpretation",
        "feature_table": feature_table,
        "activ_train": activ_train,
        "activ_val": activ_val,
        "train_df": train_pred.reset_index(drop=True),
        "val_df": val_pred.reset_index(drop=True),
        "wimhf_summary_metrics": {},
        "wimhf_auc_metrics": {},
    }


def preview(text: Any, n: int = 240) -> str:
    text = str(text or "").replace("\n", " ")
    return text if len(text) <= n else text[: n - 3] + "..."


def top_examples(axis: int, activ_val: Any, val_df: Any, n: int) -> list[dict[str, Any]]:
    import numpy as np

    values = activ_val[:, axis]
    candidates = np.where(values > 0)[0]
    order = candidates[np.argsort(-values[candidates])[:n]] if len(candidates) else np.argsort(-values)[:n]
    examples = []
    for i in order:
        row = val_df.iloc[int(i)]
        examples.append(
            {
                "disagreement_row_id": int(row["disagreement_row_id"]),
                "source_index": maybe_int(row["source_index"]),
                "activation": float(values[int(i)]),
                "delta_r_rm": float(row["delta_r_rm"]),
                "disagreement_strength": float(row["disagreement_strength"]),
                "disagree": bool(row["disagree"]),
                "prompt_preview": preview(row["prompt"]),
                "human_preferred_response_preview": preview(row["human_preferred_response"]),
                "human_rejected_response_preview": preview(row["human_rejected_response"]),
            }
        )
    return examples


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def axis_tables(native: dict[str, Any], controls: list[str], output_dir: Path, top_k: int, examples_per_axis: int) -> dict[str, Any]:
    import numpy as np
    from wimhf.frozen_basis import coefficient_vector, contribution_mass, fit_coefficients, fit_result_to_dict
    from wimhf.utils import fit_linear_model

    axes = [int(x) for x in native["feature_table"]["neuron_idx"]]
    train_df, val_df = native["train_df"], native["val_df"]
    activ_train, activ_val = native["activ_train"], native["activ_val"]
    y_train = train_df["wimhf_disagree_label"].to_numpy(dtype=int)
    y_val = val_df["wimhf_disagree_label"].to_numpy(dtype=int)
    s_train = train_df["disagreement_strength"].to_numpy(dtype=float)
    s_val = val_df["disagreement_strength"].to_numpy(dtype=float)

    disagree_fit = fit_coefficients(activ_train, y_train, axes, train_df, controls, target_type="binary_logistic")
    strength_fit = fit_coefficients(activ_train, s_train, axes, train_df, controls, target_type="continuous_regression")
    beta = coefficient_vector(disagree_fit, axes)
    strength_beta = coefficient_vector(strength_fit, axes)
    mass = contribution_mass(activ_val, beta, axes)
    strength_mass = contribution_mass(activ_val, strength_beta, axes)

    control_train = train_df[controls].to_numpy(dtype=float) if controls else np.empty((len(train_df), 0))
    control_val = val_df[controls].to_numpy(dtype=float) if controls else np.empty((len(val_df), 0))
    x_train = np.concatenate([activ_train[:, axes], control_train], axis=1)
    x_val = np.concatenate([activ_val[:, axes], control_val], axis=1)
    train_auc, val_auc = fit_linear_model(x_train, y_train, x_val, y_val, is_binary=True, standardize=True)
    train_r, val_r = fit_linear_model(x_train, s_train, x_val, s_val, is_binary=False, standardize=True)

    rows = []
    for _, feature in native["feature_table"].sort_values("neuron_idx").iterrows():
        axis = int(feature["neuron_idx"])
        coef = float(disagree_fit.coefficients.get(axis, 0.0))
        mean_abs = float(np.mean(np.abs(activ_val[:, axis])))
        rows.append(
            {
                "neuron_idx": axis,
                "interpretation": feature.get("interpretation"),
                "abbreviated_interpretation": feature.get("abbreviated_interpretation"),
                "kept": bool(feature.get("kept", True)),
                "prevalence": float(feature.get("prevalence", (activ_val[:, axis] != 0).mean())),
                "correlation": feature.get("correlation"),
                "p_value": feature.get("p_value"),
                "disagree_logit_coef": coef,
                "disagree_logit_standardized_coef": float(disagree_fit.standardized_coefficients.get(axis, 0.0)),
                "disagreement_strength_coef": float(strength_fit.coefficients.get(axis, 0.0)),
                "disagreement_strength_standardized_coef": float(strength_fit.standardized_coefficients.get(axis, 0.0)),
                "mean_abs_activation_val": mean_abs,
                "raw_contribution_score": mean_abs * abs(coef),
                "disagreement_contribution_mass": float(mass.mass_by_neuron.get(axis, 0.0)),
                "disagreement_strength_contribution_mass": float(strength_mass.mass_by_neuron.get(axis, 0.0)),
                "top_validation_examples": [],
            }
        )

    top_dis = sorted([r for r in rows if r["disagree_logit_coef"] > 0], key=lambda r: r["raw_contribution_score"], reverse=True)[:top_k]
    top_agr = sorted([r for r in rows if r["disagree_logit_coef"] < 0], key=lambda r: r["raw_contribution_score"], reverse=True)[:top_k]
    for row in rows:
        if row["neuron_idx"] in {r["neuron_idx"] for r in top_dis}:
            row["top_validation_examples"] = top_examples(row["neuron_idx"], activ_val, val_df, examples_per_axis)

    def compact(top_rows: list[dict[str, Any]], direction: str) -> list[dict[str, Any]]:
        keep = [
            "neuron_idx",
            "disagree_logit_coef",
            "disagreement_strength_coef",
            "mean_abs_activation_val",
            "raw_contribution_score",
            "disagreement_contribution_mass",
            "prevalence",
            "correlation",
            "p_value",
            "interpretation",
            "abbreviated_interpretation",
        ]
        return [{"rank": i + 1, "direction": direction, **{k: r[k] for k in keep}} for i, r in enumerate(top_rows)]

    top_dis_csv = compact(top_dis, "disagreement")
    top_agr_csv = compact(top_agr, "agreement")
    write_jsonl(output_dir / "disagreement_feature_table.jsonl", rows)
    write_csv(output_dir / "top_disagreement_axes.csv", top_dis_csv)
    write_csv(output_dir / "top_agreement_axes.csv", top_agr_csv)
    return {
        "axis_fit_metrics": {
            "full_sae_disagree_train_auc": float(train_auc),
            "full_sae_disagree_val_auc": float(val_auc),
            "full_sae_strength_train_pearson": float(train_r),
            "full_sae_strength_val_pearson": float(val_r),
        },
        "binary_disagreement_fit": fit_result_to_dict(disagree_fit),
        "disagreement_strength_fit": fit_result_to_dict(strength_fit),
        "n_feature_rows": len(rows),
        "top_disagreement_axes": top_dis_csv,
        "top_agreement_axes": top_agr_csv,
    }


def readme(output_dir: Path, command: str, summary: dict[str, Any], report: dict[str, Any], stage: str) -> None:
    (output_dir / "README.md").write_text(
        f"""# Experiment 4: Native Disagreement WIMHF

Command:

```bash
{command}
```

`response_A` is the human-preferred response and `response_B` is the
human-rejected response. `delta_r_rm = rm_reward_human_preferred -
rm_reward_human_rejected`; negative values are RM disagreement.

Rows: {summary['n_rows']}
Agree: {summary['agree']}
Disagree: {summary['disagree']}
Zero-margin: {summary['zero_margin']}

Matched before row cap: {report['matched_rows_before_max_rows']}
Matched by source_index: {report['matched_by_source_index']}
Matched by content hash fallback: {report['matched_by_content_hash']}
Unmatched RM rows: {report['unmatched_rm_rows']}

Stage: `{stage}`
"""
    )


def main() -> None:
    args = parse_args()
    base_config_path = resolve(args.base_config)
    human_path = resolve(args.human_data)
    rm_path = resolve(args.rm_data)
    output_dir = resolve(args.output_dir, output=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = read_json(base_config_path)
    rows, report = build_rows(read_jsonl(human_path), read_jsonl(rm_path))
    before_limit = len(rows)
    rows = limit_rows(rows, args.max_rows, int(base_cfg["dataset"].get("random_seed", 42)))

    examples_path = output_dir / "disagreement_examples.jsonl"
    config_file = output_dir / "disagreement_wimhf_config.json"
    write_jsonl(examples_path, rows)

    cfg = make_config(base_cfg, examples_path, output_dir, args)
    write_json(config_file, cfg)
    cache_report = seed_cache(base_cfg, output_dir)
    coverage = cache_coverage(rows, cfg, output_dir / "emb_cache") if args.cache_coverage_check else None

    summary = {
        "base_config": str(base_config_path),
        "human_data": str(human_path),
        "rm_data": str(rm_path),
        "output_dir": str(output_dir),
        "examples_path": str(examples_path),
        "wimhf_config": str(config_file),
        "run_settings": vars(args),
        "cache_seed_report": cache_report,
        "cache_coverage": coverage,
        **row_summary(rows),
    }
    report["max_rows"] = args.max_rows
    report["rows_after_max_rows"] = len(rows)
    report["summary_counts"] = row_summary(rows)

    warnings = []
    if before_limit != len(rows):
        warnings.append(f"Applied --max-rows={args.max_rows}: wrote {len(rows)} rows from {before_limit} matched rows.")
    if summary["zero_margin"]:
        warnings.append(
            f"{summary['zero_margin']} zero-margin rows were retained with agree=false, disagree=false, and wimhf_disagree_label=0."
        )
    if coverage and coverage["missing_required_texts"] and not args.allow_embedding_compute:
        warnings.append(f"Embedding cache is missing {coverage['missing_required_texts']} required texts.")

    stage = "prepare_only"
    if not args.prepare_only:
        if coverage and coverage["missing_required_texts"] and not args.allow_embedding_compute:
            raise RuntimeError("Missing embeddings; rerun with --allow-embedding-compute.")
        native = run_native(config_file, args.skip_interpretation, args.debug)
        stage = native["stage"]
        summary.update(
            {
                "wimhf_stage": stage,
                "wimhf_summary_metrics": native["wimhf_summary_metrics"],
                "wimhf_auc_metrics": native["wimhf_auc_metrics"],
                **axis_tables(native, cfg["selection"].get("controls", []), output_dir, args.top_k, args.examples_per_axis),
            }
        )
    else:
        summary["wimhf_stage"] = stage

    write_json(output_dir / "match_report.json", report)
    write_json(output_dir / "disagreement_summary.json", summary)
    (output_dir / "warnings.txt").write_text("\n".join(f"- {w}" for w in warnings) + "\n" if warnings else "No warnings.\n")
    readme(output_dir, " ".join(sys.argv), summary, report, stage)

    print(f"[disagreement-wimhf] stage={stage} rows={summary['n_rows']} disagree={summary['disagree']}")
    print(f"[disagreement-wimhf] wrote {examples_path}")
    print(f"[disagreement-wimhf] wrote {output_dir / 'disagreement_summary.json'}")


if __name__ == "__main__":
    main()
