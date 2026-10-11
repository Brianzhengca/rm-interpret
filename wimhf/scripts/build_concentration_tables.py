#!/usr/bin/env python3
"""Build standardized concentration tables for native WIMHF runs.

This script analyzes completed WIMHF runs in their own native SAE geometry. It
does not align axes across runs, retrain SAEs, or call LLM interpretation.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import io
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
WIMHF_DIR = SCRIPT_PATH.parents[1]
REPO_ROOT = WIMHF_DIR.parent

PREFERENCE_EMBEDDING_INSTRUCTION = (
    "Represent this user-assistant exchange for predicting which assistant "
    "response humans would prefer, focusing on helpfulness, correctness, "
    "harmlessness, relevance, and style."
)

TRUE_MASS = "true_contribution_mass"
PROXY_MASS = "proxy_prevalence_mass"

SUMMARY_FIELDS = [
    "run_id",
    "dataset",
    "judge",
    "epoch_or_checkpoint",
    "eval_split",
    "wimhf_auc",
    "wimhf_auc_metric",
    "num_retained_neurons",
    "num_total_neurons",
    "top_1_mass",
    "top_3_mass",
    "top_5_mass",
    "top_10_mass",
    "effective_dimension",
    "dominant_axis_interpretation",
    "dominant_semantic_cluster",
    "mass_type",
    "notes",
]

AXIS_FIELDS = [
    "run_id",
    "dataset",
    "judge",
    "epoch_or_checkpoint",
    "eval_split",
    "rank",
    "neuron_idx",
    "sign",
    "beta",
    "prevalence",
    "mean_abs_activation",
    "raw_contribution_mass",
    "normalized_contribution_mass",
    "interpretation",
    "abbreviated_interpretation",
    "semantic_cluster",
    "mass_type",
]


class TrueMassUnavailable(RuntimeError):
    """Raised when true activation mass cannot be computed from saved artifacts."""


@dataclass
class RunArtifacts:
    config_path: Path
    output_dir: Path
    config: Dict[str, Any]
    feature_table_path: Path
    summary_metrics_path: Path
    predictive_metrics_path: Path


@dataclass
class ContributionResult:
    mass_type: str
    raw_by_neuron: Dict[int, float]
    mean_abs_activation_by_neuron: Dict[int, Optional[float]]
    warnings: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build standardized contribution-mass concentration tables for "
            "completed native WIMHF runs."
        )
    )
    parser.add_argument(
        "--manifest",
        default="wimhf/configs/concentration_runs.json",
        help="JSON manifest of WIMHF runs to include.",
    )
    parser.add_argument(
        "--output-dir",
        default="wimhf/outputs/concentration_tables",
        help="Directory for concentration table outputs.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="val",
        help="Split to use for activation mass. Rows are filtered to labels {0, 1}.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "auto"],
        default="cpu",
        help="Device for SAE activation recomputation. Default avoids launching GPU work.",
    )
    parser.add_argument(
        "--mass-mode",
        choices=["auto", "true", "proxy"],
        default="auto",
        help=(
            "auto computes true activation mass when artifacts exist and falls back "
            "to proxy mass; true fails on missing artifacts; proxy never loads SAEs."
        ),
    )
    return parser.parse_args()


def resolve_path(value: str | os.PathLike[str], *base_dirs: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()

    candidates: List[Path] = []
    candidates.extend(base / path for base in base_dirs if base is not None)
    candidates.extend(
        [
            Path.cwd() / path,
            REPO_ROOT / path,
            WIMHF_DIR / path,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / path).resolve()


def resolve_config_relative(value: str | os.PathLike[str], config_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()

    candidates = [
        WIMHF_DIR / path,
        REPO_ROOT / path,
        config_path.parent / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (WIMHF_DIR / path).resolve()


def load_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        return {} if default is None else default
    with path.open() as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_manifest(path: Path) -> Dict[str, Any]:
    data = load_json(path)
    if isinstance(data, list):
        return {"runs": data}
    if not isinstance(data, dict) or "runs" not in data:
        raise ValueError(f"Manifest must be a JSON object with a 'runs' key: {path}")
    return data


def is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def clean_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def clean_row(row: Dict[str, Any], fields: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    keys = fields if fields is not None else row.keys()
    return {key: clean_value(row.get(key)) for key in keys}


def load_feature_table(path: Path, warnings: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    feature_rows = read_jsonl(path)
    retained: List[Dict[str, Any]] = []
    for row in feature_rows:
        if not is_truthy(row.get("kept")):
            continue
        beta = to_float(row.get("length_controlled_logit_coef"))
        if beta is None:
            warnings.append(
                f"{path}: retained neuron {row.get('neuron_idx')} has no finite "
                "length_controlled_logit_coef and was excluded."
            )
            continue
        out = dict(row)
        out["neuron_idx"] = int(out["neuron_idx"])
        out["length_controlled_logit_coef"] = beta
        retained.append(out)
    retained.sort(key=lambda item: int(item["neuron_idx"]))
    return feature_rows, retained


def load_semantic_clusters(
    entry: Dict[str, Any],
    manifest_path: Path,
    warnings: List[str],
) -> Dict[int, str]:
    cluster_path_value = entry.get("semantic_cluster_path")
    if not cluster_path_value:
        return {}

    cluster_path = resolve_path(cluster_path_value, manifest_path.parent)
    if not cluster_path.exists():
        warnings.append(f"{entry.get('run_id')}: semantic cluster map not found: {cluster_path}")
        return {}

    mapping: Dict[int, str] = {}
    try:
        if cluster_path.suffix == ".jsonl":
            rows = read_jsonl(cluster_path)
        elif cluster_path.suffix == ".json":
            payload = load_json(cluster_path)
            rows = payload if isinstance(payload, list) else payload.get("rows", [])
            if isinstance(payload, dict) and not rows:
                for key, value in payload.items():
                    if str(key).isdigit():
                        mapping[int(key)] = str(value)
                return mapping
        elif cluster_path.suffix == ".csv":
            with cluster_path.open(newline="") as f:
                rows = list(csv.DictReader(f))
        else:
            warnings.append(f"{entry.get('run_id')}: unsupported semantic cluster map: {cluster_path}")
            return {}
    except Exception as exc:
        warnings.append(f"{entry.get('run_id')}: failed to read semantic cluster map {cluster_path}: {exc}")
        return {}

    for row in rows:
        idx = row.get("neuron_idx")
        cluster = row.get("semantic_cluster")
        if idx is None or cluster is None:
            continue
        idx_float = to_float(idx)
        if idx_float is not None:
            mapping[int(idx_float)] = str(cluster)
    return mapping


def load_run_artifacts(entry: Dict[str, Any], manifest_path: Path) -> RunArtifacts:
    config_path = resolve_path(entry["config_path"], manifest_path.parent)
    output_dir = resolve_path(entry["output_dir"], manifest_path.parent)
    config = load_json(config_path)
    return RunArtifacts(
        config_path=config_path,
        output_dir=output_dir,
        config=config,
        feature_table_path=output_dir / "feature_table.jsonl",
        summary_metrics_path=output_dir / "summary_metrics.json",
        predictive_metrics_path=output_dir / "predictive_metrics.json",
    )


def select_auc_metric(metrics: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    preferred = to_float(metrics.get("lasso_top_10_val_auc"))
    if preferred is not None:
        return preferred, "lasso_top_10_val_auc"

    lasso_candidates: List[Tuple[int, str, float]] = []
    pattern = re.compile(r"^lasso_top_(\d+)_val_auc$")
    for key, value in metrics.items():
        match = pattern.match(key)
        parsed = to_float(value)
        if match and parsed is not None:
            lasso_candidates.append((int(match.group(1)), key, parsed))
    if lasso_candidates:
        _, key, value = max(lasso_candidates, key=lambda item: item[0])
        return value, key

    full = to_float(metrics.get("full_sae_val_auc"))
    if full is not None:
        return full, "full_sae_val_auc"
    return None, None


def connected_component_train_mask(
    rows: Sequence[Dict[str, Any]],
    split_columns: Sequence[str],
    train_frac: float,
    seed: int,
) -> np.ndarray:
    n_rows = len(rows)
    try:
        import networkx as nx  # type: ignore

        graph = nx.Graph()
        graph.add_nodes_from(range(n_rows))
        for column in split_columns:
            value_to_rows: Dict[Any, List[int]] = {}
            for idx, row in enumerate(rows):
                value_to_rows.setdefault(row[column], []).append(idx)
            for member_rows in value_to_rows.values():
                if len(member_rows) <= 1:
                    continue
                if len(member_rows) > 1000:
                    graph.add_edges_from((member_rows[0], member_rows[i]) for i in range(1, len(member_rows)))
                else:
                    graph.add_edges_from(
                        (member_rows[i], member_rows[j])
                        for i in range(len(member_rows))
                        for j in range(i + 1, len(member_rows))
                    )
        components = list(nx.connected_components(graph))
    except ImportError:
        components = connected_components_union_find(rows, split_columns)

    np.random.seed(seed)
    np.random.shuffle(components)

    train_indices: List[int] = []
    total_rows = 0
    target_train = n_rows * train_frac
    for component in components:
        if total_rows < target_train:
            component_list = list(component)
            train_indices.extend(component_list)
            total_rows += len(component_list)

    mask = np.zeros(n_rows, dtype=bool)
    mask[train_indices] = True
    return mask


def connected_components_union_find(
    rows: Sequence[Dict[str, Any]],
    split_columns: Sequence[str],
) -> List[set[int]]:
    parent = list(range(len(rows)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for column in split_columns:
        value_to_rows: Dict[Any, List[int]] = {}
        for idx, row in enumerate(rows):
            value_to_rows.setdefault(row[column], []).append(idx)
        for member_rows in value_to_rows.values():
            first = member_rows[0]
            for idx in member_rows[1:]:
                union(first, idx)

    components: Dict[int, set[int]] = {}
    for idx in range(len(rows)):
        components.setdefault(find(idx), set()).add(idx)
    return list(components.values())


def is_binary_label(value: Any) -> bool:
    return value == 0 or value == 1


def eval_rows_for_split(
    rows: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
    split: str,
) -> List[Dict[str, Any]]:
    dataset_cfg = config.get("dataset", {})
    split_columns = dataset_cfg.get("split_columns") or []
    if not split_columns:
        raise TrueMassUnavailable("config has no dataset.split_columns for split reconstruction")
    train_frac = float(dataset_cfg.get("train_split_size", 0.8))
    seed = int(dataset_cfg.get("split_random_seed", 42))
    label_column = dataset_cfg.get("label_column") or "label"

    train_mask = connected_component_train_mask(rows, split_columns, train_frac, seed)
    want_train = split == "train"
    selected = [
        row
        for idx, row in enumerate(rows)
        if bool(train_mask[idx]) == want_train and is_binary_label(row.get(label_column))
    ]
    if not selected:
        raise TrueMassUnavailable(f"no binary-label rows found for split={split!r}")
    return selected


def embedding_text(row: Dict[str, Any], config: Dict[str, Any], response_side: str) -> str:
    dataset_cfg = config.get("dataset", {})
    embedding_cfg = config.get("embedding", {})
    response_column = (
        dataset_cfg.get("response_a_column") or "response_A"
        if response_side == "A"
        else dataset_cfg.get("response_b_column") or "response_B"
    )
    prompt_column = dataset_cfg.get("prompt_column") or "prompt"
    response = str(row[response_column])
    text_mode = embedding_cfg.get("text_mode") or "response"

    if text_mode == "response":
        return response

    prompt = str(row[prompt_column])
    if text_mode == "prompt_response":
        return f"{prompt}\n\nAssistant: {response}"
    if text_mode == "instructed_prompt_response":
        instruction = (embedding_cfg.get("instruction") or PREFERENCE_EMBEDDING_INSTRUCTION).strip()
        return f"{instruction}\n\nConversation:\n{prompt}\n\nAssistant: {response}"

    raise TrueMassUnavailable(f"unsupported embedding text_mode={text_mode!r}")


def find_cache_dir(entry: Dict[str, Any], artifacts: RunArtifacts) -> Path:
    if entry.get("embedding_cache_dir"):
        return resolve_path(entry["embedding_cache_dir"], artifacts.output_dir, artifacts.config_path.parent)

    local_cache = artifacts.output_dir / "emb_cache"
    if local_cache.exists():
        return local_cache.resolve()

    runtime_cache = artifacts.config.get("runtime", {}).get("cache_dir")
    if runtime_cache:
        return resolve_config_relative(runtime_cache, artifacts.config_path)

    return local_cache.resolve()


def find_checkpoint_path(entry: Dict[str, Any], artifacts: RunArtifacts) -> Path:
    if entry.get("checkpoint_path"):
        return resolve_path(entry["checkpoint_path"], artifacts.output_dir, artifacts.config_path.parent)

    checkpoint_dirs: List[Path] = []
    local_checkpoint_dir = artifacts.output_dir / "checkpoints"
    if local_checkpoint_dir.exists():
        checkpoint_dirs.append(local_checkpoint_dir.resolve())

    runtime_checkpoint_dir = artifacts.config.get("runtime", {}).get("checkpoint_dir")
    if runtime_checkpoint_dir:
        resolved = resolve_config_relative(runtime_checkpoint_dir, artifacts.config_path)
        if resolved not in checkpoint_dirs:
            checkpoint_dirs.append(resolved)

    for checkpoint_dir in checkpoint_dirs:
        candidates = sorted(checkpoint_dir.glob("*.pt"))
        sae_candidates = [path for path in candidates if "SAE" in path.name]
        if sae_candidates:
            return sae_candidates[0].resolve()
        if candidates:
            return candidates[0].resolve()

    searched = ", ".join(str(path) for path in checkpoint_dirs) or str(local_checkpoint_dir)
    raise TrueMassUnavailable(f"no SAE checkpoint found under {searched}")


def load_embedding_subset(cache_dir: Path, needed_texts: Iterable[str]) -> Dict[str, np.ndarray]:
    needed = set(needed_texts)
    if not cache_dir.exists():
        raise TrueMassUnavailable(f"embedding cache directory not found: {cache_dir}")

    chunks = sorted(cache_dir.glob("chunk_*.npy"))
    if not chunks:
        raise TrueMassUnavailable(f"no chunk_*.npy files found in embedding cache: {cache_dir}")

    found: Dict[str, np.ndarray] = {}
    remaining = set(needed)
    for chunk in chunks:
        data = np.load(chunk, allow_pickle=True)
        for text, embedding in data:
            text_key = str(text)
            if text_key in remaining:
                found[text_key] = np.asarray(embedding, dtype=np.float32)
                remaining.remove(text_key)
        if not remaining:
            break

    if remaining:
        examples = ", ".join(repr(text[:80]) for text in list(remaining)[:3])
        raise TrueMassUnavailable(
            f"embedding cache missing {len(remaining)} of {len(needed)} required texts; examples: {examples}"
        )
    return found


def load_sae_model(checkpoint_path: Path, device: str) -> Any:
    if str(WIMHF_DIR) not in sys.path:
        sys.path.insert(0, str(WIMHF_DIR))

    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch
    from wimhf import sae as sae_module

    if device == "cpu":
        sae_module.DEVICE = torch.device("cpu")
    else:
        sae_module.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with contextlib.redirect_stdout(io.StringIO()):
        return sae_module.load_model(str(checkpoint_path))


def compute_true_contribution(
    entry: Dict[str, Any],
    artifacts: RunArtifacts,
    retained_features: Sequence[Dict[str, Any]],
    split: str,
    device: str,
) -> ContributionResult:
    dataset_path_value = artifacts.config.get("dataset", {}).get("path")
    if not dataset_path_value:
        raise TrueMassUnavailable("config has no dataset.path")
    dataset_path = resolve_config_relative(dataset_path_value, artifacts.config_path)
    if not dataset_path.exists():
        raise TrueMassUnavailable(f"dataset path not found: {dataset_path}")

    all_rows = read_jsonl(dataset_path)
    eval_rows = eval_rows_for_split(all_rows, artifacts.config, split)

    text_pairs = [
        (
            embedding_text(row, artifacts.config, "A"),
            embedding_text(row, artifacts.config, "B"),
        )
        for row in eval_rows
    ]
    needed_texts = {text for pair in text_pairs for text in pair}

    cache_dir = find_cache_dir(entry, artifacts)
    embeddings = load_embedding_subset(cache_dir, needed_texts)

    first_embedding = next(iter(embeddings.values()), None)
    if first_embedding is None:
        raise TrueMassUnavailable("no embeddings loaded from cache")
    input_dim = int(first_embedding.shape[0])
    deltas = np.empty((len(text_pairs), input_dim), dtype=np.float32)
    for idx, (text_a, text_b) in enumerate(text_pairs):
        deltas[idx] = embeddings[text_a] - embeddings[text_b]

    checkpoint_path = find_checkpoint_path(entry, artifacts)
    model = load_sae_model(checkpoint_path, device)
    activations = model.get_activations(deltas, show_progress=False)

    warnings: List[str] = []
    valid_features = [
        row
        for row in retained_features
        if 0 <= int(row["neuron_idx"]) < int(activations.shape[1])
    ]
    invalid_count = len(retained_features) - len(valid_features)
    if invalid_count:
        warnings.append(
            f"{entry.get('run_id')}: excluded {invalid_count} retained neurons outside activation width "
            f"{activations.shape[1]}."
        )
    if not valid_features:
        raise TrueMassUnavailable("no retained neurons overlap activation matrix")

    axes = np.array([int(row["neuron_idx"]) for row in valid_features], dtype=int)
    betas = np.array([float(row["length_controlled_logit_coef"]) for row in valid_features], dtype=float)
    selected_activations = activations[:, axes]
    mean_abs_activation = np.mean(np.abs(selected_activations), axis=0)
    raw_mass = np.mean(np.abs(selected_activations * betas.reshape(1, -1)), axis=0)

    raw_by_neuron = {int(axis): float(value) for axis, value in zip(axes, raw_mass)}
    mean_abs_by_neuron = {
        int(axis): float(value) for axis, value in zip(axes, mean_abs_activation)
    }

    del model, activations, selected_activations, deltas, embeddings
    gc.collect()

    return ContributionResult(
        mass_type=TRUE_MASS,
        raw_by_neuron=raw_by_neuron,
        mean_abs_activation_by_neuron=mean_abs_by_neuron,
        warnings=warnings,
    )


def compute_proxy_contribution(
    entry: Dict[str, Any],
    retained_features: Sequence[Dict[str, Any]],
) -> ContributionResult:
    raw_by_neuron: Dict[int, float] = {}
    warnings: List[str] = []
    for row in retained_features:
        neuron_idx = int(row["neuron_idx"])
        beta = float(row["length_controlled_logit_coef"])
        prevalence = to_float(row.get("prevalence"))
        if prevalence is None:
            prevalence = 0.0
            warnings.append(
                f"{entry.get('run_id')}: retained neuron {neuron_idx} has no finite prevalence; "
                "proxy mass set to zero for this axis."
            )
        raw_by_neuron[neuron_idx] = abs(beta) * prevalence
    return ContributionResult(
        mass_type=PROXY_MASS,
        raw_by_neuron=raw_by_neuron,
        mean_abs_activation_by_neuron={int(row["neuron_idx"]): None for row in retained_features},
        warnings=warnings,
    )


def beta_sign(beta: float) -> str:
    if beta > 0:
        return "positive"
    if beta < 0:
        return "negative"
    return "zero"


def normalized_mass(raw_by_neuron: Dict[int, float]) -> Tuple[Dict[int, float], float]:
    total = float(sum(max(value, 0.0) for value in raw_by_neuron.values()))
    if total <= 0:
        return {idx: 0.0 for idx in raw_by_neuron}, total
    return {idx: max(value, 0.0) / total for idx, value in raw_by_neuron.items()}, total


def effective_dimension(mass_values: Sequence[float]) -> float:
    positive = np.array([value for value in mass_values if value > 0], dtype=float)
    if positive.size == 0:
        return 0.0
    entropy = -float(np.sum(positive * np.log(positive)))
    return float(math.exp(entropy))


def top_mass(mass_values: Sequence[float], k: int) -> float:
    if not mass_values:
        return 0.0
    sorted_values = sorted(mass_values, reverse=True)
    return float(sum(sorted_values[:k]))


def process_run(
    entry: Dict[str, Any],
    manifest_path: Path,
    split: str,
    device: str,
    mass_mode: str,
    global_warnings: List[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    run_id = entry.get("run_id") or entry.get("output_dir")
    artifacts = load_run_artifacts(entry, manifest_path)
    run_warnings: List[str] = []

    if not artifacts.feature_table_path.exists():
        raise FileNotFoundError(f"{run_id}: feature_table.jsonl not found: {artifacts.feature_table_path}")

    feature_rows, retained_features = load_feature_table(artifacts.feature_table_path, run_warnings)
    if not retained_features:
        raise ValueError(f"{run_id}: no retained neurons with finite length_controlled_logit_coef")

    if mass_mode == "proxy":
        contribution = compute_proxy_contribution(entry, retained_features)
        run_warnings.append(f"{run_id}: proxy mass requested by --mass-mode=proxy.")
    else:
        try:
            contribution = compute_true_contribution(
                entry=entry,
                artifacts=artifacts,
                retained_features=retained_features,
                split=split,
                device=device,
            )
        except Exception as exc:
            if mass_mode == "true":
                raise TrueMassUnavailable(f"{run_id}: true contribution mass failed: {exc}") from exc
            run_warnings.append(
                f"{run_id}: true contribution mass unavailable ({exc}); falling back to proxy mass."
            )
            contribution = compute_proxy_contribution(entry, retained_features)

    run_warnings.extend(contribution.warnings)
    retained_features = [
        row for row in retained_features if int(row["neuron_idx"]) in contribution.raw_by_neuron
    ]
    if not retained_features:
        raise ValueError(f"{run_id}: no retained neurons remained after contribution computation")

    mass_by_neuron, total_raw = normalized_mass(contribution.raw_by_neuron)
    if total_raw <= 0:
        run_warnings.append(f"{run_id}: total contribution mass is zero; normalized masses are all zero.")

    semantic_clusters = load_semantic_clusters(entry, manifest_path, run_warnings)
    ranked_features = sorted(
        retained_features,
        key=lambda row: (
            mass_by_neuron.get(int(row["neuron_idx"]), 0.0),
            abs(float(row["length_controlled_logit_coef"])),
        ),
        reverse=True,
    )

    axis_rows: List[Dict[str, Any]] = []
    for rank, row in enumerate(ranked_features, start=1):
        neuron_idx = int(row["neuron_idx"])
        beta = float(row["length_controlled_logit_coef"])
        axis_rows.append(
            {
                "run_id": run_id,
                "dataset": entry.get("dataset"),
                "judge": entry.get("judge"),
                "epoch_or_checkpoint": entry.get("epoch_or_checkpoint"),
                "eval_split": split,
                "rank": rank,
                "neuron_idx": neuron_idx,
                "sign": beta_sign(beta),
                "beta": beta,
                "prevalence": to_float(row.get("prevalence")),
                "mean_abs_activation": contribution.mean_abs_activation_by_neuron.get(neuron_idx),
                "raw_contribution_mass": contribution.raw_by_neuron.get(neuron_idx, 0.0),
                "normalized_contribution_mass": mass_by_neuron.get(neuron_idx, 0.0),
                "interpretation": row.get("interpretation"),
                "abbreviated_interpretation": row.get("abbreviated_interpretation"),
                "semantic_cluster": semantic_clusters.get(neuron_idx, ""),
                "mass_type": contribution.mass_type,
            }
        )

    top_feature_rows = axis_rows[:10]
    predictive_metrics = load_json(artifacts.predictive_metrics_path)
    summary_metrics = load_json(artifacts.summary_metrics_path)
    wimhf_auc, wimhf_auc_metric = select_auc_metric(predictive_metrics)

    mass_values = [row["normalized_contribution_mass"] for row in axis_rows]
    dominant = axis_rows[0] if axis_rows else {}
    notes = entry.get("notes") or ""
    if run_warnings:
        notes = f"{notes} See warnings.txt." if notes else "See warnings.txt."

    summary_row = {
        "run_id": run_id,
        "dataset": entry.get("dataset"),
        "judge": entry.get("judge"),
        "epoch_or_checkpoint": entry.get("epoch_or_checkpoint"),
        "eval_split": split,
        "wimhf_auc": wimhf_auc,
        "wimhf_auc_metric": wimhf_auc_metric,
        "num_retained_neurons": len(axis_rows),
        "num_total_neurons": int(
            to_float(summary_metrics.get("quality/num_neurons_total")) or len(feature_rows)
        ),
        "top_1_mass": top_mass(mass_values, 1),
        "top_3_mass": top_mass(mass_values, 3),
        "top_5_mass": top_mass(mass_values, 5),
        "top_10_mass": top_mass(mass_values, 10),
        "effective_dimension": effective_dimension(mass_values),
        "dominant_axis_interpretation": dominant.get("interpretation"),
        "dominant_semantic_cluster": dominant.get("semantic_cluster", ""),
        "mass_type": contribution.mass_type,
        "notes": notes,
    }

    validate_run_outputs(summary_row, axis_rows, top_feature_rows)

    for warning in run_warnings:
        global_warnings.append(warning)

    manifest_used_entry = dict(entry)
    manifest_used_entry.update(
        {
            "resolved_config_path": str(artifacts.config_path),
            "resolved_output_dir": str(artifacts.output_dir),
            "mass_type": contribution.mass_type,
            "num_retained_neurons": len(axis_rows),
        }
    )
    return summary_row, axis_rows, top_feature_rows, manifest_used_entry


def validate_run_outputs(
    summary_row: Dict[str, Any],
    axis_rows: Sequence[Dict[str, Any]],
    top_feature_rows: Sequence[Dict[str, Any]],
) -> None:
    run_id = summary_row["run_id"]
    if len(top_feature_rows) > 10:
        raise AssertionError(f"{run_id}: more than 10 top-feature rows produced")

    masses = [float(row["normalized_contribution_mass"]) for row in axis_rows]
    if any(value < -1e-10 or value > 1 + 1e-10 for value in masses):
        raise AssertionError(f"{run_id}: normalized mass outside [0, 1]")
    mass_sum = sum(masses)
    if mass_sum > 0 and abs(mass_sum - 1.0) > 1e-6:
        raise AssertionError(f"{run_id}: normalized masses sum to {mass_sum}, not 1.0")

    top_values = [
        float(summary_row["top_1_mass"]),
        float(summary_row["top_3_mass"]),
        float(summary_row["top_5_mass"]),
        float(summary_row["top_10_mass"]),
    ]
    if any(value < -1e-10 or value > 1 + 1e-10 for value in top_values):
        raise AssertionError(f"{run_id}: top-k mass outside [0, 1]")
    if not all(left <= right + 1e-10 for left, right in zip(top_values, top_values[1:])):
        raise AssertionError(f"{run_id}: top-k masses are not monotone")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(clean_row(row, fields))


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(clean_row(row, fields), sort_keys=True) + "\n")


def collect_skipped_warnings(
    manifest_entries: Sequence[Dict[str, Any]],
    manifest_path: Path,
) -> List[str]:
    warnings: List[str] = []
    manifest_output_dirs = {
        resolve_path(entry["output_dir"], manifest_path.parent).resolve()
        for entry in manifest_entries
        if entry.get("output_dir")
    }

    outputs_dir = WIMHF_DIR / "outputs"
    if not outputs_dir.exists():
        return warnings

    for child in sorted(outputs_dir.iterdir()):
        if child.is_dir() and child.name.startswith("frozen_basis"):
            warnings.append(f"Skipped non-native frozen-basis output directory: {child}")

    backup = outputs_dir / "helpsteer3" / "emb_cache" / "thinking_on_backup_20260509_132246"
    if (backup / "feature_table.jsonl").exists():
        warnings.append(f"Skipped backup artifact nested inside embedding cache: {backup}")

    for feature_table in sorted(outputs_dir.glob("**/feature_table.jsonl")):
        parent = feature_table.parent.resolve()
        parts = set(feature_table.parts)
        if parent in manifest_output_dirs:
            continue
        if any(part.startswith("frozen_basis") for part in parts):
            continue
        if "thinking_on_backup_20260509_132246" in parts:
            continue
        if "concentration_tables" in parts:
            continue
        warnings.append(f"Skipped feature_table.jsonl not listed in manifest: {parent}")
    return warnings


def write_readme(
    path: Path,
    summary_rows: Sequence[Dict[str, Any]],
    manifest_path: Path,
    warnings: Sequence[str],
) -> None:
    true_runs = [row["run_id"] for row in summary_rows if row.get("mass_type") == TRUE_MASS]
    proxy_runs = [row["run_id"] for row in summary_rows if row.get("mass_type") == PROXY_MASS]

    lines = [
        "# WIMHF Concentration Tables",
        "",
        "These tables standardize contribution-mass analysis across completed native WIMHF runs.",
        "Each run is analyzed in its own retained SAE axes; no frozen-basis alignment is used.",
        "",
        "## Formula",
        "",
        "For each retained axis `j`:",
        "",
        "`C_j = mean_i | length_controlled_logit_coef_j * activation_ij |`",
        "",
        "`p_j = C_j / sum_k C_k`",
        "",
        "`p_j` is the estimated share of that run's preference signal carried by axis `j`.",
        "",
        "If cached activations cannot be recomputed from saved embeddings and SAE checkpoints,",
        "the script falls back to `abs(beta_j) * prevalence_j` and marks those rows as",
        f"`{PROXY_MASS}`.",
        "",
        "## Mass Types",
        "",
        f"- `{TRUE_MASS}`: {len(true_runs)} runs",
        f"- `{PROXY_MASS}`: {len(proxy_runs)} runs",
        "",
        "True-mass runs:",
        "",
    ]
    lines.extend(f"- `{run_id}`" for run_id in true_runs)
    if proxy_runs:
        lines.extend(["", "Proxy-mass runs:", ""])
        lines.extend(f"- `{run_id}`" for run_id in proxy_runs)
    else:
        lines.extend(["", "No runs used proxy mass in this build."])

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `concentration_summary.csv` / `.jsonl`: one row per run.",
            "- `axis_concentration.csv` / `.jsonl`: one row per retained axis.",
            "- `top_features.csv` / `.jsonl`: top 10 retained axes per run.",
            "- `run_manifest_used.json`: manifest entries resolved during this build.",
            "- `warnings.txt`: skipped runs, fallback reasons, and other non-fatal warnings.",
            "",
            "## Manifest",
            "",
            f"The manifest used here is `{manifest_path}`.",
            "To add or modify runs, edit `wimhf/configs/concentration_runs.json` and set",
            "`run_id`, `dataset`, `judge`, `epoch_or_checkpoint`, `config_path`, `output_dir`,",
            "and optional `notes`. The output directory must contain `feature_table.jsonl`,",
            "`summary_metrics.json`, `predictive_metrics.json`, `emb_cache/chunk_*.npy`, and a",
            "`checkpoints/*.pt` SAE checkpoint for true contribution mass.",
            "",
            "## Semantic Clusters",
            "",
            "Semantic clusters are blank by default. To add them later, attach",
            "`semantic_cluster_path` to a manifest entry. The file may be CSV, JSONL, or JSON",
            "with `neuron_idx` and `semantic_cluster` fields.",
            "",
            "## Warnings",
            "",
            f"`warnings.txt` contains {len(warnings)} warning lines for this build.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    manifest_path = resolve_path(args.manifest)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    runs = manifest["runs"]
    warnings = collect_skipped_warnings(runs, manifest_path)

    summary_rows: List[Dict[str, Any]] = []
    axis_rows: List[Dict[str, Any]] = []
    top_feature_rows: List[Dict[str, Any]] = []
    manifest_used_runs: List[Dict[str, Any]] = []

    for entry in runs:
        run_id = entry.get("run_id") or entry.get("output_dir")
        print(f"[concentration] processing {run_id}")
        summary, axes, top_features, manifest_used = process_run(
            entry=entry,
            manifest_path=manifest_path,
            split=args.split,
            device=args.device,
            mass_mode=args.mass_mode,
            global_warnings=warnings,
        )
        summary_rows.append(summary)
        axis_rows.extend(axes)
        top_feature_rows.extend(top_features)
        manifest_used_runs.append(manifest_used)

    write_csv(output_dir / "concentration_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_jsonl(output_dir / "concentration_summary.jsonl", summary_rows, SUMMARY_FIELDS)
    write_csv(output_dir / "axis_concentration.csv", axis_rows, AXIS_FIELDS)
    write_jsonl(output_dir / "axis_concentration.jsonl", axis_rows, AXIS_FIELDS)
    write_csv(output_dir / "top_features.csv", top_feature_rows, AXIS_FIELDS)
    write_jsonl(output_dir / "top_features.jsonl", top_feature_rows, AXIS_FIELDS)

    manifest_used = {
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "split": args.split,
        "device": args.device,
        "mass_mode": args.mass_mode,
        "runs": manifest_used_runs,
    }
    (output_dir / "run_manifest_used.json").write_text(json.dumps(manifest_used, indent=2, sort_keys=True))
    (output_dir / "warnings.txt").write_text("\n".join(warnings) + ("\n" if warnings else ""))
    write_readme(output_dir / "README.md", summary_rows, manifest_path, warnings)

    true_count = sum(1 for row in summary_rows if row.get("mass_type") == TRUE_MASS)
    proxy_count = sum(1 for row in summary_rows if row.get("mass_type") == PROXY_MASS)
    print(
        f"[concentration] wrote {len(summary_rows)} run summaries, {len(axis_rows)} axis rows, "
        f"and {len(top_feature_rows)} top-feature rows to {output_dir}"
    )
    print(f"[concentration] mass types: {true_count} true, {proxy_count} proxy")
    if warnings:
        print(f"[concentration] warnings written: {len(warnings)}")


if __name__ == "__main__":
    main()
