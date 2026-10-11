"""Frozen-basis WIMHF comparison helpers."""

from __future__ import annotations

import glob
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .utils import create_connected_component_split, get_repo_root


PREFERENCE_EMBEDDING_INSTRUCTION = (
    "Represent this user-assistant exchange for predicting which assistant "
    "response humans would prefer, focusing on helpfulness, correctness, "
    "harmlessness, relevance, and style."
)


@dataclass
class FrozenDatasetConfig:
    name: str
    path: str
    split_columns: List[str]
    train_split_size: float = 0.8
    split_random_seed: int = 42
    prompt_column: str = "prompt"
    response_a_column: str = "response_A"
    response_b_column: str = "response_B"
    label_column: str = "label"
    max_words_prompt: int = 128
    max_words_response: int = 128
    random_seed: int = 42


@dataclass
class FrozenEmbeddingConfig:
    model: str = "text-embedding-3-small"
    n_workers: int = 2
    provider: str = "openai"
    batch_size: int = 128
    text_mode: str = "response"
    instruction: str = PREFERENCE_EMBEDDING_INSTRUCTION


@dataclass
class FrozenSAEConfig:
    M: int = 32
    K: int = 4
    prefix_lengths: List[int] = field(default_factory=lambda: [8, 32])
    batch_size: int = 256
    n_epochs: int = 50
    min_epochs: int = 10
    patience: int = 3
    learning_rate: float = 5e-4
    dead_neuron_threshold_steps: int = 64
    aux_k: Optional[int] = None
    multi_k: Optional[int] = None


@dataclass
class FrozenSelectionConfig:
    controls: List[str] = field(default_factory=lambda: ["length_delta"])
    lasso_top_k: List[int] = field(default_factory=lambda: [5, 10])
    max_samples_lasso: Optional[int] = None
    classification: bool = True
    standardize: bool = True
    use_lasso: bool = True


@dataclass
class FrozenRuntimeConfig:
    checkpoint_dir: Optional[str] = None
    cache_dir: Optional[Path] = None


@dataclass
class FrozenWIMHFConfig:
    dataset: FrozenDatasetConfig
    embedding: FrozenEmbeddingConfig = field(default_factory=FrozenEmbeddingConfig)
    sae: FrozenSAEConfig = field(default_factory=FrozenSAEConfig)
    selection: FrozenSelectionConfig = field(default_factory=FrozenSelectionConfig)
    runtime: FrozenRuntimeConfig = field(default_factory=FrozenRuntimeConfig)


def load_frozen_config(path: str | Path) -> FrozenWIMHFConfig:
    """Load only the config fields needed for frozen-basis comparison."""
    with Path(path).open() as f:
        data = json.load(f)

    runtime_data = data.get("runtime", {})
    cache_dir = runtime_data.get("cache_dir")
    runtime = FrozenRuntimeConfig(
        checkpoint_dir=runtime_data.get("checkpoint_dir"),
        cache_dir=Path(cache_dir) if cache_dir else None,
    )
    return FrozenWIMHFConfig(
        dataset=FrozenDatasetConfig(**data["dataset"]),
        embedding=FrozenEmbeddingConfig(**data.get("embedding", {})),
        sae=FrozenSAEConfig(**data.get("sae", {})),
        selection=FrozenSelectionConfig(**data.get("selection", {})),
        runtime=runtime,
    )


def _load_dataframe(path: str) -> pd.DataFrame:
    if path.endswith(".json") or path.endswith(".jsonl"):
        return pd.read_json(path, orient="records", lines=True)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension for {path}")


def _ensure_length_delta(df: pd.DataFrame, cfg: FrozenDatasetConfig) -> None:
    if "length_delta" not in df.columns:
        a = df[cfg.response_a_column].astype(str).str.split().str.len()
        b = df[cfg.response_b_column].astype(str).str.split().str.len()
        df["length_delta"] = a - b


def _row_embedding_text(
    row: pd.Series,
    cfg: FrozenDatasetConfig,
    emb_cfg: FrozenEmbeddingConfig,
    response_col: str,
) -> str:
    response = str(row[response_col])
    if emb_cfg.text_mode == "response":
        return response

    prompt = str(row[cfg.prompt_column])
    if emb_cfg.text_mode == "prompt_response":
        return f"{prompt}\n\nAssistant: {response}"
    if emb_cfg.text_mode == "instructed_prompt_response":
        instruction = emb_cfg.instruction.strip()
        return f"{instruction}\n\nConversation:\n{prompt}\n\nAssistant: {response}"

    raise ValueError(
        f"Unsupported embedding text_mode {emb_cfg.text_mode!r}. "
        "Use 'response', 'prompt_response', or 'instructed_prompt_response'."
    )


def _build_deltas(
    df: pd.DataFrame,
    cfg: FrozenDatasetConfig,
    emb_cfg: FrozenEmbeddingConfig,
    text2embedding: Dict[str, np.ndarray],
) -> np.ndarray:
    resp_a = np.stack(
        [
            text2embedding[_row_embedding_text(row, cfg, emb_cfg, cfg.response_a_column)]
            for _, row in df.iterrows()
        ]
    )
    resp_b = np.stack(
        [
            text2embedding[_row_embedding_text(row, cfg, emb_cfg, cfg.response_b_column)]
            for _, row in df.iterrows()
        ]
    )
    return resp_a - resp_b


def _embed_unique_responses(
    df: pd.DataFrame,
    config: FrozenWIMHFConfig,
) -> Dict[str, np.ndarray]:
    from . import embedding

    texts = _required_embedding_texts(df, config)
    provider = config.embedding.provider.lower()
    cache_dir = config.runtime.cache_dir
    if provider == "openai":
        return embedding.get_openai_embeddings(
            texts=texts,
            model=config.embedding.model,
            batch_size=config.embedding.batch_size,
            n_workers=config.embedding.n_workers,
            cache_dir=cache_dir,
        )
    if provider in {"local", "sentence-transformers", "sentence_transformers"}:
        return embedding.get_local_embeddings(
            texts=texts,
            model=config.embedding.model,
            batch_size=config.embedding.batch_size,
            cache_dir=cache_dir,
        )
    raise ValueError(
        f"Unsupported embedding provider {config.embedding.provider!r}. "
        "Use 'openai' or 'local'."
    )


@dataclass
class FitResult:
    """Raw-unit linear/logistic coefficients for SAE axes plus controls."""

    target_type: str
    n_rows: int
    axes: List[int]
    controls: List[str]
    intercept: float
    coefficients: Dict[int, float]
    control_coefficients: Dict[str, float]
    standardized_coefficients: Dict[int, float]
    standardized_control_coefficients: Dict[str, float]
    metrics: Dict[str, float]


@dataclass
class ContributionMass:
    """Contribution-mass distribution over frozen axes."""

    contribution_by_neuron: Dict[int, float]
    mass_by_neuron: Dict[int, float]
    top_1_mass: float
    top_3_mass: float
    top_5_mass: float
    effective_dimension: float


def resolve_path(path: str | Path, base_dir: Optional[Path] = None) -> Path:
    """Resolve ``path`` relative to ``base_dir`` when it is not absolute."""
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (base_dir or get_repo_root()).joinpath(path).resolve()


def normalize_config_paths(
    config: FrozenWIMHFConfig,
    base_dir: Optional[Path] = None,
) -> FrozenWIMHFConfig:
    """Make dataset/runtime paths absolute so scripts work from any cwd."""
    base = base_dir or get_repo_root()
    config.dataset.path = str(resolve_path(config.dataset.path, base))
    if config.runtime.cache_dir is not None:
        config.runtime.cache_dir = resolve_path(config.runtime.cache_dir, base)
    if config.runtime.checkpoint_dir is not None:
        config.runtime.checkpoint_dir = str(resolve_path(config.runtime.checkpoint_dir, base))
    return config


def expand_prediction_paths(patterns: Sequence[str | Path], base_dir: Optional[Path] = None) -> List[Path]:
    """Expand shell-style glob patterns and return sorted unique prediction paths."""
    base = base_dir or get_repo_root()
    paths: List[Path] = []
    for pattern in patterns:
        pattern_text = str(pattern)
        resolved_pattern = str(resolve_path(pattern_text, base)) if not Path(pattern_text).is_absolute() else pattern_text
        matches = [Path(match).resolve() for match in glob.glob(resolved_pattern)]
        if matches:
            paths.extend(matches)
        else:
            literal = resolve_path(pattern_text, base)
            if literal.exists():
                paths.append(literal)
    unique = sorted(dict.fromkeys(paths))
    if not unique:
        raise FileNotFoundError(f"No RM prediction files matched: {list(map(str, patterns))}")
    return unique


def load_basis_feature_table(
    basis_output_dir: str | Path,
    axis_scope: str = "kept",
    base_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load the human basis feature table and filter the requested axes."""
    basis_dir = resolve_path(basis_output_dir, base_dir)
    feature_path = basis_dir / "feature_table.jsonl"
    if not feature_path.exists():
        raise FileNotFoundError(f"Frozen basis feature table not found: {feature_path}")

    table = pd.read_json(feature_path, orient="records", lines=True)
    if "neuron_idx" not in table.columns:
        raise ValueError(f"{feature_path} is missing required column 'neuron_idx'")

    if axis_scope == "kept":
        if "kept" not in table.columns:
            raise ValueError(f"{feature_path} does not contain a 'kept' column for --axis-scope kept")
        table = table[table["kept"].astype(bool)].copy()
    elif axis_scope == "all":
        table = table.copy()
    else:
        raise ValueError(f"Unsupported axis_scope={axis_scope!r}; expected 'kept' or 'all'")

    if table.empty:
        raise ValueError(f"No axes selected from {feature_path} with axis_scope={axis_scope!r}")
    return table.sort_values("neuron_idx").reset_index(drop=True)


def load_frozen_sae(
    config: FrozenWIMHFConfig,
    basis_output_dir: str | Path,
    checkpoint_path: Optional[str | Path] = None,
    base_dir: Optional[Path] = None,
):
    """Load the frozen human SAE checkpoint."""
    if checkpoint_path is not None:
        path = resolve_path(checkpoint_path, base_dir)
    else:
        if config.sae.prefix_lengths is None:
            filename = f"SAE_M={config.sae.M}_K={config.sae.K}.pt"
        else:
            prefix_str = "-".join(str(length) for length in config.sae.prefix_lengths)
            filename = f"SAE_matryoshka_M={config.sae.M}_K={config.sae.K}_prefixes={prefix_str}.pt"
        path = resolve_path(basis_output_dir, base_dir) / "checkpoints" / filename
    if not path.exists():
        raise FileNotFoundError(f"Frozen SAE checkpoint not found: {path}")
    from .sae import load_model

    return load_model(str(path))


def load_split_data(config: FrozenWIMHFConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the configured dataset and reproduce the WIMHF train/validation split."""
    df = _load_dataframe(config.dataset.path)
    _ensure_length_delta(df, config.dataset)
    return create_connected_component_split(
        df,
        config.dataset.split_columns,
        train_frac=config.dataset.train_split_size,
        seed=config.dataset.split_random_seed,
    )


def _required_embedding_texts(df: pd.DataFrame, config: FrozenWIMHFConfig) -> List[str]:
    texts: List[str] = []
    for _, row in df.iterrows():
        texts.append(_row_embedding_text(row, config.dataset, config.embedding, config.dataset.response_a_column))
        texts.append(_row_embedding_text(row, config.dataset, config.embedding, config.dataset.response_b_column))
    return list(dict.fromkeys(texts))


def load_or_compute_embeddings_for_rows(
    df: pd.DataFrame,
    config: FrozenWIMHFConfig,
    *,
    allow_compute: bool = False,
) -> Dict[str, np.ndarray]:
    """Load embeddings for rows from cache, optionally computing missing entries."""
    cache_dir = config.runtime.cache_dir
    if cache_dir is None:
        raise ValueError(
            "Embedding cache_dir is not set in the basis config. "
            "Set runtime.cache_dir or pass a config with cached basis embeddings."
        )

    required_texts = _required_embedding_texts(df, config)
    from . import embedding

    cached = embedding.load_embedding_cache(Path(cache_dir))
    missing = [text for text in required_texts if text not in cached]
    if missing and not allow_compute:
        example = missing[0][:200].replace("\n", " ")
        raise RuntimeError(
            f"Missing {len(missing)} / {len(required_texts)} required embeddings in {cache_dir}. "
            "This script is cache-only by default; rerun with --allow-embedding-compute to compute them. "
            f"First missing text starts with: {example!r}"
        )
    if not missing:
        return cached

    return _embed_unique_responses(df, config)


def compute_frozen_activations(
    df: pd.DataFrame,
    config: FrozenWIMHFConfig,
    sae,
    *,
    allow_embedding_compute: bool = False,
    show_progress: bool = True,
) -> np.ndarray:
    """Build embedding deltas for ``df`` and run the frozen SAE."""
    text2embedding = load_or_compute_embeddings_for_rows(
        df,
        config,
        allow_compute=allow_embedding_compute,
    )
    deltas = _build_deltas(df, config.dataset, config.embedding, text2embedding)
    return sae.get_activations(deltas, show_progress=show_progress)


def _prediction_join_value(row: Mapping[str, Any], join_key: str) -> Any:
    value = row.get(join_key)
    if value is None:
        raise ValueError(f"RM prediction row is missing join key {join_key!r}")
    return value


def _position_label(position: Any) -> Optional[int]:
    if position == "A":
        return 1
    if position == "B":
        return 0
    return None


def rm_margin_a_minus_b(prediction: Mapping[str, Any]) -> Optional[float]:
    """Return continuous RM margin in A-minus-B orientation when available."""
    for key in ("rm_margin_A_minus_B", "rm_margin_a_minus_b"):
        if key in prediction and prediction[key] is not None:
            return float(prediction[key])

    if "rm_reward_A" in prediction and "rm_reward_B" in prediction:
        return float(prediction["rm_reward_A"]) - float(prediction["rm_reward_B"])

    if (
        "chosen_reward" in prediction
        and "rejected_reward" in prediction
        and "chosen_position" in prediction
        and "rejected_position" in prediction
    ):
        rewards = {
            str(prediction["chosen_position"]): float(prediction["chosen_reward"]),
            str(prediction["rejected_position"]): float(prediction["rejected_reward"]),
        }
        if set(rewards) == {"A", "B"}:
            return rewards["A"] - rewards["B"]

    if "margin" in prediction and "chosen_position" in prediction and "rejected_position" in prediction:
        margin = float(prediction["margin"])
        chosen = prediction["chosen_position"]
        rejected = prediction["rejected_position"]
        if chosen == "A" and rejected == "B":
            return margin
        if chosen == "B" and rejected == "A":
            return -margin

    return None


def rm_binary_label(prediction: Mapping[str, Any]) -> Optional[int]:
    """Return binary RM-induced A/B preference label when available."""
    for key in ("rm_predicted_position", "predicted_position"):
        label = _position_label(prediction.get(key))
        if label is not None:
            return label
    if "rm_label" in prediction and prediction["rm_label"] in (0, 1):
        return int(prediction["rm_label"])
    return None


def load_rm_predictions(path: str | Path, join_key: str = "source_index") -> pd.DataFrame:
    """Load one RM prediction JSONL file with continuous and binary targets."""
    rows: List[Dict[str, Any]] = []
    path = Path(path)
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            join_value = _prediction_join_value(raw, join_key)
            margin = rm_margin_a_minus_b(raw)
            binary = rm_binary_label(raw)
            if margin is None and binary is None:
                raise ValueError(
                    f"{path}:{line_no} has neither continuous RM reward margin nor binary RM label"
                )
            rows.append(
                {
                    join_key: join_value,
                    "rm_margin_a_minus_b": margin,
                    "rm_binary_label": binary,
                    "rm_prediction_file": str(path),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No prediction rows found in {path}")
    duplicated = df[join_key].duplicated()
    if duplicated.any():
        examples = df.loc[duplicated, join_key].head(5).tolist()
        raise ValueError(f"{path} contains duplicate {join_key} values; first duplicates: {examples}")
    return df


def unique_checkpoint_names(paths: Sequence[Path]) -> Dict[Path, str]:
    """Derive stable checkpoint names from prediction file paths."""
    raw_names = [path.parent.name or path.stem for path in paths]
    counts: Dict[str, int] = {}
    names: Dict[Path, str] = {}
    for path, raw in zip(paths, raw_names):
        counts[raw] = counts.get(raw, 0) + 1
        if counts[raw] == 1 and raw_names.count(raw) == 1:
            names[path] = raw
        else:
            names[path] = f"{path.parent.parent.name}_{raw}"
    return names


def sanitize_name(name: str) -> str:
    """Make a filesystem-safe short name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "checkpoint"


def common_prediction_keys(
    val_df: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame],
    join_key: str,
) -> List[Any]:
    """Return validation join keys present in every prediction dataframe."""
    if join_key not in val_df.columns:
        raise ValueError(f"Validation dataframe is missing join key {join_key!r}")
    common = set(val_df[join_key].tolist())
    for pred_df in predictions.values():
        common &= set(pred_df[join_key].tolist())
    ordered = [value for value in val_df[join_key].tolist() if value in common]
    return ordered


def subset_val_rows(
    val_df: pd.DataFrame,
    keys: Sequence[Any],
    join_key: str,
    *,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Filter validation rows to ordered join keys, with an optional prefix limit."""
    key_set = set(keys)
    out = val_df[val_df[join_key].isin(key_set)].copy()
    order = {key: idx for idx, key in enumerate(keys)}
    out["__frozen_order__"] = out[join_key].map(order)
    out = out.sort_values("__frozen_order__").drop(columns="__frozen_order__").reset_index(drop=True)
    if max_rows is not None:
        out = out.head(max_rows).reset_index(drop=True)
    return out


def aligned_prediction_targets(
    pred_df: pd.DataFrame,
    rows_df: pd.DataFrame,
    join_key: str,
) -> Tuple[np.ndarray, str]:
    """Align one prediction dataframe to frozen validation rows."""
    indexed = pred_df.set_index(join_key)
    selected = indexed.loc[rows_df[join_key].tolist()]
    if selected["rm_margin_a_minus_b"].notna().all():
        return selected["rm_margin_a_minus_b"].to_numpy(dtype=float), "continuous_margin"
    if selected["rm_binary_label"].notna().all():
        return selected["rm_binary_label"].to_numpy(dtype=int), "binary_label"
    raise ValueError(
        "RM predictions mix missing continuous margins and missing binary labels after alignment; "
        "provide a prediction file with one complete RM target."
    )


def _raw_coefficients(
    coef_scaled: np.ndarray,
    intercept_scaled: float,
    scaler: StandardScaler,
) -> Tuple[np.ndarray, float]:
    scale = scaler.scale_.copy()
    scale[scale == 0] = 1.0
    coef_raw = coef_scaled / scale
    intercept_raw = float(intercept_scaled - np.sum(coef_scaled * scaler.mean_ / scale))
    return coef_raw, intercept_raw


def _classification_metrics(model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    probs = model.predict_proba(X)[:, 1]
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y, (probs >= 0.5).astype(int))),
    }
    if len(np.unique(y)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y, probs))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def _regression_metrics(model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    preds = model.predict(X)
    metrics = {
        "r2": float(r2_score(y, preds)),
        "rmse": float(np.sqrt(np.mean((y - preds) ** 2))),
    }
    if len(y) > 1 and np.std(y) > 0 and np.std(preds) > 0:
        metrics["pearson_r"] = float(pearsonr(y, preds)[0])
    else:
        metrics["pearson_r"] = float("nan")
    return metrics


def fit_coefficients(
    activations: np.ndarray,
    target: np.ndarray,
    axes: Sequence[int],
    rows_df: pd.DataFrame,
    controls: Sequence[str],
    *,
    target_type: str,
    logistic_c: float = 1.0,
    ridge_alpha: float = 0.0,
) -> FitResult:
    """Fit standardized model and return raw-unit coefficients."""
    axis_positions = [int(axis) for axis in axes]
    control_cols = list(controls)
    if control_cols:
        missing_controls = [col for col in control_cols if col not in rows_df.columns]
        if missing_controls:
            raise ValueError(f"Rows dataframe is missing controls: {missing_controls}")
        control_matrix = rows_df[control_cols].to_numpy(dtype=float)
        X = np.concatenate([activations[:, axis_positions], control_matrix], axis=1)
    else:
        X = activations[:, axis_positions]

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    if target_type == "binary_logistic":
        model = LogisticRegression(C=logistic_c, max_iter=2000)
        model.fit(X_std, target.astype(int))
        coef_scaled = model.coef_.reshape(-1)
        intercept_scaled = float(model.intercept_[0])
        metrics = _classification_metrics(model, X_std, target.astype(int))
    elif target_type == "continuous_regression":
        if ridge_alpha and ridge_alpha > 0:
            model = Ridge(alpha=ridge_alpha)
        else:
            model = LinearRegression()
        model.fit(X_std, target.astype(float))
        coef_scaled = model.coef_.reshape(-1)
        intercept_scaled = float(model.intercept_)
        metrics = _regression_metrics(model, X_std, target.astype(float))
    else:
        raise ValueError(f"Unsupported target_type={target_type!r}")

    coef_raw, intercept_raw = _raw_coefficients(coef_scaled, intercept_scaled, scaler)
    axis_coef_scaled = coef_scaled[: len(axis_positions)]
    axis_coef_raw = coef_raw[: len(axis_positions)]
    control_coef_scaled = coef_scaled[len(axis_positions) :]
    control_coef_raw = coef_raw[len(axis_positions) :]

    return FitResult(
        target_type=target_type,
        n_rows=int(len(target)),
        axes=axis_positions,
        controls=control_cols,
        intercept=intercept_raw,
        coefficients={axis: float(coef) for axis, coef in zip(axis_positions, axis_coef_raw)},
        control_coefficients={name: float(coef) for name, coef in zip(control_cols, control_coef_raw)},
        standardized_coefficients={axis: float(coef) for axis, coef in zip(axis_positions, axis_coef_scaled)},
        standardized_control_coefficients={
            name: float(coef) for name, coef in zip(control_cols, control_coef_scaled)
        },
        metrics=metrics,
    )


def coefficient_vector(result: FitResult, axes: Sequence[int]) -> np.ndarray:
    """Return coefficients aligned to ``axes``."""
    return np.array([result.coefficients.get(int(axis), 0.0) for axis in axes], dtype=float)


def contribution_mass(
    activations: np.ndarray,
    beta: np.ndarray,
    axes: Sequence[int],
) -> ContributionMass:
    """Compute contribution-mass distribution for a coefficient vector."""
    axis_positions = [int(axis) for axis in axes]
    Z = activations[:, axis_positions]
    contributions = np.mean(np.abs(Z * beta.reshape(1, -1)), axis=0)
    total = float(contributions.sum())
    if total > 0:
        mass = contributions / total
    else:
        mass = np.zeros_like(contributions)

    sorted_mass = np.sort(mass)[::-1]
    positive = mass[mass > 0]
    entropy = -float(np.sum(positive * np.log(positive))) if positive.size else 0.0
    return ContributionMass(
        contribution_by_neuron={axis: float(value) for axis, value in zip(axis_positions, contributions)},
        mass_by_neuron={axis: float(value) for axis, value in zip(axis_positions, mass)},
        top_1_mass=float(sorted_mass[:1].sum()),
        top_3_mass=float(sorted_mass[:3].sum()),
        top_5_mass=float(sorted_mass[:5].sum()),
        effective_dimension=float(math.exp(entropy)),
    )


def js_divergence_base2(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in base 2."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() > 0:
        p = p / p.sum()
    if q.sum() > 0:
        q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        if not mask.any():
            return 0.0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def coefficient_comparison_metrics(
    human_beta: np.ndarray,
    rm_beta: np.ndarray,
    human_mass: ContributionMass,
    rm_mass: ContributionMass,
    axes: Sequence[int],
) -> Dict[str, float]:
    """Compute compact comparison metrics between human and RM coefficients."""
    human_sign = np.sign(human_beta)
    rm_sign = np.sign(rm_beta)
    active = (human_sign != 0) & (rm_sign != 0)
    denom = float(np.linalg.norm(human_beta) * np.linalg.norm(rm_beta))
    p_h = np.array([human_mass.mass_by_neuron[int(axis)] for axis in axes], dtype=float)
    p_rm = np.array([rm_mass.mass_by_neuron[int(axis)] for axis in axes], dtype=float)

    if len(human_beta) > 1 and np.std(human_beta) > 0 and np.std(rm_beta) > 0:
        pearson = float(pearsonr(human_beta, rm_beta)[0])
        spearman = float(spearmanr(human_beta, rm_beta).correlation)
    else:
        pearson = float("nan")
        spearman = float("nan")

    return {
        "coefficient_pearson_r": pearson,
        "coefficient_spearman_r": spearman,
        "coefficient_cosine_similarity": float(np.dot(human_beta, rm_beta) / denom) if denom > 0 else float("nan"),
        "sign_agreement_rate_nonzero": float((human_sign[active] == rm_sign[active]).mean()) if active.any() else float("nan"),
        "sign_flip_fraction_nonzero": float((human_sign[active] != rm_sign[active]).mean()) if active.any() else float("nan"),
        "js_divergence_contribution_mass_base2": js_divergence_base2(p_h, p_rm),
        "human_top_1_mass": human_mass.top_1_mass,
        "human_top_3_mass": human_mass.top_3_mass,
        "human_top_5_mass": human_mass.top_5_mass,
        "human_effective_dimension": human_mass.effective_dimension,
        "rm_top_1_mass": rm_mass.top_1_mass,
        "rm_top_3_mass": rm_mass.top_3_mass,
        "rm_top_5_mass": rm_mass.top_5_mass,
        "rm_effective_dimension": rm_mass.effective_dimension,
    }


def make_axis_comparison_table(
    feature_table: pd.DataFrame,
    human_result: FitResult,
    rm_result: FitResult,
    human_mass: ContributionMass,
    rm_mass: ContributionMass,
) -> pd.DataFrame:
    """Build one-row-per-axis comparison table."""
    rows: List[Dict[str, Any]] = []
    for _, feature in feature_table.sort_values("neuron_idx").iterrows():
        axis = int(feature["neuron_idx"])
        beta_h = float(human_result.coefficients.get(axis, 0.0))
        beta_rm = float(rm_result.coefficients.get(axis, 0.0))
        sign_h = int(np.sign(beta_h))
        sign_rm = int(np.sign(beta_rm))
        p_h = float(human_mass.mass_by_neuron.get(axis, 0.0))
        p_rm = float(rm_mass.mass_by_neuron.get(axis, 0.0))
        rows.append(
            {
                "neuron_idx": axis,
                "interpretation": feature.get("interpretation"),
                "abbreviated_interpretation": feature.get("abbreviated_interpretation"),
                "prevalence": feature.get("prevalence"),
                "beta_human": beta_h,
                "beta_rm": beta_rm,
                "sign_human": sign_h,
                "sign_rm": sign_rm,
                "sign_flip": bool(sign_h != 0 and sign_rm != 0 and sign_h != sign_rm),
                "abs_beta_human": abs(beta_h),
                "abs_beta_rm": abs(beta_rm),
                "contribution_mass_human": p_h,
                "contribution_mass_rm": p_rm,
                "axis_amplification": p_rm - p_h,
            }
        )
    return pd.DataFrame(rows)


def fit_result_to_dict(result: FitResult) -> Dict[str, Any]:
    """Serialize a fit result with string keys for JSON."""
    return {
        "target_type": result.target_type,
        "n_rows": result.n_rows,
        "axes": result.axes,
        "controls": result.controls,
        "intercept": result.intercept,
        "coefficients": {str(key): value for key, value in result.coefficients.items()},
        "control_coefficients": result.control_coefficients,
        "standardized_coefficients": {
            str(key): value for key, value in result.standardized_coefficients.items()
        },
        "standardized_control_coefficients": result.standardized_control_coefficients,
        "metrics": result.metrics,
    }


def contribution_mass_to_dict(mass: ContributionMass) -> Dict[str, Any]:
    """Serialize contribution mass with string keys for JSON."""
    return {
        "contribution_by_neuron": {
            str(key): value for key, value in mass.contribution_by_neuron.items()
        },
        "mass_by_neuron": {str(key): value for key, value in mass.mass_by_neuron.items()},
        "top_1_mass": mass.top_1_mass,
        "top_3_mass": mass.top_3_mass,
        "top_5_mass": mass.top_5_mass,
        "effective_dimension": mass.effective_dimension,
    }


def write_json(data: Any, path: str | Path) -> None:
    """Write JSON with stable formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, allow_nan=True)
        f.write("\n")
