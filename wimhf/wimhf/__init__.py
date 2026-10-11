"""What's In My Human Feedback (WIMHF) package."""

from importlib import import_module

__all__ = [
    "SparseAutoencoder",
    "get_sae_checkpoint_name",
    "load_model",
    "get_openai_embeddings",
    "get_local_embeddings",
    "NeuronInterpreter",
    "InterpretConfig",
    "SamplingConfig",
    "ScoringConfig",
    "select_neurons_controlled_lasso",
    "select_neurons_controlled_ols",
    "select_neurons_lasso",
    "run_wimhf_pipeline",
    "load_config",
    "WIMHFConfig",
]

__version__ = "0.1.0"

_LAZY_EXPORTS = {
    "SparseAutoencoder": ("sae", "SparseAutoencoder"),
    "get_sae_checkpoint_name": ("sae", "get_sae_checkpoint_name"),
    "load_model": ("sae", "load_model"),
    "get_openai_embeddings": ("embedding", "get_openai_embeddings"),
    "get_local_embeddings": ("embedding", "get_local_embeddings"),
    "NeuronInterpreter": ("interpretation", "NeuronInterpreter"),
    "InterpretConfig": ("interpretation", "InterpretConfig"),
    "SamplingConfig": ("interpretation", "SamplingConfig"),
    "ScoringConfig": ("interpretation", "ScoringConfig"),
    "select_neurons_controlled_lasso": ("feature_selection", "select_neurons_controlled_lasso"),
    "select_neurons_controlled_ols": ("feature_selection", "select_neurons_controlled_ols"),
    "select_neurons_lasso": ("feature_selection", "select_neurons_lasso"),
    "run_wimhf_pipeline": ("quickstart", "run_wimhf_pipeline"),
    "load_config": ("quickstart", "load_config"),
    "WIMHFConfig": ("quickstart", "WIMHFConfig"),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module_name}"), attr_name)
    globals()[name] = value
    return value
