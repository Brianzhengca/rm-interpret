# Saved experiment results

These files are a compact export of the completed local experiments. They retain feature interpretations, coefficients, and aggregate evaluation metrics while excluding raw conversations, embeddings, model weights, and training logs. Absolute experiment-directory prefixes have been converted to relative paths. Paths inside saved metadata describe the original artifact layout; excluded artifacts must be regenerated for a new experiment.

- `native/`: feature tables, preference-prediction metrics, and coefficients from each completed native WIMHF run. `kept` indicates whether an interpretation passed the configured fidelity filter. `length_controlled_logit_coef` is a standardized, marginal coefficient from a separate feature-plus-length logistic fit.
- `safety_features.csv`: descriptions matching safety/refusal terms across native runs, including rejected interpretations. This is a keyword index, not a manually labeled harmfulness dataset.
- `rm_validation_accuracy.json`: human-preference accuracy computed from the saved validation prediction files. Usable validation populations differ across datasets and tokenizers.
- `frozen_basis_lmarena_qwen3_epochs/`: human and Qwen coefficients fitted on the same human-derived axes and the same 7,420 validation rows. Contribution masses use joint coefficients converted to raw activation units. Fit metrics use the coefficient-fitting rows.
- `disagreement_wimhf/`: aggregate agreement statistics, disagreement-prediction metrics, and feature descriptions. Raw `top_validation_examples` have been removed from the feature table; they were not needed to preserve the fitted coefficients or aggregate metrics.
- `concentration_tables/`: historical native concentration scores. These combine standardized marginal coefficients with raw activations and should be treated as diagnostic scores pending recalculation, rather than a decomposition of a joint model's prediction.

The term `true_contribution_mass` in the historical native tables distinguishes use of saved SAE activations from a prevalence-based proxy. It does not resolve the coefficient-scale and marginal-fit issues described above. Interpretation-test p-values assess description fidelity, not significance of preference coefficients.
