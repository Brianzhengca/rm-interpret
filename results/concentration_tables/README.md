# WIMHF Concentration Tables

**Historical diagnostic scores:** the native calculation combines standardized,
separately fitted feature coefficients with raw SAE activations. Its percentages
should be recalculated with consistent units and joint coefficients before being
interpreted as shares of preference signal. `true_contribution_mass` below means
that activations were used rather than a prevalence proxy; it does not correct
these methodological issues. See the repository README for the frozen-basis
comparison, which converts joint coefficients to raw activation units.

These tables standardize contribution-mass analysis across completed native WIMHF runs.
Each run is analyzed in its own retained SAE axes; no frozen-basis alignment is used.

## Formula

For each retained axis `j`:

`C_j = mean_i | length_controlled_logit_coef_j * activation_ij |`

`p_j = C_j / sum_k C_k`

`p_j` is the estimated share of that run's preference signal carried by axis `j`.

If cached activations cannot be recomputed from saved embeddings and SAE checkpoints,
the script falls back to `abs(beta_j) * prevalence_j` and marks those rows as
`proxy_prevalence_mass`.

## Mass Types

- `true_contribution_mass`: 21 runs
- `proxy_prevalence_mass`: 0 runs

True-mass runs:

- `lmarena_human`
- `lmarena_human_held_out`
- `lmarena_qwen3_8b_rm`
- `lmarena_qwen3_8b_rm_held_out`
- `lmarena_skywork_reward_v2_llama3_1_8b_rm`
- `lmarena_allenai_llama_3_1_8b_instruct_rm_rb2`
- `helpsteer3_human`
- `helpsteer3_lmarena_rm`
- `lmarena_qwen3_8b_rm_1epoch`
- `lmarena_qwen3_8b_rm_2epoch`
- `lmarena_qwen3_8b_rm_3epoch`
- `helpsteer3_qwen3_8b_rm_1epoch`
- `helpsteer3_qwen3_8b_rm_2epoch`
- `helpsteer3_qwen3_8b_rm_3epoch`
- `prism_qwen3_8b_rm_1epoch`
- `prism_qwen3_8b_rm_2epoch`
- `prism_qwen3_8b_rm_3epoch`
- `helpsteer3_lmarena_rm_checkpoint_500`
- `helpsteer3_lmarena_rm_checkpoint_1000`
- `helpsteer3_lmarena_rm_checkpoint_1500`
- `helpsteer3_lmarena_rm_checkpoint_1650`

No runs used proxy mass in this build.

## Files

- `concentration_summary.csv` / `.jsonl`: one row per run.
- `axis_concentration.csv` / `.jsonl`: one row per retained axis.
- `top_features.csv` / `.jsonl`: top 10 retained axes per run.
- `run_manifest_used.json`: manifest entries resolved during this build.
- `warnings.txt`: skipped runs, fallback reasons, and other non-fatal warnings.

## Manifest

The manifest used here is `wimhf/configs/concentration_runs.json`.
To add or modify runs, edit `wimhf/configs/concentration_runs.json` and set
`run_id`, `dataset`, `judge`, `epoch_or_checkpoint`, `config_path`, `output_dir`,
and optional `notes`. The output directory must contain `feature_table.jsonl`,
`summary_metrics.json`, `predictive_metrics.json`, `emb_cache/chunk_*.npy`, and a
`checkpoints/*.pt` SAE checkpoint for true contribution mass.

## Semantic Clusters

Semantic clusters are blank by default. To add them later, attach
`semantic_cluster_path` to a manifest entry. The file may be CSV, JSONL, or JSON
with `neuron_idx` and `semantic_cluster` fields.

## Warnings

`warnings.txt` contains 4 warning lines for this build.
