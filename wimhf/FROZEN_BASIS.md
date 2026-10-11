# Frozen-Basis WIMHF Comparison

This experiment freezes the SAE feature basis learned from human LMArena
preferences, then fits human and reward-model coefficients on the same frozen
axes and the same validation rows.

## LMArena command

Submit the full three-checkpoint LMArena/Qwen3 comparison:

```bash
cd wimhf
sbatch run_frozen_basis_lmarena_qwen3_epochs.sh
```

For a scheduler-level debug job:

```bash
sbatch run_frozen_basis_lmarena_qwen3_debug.sh
```

Both launch scripts accept environment overrides. For example:

```bash
MAX_ROWS=512 OUTPUT_DIR=/path/to/output sbatch run_frozen_basis_lmarena_qwen3_epochs.sh
DRY_RUN=1 sbatch run_frozen_basis_lmarena_qwen3_epochs.sh
```

Run from the `wimhf/` directory:

```bash
cd wimhf
source ~/.bashrc
conda activate rm_interpret

PYTHONPATH=. python scripts/run_frozen_basis_comparison.py \
  --basis-config configs/arena_human_preference_55k.json \
  --basis-output-dir outputs/arena_human_preference_55k \
  --rm-predictions '../train_rm/outputs/qwen3_8b_rm_lmarena_epoch_sweep_35249460/*epoch/eval_predictions.jsonl' \
  --output-dir outputs/frozen_basis_lmarena_qwen3_epochs
```

For a quick smoke test:

```bash
PYTHONPATH=. python scripts/run_frozen_basis_comparison.py \
  --rm-predictions ../train_rm/outputs/qwen3_8b_rm_lmarena_epoch_sweep_35249460/1epoch/eval_predictions.jsonl \
  --output-dir outputs/frozen_basis_lmarena_debug \
  --max-rows 128 \
  --debug
```

## Inputs

- Human basis artifacts in `outputs/arena_human_preference_55k/`:
  `feature_table.jsonl`, `checkpoints/SAE_matryoshka_M=32_K=4_prefixes=8-32.pt`,
  and the embedding cache under `emb_cache/`.
- The human dataset/config in `configs/arena_human_preference_55k.json`.
- One or more RM prediction JSONL files. The Qwen3 epoch files already contain
  validation rows, `source_index`, rewards for chosen/rejected responses, and
  enough position metadata to recover `rm_reward_A - rm_reward_B`.

By default the script is cache-only for embeddings. If a required embedding is
missing, it fails and reports the missing count. Use `--allow-embedding-compute`
only when you intentionally want to compute missing embeddings.

## Outputs

The output directory contains:

- `frozen_basis_feature_table.jsonl`
- `human_coefficients.json`
- `rm_coefficients_by_checkpoint.json`
- `comparison_metrics.json`
- `contribution_mass_human.json`
- `contribution_mass_rm.json`
- `axis_comparison_<checkpoint>.csv`
- `axis_comparison_<checkpoint>.jsonl`

The axis comparison tables include one row per frozen axis with interpretation,
prevalence, human/RM betas, signs, sign flips, contribution masses, and RM
axis amplification relative to the human mass distribution.

## Extending to HelpSteer3 or PRISM

Use the same command shape, but swap:

- `--basis-config` to the human config for that dataset.
- `--basis-output-dir` to the human WIMHF output directory for that dataset.
- `--rm-predictions` to the RM validation prediction JSONL files for that
  dataset/checkpoint sweep.
- `--join-key` if the dataset does not use `source_index`.

Keep `--axis-scope kept` and `--fit-rows intersection` for the clean
human-vs-RM comparison unless you intentionally want a broader diagnostic run.
