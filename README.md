# Reward Model Interpretation

This project studies which preference patterns reward models learn from human feedback, with a particular focus on safety refusals and compliance with harmful requests. It combines reward-model training with interpretable surrogate models built using [What's In My Human Feedback (WIMHF)](https://github.com/rmovva/wimhf).

The saved experiments identify a recurring concern: **LMArena-trained reward models can favor answering over refusing, including features interpreted as compliance with harmful or inappropriate requests.** These are associations in preference rankings; the experiments do not measure an unsafe-generation or jailbreak-success rate.

## Safety findings

The table below contains retained features from the saved Qwen3-8B analyses. Positive coefficients mean that a feature is associated with the reward model preferring a response; negative coefficients mean the reverse. These are standardized, separately fitted logistic coefficients controlling for response-length difference, rather than raw reward scores.

| Run | Interpreted feature | Preference coefficient | Interpretation fidelity correlation |
|---|---|---:|---:|
| [Original LMArena RM analysis](results/native/arena_human_preference_55k_qwen3_8b_rm/feature_table.jsonl) | Fulfills an inappropriate or harmful request instead of refusing | +0.562 | 0.397 |
| [Held-out LMArena RM analysis](results/native/arena_human_preference_55k_qwen3_8b_rm_held_out/feature_table.jsonl) | Refuses harmful or unethical requests instead of complying | -0.425 | 0.664 |
| [LMArena, 2 epochs](results/native/qwen3_8b_rm_lmarena_epoch_validation_wimhf/2epoch/feature_table.jsonl) | Refuses harmful, unethical, or inappropriate requests instead of providing a response | -0.521 | 0.721 |
| [LMArena, 3 epochs](results/native/qwen3_8b_rm_lmarena_epoch_validation_wimhf/3epoch/feature_table.jsonl) | Refuses the request instead of providing an answer | -0.774 | 0.669 |
| [LMArena, 3 epochs](results/native/qwen3_8b_rm_lmarena_epoch_validation_wimhf/3epoch/feature_table.jsonl) | Fulfills a harmful or explicit request instead of refusing | +0.156 | 0.455 |

Several safety-related refusal features also receive negative coefficients in the LMArena-trained Llama-3.1-8B-Instruct and OLMo-3-7B-Instruct runs. HelpSteer3-trained Qwen has weaker and mixed safety-related associations. The pattern therefore depends on the model and training data.

The [human-RM disagreement analysis](results/disagreement_wimhf/disagreement_summary.json) adds two useful observations:

- A retained feature described as providing detailed, creative responses to potentially harmful or unethical requests predicts **greater agreement** between human choices and the RM. This suggests that some preference for compliance may be present in the feedback itself.
- A retained feature describing refusal of inappropriate or harmful fictional content predicts **greater disagreement** when the human-preferred response has that feature. This association is smaller, and its examples also include ordinary fiction and answering behavior.

One inspected case, `source_index=29814`, asks the assistant to adopt an unrestricted "PersonGPT" persona. Humans preferred refusal, while the RM preferred accepting the unrestricted-persona framing, with a reward margin of about 0.826 in favor of acceptance. This is evidence about ranking that framing; no subsequent harmful generation was tested. Raw conversations are not included in this source release.

### How to interpret the safety result

The results suggest a potential conflict between learned preference rewards and safety-motivated refusal. They do not establish that every refusal is appropriate, that all compliant answers are harmful, or that models became less safe at a measured rate during training. Features can combine justified refusal, unnecessary refusal, sensitive fiction, and ordinary helpfulness. Each native run learns its own axes, so coefficients across epochs are not a direct before-and-after measurement of one fixed safety concept.

Feature descriptions are generated and evaluated by an LLM. Fidelity correlation measures how well a description matches feature activation; it is not a harmful-response rate or a significance test of the preference coefficient. The [safety feature index](results/safety_features.csv) includes both retained and rejected interpretations, with the `kept` column distinguishing them. For example, the disagreement feature about omitting disclaimers or safety warnings failed the interpretation-quality filter and should not support a confident safety claim.

A stronger follow-up would label harmful and benign prompts separately, distinguish justified refusal from over-refusal, and evaluate reward rankings and generated behavior within those groups.

## What the project does

The workflow starts with a prompt, two candidate responses, and a human preference label:

1. **Train and evaluate reward models.** `train_rm/` fine-tunes scalar reward models using chosen/rejected pairs, splits by conversation with an approximately 80/20 train/validation ratio, and exports rewards and predicted preferences. Experiments include Qwen3-8B epoch sweeps on LMArena, HelpSteer3, and PRISM; Llama and OLMo LMArena runs; and off-the-shelf Skywork and AllenAI baselines.
2. **Learn interpretable concepts.** `wimhf/` embeds instructed prompt-response exchanges, computes response-embedding differences, and learns a sparse autoencoder. The main local configurations use Qwen3-Embedding-8B, 32 SAE features with sparsity parameter 4, and Qwen3-32B to describe and evaluate features.
3. **Explain preference behavior.** Logistic probes use selected features plus length controls to predict either human preferences or RM-generated preference labels. These explanations describe input-output preference behavior through text embeddings.
4. **Compare humans and models.** Native analyses learn a separate basis for each run. The frozen-basis experiment instead uses the same human-derived SAE axes and matched rows to compare human coefficients with coefficients predicting RM reward margins.
5. **Study disagreement and concentration.** Dedicated scripts model when an RM disagrees with a human choice and summarize how concentrated the feature associations appear.

## Other saved findings

- Ten selected concepts plus response-length difference predict LMArena human preferences at about **0.699 validation AUC**, versus **0.795** for the original Qwen RM preferences. These AUCs measure different targets; a higher RM-surrogate AUC does not mean better agreement with humans.
- Qwen's agreement with human validation labels increases from **69.4% to 70.7%** on LMArena, **70.2% to 71.9%** on HelpSteer3, and **61.5% to 63.9%** on PRISM between the one- and three-epoch runs. See [validation accuracies](results/rm_validation_accuracy.json).
- Across 37,004 matched pairs in the disagreement experiment, the RM disagrees with human preference on **28.8%**; another **0.48%** have zero reward margin. A ten-concept-plus-length surrogate predicts disagreement at **0.752 validation AUC**, rising to **0.781** with all SAE features. These are overall disagreement statistics, not safety-specific error rates.
- On the [shared human-derived basis](results/frozen_basis_lmarena_qwen3_epochs/comparison_metrics.json), coefficient correlation increases from **0.793 to 0.868** across Qwen epochs 1-3, with signs agreeing on 80% of the 20 axes. Human and three-epoch RM concentration are similar: top-three contribution mass is about 56% and 55%, respectively, with about ten effective dimensions each. These coefficients are fitted on 7,420 matched validation rows; the fit metrics are diagnostic fits on those rows, not an additional held-out evaluation. Overall similarity does not establish safety alignment within harmful prompts.

### Concentration caveat

The historical [native concentration tables](results/concentration_tables/concentration_summary.csv) report stronger concentration for some RM runs. Their calculation multiplies standardized, one-feature-at-a-time coefficients by raw SAE activations. The units do not match a joint model's raw coefficients, and correlated or redundant features can be counted separately. Treat those reported percentages as historical diagnostic scores pending a corrected calculation. The frozen-basis implementation fits coefficients jointly and converts them to raw activation units before computing contributions.

## Getting started

Full model training and local interpretation require a suitable CUDA environment and substantial GPU memory. The saved results can be inspected with ordinary JSON/CSV tools without downloading models. Start from the repository root:

```bash
python -m pip install -e ./wimhf
```

Prepare LMArena data and its configuration:

```bash
cd wimhf
python scripts/prepare_arena_human_preference_55k.py
python scripts/run_wimhf.py configs/arena_human_preference_55k.json \
  --output-dir outputs/arena_human_preference_55k
cd ..
```

Train a reward model and export its validation predictions:

```bash
accelerate launch --num_processes 4 --use_fsdp train_rm/train.py \
  --model-name Qwen/Qwen3-8B \
  --data-path wimhf/data/arena_human_preference_55k_wimhf.jsonl \
  --num-train-epochs 1 \
  --output-dir train_rm/outputs/qwen3_lmarena_1epoch
```

The training script writes `eval_predictions.jsonl`. For epoch comparisons, train fresh runs for each duration. The token-length filter can change the usable row count; retain conversation/source metadata when comparing outputs.

Convert validation predictions into a WIMHF dataset and explain the RM's choices:

```bash
python wimhf/scripts/prepare_rm_prediction_dataset.py \
  --base-data wimhf/data/arena_human_preference_55k_wimhf.jsonl \
  --predictions train_rm/outputs/qwen3_lmarena_1epoch/eval_predictions.jsonl \
  --output wimhf/data/qwen3_lmarena_1epoch_wimhf.jsonl \
  --template-config wimhf/configs/arena_human_preference_55k.json \
  --config-output wimhf/configs/qwen3_lmarena_1epoch.json \
  --dataset-name Qwen3LMArena1Epoch \
  --output-dir wimhf/outputs/qwen3_lmarena_1epoch \
  --reward-model-name Qwen3LMArena1Epoch

python wimhf/scripts/run_wimhf.py wimhf/configs/qwen3_lmarena_1epoch.json \
  --output-dir wimhf/outputs/qwen3_lmarena_1epoch
```

For a comparison on the frozen human basis:

```bash
cd wimhf
python scripts/run_frozen_basis_comparison.py \
  --basis-config configs/arena_human_preference_55k.json \
  --basis-output-dir outputs/arena_human_preference_55k \
  --rm-predictions ../train_rm/outputs/qwen3_lmarena_1epoch/eval_predictions.jsonl \
  --output-dir outputs/frozen_basis_qwen3
```

Frozen-basis and disagreement experiments use cached embeddings by default. Their `--allow-embedding-compute` option explicitly allows computing missing embeddings. See [FROZEN_BASIS.md](wimhf/FROZEN_BASIS.md) and each script's `--help` for options. HelpSteer3 has a preparation script; PRISM has a dataset configuration. The `.sh` files are Slurm launch examples and require local adjustments to scheduler accounts, CUDA modules, and environment names.

The main configurations use local models. Remote WIMHF calls use the `OAI_WIMHF` environment variable; credentials should be supplied through the environment.

## Repository contents

```text
train_rm/       Reward training, evaluation, and Slurm launch examples
wimhf/          Modified WIMHF library, configuration files, and analysis scripts
results/        Compact saved feature tables, coefficients, and evaluation summaries
```

The release includes code and derived results. Model weights, optimizer states, embeddings, raw preference datasets, raw prediction conversations, training logs, and notebook outputs are excluded. Reproducing complete experiments requires downloading the datasets and models and regenerating the excluded artifacts. See [results/README.md](results/README.md) for the saved-result layout.

## Attribution and license

The WIMHF implementation derives from [rmovva/wimhf](https://github.com/rmovva/wimhf), based on *What's In My Human Feedback? Learning Interpretable Descriptions of Preference Data* by Rajiv Movva, Smitha Milli, Sewon Min, and Emma Pierson ([arXiv:2510.26202](https://arxiv.org/abs/2510.26202)). The upstream checkout used here was commit `af57741`, with local extensions for reward-model comparisons, frozen-basis analysis, and disagreement analysis. The safety observations described above come from this experiment snapshot and should not be attributed to the original WIMHF paper.

The upstream MIT copyright and license are preserved in [LICENSE](LICENSE) and [wimhf/LICENSE](wimhf/LICENSE). Dataset and model licenses remain those of their respective providers.
