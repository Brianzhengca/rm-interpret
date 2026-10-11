#!/bin/bash
#SBATCH --partition=compute
#SBATCH --account=xlab
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --job-name=frozen_basis_debug
#SBATCH --output="frozen_basis_debug.out"
RM_INTERPRET_ROOT="${RM_INTERPRET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

set -euo pipefail

WIMHF_DIR="${WIMHF_DIR:-${RM_INTERPRET_ROOT}/wimhf}"
TRAIN_SWEEP_ROOT="${TRAIN_SWEEP_ROOT:-${RM_INTERPRET_ROOT}/train_rm/outputs/qwen3_8b_rm_lmarena_epoch_sweep_35249460}"
BASIS_CONFIG="${BASIS_CONFIG:-${WIMHF_DIR}/configs/arena_human_preference_55k.json}"
BASIS_OUTPUT_DIR="${BASIS_OUTPUT_DIR:-${WIMHF_DIR}/outputs/arena_human_preference_55k}"
OUTPUT_DIR="${OUTPUT_DIR:-${WIMHF_DIR}/outputs/frozen_basis_lmarena_debug_job}"
MAX_ROWS="${MAX_ROWS:-128}"

PREDICTIONS_PATH="${PREDICTIONS_PATH:-${TRAIN_SWEEP_ROOT}/1epoch/eval_predictions.jsonl}"

echo "WIMHF_DIR=$WIMHF_DIR"
echo "PREDICTIONS_PATH=$PREDICTIONS_PATH"
echo "BASIS_CONFIG=$BASIS_CONFIG"
echo "BASIS_OUTPUT_DIR=$BASIS_OUTPUT_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "MAX_ROWS=$MAX_ROWS"

if [[ ! -s "$PREDICTIONS_PATH" ]]; then
  echo "Missing RM predictions: $PREDICTIONS_PATH" >&2
  exit 1
fi

if [[ ! -s "$BASIS_OUTPUT_DIR/feature_table.jsonl" ]]; then
  echo "Missing frozen basis feature table: $BASIS_OUTPUT_DIR/feature_table.jsonl" >&2
  exit 1
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 set; input discovery succeeded and no comparison will be launched."
  exit 0
fi

source ~/.bashrc
conda deactivate
conda activate rm_interpret

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUTPUT_DIR"
cd "$WIMHF_DIR"

PYTHONPATH=. python scripts/run_frozen_basis_comparison.py \
  --basis-config "$BASIS_CONFIG" \
  --basis-output-dir "$BASIS_OUTPUT_DIR" \
  --rm-predictions "$PREDICTIONS_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --axis-scope kept \
  --fit-rows intersection \
  --join-key source_index \
  --max-rows "$MAX_ROWS" \
  --debug

echo "Frozen-basis debug comparison completed under $OUTPUT_DIR"
