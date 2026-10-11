#!/bin/bash
#SBATCH --partition=gpu-l40s
#SBATCH --account=ark
#SBATCH --mem=32G
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --job-name=frozen_basis_lmarena
#SBATCH --output="frozen_basis_lmarena.out"
RM_INTERPRET_ROOT="${RM_INTERPRET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

WIMHF_DIR="${WIMHF_DIR:-${RM_INTERPRET_ROOT}/wimhf}"
TRAIN_SWEEP_ROOT="${TRAIN_SWEEP_ROOT:-${RM_INTERPRET_ROOT}/train_rm/outputs/qwen3_8b_rm_lmarena_epoch_sweep_35249460}"
BASIS_CONFIG="${BASIS_CONFIG:-${WIMHF_DIR}/configs/arena_human_preference_55k.json}"
BASIS_OUTPUT_DIR="${BASIS_OUTPUT_DIR:-${WIMHF_DIR}/outputs/arena_human_preference_55k}"
OUTPUT_DIR="${OUTPUT_DIR:-${WIMHF_DIR}/outputs/frozen_basis_lmarena_qwen3_epochs}"
AXIS_SCOPE="${AXIS_SCOPE:-kept}"
JOIN_KEY="${JOIN_KEY:-source_index}"
MAX_ROWS="${MAX_ROWS:-}"
ALLOW_EMBEDDING_COMPUTE="${ALLOW_EMBEDDING_COMPUTE:-0}"

echo "WIMHF_DIR=$WIMHF_DIR"
echo "TRAIN_SWEEP_ROOT=$TRAIN_SWEEP_ROOT"
echo "BASIS_CONFIG=$BASIS_CONFIG"
echo "BASIS_OUTPUT_DIR=$BASIS_OUTPUT_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "AXIS_SCOPE=$AXIS_SCOPE"
echo "JOIN_KEY=$JOIN_KEY"

if [[ ! -s "$BASIS_CONFIG" ]]; then
  echo "Missing basis config: $BASIS_CONFIG" >&2
  exit 1
fi

if [[ ! -s "$BASIS_OUTPUT_DIR/feature_table.jsonl" ]]; then
  echo "Missing frozen basis feature table: $BASIS_OUTPUT_DIR/feature_table.jsonl" >&2
  exit 1
fi

if [[ ! -d "$BASIS_OUTPUT_DIR/checkpoints" ]]; then
  echo "Missing frozen basis checkpoint directory: $BASIS_OUTPUT_DIR/checkpoints" >&2
  exit 1
fi

PREDICTION_FILES=(
  "$TRAIN_SWEEP_ROOT"/1epoch/eval_predictions.jsonl
  "$TRAIN_SWEEP_ROOT"/2epoch/eval_predictions.jsonl
  "$TRAIN_SWEEP_ROOT"/3epoch/eval_predictions.jsonl
)

MISSING=0
for PREDICTIONS_PATH in "${PREDICTION_FILES[@]}"; do
  if [[ ! -s "$PREDICTIONS_PATH" ]]; then
    echo "Missing required RM predictions: $PREDICTIONS_PATH" >&2
    MISSING=1
  else
    echo "Found RM predictions: $PREDICTIONS_PATH"
  fi
done

if [[ "$MISSING" -ne 0 ]]; then
  echo "Expected Qwen3 LMArena validation predictions for 1epoch, 2epoch, and 3epoch." >&2
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

CMD=(
  python scripts/run_frozen_basis_comparison.py
  --basis-config "$BASIS_CONFIG"
  --basis-output-dir "$BASIS_OUTPUT_DIR"
  --rm-predictions "${PREDICTION_FILES[@]}"
  --output-dir "$OUTPUT_DIR"
  --axis-scope "$AXIS_SCOPE"
  --fit-rows intersection
  --join-key "$JOIN_KEY"
)

if [[ -n "$MAX_ROWS" ]]; then
  CMD+=(--max-rows "$MAX_ROWS")
fi

if [[ "$ALLOW_EMBEDDING_COMPUTE" == "1" ]]; then
  CMD+=(--allow-embedding-compute)
fi

echo "Running frozen-basis comparison:"
printf ' %q' PYTHONPATH=. "${CMD[@]}"
echo

PYTHONPATH=. "${CMD[@]}"

echo "Frozen-basis comparison completed under $OUTPUT_DIR"
