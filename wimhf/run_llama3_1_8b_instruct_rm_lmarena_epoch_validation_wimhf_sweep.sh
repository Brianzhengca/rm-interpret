#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --account=xlab
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-gpu=5
#SBATCH --gres=gpu:4
#SBATCH --time=23:00:00
#SBATCH --job-name=llama3_1_lmarena_val_wimhf
#SBATCH --output="llama3_1_lmarena_val_wimhf.out"
RM_INTERPRET_ROOT="${RM_INTERPRET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

WIMHF_DIR="${WIMHF_DIR:-${RM_INTERPRET_ROOT}/wimhf}"
TRAIN_SWEEP_ROOT="${TRAIN_SWEEP_ROOT:-${RM_INTERPRET_ROOT}/train_rm/outputs/llama3_1_8b_instruct_rm_lmarena_epoch_sweep_35607842}"
BASE_DATA="${BASE_DATA:-${WIMHF_DIR}/data/arena_human_preference_55k_wimhf.jsonl}"
TEMPLATE_CONFIG="${TEMPLATE_CONFIG:-${WIMHF_DIR}/configs/arena_human_preference_55k_qwen3_8b_rm.json}"
ROOT_OUTPUT="${ROOT_OUTPUT:-${WIMHF_DIR}/outputs/llama3_1_8b_instruct_rm_lmarena_epoch_validation_wimhf}"
REPO_ROOT="${REPO_ROOT:-$(cd "${WIMHF_DIR}/../.." && pwd)}"

EPOCHS=("1epoch" "2epoch" "3epoch")

echo "WIMHF_DIR=$WIMHF_DIR"
echo "TRAIN_SWEEP_ROOT=$TRAIN_SWEEP_ROOT"
echo "ROOT_OUTPUT=$ROOT_OUTPUT"
echo "REPO_ROOT=$REPO_ROOT"

MISSING=0
for EPOCH in "${EPOCHS[@]}"; do
  PREDICTIONS_PATH="${TRAIN_SWEEP_ROOT}/${EPOCH}/eval_predictions.jsonl"
  if [[ ! -s "$PREDICTIONS_PATH" ]]; then
    echo "Missing required validation predictions: $PREDICTIONS_PATH" >&2
    MISSING=1
  else
    echo "Found ${EPOCH} validation predictions: $PREDICTIONS_PATH"
  fi
done

if [[ "$MISSING" -ne 0 ]]; then
  echo "Expected validation predictions for all epochs; refusing to run WIMHF." >&2
  exit 1
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 set; validation prediction discovery succeeded and no WIMHF work will be launched."
  exit 0
fi

source ~/.bashrc
conda deactivate
conda activate rm_interpret

module load cuda/12.9.1

export CUDA_HOME=/mmfs1/sw/cuda/12.9.1
export CUDA_PATH="$CUDA_HOME"
export CUDA_INSTALL_PATH="$CUDA_HOME"
export CUDA_LIB_PATH="$CUDA_HOME/targets/x86_64-linux/lib"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PYTORCH_NVCC="$CUDA_HOME/bin/nvcc"
export FLASHINFER_NVCC="$CUDA_HOME/bin/nvcc"
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "CUDA_HOME=$CUDA_HOME"
echo "FLASHINFER_NVCC=$FLASHINFER_NVCC"
echo "VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"
which nvcc
nvcc --version

mkdir -p "$ROOT_OUTPUT"
cd "$WIMHF_DIR"

for EPOCH in "${EPOCHS[@]}"; do
  PREDICTIONS_PATH="${TRAIN_SWEEP_ROOT}/${EPOCH}/eval_predictions.jsonl"
  OUTPUT_DIR="${ROOT_OUTPUT}/${EPOCH}"
  DATASET_PATH="${OUTPUT_DIR}/arena_human_preference_55k_llama3_1_8b_instruct_rm_lmarena_${EPOCH}_validation_wimhf.jsonl"
  CONFIG_PATH="${OUTPUT_DIR}/wimhf_config.json"
  DATASET_NAME="ArenaHumanPreference55K_Llama3_1_8B_Instruct_RM_LMArena_${EPOCH}_Validation"

  echo "===== ${EPOCH}: creating WIMHF dataset/config from validation predictions ====="
  mkdir -p "$OUTPUT_DIR"
  python "${WIMHF_DIR}/scripts/prepare_arena_rm_predictions.py" \
    --base-data "$BASE_DATA" \
    --predictions "$PREDICTIONS_PATH" \
    --output "$DATASET_PATH" \
    --template-config "$TEMPLATE_CONFIG" \
    --config-output "$CONFIG_PATH" \
    --dataset-name "$DATASET_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --reward-model-name "meta-llama/Llama-3.1-8B-Instruct ${EPOCH} fresh LMArena RM"

  if [[ "$?" -ne 0 ]]; then
    echo "Failed to prepare WIMHF inputs for ${EPOCH}; stopping." >&2
    exit 1
  fi

  echo "===== ${EPOCH}: running WIMHF ====="
  PYTHONPATH="$REPO_ROOT" python "${WIMHF_DIR}/scripts/run_wimhf.py" "$CONFIG_PATH" --output-dir "$OUTPUT_DIR"

  if [[ "$?" -ne 0 ]]; then
    echo "WIMHF failed for ${EPOCH}; stopping." >&2
    exit 1
  fi

  echo "===== ${EPOCH}: completed ====="
done

echo "All epoch validation WIMHF runs completed under $ROOT_OUTPUT"
