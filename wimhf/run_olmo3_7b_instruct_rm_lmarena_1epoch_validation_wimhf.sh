#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --account=xlab
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-gpu=5
#SBATCH --gres=gpu:4
#SBATCH --time=23:00:00
#SBATCH --job-name=olmo3_7b_lmarena_val_wimhf
#SBATCH --output="olmo3_7b_lmarena_val_wimhf.out"
RM_INTERPRET_ROOT="${RM_INTERPRET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

WIMHF_DIR="${WIMHF_DIR:-${RM_INTERPRET_ROOT}/wimhf}"
PREDICTIONS_PATH="${PREDICTIONS_PATH:-${RM_INTERPRET_ROOT}/train_rm/outputs/olmo3_7b_instruct_rm_lmarena_1epoch_35697642/eval_predictions.jsonl}"
BASE_DATA="${BASE_DATA:-${WIMHF_DIR}/data/arena_human_preference_55k_wimhf.jsonl}"
TEMPLATE_CONFIG="${TEMPLATE_CONFIG:-${WIMHF_DIR}/configs/arena_human_preference_55k_qwen3_8b_rm.json}"
ROOT_OUTPUT="${ROOT_OUTPUT:-${WIMHF_DIR}/outputs/olmo3_7b_instruct_rm_lmarena_1epoch_validation_wimhf}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_OUTPUT}/1epoch}"
DATASET_PATH="${DATASET_PATH:-${OUTPUT_DIR}/arena_human_preference_55k_olmo3_7b_instruct_rm_lmarena_1epoch_validation_wimhf.jsonl}"
CONFIG_PATH="${CONFIG_PATH:-${OUTPUT_DIR}/wimhf_config.json}"
DATASET_NAME="${DATASET_NAME:-ArenaHumanPreference55K_OLMo3_7B_Instruct_RM_LMArena_1epoch_Validation}"
REWARD_MODEL_NAME="${REWARD_MODEL_NAME:-allenai/Olmo-3-7B-Instruct 1epoch fresh LMArena RM}"
REPO_ROOT="${REPO_ROOT:-$(cd "${WIMHF_DIR}/../.." && pwd)}"

echo "WIMHF_DIR=$WIMHF_DIR"
echo "PREDICTIONS_PATH=$PREDICTIONS_PATH"
echo "BASE_DATA=$BASE_DATA"
echo "TEMPLATE_CONFIG=$TEMPLATE_CONFIG"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "REPO_ROOT=$REPO_ROOT"

if [[ ! -s "$PREDICTIONS_PATH" ]]; then
  echo "Missing required OLMo validation predictions: $PREDICTIONS_PATH" >&2
  echo "Run rm_interpret/train_rm/run_olmo3_7b_instruct_rm_lmarena_1epoch.sh first." >&2
  exit 1
fi

if [[ ! -s "$BASE_DATA" ]]; then
  echo "Missing required base WIMHF data: $BASE_DATA" >&2
  exit 1
fi

if [[ ! -s "$TEMPLATE_CONFIG" ]]; then
  echo "Missing required WIMHF template config: $TEMPLATE_CONFIG" >&2
  exit 1
fi

echo "Found OLMo validation predictions: $PREDICTIONS_PATH"

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

mkdir -p "$OUTPUT_DIR"
cd "$WIMHF_DIR"

echo "===== 1epoch: creating WIMHF dataset/config from OLMo validation predictions ====="
python "${WIMHF_DIR}/scripts/prepare_arena_rm_predictions.py" \
  --base-data "$BASE_DATA" \
  --predictions "$PREDICTIONS_PATH" \
  --output "$DATASET_PATH" \
  --template-config "$TEMPLATE_CONFIG" \
  --config-output "$CONFIG_PATH" \
  --dataset-name "$DATASET_NAME" \
  --output-dir "$OUTPUT_DIR" \
  --reward-model-name "$REWARD_MODEL_NAME"

if [[ "$?" -ne 0 ]]; then
  echo "Failed to prepare OLMo WIMHF inputs; stopping." >&2
  exit 1
fi

echo "===== 1epoch: running WIMHF for OLMo LMArena reward-model labels ====="
PYTHONPATH="$REPO_ROOT" python "${WIMHF_DIR}/scripts/run_wimhf.py" "$CONFIG_PATH" --output-dir "$OUTPUT_DIR"

if [[ "$?" -ne 0 ]]; then
  echo "WIMHF failed for OLMo LMArena reward-model labels." >&2
  exit 1
fi

echo "OLMo LMArena WIMHF run completed under $OUTPUT_DIR"
