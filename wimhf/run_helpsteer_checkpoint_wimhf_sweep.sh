#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --account=xlab
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=5
#SBATCH --mem-per-gpu=32G
#SBATCH --time=23:00:00
#SBATCH --job-name=helpsteer_ckpt_wimhf
#SBATCH --output="helpsteer_ckpt_wimhf.out"
RM_INTERPRET_ROOT="${RM_INTERPRET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

WIMHF_DIR="${WIMHF_DIR:-${RM_INTERPRET_ROOT}/wimhf}"
TRAIN_RM_DIR="${TRAIN_RM_DIR:-${RM_INTERPRET_ROOT}/train_rm}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${TRAIN_RM_DIR}/outputs/arena_human_preference_55k_qwen3_8b_full_ft}"
BASE_DATA="${BASE_DATA:-${WIMHF_DIR}/data/helpsteer3_preference_wimhf.jsonl}"
TEMPLATE_CONFIG="${TEMPLATE_CONFIG:-${WIMHF_DIR}/configs/helpsteer3_arena_rm.json}"
ROOT_OUTPUT="${ROOT_OUTPUT:-${WIMHF_DIR}/outputs/checkpoint_outputs}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
BATCH_SIZE="${BATCH_SIZE:-4}"

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
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

echo "CUDA_HOME=$CUDA_HOME"
echo "FLASHINFER_NVCC=$FLASHINFER_NVCC"
echo "VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"
which nvcc
nvcc --version

mapfile -t CHECKPOINTS < <(find "$CHECKPOINT_ROOT" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)

echo "Discovered ${#CHECKPOINTS[@]} checkpoints under $CHECKPOINT_ROOT"
printf '  %s\n' "${CHECKPOINTS[@]}"

if [[ "${#CHECKPOINTS[@]}" -ne 4 ]]; then
  echo "Expected exactly 4 checkpoint-* directories; refusing to run." >&2
  exit 1
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 set; checkpoint discovery succeeded and no work will be launched."
  exit 0
fi

mkdir -p "$ROOT_OUTPUT"
cd "$WIMHF_DIR"

for CHECKPOINT_DIR in "${CHECKPOINTS[@]}"; do
  CHECKPOINT_NAME="$(basename "$CHECKPOINT_DIR")"
  OUTPUT_DIR="${ROOT_OUTPUT}/${CHECKPOINT_NAME}"
  EVAL_PATH="${OUTPUT_DIR}/eval_results_helpsteer3.jsonl"
  DATASET_PATH="${OUTPUT_DIR}/helpsteer3_${CHECKPOINT_NAME}_rm_wimhf.jsonl"
  CONFIG_PATH="${OUTPUT_DIR}/wimhf_config.json"
  DATASET_NAME="HelpSteer3_${CHECKPOINT_NAME}_RM"

  echo "===== ${CHECKPOINT_NAME}: resetting ${OUTPUT_DIR} ====="
  rm -rf "$OUTPUT_DIR"
  mkdir -p "$OUTPUT_DIR"

  echo "===== ${CHECKPOINT_NAME}: evaluating reward model on HelpSteer3 ====="
  python "${TRAIN_RM_DIR}/evaluate.py" \
    --model-path "$CHECKPOINT_DIR" \
    --data-path "$BASE_DATA" \
    --batch-size "$BATCH_SIZE" \
    --max-length "$MAX_LENGTH" \
    --output-path "$EVAL_PATH"

  echo "===== ${CHECKPOINT_NAME}: creating WIMHF dataset/config ====="
  python "${WIMHF_DIR}/scripts/prepare_rm_prediction_dataset.py" \
    --base-data "$BASE_DATA" \
    --predictions "$EVAL_PATH" \
    --output "$DATASET_PATH" \
    --template-config "$TEMPLATE_CONFIG" \
    --config-output "$CONFIG_PATH" \
    --dataset-name "$DATASET_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --reward-model-name "$CHECKPOINT_DIR"

  echo "===== ${CHECKPOINT_NAME}: running WIMHF ====="
  PYTHONPATH=. python scripts/run_wimhf.py "$CONFIG_PATH" --output-dir "$OUTPUT_DIR"

  echo "===== ${CHECKPOINT_NAME}: completed ====="
done

echo "All checkpoint WIMHF runs completed under $ROOT_OUTPUT"
