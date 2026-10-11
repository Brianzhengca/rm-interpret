#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --account=xlab
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-gpu=5
#SBATCH --gres=gpu:4
#SBATCH --time=23:00:00
#SBATCH --job-name=run_arena_allenai_rb2_rm
#SBATCH --output="run_arena_allenai_rb2_rm.out"
RM_INTERPRET_ROOT="${RM_INTERPRET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

WIMHF_DIR="${WIMHF_DIR:-${RM_INTERPRET_ROOT}/wimhf}"
BASE_DATA="${BASE_DATA:-${WIMHF_DIR}/data/arena_human_preference_55k_wimhf.jsonl}"
TEMPLATE_CONFIG="${TEMPLATE_CONFIG:-${WIMHF_DIR}/configs/arena_human_preference_55k_qwen3_8b_rm.json}"
PREDICTIONS_PATH="${PREDICTIONS_PATH:-${RM_INTERPRET_ROOT}/train_rm/outputs/allenai_llama_3_1_8b_instruct_rm_rb2/eval_results.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${WIMHF_DIR}/outputs/arena_human_preference_55k_allenai_llama_3_1_8b_instruct_rm_rb2_rm}"
DATASET_PATH="${DATASET_PATH:-${WIMHF_DIR}/data/arena_human_preference_55k_allenai_llama_3_1_8b_instruct_rm_rb2_rm_wimhf.jsonl}"
CONFIG_PATH="${CONFIG_PATH:-${WIMHF_DIR}/configs/arena_human_preference_55k_allenai_llama_3_1_8b_instruct_rm_rb2_rm.json}"
DATASET_NAME="${DATASET_NAME:-ArenaHumanPreference55K_AllenAI_Llama_3_1_8B_Instruct_RM_RB2_RM}"
REWARD_MODEL_NAME="${REWARD_MODEL_NAME:-allenai/Llama-3.1-8B-Instruct-RM-RB2}"

echo "WIMHF_DIR=$WIMHF_DIR"
echo "BASE_DATA=$BASE_DATA"
echo "PREDICTIONS_PATH=$PREDICTIONS_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"

if [[ ! -s "$PREDICTIONS_PATH" ]]; then
  echo "Missing required AllenAI reward-model predictions: $PREDICTIONS_PATH" >&2
  echo "Run rm_interpret/train_rm/eval_allenai_llama_3_1_8b_instruct_rm_rb2_ots.sh first." >&2
  exit 1
fi

python - "$BASE_DATA" "$PREDICTIONS_PATH" <<'PY'
import json
import sys
from pathlib import Path

base_path = Path(sys.argv[1])
predictions_path = Path(sys.argv[2])

expected = 0
with base_path.open() as f:
    for line_no, line in enumerate(f, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{base_path}:{line_no} is not valid JSON: {exc}") from exc
        if row.get("label") in (0, 1):
            expected += 1

seen = 0
with predictions_path.open() as f:
    for line_no, line in enumerate(f, start=1):
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"{predictions_path}:{line_no} is not valid JSON. "
                "The reward-model eval likely failed before finishing; rerun the eval job "
                "after resolving the underlying failure. "
                f"JSON error: {exc}"
            ) from exc
        seen += 1

if seen != expected:
    raise SystemExit(
        f"{predictions_path} has {seen} prediction rows but {base_path} has "
        f"{expected} binary preference rows. Refusing to create a partial WIMHF dataset."
    )

print(f"Validated {seen} AllenAI prediction rows for {expected} LMArena rows.")
PY

if [[ "$?" -ne 0 ]]; then
  echo "AllenAI prediction validation failed; stopping before WIMHF conversion." >&2
  exit 1
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 set; prediction validation succeeded and no WIMHF work will be launched."
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

echo "===== creating WIMHF dataset/config from AllenAI LMArena predictions ====="
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
  echo "Failed to prepare AllenAI WIMHF inputs; stopping." >&2
  exit 1
fi

echo "===== running WIMHF for AllenAI LMArena reward-model labels ====="
PYTHONPATH="$WIMHF_DIR" python "${WIMHF_DIR}/scripts/run_wimhf.py" "$CONFIG_PATH" --output-dir "$OUTPUT_DIR"

if [[ "$?" -ne 0 ]]; then
  echo "WIMHF failed for AllenAI LMArena reward-model labels." >&2
  exit 1
fi

echo "AllenAI LMArena WIMHF run completed under $OUTPUT_DIR"
