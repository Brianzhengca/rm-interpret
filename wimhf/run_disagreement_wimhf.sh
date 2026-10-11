#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --account=xlab
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-gpu=5
#SBATCH --gres=gpu:4
#SBATCH --time=23:00:00
#SBATCH --job-name=disagreement_wimhf
#SBATCH --output="disagreement_wimhf.out"
RM_INTERPRET_ROOT="${RM_INTERPRET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

WIMHF_DIR="${WIMHF_DIR:-${RM_INTERPRET_ROOT}/wimhf}"
OUTPUT_DIR="${OUTPUT_DIR:-${WIMHF_DIR}/outputs/disagreement_wimhf}"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-${WIMHF_DIR}/outputs/disagreement_wimhf_smoke}"
MAX_ROWS="${MAX_ROWS:-256}"

# Modes:
#   prepare: build the full derived disagreement dataset and reports only.
#   smoke:   train a tiny debug SAE without LLM interpretation.
#   full:    train and interpret the native disagreement WIMHF run.
MODE="${MODE:-prepare}"

case "$MODE" in
  prepare)
    OUTPUT="$OUTPUT_DIR"
    ARGS=(--prepare-only)
    ;;
  smoke)
    OUTPUT="$SMOKE_OUTPUT_DIR"
    ARGS=(--debug --max-rows "$MAX_ROWS" --skip-interpretation)
    ;;
  full)
    OUTPUT="$OUTPUT_DIR"
    ARGS=()
    ;;
  *)
    echo "Unsupported MODE='$MODE'. Use prepare, smoke, or full." >&2
    exit 2
    ;;
esac

echo "WIMHF_DIR=$WIMHF_DIR"
echo "MODE=$MODE"
echo "OUTPUT=$OUTPUT"
echo "MAX_ROWS=$MAX_ROWS"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 set; command would be:"
  echo "cd '$WIMHF_DIR' && PYTHONPATH=. python scripts/run_disagreement_wimhf.py --output-dir '$OUTPUT' ${ARGS[*]}"
  exit 0
fi

source ~/.bashrc
conda deactivate
conda activate rm_interpret

if command -v module >/dev/null 2>&1; then
  module load cuda/12.9.1 || true
fi

export CUDA_HOME="${CUDA_HOME:-/mmfs1/sw/cuda/12.9.1}"
export CUDA_PATH="$CUDA_HOME"
export CUDA_INSTALL_PATH="$CUDA_HOME"
export CUDA_LIB_PATH="$CUDA_HOME/targets/x86_64-linux/lib"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PYTORCH_NVCC="$CUDA_HOME/bin/nvcc"
export FLASHINFER_NVCC="$CUDA_HOME/bin/nvcc"
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_GPU:-${SLURM_CPUS_PER_TASK:-5}}"
export PYTHONUNBUFFERED=1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "$OUTPUT"
cd "$WIMHF_DIR"

PYTHONPATH=. python scripts/run_disagreement_wimhf.py \
  --output-dir "$OUTPUT" \
  "${ARGS[@]}"

echo "Disagreement WIMHF $MODE run completed under $OUTPUT"
