#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --account=xlab
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=5
#SBATCH --mem-per-gpu=32G
#SBATCH --time=1-12:00:00
#SBATCH --job-name=olmo3_7b_instruct_rm_lmarena_1epoch
#SBATCH --output="olmo3_7b_instruct_rm_lmarena_1epoch.out"
RM_INTERPRET_ROOT="${RM_INTERPRET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source ~/.bashrc
conda deactivate
conda activate rm_interpret

module load cuda/12.9.1

export CUDA_HOME=/mmfs1/sw/cuda/12.9.1
export CUDA_PATH="$CUDA_HOME"
export CUDA_INSTALL_PATH="$CUDA_HOME"
export CUDA_LIB_PATH="$CUDA_HOME/targets/x86_64-linux/lib"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

RUN_STAMP=${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}
MASTER_PORT=$((10000 + (${SLURM_JOB_ID:-0} % 50000)))
OUTPUT_DIR="${RM_INTERPRET_ROOT}/train_rm/outputs/olmo3_7b_instruct_rm_lmarena_1epoch_${RUN_STAMP}"
RUN_NAME="olmo3_7b_instruct_rm_lmarena_1epoch_fresh_${RUN_STAMP}"

echo "Starting fresh OLMo 3 7B Instruct RM run for 1 epoch"
echo "Output dir: ${OUTPUT_DIR}"

accelerate launch \
  --num_processes 4 \
  --use_fsdp \
  --fsdp_transformer_layer_cls_to_wrap Olmo3DecoderLayer \
  --main_process_port "$MASTER_PORT" \
  ${RM_INTERPRET_ROOT}/train_rm/train.py \
  --model-name allenai/Olmo-3-7B-Instruct \
  --data-path ${RM_INTERPRET_ROOT}/wimhf/data/arena_human_preference_55k_wimhf.jsonl \
  --num-train-epochs 1 \
  --output-dir "$OUTPUT_DIR" \
  --run-name "$RUN_NAME" \
  --format-mode chat \
  --wandb
