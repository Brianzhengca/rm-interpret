#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --account=xlab
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=80G
#SBATCH --time=23:00:00
#SBATCH --job-name=qwen3_8b_rm_full
#SBATCH --output="qwen3_8b_rm_full.out"
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

MASTER_PORT=$((10000 + (${SLURM_JOB_ID:-0} % 50000)))

accelerate launch \
  --num_processes 4 \
  --use_fsdp \
  --main_process_port "$MASTER_PORT" \
  ${RM_INTERPRET_ROOT}/train_rm/train.py \
  --wandb
