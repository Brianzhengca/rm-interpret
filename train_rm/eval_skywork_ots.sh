#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --account=xlab
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=5
#SBATCH --mem-per-gpu=32G
#SBATCH --time=23:00:00
#SBATCH --job-name=skywork_ots_eval
#SBATCH --output="skywork_ots_eval.out"
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

python ${RM_INTERPRET_ROOT}/train_rm/evaluate_ots.py \
  --model-name Skywork/Skywork-Reward-V2-Llama-3.1-8B \
  --data-path ${RM_INTERPRET_ROOT}/wimhf/data/arena_human_preference_55k_wimhf.jsonl \
  --batch-size 4 \
  --max-length 16384 \
  --output-path ${RM_INTERPRET_ROOT}/train_rm/outputs/skywork_reward_v2_llama3_1_8b/eval_results.jsonl
