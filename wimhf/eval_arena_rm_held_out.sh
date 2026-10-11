#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --account=xlab
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-gpu=5
#SBATCH --gres=gpu:4
#SBATCH --time=13:00:00
#SBATCH --job-name=eval_arena_rm_held_out
#SBATCH --output="eval_arena_rm_held_out.out"
RM_INTERPRET_ROOT="${RM_INTERPRET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

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

PYTHONPATH=. python scripts/run_wimhf.py configs/arena_human_preference_55k_qwen3_8b_rm_held_out.json --output-dir outputs/arena_human_preference_55k_qwen3_8b_rm_held_out
