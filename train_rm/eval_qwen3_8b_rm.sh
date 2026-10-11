#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --account=xlab
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=80G
#SBATCH --time=23:00:00
#SBATCH --job-name=qwen3_8b_rm_full_eval
#SBATCH --output="qwen3_8b_rm_full_eval.out"
RM_INTERPRET_ROOT="${RM_INTERPRET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source ~/.bashrc
conda deactivate 
conda activate rm_interpret

python evaluate.py --batch-size 4
