#!/bin/bash
#SBATCH --job-name=debug-qwen
#SBATCH --partition=coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=logs/debug_qwen.out
#SBATCH --error=logs/debug_qwen.err
#SBATCH --nodelist=atl1-1-01-005-5-0

export CONDA_PREFIX=/storage/ice1/6/3/jyoon370/miniconda3/envs/llm1b
export PATH=$CONDA_PREFIX/bin:$PATH
module load cudnn/9.2.0.82-12-cuda

python /home/hice1/jyoon370/scratch/llm-interpretability/debug_qwen.py
