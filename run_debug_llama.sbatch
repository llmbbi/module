#!/bin/bash
#SBATCH --job-name=debug-llama
#SBATCH --partition=coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=logs/debug_llama.out
#SBATCH --error=logs/debug_llama.err
#SBATCH --nodelist=atl1-1-01-005-5-0

export CONDA_PREFIX=/storage/ice1/6/3/jyoon370/miniconda3/envs/llm1b
export PATH=$CONDA_PREFIX/bin:$PATH

python /home/hice1/jyoon370/scratch/llm-interpretability/debug_llama.py
