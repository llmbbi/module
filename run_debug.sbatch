#!/bin/bash
#SBATCH --job-name=xai-debug
#SBATCH --partition=coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/debug_%j.out
#SBATCH --error=logs/debug_%j.err

# Manual environment activation
export CONDA_PREFIX=/storage/ice1/6/3/jyoon370/miniconda3/envs/llm1b
export PATH=$CONDA_PREFIX/bin:$PATH

python debug_xai_methods.py
