#!/bin/bash
#SBATCH --job-name=multi-family-xai
#SBATCH --partition=coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=interpretability_lib_sst2/logs/multi_family_%j.out
#SBATCH --error=interpretability_lib_sst2/logs/multi_family_%j.err

# Manual environment activation (Workaround for broken conda)
export CONDA_PREFIX=/storage/ice1/6/3/jyoon370/miniconda3/envs/llm1b
export PATH=$CONDA_PREFIX/bin:$PATH
module load cudnn/9.2.0.82-12-cuda

# Ensure we are in the correct directory
cd /home/hice1/jyoon370/scratch/llm-interpretability

# Make the helper script executable
chmod +x run_multi_sst2.sh

# Run the multi-family batch script
./run_multi_sst2.sh
