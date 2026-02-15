#!/bin/bash
#SBATCH --job-name=childguard-pipeline
#SBATCH --partition=coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=interpretability_lib_childguard/logs/childguard_pipeline_%j.out
#SBATCH --error=interpretability_lib_childguard/logs/childguard_pipeline_%j.err
# #SBATCH --nodelist=atl1-1-01-005-5-0

export CONDA_PREFIX=/storage/ice1/6/3/jyoon370/miniconda3/envs/llm1b
export PATH=$CONDA_PREFIX/bin:$PATH

# Increase HuggingFace download timeout
export HF_HUB_DOWNLOAD_TIMEOUT=120

# Navigate to project root (assuming this script is in interpretability_lib_childguard/)
cd /home/hice1/jyoon370/scratch/llm-interpretability

# Create logs directory if it doesn't exist
mkdir -p interpretability_lib_childguard/logs

# Run the ChildGuard multi-family pipeline
bash run_multi_childguard.sh
