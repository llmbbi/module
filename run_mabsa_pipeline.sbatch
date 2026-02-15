#!/bin/bash
#SBATCH --job-name=mabsa-pipeline
#SBATCH --partition=coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=interpretability_lib_mabsa/logs/mabsa_pipeline.out
#SBATCH --error=interpretability_lib_mabsa/logs/mabsa_pipeline.err
#SBATCH --nodelist=atl1-1-01-005-5-0

export CONDA_PREFIX=/storage/ice1/6/3/jyoon370/miniconda3/envs/llm1b
export PATH=$CONDA_PREFIX/bin:$PATH

# Increase HuggingFace download timeout (default 10s is too short on cluster)
export HF_HUB_DOWNLOAD_TIMEOUT=120

cd /home/hice1/jyoon370/scratch/llm-interpretability

# Run M-ABSA multi-family comparison
bash run_mabsa_multi_families.sh