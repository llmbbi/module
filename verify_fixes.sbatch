#!/bin/bash
#SBATCH --job-name=verify_fixes
#SBATCH --partition=coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/verify_fixes_%j.out
#SBATCH --error=logs/verify_fixes_%j.err

cd /home/hice1/jyoon370/scratch/llm-interpretability

# Manual environment activation (Workaround for broken conda)
export CONDA_PREFIX=/storage/ice1/6/3/jyoon370/miniconda3/envs/llm1b
export PATH=$CONDA_PREFIX/bin:$PATH
module load cudnn/9.2.0.82-12-cuda

# Use the direct path to the python executable in the environment
python run_pipeline_modular.py \
    --model-name unsloth/Llama-3.2-1B \
    --train-size 4 \
    --eval-sample-size 2 \
    --epochs 0.01 \
    --output-dir outputs/verification_fix
