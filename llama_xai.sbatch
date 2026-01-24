#!/bin/bash
#SBATCH --job-name=llama-final
#SBATCH --partition=coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Manual environment activation (Workaround for broken conda)
export CONDA_PREFIX=/storage/ice1/6/3/jyoon370/miniconda3/envs/llm1b
export PATH=$CONDA_PREFIX/bin:$PATH
module load cudnn/9.2.0.82-12-cuda

python /storage/ice1/6/3/jyoon370/llm-interpretability/llama_3.2_1B_xai_3.py \
    --model-name unsloth/Llama-3.2-1B-Instruct  \
    --train-size 0 \
    --epochs 1 \
    --output-dir outputs/run_$(date +%m%d_%H%M) \
    --finetune \
    --run-lime \
    --run-xai \
    --load-in-4bit
