#!/bin/bash
#SBATCH --job-name=roar_exp
#SBATCH --partition=coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/roar_%j.out
#SBATCH --error=logs/roar_%j.err
#SBATCH --nodelist=atl1-1-01-005-5-0

# Activate environment (Manual Workaround for broken conda)
export CONDA_PREFIX=/storage/ice1/6/3/jyoon370/miniconda3/envs/llm1b
export PATH=$CONDA_PREFIX/bin:$PATH
module load cudnn/9.2.0.82-12-cuda

# Run the ROAR experiment
# We use train-size 500 to keep runtime reasonable (LIME generation is slow)
# We use mask-percentage 0.2 (Remove top 20% of words)

python /storage/ice1/6/3/jyoon370/llm-interpretability/llama_3.2_1B_roar.py \
    --model-name meta-llama/Llama-3.2-1B \
    --output-dir outputs/roar_results_$(date +%m%d_%H%M) \
    --train-size 500 \
    --mask-percentage 0.2
