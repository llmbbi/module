#!/bin/bash
#SBATCH --job-name=deepseek-bias
#SBATCH --partition=coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodelist=atl1-1-01-005-5-0

# Manual environment activation
export CONDA_PREFIX=/storage/ice1/6/3/jyoon370/miniconda3/envs/llm1b
export PATH=$CONDA_PREFIX/bin:$PATH
module load cudnn/9.2.0.82-12-cuda

# Paths
SCRIPT_PATH=/home/hice1/jyoon370/scratch/llm-interpretability/run_pipeline_modular.py
ADAPTER_PATH=/home/hice1/jyoon370/scratch/llm-interpretability/outputs/family_comparison_0202_2319/DeepSeek-R1-Distill-Qwen-1.5B/lora_adapters
OUTPUT_DIR=outputs/robust_bias_analysis/DeepSeek-R1-Distill-Qwen-1.5B

echo "[INFO] Starting Robust Bias Analysis for DeepSeek-R1-Distill-Qwen-1.5B"
echo "[INFO] Loading adapters from: $ADAPTER_PATH"
echo "[INFO] Output directory: $OUTPUT_DIR"

python "$SCRIPT_PATH" \
  --model-name unsloth/DeepSeek-R1-Distill-Qwen-1.5B \
  --load-adapter "$ADAPTER_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --bias-sample-size 500 \
  --run-shift-analysis

echo "[INFO] Job Complete"
