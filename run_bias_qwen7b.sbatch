#!/bin/bash
#SBATCH --job-name=bias-qwen7b
#SBATCH --partition=coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=logs/bias_qwen7b.out
#SBATCH --error=logs/bias_qwen7b.err
#SBATCH --nodelist=atl1-1-01-005-5-0

export CONDA_PREFIX=/storage/ice1/6/3/jyoon370/miniconda3/envs/llm1b
export PATH=$CONDA_PREFIX/bin:$PATH
export HF_HUB_DOWNLOAD_TIMEOUT=120

cd /home/hice1/jyoon370/scratch/llm-interpretability

OUTPUT_DIR="outputs/bias_qwen7b_$(date +%m%d_%H%M)"
ADAPTER_DIR="outputs/mabsa_family_comparison_0208_1038/Qwen2.5-7B/lora_adapters"
mkdir -p "$OUTPUT_DIR"

echo "Running bias detection (zero-shot + fine-tuned) for Qwen2.5-7B"
echo "Using existing adapters from: $ADAPTER_DIR"
echo "Output: $OUTPUT_DIR"

python run_pipeline_mabsa.py \
    --model-name "Qwen/Qwen2.5-7B" \
    --output-dir "$OUTPUT_DIR" \
    --load-adapter "$ADAPTER_DIR" \
    --train-size 10000 \
    --val-size 500 \
    --test-size 500 \
    --eval-sample-size 50 \
    --lora-r 8 \
    --lora-dropout 0.1 \
    --learning-rate 1e-4 \
    --epochs 5.0 \
    --balanced \
    --run-xai \
    --extended-xai \
    --run-shift-analysis \
    --bias-sample-size 100 \
    --languages "en,de,fr,es,nl,pt,ru,tr,ar,zh,ja,ko,hi,th,vi"

echo "Done!"
