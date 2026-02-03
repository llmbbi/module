#!/bin/bash
# run_multi_families.sh
# Iterates through selected models for cross-family comparison

# Models: Llama 3.2 1B, Qwen 2.5 1.5B, Phi 1.5, DeepSeek 1.3B, Gemma 3 1B
MODELS=(
    # "unsloth/Llama-3.2-1B"
    # "unsloth/Qwen2.5-1.5B-Instruct"
    # "microsoft/phi-1_5"
    "deepseek-ai/deepseek-coder-1.3b-instruct"
    "unsloth/gemma-3-1b-it"
)

# Output root directory
OUTPUT_ROOT="outputs/family_comparison_$(date +%m%d_%H%M)"
mkdir -p "$OUTPUT_ROOT"

echo "Starting Multi-Family Model Comparison Pipeline..."
echo "Results will be saved to: $OUTPUT_ROOT"

for MODEL in "${MODELS[@]}"; do
    MODEL_NAME_BASE=$(echo $MODEL | cut -d'/' -f2)
    MODEL_OUT_DIR="$OUTPUT_ROOT/$MODEL_NAME_BASE"
    
    echo "============================================================"
    echo "PROCESSING MODEL: $MODEL"
    echo "============================================================"
    
    python run_pipeline_modular.py \
        --model-name "$MODEL" \
        --output-dir "$MODEL_OUT_DIR" \
        --train-size 0 \
        --eval-sample-size 50 \
        --epochs 1 \
        --finetune \
        --load-in-4bit \
        --run-xai \
        --extended-xai \
        --run-shift-analysis
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] Pipeline failed for $MODEL. Continuing to next model..."
    fi
done

echo "Multi-Family Model Comparison Complete!"
