#!/bin/bash
# run_multi_childguard.sh
# Iterates through selected models for ChildGuard fine-tuning and interpretability

# Models to test
MODELS=(
    # "unsloth/Llama-3.2-1B"
    # "microsoft/phi-1_5"
    "unsloth/Qwen2.5-1.5B"
    "unsloth/gemma-3-1b-it"

    "meta-llama/Llama-3.2-3B"
    "tiiuae/Falcon3-3B-Instruct"
    "ibm-granite/granite-3.1-3b-a800m-instruct"
    "stabilityai/stablelm-zephyr-3b"
    "ibm-research/PowerLM-3b"

    "tiiuae/Falcon3-7B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "nvidia/AceInstruct-7B"
    "Intel/neural-chat-7b-v3-1"
    "berkeley-nest/Starling-LM-7B-alpha"
    "mistralai/Mistral-7B-Instruct-v0.3"
)

# Output root directory (fixed name for resumability)
OUTPUT_ROOT="interpretability_lib_childguard/outputs/family_comparison"
mkdir -p "$OUTPUT_ROOT"

# Ensure logs directory exists
mkdir -p "interpretability_lib_childguard/logs"

echo "Starting ChildGuard Multi-Family Pipeline..."
echo "Results will be saved to: $OUTPUT_ROOT"

for MODEL in "${MODELS[@]}"; do
    MODEL_NAME_BASE=$(echo $MODEL | cut -d'/' -f2)
    MODEL_OUT_DIR="$OUTPUT_ROOT/$MODEL_NAME_BASE"
    DONE_MARKER="$MODEL_OUT_DIR/.completed"
    
    # Skip already-completed models for resumability
    if [ -f "$DONE_MARKER" ]; then
        echo "[SKIP] $MODEL already completed. Remove $DONE_MARKER to re-run."
        continue
    fi
    
    echo "============================================================"
    echo "PROCESSING MODEL: $MODEL"
    echo "============================================================"
    
    python3 run_pipeline_childguard.py \
        --model-name "$MODEL" \
        --output-dir "$MODEL_OUT_DIR" \
        --data-path "interpretability_lib_childguard/data/ChildGuard/ChildGuard.csv" \
        --train-size 50000 \
        --eval-sample-size 50 \
        --epochs 3.0 \
        --learning-rate 5e-5 \
        --lora-r 32 \
        --lora-alpha 64 \
        --finetune \
        --load-in-4bit \
        --run-xai \
        --extended-xai \
        --run-shift-analysis \
        --bias-sample-size 100
    
    if [ $? -eq 0 ]; then
        touch "$DONE_MARKER"
        echo "[DONE] $MODEL completed successfully."
    else
        echo "[ERROR] Pipeline failed for $MODEL. Will retry on next run."
    fi
done

echo "ChildGuard Pipeline Complete!"
