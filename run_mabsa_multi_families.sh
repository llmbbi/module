#!/bin/bash
# run_mabsa_multi_families.sh
# Iterates through selected models for M-ABSA cross-family comparison
#
# Usage:
#   bash run_mabsa_multi_families.sh                  # Fresh run
#   bash run_mabsa_multi_families.sh --resume <dir>   # Resume from existing output dir
#
# Models chosen for strong multilingual support across the 21 M-ABSA languages:
#   ar, da, de, en, es, fr, hi, hr, id, ja, ko, nl, pt, ru, sk, sv, sw, th, tr, vi, zh

# ── Multilingual-capable models (7B class, base/non-instruct) ──────────
MODELS=(
    # "Qwen/Qwen2.5-7B"                        # Best multilingual 7B; 29+ languages
    "meta-llama/Meta-Llama-3.1-8B"            # 8 official langs, strong cross-lingual transfer
    "mistralai/Mistral-7B-v0.3"               # Improved tokenizer over v0.1, better non-English
    "tiiuae/falcon-7b"                        # 30+ lang pre-training data
    # Older / less multilingual (kept for reference, uncomment to include):
    # "mistralai/Mistral-7B-v0.1"             # Weak multilingual (English-heavy tokenizer)
    # "meta-llama/Llama-2-7b-hf"              # Weak multilingual (English-heavy)
    # "Qwen/Qwen1.5-7B"                       # Good multilingual, superseded by Qwen2.5
)

# ── Language filter ────────────────────────────────────────────────────
LANG_FILTER="en,de,fr,es,nl,pt,ru,tr,ar,zh,ja,ko,hi,th,vi"

# ── Handle --resume flag ──────────────────────────────────────────────
if [ "$1" = "--resume" ] && [ -n "$2" ]; then
    OUTPUT_ROOT="$2"
    echo "RESUMING from existing output directory: $OUTPUT_ROOT"
else
    OUTPUT_ROOT="outputs/mabsa_family_comparison_$(date +%m%d_%H%M)"
fi
mkdir -p "$OUTPUT_ROOT"

echo "Starting M-ABSA Multi-Family Model Comparison Pipeline..."
echo "Results will be saved to: $OUTPUT_ROOT"

for MODEL in "${MODELS[@]}"; do
    MODEL_NAME_BASE=$(echo $MODEL | cut -d'/' -f2)
    MODEL_OUT_DIR="$OUTPUT_ROOT/$MODEL_NAME_BASE"

    # ── Skip completed models ─────────────────────────────────────
    RESULTS_FILE="$MODEL_OUT_DIR/mabsa_pipeline_results.json"
    if [ -f "$RESULTS_FILE" ]; then
        # Check if it has test_performance (= fully complete)
        if grep -q '"test_performance"' "$RESULTS_FILE" 2>/dev/null; then
            echo "============================================================"
            echo "SKIPPING MODEL (already complete): $MODEL"
            echo "  Results at: $RESULTS_FILE"
            echo "============================================================"
            continue
        else
            echo "============================================================"
            echo "RESUMING PARTIAL RUN: $MODEL"
            echo "============================================================"
        fi
    else
        echo "============================================================"
        echo "PROCESSING MODEL: $MODEL"
        echo "============================================================"
    fi

    # Build language filter argument
    LANG_ARG=""
    if [ -n "$LANG_FILTER" ]; then
        LANG_ARG="--languages $LANG_FILTER"
    fi

    python run_pipeline_mabsa.py \
        --model-name "$MODEL" \
        --output-dir "$MODEL_OUT_DIR" \
        --train-size 30000 \
        --val-size 500 \
        --test-size 500 \
        --eval-sample-size 50 \
        --lora-r 32 \
        --lora-alpha 64 \
        --lora-dropout 0.05 \
        --learning-rate 5e-5 \
        --epochs 10.0 \
        --max-seq-length 1024 \
        --balanced \
        --finetune \
        --run-xai \
        --extended-xai \
        --run-shift-analysis \
        --bias-sample-size 100 \
        $LANG_ARG

    if [ $? -ne 0 ]; then
        echo "[ERROR] M-ABSA Pipeline failed for $MODEL. Continuing to next model..."
    fi
done

echo "M-ABSA Multi-Family Model Comparison Complete!"