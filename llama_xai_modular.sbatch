#!/bin/bash
#SBATCH --job-name=llama-modular
#SBATCH --partition=coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodelist=atl1-1-01-005-5-0

# Manual environment activation (Workaround for broken conda)
export CONDA_PREFIX=/storage/ice1/6/3/jyoon370/miniconda3/envs/llm1b
export PATH=$CONDA_PREFIX/bin:$PATH
module load cudnn/9.2.0.82-12-cuda

# Paths
SCRATCH_BASE=${SLURM_TMPDIR:-/home/hice1/jyoon370/scratch}
RUN_ID=$(date +%m%d_%H%M)
SCRATCH_OUT_DIR="$SCRATCH_BASE/llm-interpretability/outputs/modular_run_${RUN_ID}"
STORAGE_OUT_ROOT=/storage/ice1/6/3/jyoon370/llm-interpretability/outputs
STORAGE_OUT_DIR="$STORAGE_OUT_ROOT/modular_run_${RUN_ID}"

mkdir -p "$SCRATCH_OUT_DIR" "$STORAGE_OUT_ROOT"

# Use the updated script from scratch workspace to ensure latest changes
SCRIPT_PATH=/home/hice1/jyoon370/scratch/llm-interpretability/run_pipeline_modular.py

echo "[INFO] Writing outputs to scratch: $SCRATCH_OUT_DIR"
python "$SCRIPT_PATH" \
    --model-name unsloth/Llama-3.2-1B-Instruct  \
    --train-size 0 \
    --eval-sample-size 50 \
    --epochs 1 \
    --output-dir "$SCRATCH_OUT_DIR" \
    --finetune \
    --load-in-4bit

STATUS=$?
echo "[INFO] Syncing outputs to storage: $STORAGE_OUT_DIR"
rsync -avh "$SCRATCH_OUT_DIR"/ "$STORAGE_OUT_DIR"/
exit $STATUS
