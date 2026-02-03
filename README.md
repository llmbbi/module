# LLM Interpretability: Zero-Shot vs. Fine-Tuned Comparison

## Overview

This repository contains a comprehensive framework for comparing the interpretability of large language models (LLMs) across zero-shot and task-specific fine-tuned configurations. Using the Llama 3.2 1B model, we perform token-level attributional analysis through SHAP (SHapley Additive exPlanations) and provide extensive evaluation metrics including model performance and interpretability statistics.

The implementation is designed to be accessible and reproducible, with Colab-optimized code that can be executed on free-tier GPU resources.

## Key Features

- **Binary Classification Experiments**: Token-level interpretability analysis on the The Stanford Sentiment Treebank (https://huggingface.co/datasets/stanfordnlp/sst2)
- **Dual-Path Evaluation**: Direct comparison of zero-shot model behavior versus LoRA-adapted fine-tuned variants
- **SHAP-Based Attribution**: Token-level explanation extraction with statistical aggregation
- **Comprehensive Metrics**: Performance evaluation (accuracy, precision, recall, F1), interpretability metrics (sparsity, entropy, agreement), and effect size analysis
- **LoRA Adapter Support**: Lightweight fine-tuning with persistent adapter storage

## Interpretability Methods

This implementation references three major approaches for model explanation:

- **LIME (Local Interpretable Model-agnostic Explanations)**: A perturbation-based technique that trains local surrogate models to identify feature importance. While model-agnostic and easy to apply to text, it can be sensitive to sampling strategies

- **KernelSHAP**: Approximates Shapley values using a weighted kernel that emphasizes coalitions near the original input. It inherits desirable properties (local accuracy, missingness, consistency) while remaining practical for deep networks through sampling.
  
- **ROAR (Remove And Retrain)**: A direct faithfulness evaluation method that measures how much model accuracy drops when top-ranked features are iteratively removed and the model is retrained. While computationally expensive for large models, ROAR provides a strong empirical measure of explanation fidelity.

- **Integrated Gradients (IG)**: Accumulates gradients from baseline (masked/zero embeds) to input via Riemann integration (20 steps). Captures full attribution paths satisfying completeness/sensitivity axioms. 
  
- **IG Occlusion**: Hybrid—uses IG attribution map to guide occlusion mask placement, computing faithfulness via masked logit importance (steps=5, baseline='zeros'). 
  
For classification evaluation, we complement accuracy-based metrics with the **Matthews Correlation Coefficient (MCC)**, which provides a balanced score (−1 to +1) that remains informative under class imbalance. An MCC of +1 indicates perfect predictions, 0 matches random guessing, and −1 reflects complete disagreement.

Multi-Family Experiments
The multi-family setup is intended for sweeping interpretability and bias analyses over multiple models and datasets in a single run. It reuses the modular pipeline while varying configuration via shell and SLURM wrappers.
​
Entry points
run_multi_families.sh: Local launcher for running multiple model families and configurations in sequence (e.g., different --model-name, dataset splits, or XAI settings).
​
run_multi_families.sbatch: SLURM submission script that mirrors the shell launcher but targets the ICE/PACE GPUs for large-scale sweeps.

run_pipeline_modular.py: Core Python entry point; each job invoked by the multi-family scripts wraps this script with different flags.

# Local multi-family sweep (example)
```bash
bash run_multi_families.sh
```

Each sub-run calls:
```bash
python run_pipeline_modular.py \
  --model-name <MODEL_NAME> \
  --output-dir outputs/<EXPERIMENT_TAG> \
  --train-size <N_TRAIN> \
  --eval-sample-size <N_EVAL> \
  --epochs 1.0 \
  --finetune \
  --load-in-4bit \
  --run-xai
```

Dynamically Computed Properties

These scores are calculated during execution based on the specific Llama-3.2-1B model and SST-2 dataset: F1.1 Scope: Measures the diversity of features selected across the evaluation set. F1.4 Practicality: Calculated as the average of F11 (Speed) and F1.3 (Access). F3 Selectivity: Measures feature selectivity using Gini coefficient based on LIME/SHAP weights. F4.1 Contrastivity: Measures whether explanations contain both positive and negative contributions. F4.2 Target Sensitivity: Measures divergence in LIME explanations between opposite class labels. F6.1 Fidelity Check: Measures how well LIME/SHAP approximations match the model's predicted probabilities (scaled 0-5). F6.2 Surrogate Agreement: Measures fidelity on the original scale (0-1) for the local linear approximation. F7.1 Incremental Deletion: Measures prediction drop when top features are iteratively masked from the input. F9.1 Similarity (Stability): Measures consistency of explanations under input perturbations. F9.2 Identity: Measures consistency of feature rankings across multiple explanation runs. F11 Speed: Measures execution time per explanation (scaled to 1-5).

## Modular Interpretability Pipeline (`run_pipeline_modular.py`)

This script provides a **single entry point** for running a modular interpretability and bias analysis pipeline on SST-2 using `unsloth/Llama-3.2-1B-Instruct`. It supports zero-shot evaluation, LoRA fine-tuning, LIME/SHAP attribution, functionally grounded XAI metrics, and bias analysis within one workflow. [github](https://github.com/llmbbi/module/blob/main/run_pipeline_modular.py)

### Functionality

`run_pipeline_modular.py` performs the following steps: [github](https://github.com/llmbbi/module/blob/main/run_pipeline_modular.py)

- Loads SST-2 (`stanfordnlp/sst2`) and evaluates the base model on the validation split (zero-shot).  
- Computes LIME and SHAP explanations on sampled validation examples and aggregates them into XAI property scores (faithfulness, contrastivity, stability, etc.).  
- Runs bias analysis by inspecting attributions on sentences containing gendered tokens (e.g., “he”, “she”, “woman”, “man”).  
- Optionally fine-tunes the model with LoRA on the SST-2 train split and repeats performance, XAI, and bias analysis for the fine-tuned model.  
- Saves a consolidated JSON results file and comparison plots (zero-shot vs fine-tuned). [github](https://github.com/llmbbi/module/blob/main/run_pipeline_modular.py)

### Arguments

Run:

```bash
python run_pipeline_modular.py [FLAGS...]
```

Core flags: [github](https://github.com/llmbbi/module/blob/main/run_pipeline_modular.py)

- `--model-name`  
  Hugging Face model name, default: `unsloth/Llama-3.2-1B-Instruct`.

- `--output-dir`  
  Directory to store all results and plots, default: `outputs/modular_pipeline_fixed`.

- `--train-size`  
  Number of training samples from SST-2 `train` split.  
  - `> 0`: subsample this many examples.  
  - `<= 0`: use the full training split.

- `--eval-sample-size`  
  Number of validation examples used for attribution/XAI metrics, default: `50`. [github](https://github.com/llmbbi/module/blob/main/run_pipeline_modular.py)

- `--epochs`  
  Number of epochs for LoRA fine-tuning, default: `1.0`. [github](https://github.com/llmbbi/module/blob/main/run_pipeline_modular.py)

- `--finetune`  
  Enable LoRA fine-tuning and post-finetuning evaluation.  
  Flag is `store_true` with default `True`, so fine-tuning runs unless you pass `--no-finetune`. [github](https://github.com/llmbbi/module/blob/main/run_pipeline_modular.py)

- `--max-seq-length`  
  Maximum sequence length for tokenization, default: `512`. [github](https://github.com/llmbbi/module/blob/main/run_pipeline_modular.py)

- `--load-in-4bit`  
  Load base model in 4-bit for memory efficiency.  
  `store_true`, default `True`. [github](https://github.com/llmbbi/module/blob/main/run_pipeline_modular.py)

- `--run-xai`  
  Compute LIME/SHAP-based XAI metrics (zero-shot and, if enabled, fine-tuned).  
  `store_true`, default `True`. [github](https://github.com/llmbbi/module/blob/main/run_pipeline_modular.py)

### Example commands

Full pipeline (zero-shot + LoRA fine-tuning + XAI + bias):

```bash
python run_pipeline_modular.py \
  --model-name unsloth/Llama-3.2-1B-Instruct \
  --output-dir outputs/modular_pipeline_full \
  --train-size 2000 \
  --eval-sample-size 50 \
  --epochs 1.0 \
  --finetune \
  --load-in-4bit \
  --run-xai
```

Zero-shot only (no fine-tuning):

```bash
python run_pipeline_modular.py \
  --model-name unsloth/Llama-3.2-1B-Instruct \
  --output-dir outputs/modular_zero_shot_only \
  --train-size 0 \
  --eval-sample-size 50 \
  --epochs 1.0 \
  --load-in-4bit \
  --run-xai \
  --no-finetune
```

Quick diagnostic (zero-shot, no XAI metrics):

```bash
python run_pipeline_modular.py \
  --output-dir outputs/modular_quick \
  --no-finetune \
  --no-run-xai
```

### Outputs

All artifacts are stored in `--output-dir`: [github](https://github.com/llmbbi/module/blob/main/run_pipeline_modular.py)

- `modular_pipeline_results.json`  
  Contains:
  - `zero_shot_performance`, `finetuned_performance` (accuracy, precision, recall, F1, MCC)  
  - `zero_shot_attributions`, `finetuned_attributions` (XAI property scores for LIME and SHAP)  
  - `zero_shot_bias`, `finetuned_bias` (bias metrics from demographic-term analysis). [github](https://github.com/llmbbi/module/blob/main/run_pipeline_modular.py)

- `model_performance_comparison.png`  
  Bar chart comparing zero-shot vs fine-tuned metrics. [github](https://github.com/llmbbi/module/blob/main/run_pipeline_modular.py)

- `xai_properties_comparison_zero-shot.png`  
- `xai_properties_comparison_fine-tuned.png`  
  Horizontal bar charts for XAI properties (e.g., Scope, Selectivity, Fidelity, Deletion, Sufficiency, Similarity, Identity, Monotonicity, Compactness, Speed). [github](https://github.com/llmbbi/module/blob/main/run_pipeline_modular.py)

Bias analysis currently focuses on gendered tokens, but the same pattern can be extended in `detect_bias` to capture other demographic categories (race, age, etc.). [github](https://github.com/llmbbi/module/blob/main/run_pipeline_modular.py)
## Getting Started

### Prerequisites
- Python 3.8+
- GPU with at least 8GB VRAM (16GB recommended for fine-tuning)
- Hugging Face account and authentication token (for gated models)

### Installation

```bash
git clone https://github.com/jamesjyoon/llm_interpretability.git
cd llm_interpretability
pip install -r requirements.txt
```

   
## Output Artifacts

The pipeline generates the following outputs in the specified `--output-dir`:

### Performance Metrics
- **`zero_shot_metrics.json`** / **`fine_tuned_metrics.json`**: Accuracy, precision, recall, F1 scores (per-class and aggregated), confusion matrix, and probability distributions
- **`metrics_comparison.png`**: Side-by-side visualization of zero-shot vs. fine-tuned performance

### Interpretability Analysis  
- **`zero_shot_shap.json`** / **`fine_tuned_shap.json`**: Serialized SHAP explanations for all evaluated examples
- **`zero_shot_shap_summary.png`** / **`fine_tuned_shap_summary.png`**: Aggregated token importance visualizations
- **`interpretability_metrics.json`**: Comparative statistics including:
  - Average absolute token importance and sparsity (Gini coefficients)
  - Entropy of attribution distributions
  - Top-5 and top-10 token overlap rates
  - Cosine and Spearman correlation of explanations
  - Per-example mean-importance deltas and Cohen's d effect size
  - Per-example SHAP summaries for downstream analysis

### Model Artifacts
- **`lora_adapter/`**: Trained LoRA adapter weights for reuse via `PeftModel.from_pretrained()`

## Configuration Options

### Model Selection
- `--model-name` (default: `meta-llama/Llama-3.2-1B`): Model checkpoint to use. Swap to an open-access alternative if preferred.

### Data Configuration
- `--train-subset` (default: 2000): Number of training examples to sample
- `--eval-subset` (default: 1000): Number of evaluation examples to sample
- `--train-split` (default: `train`): Training data split name
- `--eval-split` (default: `test`): Evaluation data split name
- `--text-field` (default: `text`): Column name for input text
- `--label-field` (default: `label`): Column name for labels
- `--label-space`: Explicitly specify numeric labels to include (e.g., to exclude neutral class)

### Experiment Options
- `--finetune`: Enable LoRA fine-tuning on the training subset
- `--run-shap`: Extract SHAP attributions (disable with `--no-run-shap` for faster runs)
- `--load-in-4bit` (default: True): Use 4-bit quantization. Disable with `--no-load-in-4bit` for full precision on high-memory GPUs
- `--output-dir`: Directory for output artifacts
- `--huggingface-token`: Hugging Face authentication token for gated models

## Notes

- **Model Access**: The default Llama 3.2 1B model is gated. [Request access](https://huggingface.co/meta-llama/Llama-3.2-1B) and authenticate via token before running.
- **Memory Efficiency**: By default, we sample 2,000 training and 1,000 evaluation examples to maintain Colab compatibility. Adjust `--train-subset` and `--eval-subset` as needed, or set to `None` for the full dataset.
- **Fine-Tuning Details**: During fine-tuning, instruction tokens are masked with `-100` so the loss only supervises the appended label token, keeping LoRA updates focused on classification decisions.
- **Output Visualization**: Charts render inline in Colab notebooks when available; otherwise they are saved as PNG files.

## Implementation Details: llama_3.2_1B_roar.py

The `llama_3.2_1B_roar.py` script implements Remove And Retrain (ROAR) based faithfulness analysis for the Llama 3.2 1B model. ROAR is a direct faithfulness evaluation method that measures the impact of removing important features on model performance.

### Key Capabilities

- **ROAR Faithfulness Evaluation**: Iteratively removes top-ranked features and retrains the model to measure accuracy drops
- **Attribution Integration**: Uses LIME and KernelSHAP feature importances as the basis for removal ranking
- **Dual Path Evaluation**: Evaluates both zero-shot and fine-tuned models
- **Metric Aggregation**: Computes fidelity scores by analyzing performance degradation curves

### Script Arguments

- `--model-name`: Llama model checkpoint (default: `meta-llama/Llama-3.2-1B`)
- `--output-dir`: Directory for saving ROAR analysis results
- `--train-size`: Number of training samples for retraining (default: 1000)
- `--eval-sample-size`: Number of evaluation samples for ROAR computation (default: 50)
- `--finetune`: Enable LoRA fine-tuning before ROAR analysis
- `--load-in-4bit`: Use 4-bit quantization for memory efficiency
- `--huggingface-token`: Authentication token for gated models

### Output Files

- `roar_metrics.json`: ROAR faithfulness scores for LIME and KernelSHAP
- `roar_comparison.png`: Visualization of feature removal impact on accuracy
- `roar_analysis_results.json`: Detailed results including per-feature removal metrics

## Implementation Details: llama_3.2_1B_xai.py

The `llama_3.2_1B_xai.py` script provides a focused implementation for XAI (Explainable AI) property analysis on the Llama 3.2 1B model. This file is specifically designed for detailed evaluation of model interpretability across zero-shot and fine-tuned configurations using functionally-grounded explanations.

### Key Capabilities

- **Functionally-Grounded Properties (F1-F11)**: Computes interpretability metrics based on the framework by Mohseni et al., including scope, structure, selectivity, contrastivity, interactivity, fidelity, faithfulness, rationality, alignment, uncertainty, and speed.
- **LIME Explainer Integration**: Local Interpretable Model-agnostic Explanations for feature-level attribution analysis.
- **KernelSHAP Integration**: Kernel-based Shapley value approximation for robust feature importance estimation.
- **Standalone Execution**: Can be run independently or as part of the full pipeline for targeted XAI analysis.

### Script Arguments

- `--model-name`: Llama model checkpoint (default: `meta-llama/Llama-3.2-1B`)
- `--output-dir`: Directory for saving metrics and visualizations
- `--train-size`: Number of training samples for fine-tuning (default: 1000)
- `--eval-sample-size`: Number of evaluation samples for XAI property computation (default: 50)
- `--epochs`: Number of fine-tuning epochs (default: 1.0)
- `--finetune`: Enable LoRA fine-tuning
- `--run-xai`: Compute functionally-grounded XAI properties (default: True)
- `--load-in-4bit`: Use 4-bit quantization for memory efficiency
- `--huggingface-token`: Authentication token for gated models

### Output Files

- `metrics_comparison.json`: Performance metrics (accuracy, precision, recall, F1, MCC) for both zero-shot and fine-tuned models
- `xai_properties_results.json`: Functionally-grounded property scores for LIME and KernelSHAP
- `model_performance_comparison.png`: Bar chart comparing zero-shot vs. fine-tuned performance
- `xai_properties_comparison_*.png`: Horizontal bar charts showing property scores across explainability methods
- `ft_example_lime_*.png`: Visual LIME explanations for sample fine-tuned model predictions

## Running on Georgia Tech ICE PACE Cluster

The ICE (Interactive Cluster Environment) and PACE (Partnership for an Advanced Computing Environment) clusters at Georgia Tech provide high-performance GPU resources ideal for model training and interpretation analysis.

### ICE PACE Setup

1. **SSH into the cluster**:
   ```bash
   ssh <gt_username>@ice.cc.gatech.edu  # or ice-login.cc.gatech.edu
   ```

2. **Load necessary modules**:
   ```bash
   module load anaconda3/2023.09
   module load cuda/12.2.0
   module load cudnn/8.9.4
   ```

3. **Set up Python environment**:
   ```bash
   conda create -n llm_interpretability python=3.10
   conda activate llm_interpretability
   ```

4. **Clone repository and install dependencies**:
   ```bash
   git clone https://github.com/jamesjyoon/llm_interpretability.git
   cd llm_interpretability
   pip install -r requirements.txt
   ```

### Submitting a Job via SLURM

Create a submission script (e.g., `llama_xai.sbatch`):

```bash
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

source /storage/ice1/6/3/jyoon370/miniconda3/etc/profile.d/conda.sh
conda activate llm1b

python /storage/ice1/6/3/jyoon370/llm-interpretability/llama_3.2_1B_xai.py and llama_3.2_1B_xai_2.py \
    --model-name meta-llama/Llama-3.2-1B \
    --train-size -1 \
    --epochs 3 \
    --output-dir outputs/run_$(date +%m%d_%H%M) \
    --finetune \
    --run-lime \
    --run-xai \
    --load-in-4bit
    --huggingface-token <your_hf_token>
```

Submit the job:
```bash
sbatch llama_xai.sbatch
```

### Monitoring Jobs

```bash
# Check job status
squeue -u <gt_username>

# View job details
scontrol show job <job_id>

# Cancel a job
scancel <job_id>
```

### Transferring Results

Transfer results from PACE to local machine:
```bash
scp -r <gt_username>@ice.cc.gatech.edu:/home/<gt_username>/llm_interpretability/outputs/pace_xai_analysis ./results_from_pace/
```

### PACE-Specific Considerations

- **GPU Allocation**: Request appropriate GPU resources (A100 for larger models, V100/H100 for standard tasks)
- **Memory Management**: Monitor memory usage with `nvidia-smi` in interactive sessions
- **Job Time Limits**: Default partition has 4-hour limit; longer jobs may need `--partition=gpu-long` (up to 24 hours)
- **Storage**: Store large datasets in `/scratch` directory for faster I/O; results in `$HOME` are automatically backed up
- **Data Transfer**: For large models/datasets, use `rsync` or `globus` instead of `scp` for improved performance

### Example: Interactive GPU Session

For development and debugging:
```bash
interactive -p gpu -N 1 --gres=gpu:A100:1 -c 8 --mem=64G -t 120
```

Then run the script directly:
```bash
python llama_3.2_1B_xai.py --model-name meta-llama/Llama-3.2-1B --output-dir outputs/debug_run --train-size 100 --eval-sample-size 10
```

### Troubleshooting on PACE

- **CUDA out of memory**: Reduce `--train-size` or `--eval-sample-size`, or enable `--load-in-4bit`
- **Module not found**: Verify all modules are loaded and environment is activated
- **Slow I/O**: Use `/scratch` for intermediate files instead of `$HOME`
- **Job timeout**: Increase `--time` in SLURM header or reduce model size/data volume

## Citations

This project utilizes the Functionally-Grounded Evaluation framework to assess XAI methods (LIME and KernelSHAP), focusing on proxy tasks and quantitative metrics rather than human-subject studies to evaluate explanation quality.

If you use this code, please cite the foundational work on this evaluation strategy:

```bibtex
@article{doshi2017towards,
  title={Towards a rigorous science of interpretable machine learning},
  author={Doshi-Velez, Finale and Kim, Been},
  journal={arXiv preprint arXiv:1702.08608},
  year={2017}
}

@article{nauta2023anecdotal,
  title={From anecdotal evidence to quantitative evaluation methods: A systematic review on evaluating explainable AI},
  author={Nauta, Meike and Trienes, Jan and Pathak, Shreyasi and Nguyen, Elisa and Peters, Michelle and Schmitt, Yasmin and Schl{\"o}tterer, J{\"o}rg and van Keulen, Maurice and Seifert, Christin},
  journal={ACM Computing Surveys},
  volume={55},
  number={13s},
  pages={1--42},
  year={2023},
  publisher={ACM New York, NY}
}
```

If you use this work in your research, please cite:

```bibtex
@software{yoon2025llm_interpretability,
  author = {Yoon, James},
  title = {LLM Interpretability: Zero-Shot vs. Fine-Tuned Comparison},
  year = {2025},
  url = {https://github.com/jamesjyoon/llm_interpretability}
}
```

## Related Work

This project builds on foundational research in model interpretability and attribution methods:

- **SHAP/Shapley Values**: Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"
- **LIME**: Ribeiro et al. (2016) "'Why Should I Trust You?': Explaining the Predictions of Any Classifier"
- **LoRA**: Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"

## License

MIT License. See LICENSE file for details.

## Contact

For questions or feedback, please open an issue or contact [jamesjyoon](https://github.com/jamesjyoon).
