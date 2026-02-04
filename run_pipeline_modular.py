#!/usr/bin/env python3
"""
Modular Interpretability Pipeline (Enhanced)

Demonstrates using the interpretability_lib components for:
- Fine-tuning with LoRA
- Multi-method feature attribution (LIME, SHAP, Integrated Gradients, Attention Rollout, Occlusion)
- Extended faithfulness & stability metrics
- Bias detection with amplification analysis
- Before/after attribution shift analysis
"""

import unsloth
import warnings
# Suppress sklearn ConvergenceWarning from LIME/Lasso
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import argparse
import json
import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
import torch
from transformers import set_seed
from interpretability_lib import (
    LoRAFineTuner, 
    FeatureAttributor, 
    MetricsCalculator, 
    BiasDetector,
    AttributionShiftAnalyzer
)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="unsloth/Llama-3.2-1B")
    # Forcing default output-dir to ensure consistent comparison if not specified
    parser.add_argument("--output-dir", default="outputs/modular_pipeline_enhanced")
    parser.add_argument("--train-size", type=int, default=0, help="Number of training samples to use. >0 subsamples, 0 or negative = full dataset")
    parser.add_argument("--eval-sample-size", type=int, default=50)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--finetune", action="store_true", default=True)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--run-xai", action="store_true", default=True)
    parser.add_argument("--extended-xai", action="store_true", default=True, help="Run extended attribution methods (IG, Attention, Occlusion)")
    parser.add_argument("--run-shift-analysis", action="store_true", default=True, help="Run attribution shift analysis")
    parser.add_argument("--bias-sample-size", type=int, default=100, help="Maximum samples to use for bias analysis")
    parser.add_argument("--load-adapter", type=str, default=None, help="Path to existing LoRA adapter to load for fine-tuned evaluation")
    return parser.parse_args()


def evaluate_performance(predict_fn, data):
    """Evaluate model performance on dataset."""
    texts = data["sentence"]
    labels = data["label"]
    
    print("  Calculating predictions...")
    probs_list = []
    batch_size = 16
    for i in tqdm(range(0, len(texts), batch_size), desc="Performance Eval"):
        batch_probs = predict_fn(texts[i:i+batch_size])[:, 1]
        probs_list.extend(batch_probs)
        
    probs = np.array(probs_list)
    preds = (probs > 0.5).astype(int)
    
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    mcc = matthews_corrcoef(labels, preds)
    
    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "mcc": round(float(mcc), 4)
    }


def plot_xai_properties(results, output_dir, phase_name=""):
    """Visualizes XAI properties in a horizontal bar chart similar to llama_3.2_1B_xai_3.py."""
    if not results: return
    
    methods = sorted(list(results.keys()))
    properties = sorted(list(results[methods[0]].keys()))
    
    y = np.arange(len(properties))
    height = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f']
    for i, method in enumerate(methods):
        vals = [results[method].get(p, 0) for p in properties]
        ax.barh(y + i*height - (len(methods)-1)*height/2, vals, height, label=method, color=colors[i % len(colors)], alpha=0.8)
    
    ax.set_xlabel("Score / Value")
    ax.set_title(f"Calculated XAI Properties: {phase_name} Model")
    ax.set_yticks(y)
    ax.set_yticklabels([p.replace('_', ' ') for p in properties], fontsize=9)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/xai_properties_comparison_{phase_name.lower().replace(' ', '_')}.png", dpi=300)
    plt.close()


def plot_performance_comparison(final_results, output_dir):
    """Plots model performance metrics for zero-shot and fine-tuned models."""
    metrics = ["accuracy", "precision", "recall", "f1", "mcc"]
    zs_perf = final_results.get("zero_shot_performance", {})
    ft_perf = final_results.get("finetuned_performance", {})
    
    if not zs_perf: return

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, [zs_perf.get(m, 0) for m in metrics], width, label="Zero-Shot", color="#1f77b4", alpha=0.8)
    
    if ft_perf:
        bars2 = ax.bar(x + width/2, [ft_perf.get(m, 0) for m in metrics], width, label="Fine-Tuned", color="#ff7f0e", alpha=0.8)
    
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    add_labels(bars1)
    if ft_perf:
        add_labels(bars2)
            
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_performance_comparison.png", dpi=300)
    plt.close()


def compute_attribution_metrics(attributor, metrics_calc, eval_data, sample_size, phase_name="", 
                                  extended_methods=True, shift_analyzer=None):
    """Compute comprehensive feature attribution and interpretability metrics."""
    print(f"\nComputing attribution metrics for {phase_name}...")
    
    safe_sample_size = min(sample_size, len(eval_data))
    indices = random.sample(range(len(eval_data)), safe_sample_size)
    sample_texts = [eval_data[i]["sentence"] for i in indices]
    sample_labels = [eval_data[i]["label"] for i in indices]
    
    predict_fn = attributor.get_predict_proba_fn()
    
    # Define attribution methods - baseline + extended
    methods = {
        "LIME": lambda t: attributor.explain_lime(t, predict_fn, num_features=5, num_samples=50),
        "SHAP": lambda t: attributor.explain_shap(t, predict_fn, nsamples=40)
    }
    
    # Add extended methods if requested
    if extended_methods:
        methods.update({
            "IntegratedGradients": lambda t: attributor.explain_integrated_gradients(t, steps=30),
            "Occlusion": lambda t: attributor.explain_occlusion(t, predict_fn)
        })
    
    raw_metrics = {m: {
        "faithfulness": [], "comprehensiveness": [], "sufficiency": [], 
        "gini": [], "monotonicity": [], "compactness": [], "contrastivity": [],
        "time": [], "identity": [], "stability": [], "fidelity": []
    } for m in methods}
    
    lime_features_all = set()
    cross_method_agreement = []
    
    loop_limit = min(10, len(sample_texts))
    
    # Determine phase for shift analysis
    phase_key = "pre_finetune" if "zero" in phase_name.lower() else "post_finetune"
    
    for i, text in tqdm(enumerate(sample_texts[:loop_limit]), total=loop_limit, desc=f"Analyzing {phase_name}"):
        orig_prob = predict_fn([text])[0][1]
        text_words = text.split()
        sample_id = f"sample_{indices[i]}"
        
        explanations = {}
        for name, explain_fn in methods.items():
            start_time = time.time()
            explanation = explain_fn(text)
            elapsed = time.time() - start_time
            explanations[name] = explanation
            
            top_features = attributor.get_top_features(explanation, top_k=3)
            weights = [x[1] for x in explanation]
            
            if name == "LIME":
                lime_features_all.update([x[0] for x in explanation])
            
            m = metrics_calc.compute_all_metrics(text, predict_fn, orig_prob, weights, top_features)
            for k, v in m.items():
                raw_metrics[name][k].append(v)
            
            raw_metrics[name]["time"].append(elapsed)
            
            # Store attributions for shift analysis
            if shift_analyzer is not None:
                shift_analyzer.store_attributions(
                    phase=phase_key,
                    sample_id=sample_id,
                    text=text,
                    method=name,
                    attribution_list=explanation,
                    prediction_prob=orig_prob,
                    label=sample_labels[i] if i < len(sample_labels) else None
                )
            
            # Fidelity (Approximation quality)
            if name == "LIME":
                # LIME explainer doesn't directly give us the surrogate's prediction in this modular setup easily without re-running
                # But we can estimate it or just use the faithfulness as a proxy if we had to.
                # In llama_3.2_1B_xai_3.py, it uses exp.predict_proba[1]
                # For simplicity here, let's use a placeholder or stick to what we can get.
                raw_metrics[name]["fidelity"].append(m["faithfulness"]) # Placeholder for now
            else:
                # For SHAP, we can compute fidelity if we have the expected value, but we don't store it in FeatureAttributor.
                raw_metrics[name]["fidelity"].append(m["faithfulness"]) # Placeholder
            
            # Identity and Stability (Sub-samples)
            if i < 3:
                # Identity: explain same text again
                exp2 = explain_fn(text)
                feats1 = [x[0] for x in explanation]
                feats2 = [x[0] for x in exp2]
                raw_metrics[name]["identity"].append(metrics_calc.calculate_jaccard(feats1, feats2))
                
                # Stability: explain perturbed text
                perturbed_text = text + " "
                exp_p = explain_fn(perturbed_text)
                feats_p = [x[0] for x in exp_p]
                raw_metrics[name]["stability"].append(metrics_calc.calculate_jaccard(feats1, feats_p))

        # Cross-method agreement
        if "LIME" in explanations and "SHAP" in explanations:
            lime_feats = [x[0] for x in explanations["LIME"]]
            shap_feats = [x[0] for x in explanations["SHAP"]]
            cross_method_agreement.append(metrics_calc.calculate_feature_agreement(lime_feats, shap_feats))
    
    # Final scoring and mapping (similar to llama_3.2_1B_xai_3.py)
    def map_speed(t):
        if t < 0.5: return 5
        elif t < 2.0: return 4
        elif t < 5.0: return 3
        elif t < 10.0: return 2
        else: return 1

    final_results = {}
    for m in methods:
        final_results[m] = {
            "F1.1_Scope": round(min(len(lime_features_all) / 20 * 5, 5), 1) if m == "LIME" else 0, # Scope usually for LIME features discovered
            "F3_Selectivity": round(np.mean(raw_metrics[m]["gini"]) * 5, 2) if raw_metrics[m]["gini"] else 0,
            "F4.1_Contrastivity": round(np.mean(raw_metrics[m]["contrastivity"]) * 5, 2) if raw_metrics[m]["contrastivity"] else 0,
            "F6.1_Fidelity": round(np.mean(raw_metrics[m]["fidelity"]) * 5, 2) if raw_metrics[m]["fidelity"] else 0,
            "F7.1_Deletion": round(np.mean(raw_metrics[m]["faithfulness"]) * 10, 2) if raw_metrics[m]["faithfulness"] else 0,
            "F7.2_Comprehensiveness": round(np.mean(raw_metrics[m]["comprehensiveness"]) * 5, 2) if raw_metrics[m]["comprehensiveness"] else 0,
            "F7.3_Sufficiency": round(np.mean(raw_metrics[m]["sufficiency"]) * 5, 2) if raw_metrics[m]["sufficiency"] else 0,
            "F9.1_Similarity": round(np.mean(raw_metrics[m]["stability"]) * 5, 2) if raw_metrics[m]["stability"] else 0,
            "F9.2_Identity": round(np.mean(raw_metrics[m]["identity"]) * 5, 2) if raw_metrics[m]["identity"] else 0,
            "F10.1_Monotonicity": round(np.mean(raw_metrics[m]["monotonicity"]) * 5, 2) if raw_metrics[m]["monotonicity"] else 0,
            "F10.2_Compactness": round(np.mean(raw_metrics[m]["compactness"]) * 5, 2) if raw_metrics[m]["compactness"] else 0,
            "F11_Speed": map_speed(np.mean(raw_metrics[m]["time"])) if raw_metrics[m]["time"] else 0,
        }
        if cross_method_agreement:
            final_results[m]["F12_CrossMethodAgreement"] = round(np.mean(cross_method_agreement) * 5, 2)

    # Specific fix for SHAP Scope (can set it same as LIME if desired, or 0)
    if "SHAP" in final_results:
         final_results["SHAP"]["F1.1_Scope"] = final_results["LIME"]["F1.1_Scope"]

    return final_results


def detect_bias(attributor, eval_data, sample_size=100):
    """Analyze potential bias in feature attributions by specifically searching for demographic terms."""
    bias_results, _, _ = detect_bias_with_attributions(attributor, eval_data, sample_size)
    return bias_results


def detect_bias_with_attributions(attributor, eval_data, sample_size=100):
    """
    Analyze potential bias in feature attributions and return raw attributions.
    
    Returns:
        Tuple of (bias_results dict, list of attributions for shift analysis, list of raw texts)
    """
    print(f"\nDetecting bias in attributions (sample size limit: {sample_size})...")
    
    predict_fn = attributor.get_predict_proba_fn()
    bias_detector = BiasDetector()
    
    all_demographic_tokens = []
    for dimension in bias_detector.demographic_groups.values():
        for group_tokens in dimension.values():
            all_demographic_tokens.extend(group_tokens)
    
    # Filter for relevant sentences first
    relevant_indices = [i for i, item in enumerate(eval_data) 
                        if any(t in item["sentence"].lower().split() for t in all_demographic_tokens)]
    
    if not relevant_indices:
        print("  WARNING: No sentences with demographic tokens found in the evaluation set.")
        # Return empty results following the structure
        return bias_detector.analyze_multiple_groups([]), [], []

    # Take a sample from the relevant sentences
    actual_sample_size = min(sample_size, len(relevant_indices))
    selected_indices = random.sample(relevant_indices, actual_sample_size)
    sample_texts = [eval_data[i]["sentence"] for i in selected_indices]
    
    all_attributions = []
    print(f"  Analyzing {len(sample_texts)} sentences containing demographic terms...")
    for text in tqdm(sample_texts, desc="Bias Analysis"):
        lime_exp = attributor.explain_lime(text, predict_fn, num_features=50, num_samples=50)
        all_attributions.append(lime_exp)
    
    # Run multi-group analysis
    bias_results = bias_detector.analyze_multiple_groups(all_attributions)
    
    # Add WAT scores
    for dimension, groups in bias_detector.demographic_groups.items():
        if dimension in bias_results:
            for group_name, tokens in groups.items():
                wat_res = bias_detector.compute_wat_score(all_attributions, tokens)
                bias_results[dimension][f"wat_{group_name}"] = wat_res
    
    return bias_results, all_attributions, sample_texts


def main():
    args = parse_arguments()
    
    # If loading adapter, disable finetuning to prevent conflict
    if args.load_adapter:
        args.finetune = False
        
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(42)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("stanfordnlp/sst2")
    eval_data = dataset["validation"]
    
    # Initialize Fine-Tuner
    print("\n" + "="*50)
    print("INITIALIZING FINE-TUNER")
    print("="*50)
    
    finetuner = LoRAFineTuner(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length
    )
    
    # Load model (without LoRA for zero-shot evaluation)
    print("Loading model...")
    finetuner.load_model(load_in_4bit=args.load_in_4bit)
    
    # Enable inference mode for zero-shot evaluation
    finetuner.enable_inference_mode()
    
    # Initialize Attribution and Metrics
    attributor = FeatureAttributor(finetuner.model, finetuner.tokenizer)
    predict_fn = attributor.get_predict_proba_fn()
    metrics_calc = MetricsCalculator(predict_fn)
    
    final_results = {}
    
    # Zero-shot evaluation
    print("\n" + "="*50)
    print("ZERO-SHOT EVALUATION")
    print("="*50)
    
    zs_perf = evaluate_performance(predict_fn, eval_data)
    print(f"Zero-Shot Performance: {zs_perf}")
    final_results["zero_shot_performance"] = zs_perf
    
    # Initialize shift analyzer for before/after comparison
    shift_analyzer = None
    if args.run_shift_analysis and args.finetune:
        shift_analyzer = AttributionShiftAnalyzer(args.output_dir)
        print("  Attribution Shift Analyzer initialized for before/after comparison")
    
    # Store zero-shot attributions for bias shift analysis
    zs_attributions = []  # For bias shift report
    
    # Zero-shot attribution metrics
    if args.run_xai:
        zs_attrs = compute_attribution_metrics(
            attributor, metrics_calc, eval_data, args.eval_sample_size, "Zero-Shot",
            extended_methods=args.extended_xai, shift_analyzer=shift_analyzer
        )
        print(f"Zero-Shot XAI Properties: {zs_attrs}")
        final_results["zero_shot_attributions"] = zs_attrs
        plot_xai_properties(zs_attrs, args.output_dir, "Zero-Shot")
    
    # Bias detection (zero-shot) - collect attributions for shift report
    zs_bias, zs_attributions, _ = detect_bias_with_attributions(attributor, eval_data, sample_size=args.bias_sample_size)
    print(f"Zero-Shot Bias Analysis: {zs_bias}")
    final_results["zero_shot_bias"] = zs_bias
    
    # Fine-tuning or Loading Adapter
    if args.finetune or args.load_adapter:
        print("\n" + "="*50)
        print("FINE-TUNING / ADAPTER EVALUATION")
        print("="*50)
        
        if args.finetune:
            # Configure LoRA adapters for training (switches model to training mode)
            print("Configuring LoRA adapters...")
            finetuner.configure_lora()
            
            full_train_ds = dataset["train"].shuffle(seed=42)
            total_train = len(full_train_ds)
            if args.train_size > 0 and args.train_size < total_train:
                print(f"Subsampling training data to {args.train_size} of {total_train} samples...")
                train_ds = full_train_ds.select(range(args.train_size))
            else:
                print(f"Using FULL training dataset ({total_train} samples).")
                train_ds = full_train_ds
            
            finetuner.train(train_ds, epochs=args.epochs)
            
            # Free up memory after training
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            # Switch model to inference mode for evaluation
            print("Enabling inference mode after fine-tuning...")
            finetuner.enable_inference_mode()
        
        elif args.load_adapter:
            print(f"Loading existing adapter from: {args.load_adapter}")
            # Free up memory to reload logic cleanliness
            del finetuner.model
            del finetuner.tokenizer
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            # Re-initialize to load adapter
            finetuner = LoRAFineTuner(
                model_name=args.load_adapter,
                output_dir=args.output_dir,
                max_seq_length=args.max_seq_length
            )
            finetuner.load_model(load_in_4bit=args.load_in_4bit)
            finetuner.enable_inference_mode()
        
        # Fine-tuned evaluation
        print("\n" + "="*50)
        print("FINE-TUNED EVALUATION")
        print("="*50)
        
        # Reinitialize attributor with fine-tuned model
        attributor = FeatureAttributor(finetuner.model, finetuner.tokenizer)
        predict_fn = attributor.get_predict_proba_fn()
        metrics_calc = MetricsCalculator(predict_fn)
        
        ft_perf = evaluate_performance(predict_fn, eval_data)
        print(f"Fine-Tuned Performance: {ft_perf}")
        final_results["finetuned_performance"] = ft_perf
        
        # Fine-tuned attribution metrics
        if args.run_xai:
            ft_attrs = compute_attribution_metrics(
                attributor, metrics_calc, eval_data, args.eval_sample_size, "Fine-Tuned",
                extended_methods=args.extended_xai, shift_analyzer=shift_analyzer
            )
            print(f"Fine-Tuned XAI Properties: {ft_attrs}")
            final_results["finetuned_attributions"] = ft_attrs
            plot_xai_properties(ft_attrs, args.output_dir, "Fine-Tuned")
        
        # Bias detection (fine-tuned) - collect attributions for shift report
        ft_bias, ft_attributions, _ = detect_bias_with_attributions(attributor, eval_data, sample_size=args.bias_sample_size)
        print(f"Fine-Tuned Bias Analysis: {ft_bias}")
        final_results["finetuned_bias"] = ft_bias
        
        # Generate bias shift report
        if zs_attributions and ft_attributions:
            print("\n  Computing bias amplification analysis...")
            bias_detector = BiasDetector()
            bias_shift_report = bias_detector.generate_bias_shift_report(zs_attributions, ft_attributions)
            print(f"  Bias Shift Summary: {bias_shift_report.get('summary', {})}")
            final_results["bias_shift_report"] = bias_shift_report
        
        # Generate attribution shift report
        if shift_analyzer is not None and args.run_shift_analysis:
            print("\n" + "="*50)
            print("ATTRIBUTION SHIFT ANALYSIS")
            print("="*50)
            
            shift_report = shift_analyzer.generate_shift_report()
            print(f"Attribution Shift Analysis generated for {shift_report.get('num_samples', 0)} samples")
            
            if shift_report.get("method_reports"):
                for method, method_report in shift_report["method_reports"].items():
                    print(f"\n  {method}:")
                    print(f"    Rank Correlation: {method_report.get('rank_correlation_mean', 'N/A'):.3f}" 
                          if isinstance(method_report.get('rank_correlation_mean'), (int, float)) else f"    Rank Correlation: N/A")
                    print(f"    Top-3 Overlap: {method_report.get('top_3_overlap_mean', 'N/A'):.3f}"
                          if isinstance(method_report.get('top_3_overlap_mean'), (int, float)) else f"    Top-3 Overlap: N/A")
            
            final_results["attribution_shift_report"] = shift_report
            
            # Save attribution data (HDF5 or JSON)
            try:
                shift_analyzer.save_to_hdf5("attribution_shifts.h5")
            except Exception as e:
                print(f"  Note: Could not save to HDF5 ({e}), falling back to JSON")
                shift_analyzer.save_to_json("attribution_shifts.json")
    
    # Save results
    print("\n" + "="*50)
    print("SAVING RESULTS")
    print("="*50)
    
    # Generate aggregate performance plot
    plot_performance_comparison(final_results, args.output_dir)
    
    output_file = f"{args.output_dir}/modular_pipeline_results.json"
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=2, cls=NumpyEncoder)
    
    print(f"Results saved to: {output_file}")
    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
