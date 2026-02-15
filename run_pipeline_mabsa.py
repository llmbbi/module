#!/usr/bin/env python3
"""
M-ABSA Interpretability Pipeline

Adapted from Modular Interpretability Pipeline for Multilingual Aspect-Based Sentiment Analysis (M-ABSA)
- Fine-tuning with LoRA
- Multi-method feature attribution (LIME, SHAP, Integrated Gradients, Attention Rollout, Occlusion)
- Extended faithfulness & stability metrics
- Bias detection with amplification analysis
- Before/after attribution shift analysis

Dataset: Multilingual-NLP/M-ABSA with 3 classes (0=negative, 1=neutral, 2=positive)
"""

import unsloth
import warnings
# Suppress sklearn ConvergenceWarning from LIME/Lasso
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Suppress torch dynamo/inductor errors (e.g. BLOOM alibi tensor fp32)
# so the model falls back to eager mode instead of crashing
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import argparse
import ast
import json
import os
import re
import random
import time
import unicodedata
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from datasets import load_dataset, Dataset, concatenate_datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef, confusion_matrix
import torch
from transformers import set_seed
from interpretability_lib_mabsa import (
    LoRAFineTunerMABSA as LoRAFineTuner,
    FeatureAttributorMABSA as FeatureAttributor,
    MetricsCalculatorMABSA as MetricsCalculator,
    BiasDetector,
    AttributionShiftAnalyzer
)

# Sentiment string -> integer mapping: 0=negative, 1=neutral, 2=positive
SENTIMENT_MAP = {"negative": 0, "neutral": 1, "positive": 2}

# All 21 M-ABSA languages (ISO 639-1)
ALL_MABSA_LANGS = ["ar", "da", "de", "en", "es", "fr", "hi", "hr", "id",
                   "ja", "ko", "nl", "pt", "ru", "sk", "sv", "sw", "th",
                   "tr", "vi", "zh"]


def detect_language(text):
    """Detect language of a text using Unicode script heuristics.

    Returns ISO 639-1 code or 'unk' if detection is uncertain.
    For Latin-script languages we fall back to langdetect if available,
    otherwise return 'latin' (the caller can then accept all Latin-script
    texts when filtering).
    """
    if not text or not text.strip():
        return "unk"

    # Count characters by Unicode script category
    script_counts = Counter()
    for ch in text:
        if ch.isspace() or ch in '####[]\'",.:;!?()-':
            continue
        cp = ord(ch)
        # CJK Unified Ideographs
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF:
            script_counts["cjk"] += 1
        # Japanese Hiragana / Katakana
        elif 0x3040 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF:
            script_counts["ja"] += 1
        # Korean Hangul
        elif 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF or 0x3130 <= cp <= 0x318F:
            script_counts["ko"] += 1
        # Arabic
        elif 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0xFB50 <= cp <= 0xFDFF:
            script_counts["ar"] += 1
        # Devanagari (Hindi)
        elif 0x0900 <= cp <= 0x097F:
            script_counts["hi"] += 1
        # Thai
        elif 0x0E00 <= cp <= 0x0E7F:
            script_counts["th"] += 1
        # Vietnamese diacritics (Latin + special marks)
        elif unicodedata.name(ch, '').startswith('LATIN') and any(
            unicodedata.combining(c) for c in unicodedata.normalize('NFD', ch)
        ):
            script_counts["vi_hint"] += 1
            script_counts["latin"] += 1
        # Cyrillic (Russian, etc.)
        elif 0x0400 <= cp <= 0x04FF:
            script_counts["cyrillic"] += 1
        # Latin
        elif unicodedata.category(ch).startswith('L'):
            script_counts["latin"] += 1

    if not script_counts:
        return "unk"

    dominant = script_counts.most_common(1)[0][0]

    # Non-Latin scripts can be mapped directly
    if dominant == "ko":
        return "ko"
    if dominant == "ja":
        return "ja"
    if dominant == "cjk":
        # CJK without kana/hangul => Chinese (could be ja without kana but unlikely)
        if script_counts.get("ja", 0) > 0:
            return "ja"
        return "zh"
    if dominant == "ar":
        return "ar"
    if dominant == "hi":
        return "hi"
    if dominant == "th":
        return "th"
    if dominant == "cyrillic":
        return "ru"  # M-ABSA only has Russian for Cyrillic

    # Latin-script: try langdetect if available
    if dominant == "latin":
        try:
            from langdetect import detect as _ld_detect
            detected = _ld_detect(text)
            # Map langdetect output to M-ABSA language codes
            if detected in ALL_MABSA_LANGS:
                return detected
            # Common aliases
            return {"nb": "da", "nn": "da"}.get(detected, detected)
        except Exception:
            return "latin"  # couldn't determine which Latin language

    return "unk"


def filter_dataset_by_languages(dataset, languages, quiet=False):
    """Filter a HuggingFace Dataset to keep only rows whose detected language
    is in the given set.  `languages` is a set of ISO 639-1 codes.

    If a language code 'latin' is in the set, all Latin-script texts whose
    specific language couldn't be determined are kept.
    """
    keep_indices = []
    lang_counts = Counter()
    sentences = dataset["sentence"]
    for idx, sent in enumerate(sentences):
        lang = detect_language(sent)
        lang_counts[lang] += 1
        if lang in languages:
            keep_indices.append(idx)
    if not quiet:
        print(f"  Language detection summary ({len(sentences)} texts):")
        for lang, cnt in lang_counts.most_common():
            marker = " [kept]" if lang in languages else " [filtered]"
            print(f"    {lang}: {cnt}{marker}")
        print(f"  Kept {len(keep_indices)}/{len(sentences)} samples")
    return dataset.select(keep_indices)


def parse_mabsa_entry(raw_text):
    """Parse a single M-ABSA entry of the form:
       <sentence>####[['aspect_term', 'aspect_category', 'sentiment'], ...]
    Returns (sentence, label) where label is 0/1/2.
    Entries with no aspects default to label 1 (neutral).
    When multiple aspects exist, the most common sentiment wins.
    """
    if "####" not in raw_text:
        return raw_text.strip(), 1  # no annotation -> neutral

    sentence, annotation_str = raw_text.rsplit("####", 1)
    sentence = sentence.strip().lstrip("\u200e")  # strip LRM marks

    try:
        aspects = ast.literal_eval(annotation_str.strip())
    except (ValueError, SyntaxError):
        return sentence, 1  # unparseable -> neutral

    if not aspects:  # empty list
        return sentence, 1

    sentiments = []
    for aspect in aspects:
        if isinstance(aspect, (list, tuple)) and len(aspect) >= 3:
            sent_str = aspect[2].lower().strip()
            if sent_str in SENTIMENT_MAP:
                sentiments.append(SENTIMENT_MAP[sent_str])

    if not sentiments:
        return sentence, 1  # no valid sentiments -> neutral

    # Most common sentiment label
    label = Counter(sentiments).most_common(1)[0][0]
    return sentence, label


def preprocess_mabsa_dataset(dataset_split):
    """Convert raw M-ABSA split into a Dataset with 'sentence' and 'label' columns."""
    sentences = []
    labels = []
    for row in dataset_split:
        sentence, label = parse_mabsa_entry(row["text"])
        if sentence:  # skip empty
            sentences.append(sentence)
            labels.append(label)

    print(f"  Parsed {len(sentences)} samples from {len(dataset_split)} raw entries")
    print(f"  Label distribution: {Counter(labels)}")
    return Dataset.from_dict({"sentence": sentences, "label": labels})


def stratified_sample(dataset, n, seed=42):
    """Return a stratified subsample of size `n` from `dataset`.
    Preserves the original class proportions (label column).
    If `n` >= len(dataset), returns the full dataset shuffled.
    """
    if n >= len(dataset):
        return dataset.shuffle(seed=seed)

    rng = random.Random(seed)
    labels = dataset["label"]
    label_indices = {}
    for idx, lab in enumerate(labels):
        label_indices.setdefault(lab, []).append(idx)

    selected = []
    # Allocate proportionally, at least 1 per class
    remaining = n
    alloc = {}
    for lab, idxs in label_indices.items():
        alloc[lab] = max(1, int(round(len(idxs) / len(labels) * n)))
    # Adjust to hit exactly n
    total_alloc = sum(alloc.values())
    if total_alloc > n:
        # Trim from largest class
        sorted_labs = sorted(alloc, key=lambda l: alloc[l], reverse=True)
        for lab in sorted_labs:
            if total_alloc <= n:
                break
            reduce = min(alloc[lab] - 1, total_alloc - n)
            alloc[lab] -= reduce
            total_alloc -= reduce
    elif total_alloc < n:
        sorted_labs = sorted(alloc, key=lambda l: len(label_indices[l]), reverse=True)
        for lab in sorted_labs:
            if total_alloc >= n:
                break
            add = min(len(label_indices[lab]) - alloc[lab], n - total_alloc)
            alloc[lab] += add
            total_alloc += add

    for lab, idxs in label_indices.items():
        k = min(alloc[lab], len(idxs))
        selected.extend(rng.sample(idxs, k))

    rng.shuffle(selected)
    return dataset.select(selected)


def balanced_sample(dataset, n, seed=42):
    """Return a class-balanced subsample of size `n` from `dataset`.
    Each class gets n // num_classes samples (oversampled if needed).
    """
    rng = random.Random(seed)
    labels = dataset["label"]
    label_indices = {}
    for idx, lab in enumerate(labels):
        label_indices.setdefault(lab, []).append(idx)

    num_classes = len(label_indices)
    per_class = n // num_classes

    selected = []
    for lab, idxs in label_indices.items():
        rng.shuffle(idxs)
        if len(idxs) >= per_class:
            selected.extend(rng.sample(idxs, per_class))
        else:
            # Oversample: repeat indices to reach per_class
            repeats = (per_class // len(idxs)) + 1
            pool = (idxs * repeats)[:per_class]
            selected.extend(pool)

    rng.shuffle(selected)
    return dataset.select(selected)


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
    parser.add_argument("--output-dir", default="interpretability_lib_mabsa/outputs/mabsa_pipeline")
    parser.add_argument("--train-size", type=int, default=30000, help="Number of stratified training samples (default 30000). 0 = full dataset")
    parser.add_argument("--val-size", type=int, default=500, help="Number of stratified validation samples (default 500). 0 = full validation split")
    parser.add_argument("--test-size", type=int, default=500, help="Number of stratified test samples for final evaluation (default 500). 0 = full test split")
    parser.add_argument("--balanced", action="store_true", default=False, help="Use class-balanced oversampling for training data")
    parser.add_argument("--eval-sample-size", type=int, default=50)
    parser.add_argument("--perf-eval-size", type=int, default=0, help="Max samples for performance evaluation. 0 = use val set as-is")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank (default 32)")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha (default 64)")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout (default 0.05)")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Fine-tuning learning rate (default 5e-5)")
    parser.add_argument("--epochs", type=float, default=10.0)
    parser.add_argument("--finetune", action="store_true", default=True)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--run-xai", action="store_true", default=True)
    parser.add_argument("--extended-xai", action="store_true", default=True, help="Run extended attribution methods (IG, Attention, Occlusion)")
    parser.add_argument("--run-shift-analysis", action="store_true", default=True, help="Run attribution shift analysis")
    parser.add_argument("--bias-sample-size", type=int, default=100, help="Maximum samples to use for bias analysis")
    parser.add_argument("--load-adapter", type=str, default=None, help="Path to existing LoRA adapter to load for fine-tuned evaluation")
    parser.add_argument("--languages", type=str, default=None,
                        help="Comma-separated ISO 639-1 codes to keep (e.g. 'en,de,fr,es,zh'). "
                             "Default: use all 21 M-ABSA languages.")
    return parser.parse_args()


def evaluate_performance_mabsa(predict_fn, data):
    """Evaluate model performance on preprocessed M-ABSA dataset with 3 classes.
    Expects data with 'sentence' and 'label' columns (already parsed).
    """
    texts = data["sentence"]
    labels = data["label"]

    print(f"  Evaluating on {len(texts)} samples")
    print(f"  Label distribution: {Counter(labels)}")

    print("  Calculating predictions...")
    all_preds = []
    batch_size = 16
    for i in tqdm(range(0, len(texts), batch_size), desc="Performance Eval"):
        batch_texts = texts[i:i+batch_size]
        batch_probs = predict_fn(batch_texts)  # Shape: (batch_size, num_classes)
        batch_preds = np.argmax(batch_probs, axis=1)
        all_preds.extend(batch_preds)

    preds = np.array(all_preds)

    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    mcc = matthews_corrcoef(labels, preds)

    # Per-class metrics and confusion matrix
    p_cls, r_cls, f1_cls, sup = precision_recall_fscore_support(
        labels, preds, labels=[0, 1, 2], average=None, zero_division=0
    )
    class_names = ["negative", "neutral", "positive"]
    print("  Per-class metrics:")
    print(f"    {'Class':>10}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Support':>7}")
    for i, name in enumerate(class_names):
        print(f"    {name:>10}  {p_cls[i]:6.3f}  {r_cls[i]:6.3f}  {f1_cls[i]:6.3f}  {int(sup[i]):>7}")

    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    print("  Confusion Matrix (rows=true, cols=predicted):")
    print(f"    {'':>10}  {'neg':>6}  {'neu':>6}  {'pos':>6}")
    for i, name in enumerate(class_names):
        print(f"    {name:>10}  {cm[i][0]:>6}  {cm[i][1]:>6}  {cm[i][2]:>6}")

    result = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "mcc": round(float(mcc), 4)
    }

    # ── Per-language breakdown ──────────────────────────────────────
    lang_labels_map = {}  # lang -> (true_labels, pred_labels)
    for idx, (txt, true_lab, pred_lab) in enumerate(zip(texts, labels, preds)):
        lang = detect_language(txt)
        lang_labels_map.setdefault(lang, ([], []))
        lang_labels_map[lang][0].append(true_lab)
        lang_labels_map[lang][1].append(int(pred_lab))

    if len(lang_labels_map) > 1:
        print("\n  Per-language performance:")
        print(f"    {'Lang':>6}  {'N':>5}  {'Acc':>6}  {'F1':>6}  {'MCC':>6}")
        per_lang_results = {}
        for lang in sorted(lang_labels_map.keys()):
            y_true, y_pred = lang_labels_map[lang]
            n = len(y_true)
            lang_acc = accuracy_score(y_true, y_pred)
            _, _, lang_f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0
            )
            lang_mcc = matthews_corrcoef(y_true, y_pred) if n >= 2 else 0.0
            print(f"    {lang:>6}  {n:>5}  {lang_acc:6.3f}  {lang_f1:6.3f}  {lang_mcc:6.3f}")
            per_lang_results[lang] = {
                "n": n, "accuracy": round(float(lang_acc), 4),
                "f1": round(float(lang_f1), 4), "mcc": round(float(lang_mcc), 4)
            }
        result["per_language"] = per_lang_results

    return result


def plot_xai_properties(results, output_dir, phase_name=""):
    """Visualizes XAI properties in a horizontal bar chart."""
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
    """Plots model performance metrics for zero-shot, fine-tuned, and test models."""
    metrics = ["accuracy", "precision", "recall", "f1", "mcc"]
    zs_perf = final_results.get("zero_shot_performance", {})
    ft_perf = final_results.get("finetuned_performance", {})
    test_perf = final_results.get("test_performance", {})

    if not zs_perf and not ft_perf and not test_perf:
        print("  [WARNING] No performance data found — skipping mabsa_model_performance_comparison.png")
        return

    # Determine how many bars and their positions
    phases = {}
    if zs_perf:
        phases["Zero-Shot"] = (zs_perf, "#1f77b4")
    if ft_perf:
        phases["Fine-Tuned"] = (ft_perf, "#ff7f0e")
    if test_perf:
        phases["Test"] = (test_perf, "#2ca02c")

    n_phases = len(phases)
    x = np.arange(len(metrics))
    width = 0.8 / max(n_phases, 1)
    fig, ax = plt.subplots(figsize=(10, 6))

    all_bars = []
    for i, (label, (perf, color)) in enumerate(phases.items()):
        offset = (i - (n_phases - 1) / 2) * width
        bars = ax.bar(x + offset, [perf.get(m, 0) for m in metrics], width, label=label, color=color, alpha=0.8)
        all_bars.append(bars)

    ax.set_ylabel("Score")
    ax.set_title("M-ABSA Model Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend()
    ax.set_ylim(0, 1.1)

    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    for bars in all_bars:
        add_labels(bars)

    plt.tight_layout()
    out_path = f"{output_dir}/mabsa_model_performance_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Performance comparison chart saved to: {out_path}")


def plot_bias_results(bias_results, output_dir, phase_name=""):
    """Generate a grouped bar chart for bias analysis results (per-method)."""
    if not bias_results:
        return

    # bias_results structure: {method: {dimension: {mean_mass_a, ...}}, "summary": ...}
    methods = [m for m in bias_results if m != "summary"]
    if not methods:
        return

    for method in methods:
        method_data = bias_results[method]
        dimensions = [d for d in method_data if d not in ("summary",)]
        if not dimensions:
            continue

        fig, axes = plt.subplots(1, len(dimensions), figsize=(6 * len(dimensions), 6))
        if len(dimensions) == 1:
            axes = [axes]

        colors_a = '#3498db'
        colors_b = '#e74c3c'

        for ax, dim in zip(axes, dimensions):
            data = method_data[dim]
            group_a = data.get('group_a_name', 'Group A')
            group_b = data.get('group_b_name', 'Group B')
            mass_a = data.get('mean_mass_a', 0)
            mass_b = data.get('mean_mass_b', 0)
            cohen_d = data.get('cohen_d', 0)
            p_val = data.get('p_value', 1.0)

            bars = ax.bar([group_a, group_b], [mass_a, mass_b], color=[colors_a, colors_b], alpha=0.8)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., h, f'{h:.4f}',
                        ha='center', va='bottom', fontsize=9)

            ax.set_title(f"{dim.title()}\nCohen's d={cohen_d:.3f}  p={p_val:.3f}", fontsize=10)
            ax.set_ylabel('Mean Attribution Mass')

        fig.suptitle(f"Bias Analysis ({method}): {phase_name}", fontsize=13, fontweight='bold')
        plt.tight_layout()
        fname = f"{output_dir}/bias_results_{method.lower()}_{phase_name.lower().replace(' ', '_')}.png"
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"  Bias chart saved to: {fname}")


def compute_attribution_metrics(attributor, metrics_calc, eval_data, sample_size, phase_name="",
                                  extended_methods=True, shift_analyzer=None):
    """Compute comprehensive feature attribution and interpretability metrics.
    Expects eval_data with 'sentence' and 'label' columns (already parsed).
    """
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
        orig_probs = predict_fn([text])[0]  # Shape: (3,) for 3 classes
        orig_pred = np.argmax(orig_probs)
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

            m = metrics_calc.compute_all_metrics(text, predict_fn, orig_probs[orig_pred], weights, top_features, attribution_list=explanation)
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
                    prediction_prob=orig_probs[orig_pred],
                    label=sample_labels[i] if i < len(sample_labels) else None
                )

            # Fidelity (Approximation quality)
            if name == "LIME":
                raw_metrics[name]["fidelity"].append(m["faithfulness"])
            else:
                raw_metrics[name]["fidelity"].append(m["faithfulness"])

            # Identity and Stability (Sub-samples)
            if i < 3:
                # Identity: Same input should give same explanation
                explanation2 = explain_fn(text)
                identity_score = attributor.compute_identity(explanation, explanation2)
                raw_metrics[name]["identity"].append(identity_score)

                # Stability: Similar inputs should give similar explanations
                if len(text_words) > 5:
                    # Remove one word and check stability
                    perturbed_text = " ".join(text_words[:-1])
                    explanation_perturbed = explain_fn(perturbed_text)
                    stability_score = attributor.compute_stability(explanation, explanation_perturbed)
                    raw_metrics[name]["stability"].append(stability_score)
                else:
                    raw_metrics[name]["stability"].append(0.0)

    # Aggregate metrics
    aggregated_metrics = {}
    for method, metrics_dict in raw_metrics.items():
        aggregated_metrics[method] = {}
        for metric_name, values in metrics_dict.items():
            if values:
                aggregated_metrics[method][metric_name] = round(float(np.mean(values)), 4)
            else:
                aggregated_metrics[method][metric_name] = 0.0

    # Cross-method agreement
    if len(explanations) > 1:
        method_names = list(explanations.keys())
        for i in range(len(method_names)):
            for j in range(i+1, len(method_names)):
                method1, method2 = method_names[i], method_names[j]
                agreement = attributor.compute_cross_method_agreement(
                    explanations[method1], explanations[method2]
                )
                cross_method_agreement.append({
                    "methods": f"{method1}_vs_{method2}",
                    "agreement": round(float(agreement), 4)
                })

    return aggregated_metrics, cross_method_agreement


def detect_bias_with_attributions(attributor, eval_data, sample_size=100):
    """Detect bias patterns in model attributions.
    Expects eval_data with 'sentence' and 'label' columns (already parsed).
    """
    print("  Detecting bias patterns...")

    safe_sample_size = min(sample_size, len(eval_data))
    indices = random.sample(range(len(eval_data)), safe_sample_size)
    sample_texts = [eval_data[i]["sentence"] for i in indices]
    sample_labels = [eval_data[i]["label"] for i in indices]

    predict_fn = attributor.get_predict_proba_fn()
    bias_detector = BiasDetector()

    # Only analyze gender and age dimensions (exclude race and religion)
    filtered_groups = {k: v for k, v in bias_detector.demographic_groups.items() if k in ("gender", "age")}

    all_attributions = {}
    methods = ["LIME", "SHAP"]

    for method_name in methods:
        attributions = []
        for text in tqdm(sample_texts, desc=f"Bias analysis ({method_name})"):
            if method_name == "LIME":
                explanation = attributor.explain_lime(text, predict_fn, num_features=10, num_samples=50)
            else:  # SHAP
                explanation = attributor.explain_shap(text, predict_fn, nsamples=40)

            attributions.append({
                "text": text,
                "attribution": explanation,
                "prediction": np.argmax(predict_fn([text])[0]),
                "label": sample_labels[len(attributions)]
            })

        all_attributions[method_name] = attributions

    # Analyze bias patterns (gender and age only)
    bias_results = bias_detector.analyze_bias_patterns(all_attributions, groups=filtered_groups)

    return bias_results, all_attributions, sample_texts


def _save_checkpoint(final_results, output_dir):
    """Incrementally save results so the pipeline can resume after a crash."""
    ckpt_file = os.path.join(output_dir, "mabsa_pipeline_results.json")
    with open(ckpt_file, "w") as f:
        json.dump(final_results, f, indent=2, cls=NumpyEncoder)


def main():
    args = parse_arguments()

    # If loading adapter explicitly, disable finetuning to prevent conflict
    if args.load_adapter:
        args.finetune = False

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(42)

    # ── Resume support: load partial results if they exist ────────
    ckpt_file = os.path.join(args.output_dir, "mabsa_pipeline_results.json")
    if os.path.isfile(ckpt_file):
        with open(ckpt_file) as f:
            final_results = json.load(f)
        completed = set(final_results.keys())
        print(f"Resuming from checkpoint — completed stages: {sorted(completed)}")
    else:
        final_results = {}
        completed = set()

    # Load M-ABSA dataset
    print("Loading M-ABSA dataset...")
    raw_dataset = load_dataset("Multilingual-NLP/M-ABSA")
    print(f"Raw dataset splits: {list(raw_dataset.keys())}")

    # Preprocess: parse text####[[aspect, category, sentiment]] into sentence + label
    print("\nPreprocessing M-ABSA data (parsing text####annotations)...")
    print("  Sentiment mapping: negative=0, neutral=1, positive=2")

    dataset = {}
    for split_name in raw_dataset:
        print(f"\n  Processing '{split_name}' split...")
        dataset[split_name] = preprocess_mabsa_dataset(raw_dataset[split_name])

    # ── Optional language filtering ───────────────────────────────────
    if args.languages:
        lang_set = set(l.strip().lower() for l in args.languages.split(","))
        print(f"\n{'='*50}")
        print(f"LANGUAGE FILTERING: keeping {sorted(lang_set)}")
        print("="*50)
        for split_name in list(dataset.keys()):
            print(f"\n  Filtering '{split_name}' split...")
            dataset[split_name] = filter_dataset_by_languages(
                dataset[split_name], lang_set
            )
        # Store which languages were requested
    else:
        lang_set = None
        print("\n  No --languages filter; using all 21 M-ABSA languages.")

    # ── Stratified sub-sampling for train / val / test ─────────────────
    print("\n" + "="*50)
    print("DATA SPLITTING (stratified sampling)")
    print("="*50)

    # Training set
    full_train = dataset["train"]
    if args.train_size > 0 and args.train_size < len(full_train):
        if args.balanced:
            train_data = balanced_sample(full_train, args.train_size, seed=42)
            print(f"  Train: {len(train_data)} balanced samples from {len(full_train)}")
        else:
            train_data = stratified_sample(full_train, args.train_size, seed=42)
            print(f"  Train: {len(train_data)} stratified samples from {len(full_train)}")
    else:
        train_data = full_train.shuffle(seed=42)
        print(f"  Train: full dataset ({len(train_data)} samples)")
    print(f"    Label dist: {Counter(train_data['label'])}")

    # Validation set (used for XAI analysis, bias detection, and intermediate eval)
    if "validation" in dataset:
        full_val = dataset["validation"]
    else:
        # Fallback: carve 10% from train
        full_val = full_train.shuffle(seed=99).select(range(min(5000, len(full_train) // 10)))

    if args.val_size > 0 and args.val_size < len(full_val):
        val_data = stratified_sample(full_val, args.val_size, seed=42)
        print(f"  Val:   {len(val_data)} stratified samples from {len(full_val)}")
    else:
        val_data = full_val.shuffle(seed=42)
        print(f"  Val:   full split ({len(val_data)} samples)")
    print(f"    Label dist: {Counter(val_data['label'])}")

    # Test set (held-out final evaluation — never used for XAI/bias)
    if "test" in dataset:
        full_test = dataset["test"]
    else:
        full_test = full_val  # fallback

    if args.test_size > 0 and args.test_size < len(full_test):
        test_data = stratified_sample(full_test, args.test_size, seed=42)
        print(f"  Test:  {len(test_data)} stratified samples from {len(full_test)}")
    else:
        test_data = full_test.shuffle(seed=42)
        print(f"  Test:  full split ({len(test_data)} samples)")
    print(f"    Label dist: {Counter(test_data['label'])}")

    # eval_data is the validation set for XAI / bias / intermediate perf
    eval_data = val_data

    # Optionally further subsample for performance evaluation
    if args.perf_eval_size > 0 and args.perf_eval_size < len(eval_data):
        eval_data = stratified_sample(eval_data, args.perf_eval_size, seed=42)
        print(f"\n  Perf-eval subset: {len(eval_data)} samples (--perf-eval-size {args.perf_eval_size})")

    print(f"\nEval columns: {eval_data.column_names}")
    print(f"Sample: {eval_data[0]}")

    # Store split info in results
    split_info = {
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "train_label_dist": dict(Counter(train_data["label"])),
        "val_label_dist": dict(Counter(val_data["label"])),
        "test_label_dist": dict(Counter(test_data["label"])),
        "languages_filter": sorted(lang_set) if lang_set else "all_21",
    }

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

    final_results["split_info"] = split_info

    # Zero-shot evaluation
    if "zero_shot_performance" in completed:
        print("\n[RESUME] Skipping zero-shot evaluation (already done)")
        zs_performance = final_results["zero_shot_performance"]
    else:
        print("\n" + "="*50)
        print("ZERO-SHOT EVALUATION")
        print("="*50)

        zs_performance = evaluate_performance_mabsa(predict_fn, eval_data)
        print(f"Zero-Shot Performance: {zs_performance}")
        final_results["zero_shot_performance"] = zs_performance
        _save_checkpoint(final_results, args.output_dir)

    # Zero-shot XAI analysis
    shift_analyzer = None
    if args.run_xai:
        if "zero_shot_attributions" in completed:
            print("[RESUME] Skipping zero-shot XAI analysis (already done)")
        else:
            print("\n" + "="*50)
            print("ZERO-SHOT XAI ANALYSIS")
            print("="*50)

            shift_analyzer = AttributionShiftAnalyzer(output_dir=args.output_dir) if args.run_shift_analysis else None

            zs_attrs, zs_agreement = compute_attribution_metrics(
                attributor, metrics_calc, eval_data, args.eval_sample_size,
                "Zero-Shot", args.extended_xai, shift_analyzer
            )
            print(f"Zero-Shot XAI Properties: {zs_attrs}")
            final_results["zero_shot_attributions"] = zs_attrs
            final_results["zero_shot_agreement"] = zs_agreement
            _save_checkpoint(final_results, args.output_dir)

            plot_xai_properties(zs_attrs, args.output_dir, "Zero-Shot")

    # Zero-shot bias detection (collect raw attributions for shift report)
    zs_raw_attributions = {}
    if "zero_shot_bias" in completed:
        print("[RESUME] Skipping zero-shot bias detection (already done)")
        zs_bias = final_results["zero_shot_bias"]
    else:
        zs_bias, zs_raw_attributions, _ = detect_bias_with_attributions(
            attributor, eval_data, sample_size=args.bias_sample_size
        )
        print(f"Zero-Shot Bias Analysis: {zs_bias}")
        final_results["zero_shot_bias"] = zs_bias
        _save_checkpoint(final_results, args.output_dir)
        plot_bias_results(zs_bias, args.output_dir, "Zero-Shot")

    # Fine-tuning or adapter loading
    # Determine adapter path: explicit --load-adapter, or auto-detect in output_dir
    adapter_path = args.load_adapter
    if adapter_path is None:
        candidate = os.path.join(args.output_dir, "lora_adapters")
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "adapter_config.json")):
            adapter_path = candidate
            print(f"\nFound existing adapters at {adapter_path}")

    if adapter_path and os.path.isdir(adapter_path):
        # ── Load previously trained adapters (skip training) ─────────
        print("\n" + "="*50)
        print("LOADING EXISTING LORA ADAPTERS (skipping training)")
        print("="*50)
        print(f"  Adapter path: {adapter_path}")
        finetuner.load_adapters(adapter_path)

    elif args.finetune:
        print("\n" + "="*50)
        print("FINE-TUNING")
        print("="*50)

        # Override prompt for 3-class M-ABSA (original is binary SST2)
        def _mabsa_format_prompt(text, label=None):
            prompt = ("Classify the sentiment as 0 (negative), 1 (neutral), or 2 (positive).\n"
                      f"Text: {text}\nSentiment:")
            if label is not None:
                prompt += f" {label}"
            return prompt

        finetuner._format_prompt = _mabsa_format_prompt

        # Configure LoRA adapters for training
        print("Configuring LoRA adapters...")
        finetuner.configure_lora(r=args.lora_r, lora_alpha=args.lora_alpha, dropout=args.lora_dropout)

        # Use the stratified train_data prepared above
        print(f"Using stratified training set: {len(train_data)} samples")
        print(f"  Label distribution: {Counter(train_data['label'])}")

        print(f"Training for {args.epochs} epochs (lr={args.learning_rate})")
        finetuner.train(train_data, eval_dataset=val_data, epochs=args.epochs,
                        learning_rate=args.learning_rate)

        # Free memory after training
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # Switch to inference mode
        print("Enabling inference mode after fine-tuning...")
        finetuner.enable_inference_mode()

    # Run fine-tuned evaluation if we have adapters (loaded or just trained)
    if adapter_path or args.finetune:
        # Reinitialize attributor with fine-tuned model
        attributor = FeatureAttributor(finetuner.model, finetuner.tokenizer)
        predict_fn = attributor.get_predict_proba_fn()
        metrics_calc = MetricsCalculator(predict_fn)

        # Fine-tuned evaluation
        if "finetuned_performance" in completed:
            print("\n[RESUME] Skipping fine-tuned evaluation (already done)")
        else:
            print("\n" + "="*50)
            print("FINE-TUNED EVALUATION")
            print("="*50)

            ft_performance = evaluate_performance_mabsa(predict_fn, eval_data)
            print(f"Fine-Tuned Performance: {ft_performance}")
            final_results["finetuned_performance"] = ft_performance
            _save_checkpoint(final_results, args.output_dir)

        # Fine-tuned XAI analysis
        if args.run_xai:
            if "finetuned_attributions" in completed:
                print("[RESUME] Skipping fine-tuned XAI analysis (already done)")
            else:
                print("\n" + "="*50)
                print("FINE-TUNED XAI ANALYSIS")
                print("="*50)

                ft_attrs, ft_agreement = compute_attribution_metrics(
                    attributor, metrics_calc, eval_data, args.eval_sample_size,
                    "Fine-Tuned", args.extended_xai, shift_analyzer
                )
                print(f"Fine-Tuned XAI Properties: {ft_attrs}")
                final_results["finetuned_attributions"] = ft_attrs
                final_results["finetuned_agreement"] = ft_agreement
                _save_checkpoint(final_results, args.output_dir)

                plot_xai_properties(ft_attrs, args.output_dir, "Fine-Tuned")

        # Bias detection (fine-tuned) - collect attributions for shift report
        ft_attributions = {}
        if "finetuned_bias" in completed:
            print("[RESUME] Skipping fine-tuned bias detection (already done)")
        else:
            ft_bias, ft_attributions, _ = detect_bias_with_attributions(attributor, eval_data, sample_size=args.bias_sample_size)
            print(f"Fine-Tuned Bias Analysis: {ft_bias}")
            final_results["finetuned_bias"] = ft_bias
            _save_checkpoint(final_results, args.output_dir)
            plot_bias_results(ft_bias, args.output_dir, "Fine-Tuned")

        # Generate bias shift report
        if "bias_shift_report" in completed:
            print("[RESUME] Skipping bias shift report (already done)")
        elif zs_raw_attributions and ft_attributions:
            print("\n  Computing bias amplification analysis...")
            bias_detector = BiasDetector()
            shared_methods = set(zs_raw_attributions.keys()) & set(ft_attributions.keys())
            bias_shift_report = {}
            for method in shared_methods:
                pre_lists = [d["attribution"] for d in zs_raw_attributions[method]]
                post_lists = [d["attribution"] for d in ft_attributions[method]]
                min_len = min(len(pre_lists), len(post_lists))
                if min_len > 0:
                    method_report = bias_detector.generate_bias_shift_report(
                        pre_lists[:min_len],
                        post_lists[:min_len]
                    )
                    bias_shift_report[method] = method_report
            print(f"  Bias Shift Methods Analyzed: {list(bias_shift_report.keys())}")
            final_results["bias_shift_report"] = bias_shift_report
            _save_checkpoint(final_results, args.output_dir)

        # Generate attribution shift report
        if "attribution_shift_report" in completed:
            print("[RESUME] Skipping attribution shift report (already done)")
        elif shift_analyzer is not None and args.run_shift_analysis:
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
            _save_checkpoint(final_results, args.output_dir)

            # Save attribution data (HDF5 or JSON)
            try:
                shift_analyzer.save_to_hdf5("attribution_shifts_mabsa.h5")
            except Exception as e:
                print(f"  Note: Could not save to HDF5 ({e}), falling back to JSON")
                shift_analyzer.save_to_json("attribution_shifts_mabsa.json")

    # ── Held-out TEST evaluation ─────────────────────────────────────
    if "test_performance" in completed:
        print("\n[RESUME] Skipping test evaluation (already done)")
    else:
        print("\n" + "="*50)
        print("HELD-OUT TEST EVALUATION")
        print("="*50)

        test_performance = evaluate_performance_mabsa(predict_fn, test_data)
        phase_label = "Fine-Tuned" if args.finetune else "Zero-Shot"
        print(f"Test Performance ({phase_label}): {test_performance}")
        final_results["test_performance"] = test_performance
        final_results["test_performance_phase"] = phase_label

    # Save final results
    print("\n" + "="*50)
    print("SAVING RESULTS")
    print("="*50)

    # Generate aggregate performance plot
    plot_performance_comparison(final_results, args.output_dir)

    output_file = f"{args.output_dir}/mabsa_pipeline_results.json"
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=2, cls=NumpyEncoder)

    print(f"Results saved to: {output_file}")
    print("\nM-ABSA Pipeline complete!")


if __name__ == "__main__":
    main()