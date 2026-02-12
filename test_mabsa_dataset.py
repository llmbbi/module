#!/usr/bin/env python3
"""
Test script for M-ABSA dataset loading and basic functionality
"""

import sys
import os
sys.path.append('/home/hice1/jyoon370/scratch/miniconda3/envs/llm1b/lib/python3.10/site-packages')

try:
    from datasets import load_dataset
    import numpy as np

    print("Loading M-ABSA dataset...")
    dataset = load_dataset("Multilingual-NLP/M-ABSA")

    print("Dataset splits:", list(dataset.keys()))

    # Check train split
    train_data = dataset["train"]
    print(f"Train split size: {len(train_data)}")
    print("Train columns:", train_data.column_names)

    # Sample a few examples
    print("\nSample entries from train split:")
    for i in range(min(5, len(train_data))):
        entry = train_data[i]
        print(f"\nEntry {i}:")
        for key, value in entry.items():
            print(f"  {key}: {value}")

    # Check label distribution
    if "label" in train_data.column_names:
        labels = train_data["label"]
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"\nLabel distribution: {dict(zip(unique_labels, counts))}")
        print("Expected: 0=negative, 1=neutral, 2=positive")

    print("\nM-ABSA dataset loaded successfully!")

except Exception as e:
    print(f"Error loading M-ABSA dataset: {e}")
    import traceback
    traceback.print_exc()