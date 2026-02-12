#!/usr/bin/env python3
"""
Simple test to check M-ABSA dataset structure
"""

import sys
import os
sys.path.append('/home/hice1/jyoon370/scratch/miniconda3/envs/llm1b/lib/python3.10/site-packages')

try:
    from datasets import load_dataset
    print("Loading M-ABSA dataset...")

    # Load just a small portion to check structure
    dataset = load_dataset("Multilingual-NLP/M-ABSA", split="validation[:10]")

    print(f"Dataset type: {type(dataset)}")
    print(f"Number of samples: {len(dataset)}")
    print(f"Column names: {dataset.column_names}")

    print("\nFirst 3 samples:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        for key, value in sample.items():
            print(f"  {key}: {value} (type: {type(value)})")

    # Check if there are any list-type columns that might contain sentiments
    print("\nAnalyzing column types:")
    for col in dataset.column_names:
        sample_value = dataset[0][col]
        print(f"  {col}: {type(sample_value)} - {sample_value}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()