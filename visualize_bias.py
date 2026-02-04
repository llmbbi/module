
import os
import json
import argparse
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

def load_results(output_root: str) -> pd.DataFrame:
    """Scans output directory for modular_pipeline_results.json and aggregates bias data."""
    results_files = glob.glob(os.path.join(output_root, "**/modular_pipeline_results.json"), recursive=True)
    
    data = []
    
    for file_path in results_files:
        try:
            with open(file_path, 'r') as f:
                res = json.load(f)
            
            # Infer model name from path (e.g. outputs/family_comparison/.../ModelName/modular_...json)
            # This is a heuristic, might need adjustment based on directory structure
            path_parts = file_path.split(os.sep)
            # Assuming structure: .../experiment_name/ModelName/modular_pipeline_results.json
            if len(path_parts) >= 3:
                model_name = path_parts[-2]
                experiment = path_parts[-3]
            else:
                model_name = "Unknown"
                experiment = "Unknown"
                
            # Extract Zero-Shot Bias
            if "zero_shot_bias" in res:
                for dim, metrics in res["zero_shot_bias"].items():
                    if dim == "summary": continue
                    
                    row = {
                        "Model": model_name,
                        "Experiment": experiment,
                        "State": "Base (Zero-Shot)",
                        "Dimension": dim,
                        "Cohen_d": metrics.get("cohen_d", 0.0),
                        "Mean_Difference": metrics.get("mean_difference", 0.0),
                        "Significant": metrics.get("significant_bias", False)
                    }
                    
                    # WAT Scores (aggregated if available)
                    # Currently structure has wat_groupA, wat_groupB dicts
                    # We can take the absolute difference or max absolute WAT score as a proxy for association strength?
                    # Or just plot one group's WAT. Let's try to extract group A's WAT.
                    group_a = metrics.get("group_a_name", "group_a")
                    wat_key = f"wat_{group_a}"
                    if wat_key in metrics:
                         row["WAT_Score"] = metrics[wat_key].get("wat_score", 0.0)
                    else:
                        row["WAT_Score"] = 0.0
                        
                    data.append(row)

            # Extract Fine-Tuned Bias
            if "finetuned_bias" in res:
                for dim, metrics in res["finetuned_bias"].items():
                    if dim == "summary": continue
                    
                    row = {
                        "Model": model_name,
                        "Experiment": experiment,
                        "State": "Fine-Tuned",
                        "Dimension": dim,
                        "Cohen_d": metrics.get("cohen_d", 0.0),
                        "Mean_Difference": metrics.get("mean_difference", 0.0),
                        "Significant": metrics.get("significant_bias", False)
                    }
                    
                    group_a = metrics.get("group_a_name", "group_a")
                    wat_key = f"wat_{group_a}"
                    if wat_key in metrics:
                         row["WAT_Score"] = metrics[wat_key].get("wat_score", 0.0)
                    else:
                        row["WAT_Score"] = 0.0
                    
                    data.append(row)
                    
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    return pd.DataFrame(data)

def plot_bias_comparison(df: pd.DataFrame, output_dir: str):
    """Generates comparison plots."""
    if df.empty:
        print("No data found to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Setup aesthetic
    sns.set_theme(style="whitegrid")
    
    # 1. Cohen's d Comparison (Facet by Dimension)
    g = sns.catplot(
        data=df, kind="bar",
        x="Model", y="Cohen_d", hue="State", col="Dimension",
        palette="viridis", height=5, aspect=1.2, sharey=False
    )
    g.set_axis_labels("Model", "Effect Size (Cohen's d)")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Bias Magnitude (Cohen\'s d) across Models and Dimensions')
    plt.savefig(os.path.join(output_dir, "bias_cohens_d_comparison.png"))
    plt.close()
    
    # 2. WAT Score Comparison
    g = sns.catplot(
        data=df, kind="bar",
        x="Model", y="WAT_Score", hue="State", col="Dimension",
        palette="magma", height=5, aspect=1.2, sharey=False
    )
    g.set_axis_labels("Model", "WAT Score (Word Association)")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Word Attribution Association (WAT) Scores')
    plt.savefig(os.path.join(output_dir, "bias_wat_score_comparison.png"))
    plt.close()
    
    print(f"Plots saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Visualize Bias Metrics")
    parser.add_argument("--input-dir", type=str, default="outputs", help="Root directory searching for results")
    parser.add_argument("--output-dir", type=str, default="results/bias_charts", help="Output directory for charts")
    args = parser.parse_args()
    
    df = load_results(args.input_dir)
    print(f"Loaded {len(df)} records.")
    
    if not df.empty:
        print("Data Sample:")
        print(df.head())
        plot_bias_comparison(df, args.output_dir)
        
        # Save aggregated CSV
        csv_path = os.path.join(args.output_dir, "aggregated_bias_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"Aggregated data saved to {csv_path}")

if __name__ == "__main__":
    main()
