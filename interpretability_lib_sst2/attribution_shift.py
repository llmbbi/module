"""
Attribution Shift Analysis Module

Analyzes how feature attributions change between different model phases
(e.g., before vs after fine-tuning). Uses HDF5 for efficient storage
of large attribution datasets.

Based on research questions from recent interpretability work:
- Do feature attributions shift toward task-relevant tokens after fine-tuning?
- Does explanation stability improve or degrade?
- Are there systematic patterns in attribution shifts across architectures?
"""

import numpy as np
import json
import os
from datetime import datetime
from scipy import stats as scipy_stats
from typing import Dict, List, Tuple, Optional, Any

# Try to import h5py for HDF5 storage, fall back to JSON if not available
try:
    import h5py
    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False
    print("Warning: h5py not installed. Attribution storage will use JSON (less efficient for large datasets).")


class AttributionShiftAnalyzer:
    """
    Analyzes how feature attributions change between model phases.
    
    This class enables systematic comparison of explanations before and after
    fine-tuning to answer questions like:
    - Do models attend to more task-relevant tokens after fine-tuning?
    - Does explanation stability change?
    - Are there sign flips in important features?
    """
    
    def __init__(self, output_dir: str, use_hdf5: bool = True):
        """
        Initialize the analyzer.
        
        Args:
            output_dir: Directory to store attribution data
            use_hdf5: Whether to use HDF5 storage (falls back to JSON if False or h5py unavailable)
        """
        self.output_dir = output_dir
        self.use_hdf5 = use_hdf5 and HAS_HDF5
        os.makedirs(output_dir, exist_ok=True)
        
        self.attributions = {
            "pre_finetune": {},
            "post_finetune": {}
        }
        self.metadata = {
            "created": datetime.now().isoformat(),
            "samples": []
        }
    
    def store_attributions(
        self,
        phase: str,
        sample_id: str,
        text: str,
        method: str,
        attribution_list: List[Tuple[str, float]],
        prediction_prob: float,
        label: Optional[int] = None
    ):
        """
        Store attributions for a sample in a specific phase.
        
        Args:
            phase: "pre_finetune" or "post_finetune"
            sample_id: Unique identifier for the sample
            text: Original input text
            method: Attribution method used (e.g., "LIME", "SHAP", "IG")
            attribution_list: List of (word, score) tuples
            prediction_prob: Model prediction probability
            label: Optional true label
        """
        if phase not in self.attributions:
            self.attributions[phase] = {}
        
        if sample_id not in self.attributions[phase]:
            self.attributions[phase][sample_id] = {
                "text": text,
                "label": label,
                "methods": {}
            }
        
        self.attributions[phase][sample_id]["methods"][method] = {
            "attributions": attribution_list,
            "prediction_prob": prediction_prob
        }
        
        # Track samples
        if sample_id not in self.metadata["samples"]:
            self.metadata["samples"].append(sample_id)
    
    def compute_shift_metrics(
        self,
        sample_id: str,
        method: str
    ) -> Dict[str, float]:
        """
        Compute shift metrics for a single sample between phases.
        
        Args:
            sample_id: Sample identifier
            method: Attribution method to analyze
            
        Returns:
            Dictionary of shift metrics
        """
        pre = self.attributions.get("pre_finetune", {}).get(sample_id, {})
        post = self.attributions.get("post_finetune", {}).get(sample_id, {})
        
        if not pre or not post:
            return {}
        
        pre_method = pre.get("methods", {}).get(method, {})
        post_method = post.get("methods", {}).get(method, {})
        
        if not pre_method or not post_method:
            return {}
        
        pre_attrs = pre_method.get("attributions", [])
        post_attrs = post_method.get("attributions", [])
        
        if not pre_attrs or not post_attrs:
            return {}
        
        # Convert to dictionaries for easier comparison
        pre_dict = {w: s for w, s in pre_attrs}
        post_dict = {w: s for w, s in post_attrs}
        
        # Find common words
        common_words = set(pre_dict.keys()) & set(post_dict.keys())
        
        metrics = {}
        
        # 1. Spearman Rank Correlation
        if len(common_words) >= 2:
            pre_vals = [pre_dict[w] for w in common_words]
            post_vals = [post_dict[w] for w in common_words]
            try:
                corr, p_val = scipy_stats.spearmanr(pre_vals, post_vals)
                if not np.isnan(corr):
                    metrics["rank_correlation"] = float(corr)
                    metrics["rank_correlation_pval"] = float(p_val)
            except:
                pass
        
        # 2. Top-K Overlap (for k=3, 5)
        for k in [3, 5]:
            pre_top_k = set(w for w, _ in sorted(pre_attrs, key=lambda x: abs(x[1]), reverse=True)[:k])
            post_top_k = set(w for w, _ in sorted(post_attrs, key=lambda x: abs(x[1]), reverse=True)[:k])
            
            if pre_top_k and post_top_k:
                overlap = len(pre_top_k & post_top_k) / len(pre_top_k | post_top_k)
                metrics[f"top_{k}_overlap"] = float(overlap)
        
        # 3. Sign Flips
        sign_flips = 0
        for word in common_words:
            pre_sign = np.sign(pre_dict[word])
            post_sign = np.sign(post_dict[word])
            if pre_sign != post_sign and pre_sign != 0 and post_sign != 0:
                sign_flips += 1
        metrics["sign_flips"] = sign_flips
        metrics["sign_flip_ratio"] = sign_flips / max(1, len(common_words))
        
        # 4. Attribution Mass Shift
        pre_mass = sum(abs(s) for _, s in pre_attrs)
        post_mass = sum(abs(s) for _, s in post_attrs)
        metrics["pre_attribution_mass"] = float(pre_mass)
        metrics["post_attribution_mass"] = float(post_mass)
        if pre_mass > 0:
            metrics["mass_change_ratio"] = float((post_mass - pre_mass) / pre_mass)
        
        # 5. Concentration (Gini-like measure)
        def gini_coefficient(values):
            values = np.abs(values)
            if len(values) == 0 or np.sum(values) == 0:
                return 0
            values = np.sort(values)
            n = len(values)
            cumsum = np.cumsum(values)
            return (2 * np.sum((np.arange(1, n+1) * values)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])
        
        if len(pre_dict) > 0:
            metrics["pre_concentration"] = float(gini_coefficient(list(pre_dict.values())))
        if len(post_dict) > 0:
            metrics["post_concentration"] = float(gini_coefficient(list(post_dict.values())))
        
        # 6. Prediction probability change
        pre_prob = pre_method.get("prediction_prob", 0)
        post_prob = post_method.get("prediction_prob", 0)
        metrics["pre_prediction_prob"] = float(pre_prob)
        metrics["post_prediction_prob"] = float(post_prob)
        metrics["prediction_change"] = float(post_prob - pre_prob)
        
        return metrics
    
    def identify_shifted_tokens(
        self,
        sample_id: str,
        method: str,
        threshold: float = 0.1
    ) -> Dict[str, List[Tuple[str, float, float]]]:
        """
        Identify which tokens gained or lost importance after fine-tuning.
        
        Args:
            sample_id: Sample identifier
            method: Attribution method
            threshold: Minimum absolute change to report
            
        Returns:
            Dictionary with "gained_importance" and "lost_importance" lists
        """
        pre = self.attributions.get("pre_finetune", {}).get(sample_id, {})
        post = self.attributions.get("post_finetune", {}).get(sample_id, {})
        
        if not pre or not post:
            return {"gained_importance": [], "lost_importance": []}
        
        pre_attrs = pre.get("methods", {}).get(method, {}).get("attributions", [])
        post_attrs = post.get("methods", {}).get(method, {}).get("attributions", [])
        
        pre_dict = {w: s for w, s in pre_attrs}
        post_dict = {w: s for w, s in post_attrs}
        
        gained = []
        lost = []
        
        all_words = set(pre_dict.keys()) | set(post_dict.keys())
        
        for word in all_words:
            pre_score = pre_dict.get(word, 0)
            post_score = post_dict.get(word, 0)
            
            change = abs(post_score) - abs(pre_score)
            
            if change > threshold:
                gained.append((word, pre_score, post_score))
            elif change < -threshold:
                lost.append((word, pre_score, post_score))
        
        # Sort by magnitude of change
        gained.sort(key=lambda x: abs(x[2]) - abs(x[1]), reverse=True)
        lost.sort(key=lambda x: abs(x[1]) - abs(x[2]), reverse=True)
        
        return {
            "gained_importance": gained[:10],  # Top 10
            "lost_importance": lost[:10]
        }
    
    def compute_aggregate_shift_analysis(self, method: str = "LIME") -> Dict[str, Any]:
        """
        Compute aggregate shift metrics across all samples.
        
        Args:
            method: Attribution method to analyze
            
        Returns:
            Dictionary with aggregate statistics
        """
        all_metrics = []
        all_shifted_tokens = {"gained": {}, "lost": {}}
        
        for sample_id in self.metadata["samples"]:
            metrics = self.compute_shift_metrics(sample_id, method)
            if metrics:
                all_metrics.append(metrics)
            
            shifted = self.identify_shifted_tokens(sample_id, method)
            for word, _, _ in shifted["gained_importance"]:
                all_shifted_tokens["gained"][word] = all_shifted_tokens["gained"].get(word, 0) + 1
            for word, _, _ in shifted["lost_importance"]:
                all_shifted_tokens["lost"][word] = all_shifted_tokens["lost"].get(word, 0) + 1
        
        if not all_metrics:
            return {}
        
        # Aggregate statistics
        aggregate = {
            "num_samples": len(all_metrics),
            "method": method
        }
        
        # Compute mean and std for each metric
        metric_keys = set()
        for m in all_metrics:
            metric_keys.update(m.keys())
        
        for key in metric_keys:
            values = [m.get(key) for m in all_metrics if m.get(key) is not None and not np.isnan(m.get(key, 0))]
            if values:
                aggregate[f"{key}_mean"] = float(np.mean(values))
                aggregate[f"{key}_std"] = float(np.std(values))
        
        # Most frequently shifted tokens
        aggregate["most_gained_tokens"] = sorted(
            all_shifted_tokens["gained"].items(), key=lambda x: x[1], reverse=True
        )[:10]
        aggregate["most_lost_tokens"] = sorted(
            all_shifted_tokens["lost"].items(), key=lambda x: x[1], reverse=True
        )[:10]
        
        return aggregate
    
    def save_to_hdf5(self, filename: str = "attribution_data.h5"):
        """Save all attribution data to HDF5 file."""
        if not HAS_HDF5:
            print("h5py not available, falling back to JSON")
            return self.save_to_json(filename.replace(".h5", ".json"))
        
        filepath = os.path.join(self.output_dir, filename)
        
        with h5py.File(filepath, 'w') as f:
            # Store metadata
            meta_grp = f.create_group("metadata")
            meta_grp.attrs["created"] = self.metadata["created"]
            meta_grp.create_dataset("samples", data=np.array(self.metadata["samples"], dtype='S'))
            
            # Store attributions for each phase
            for phase, samples in self.attributions.items():
                phase_grp = f.create_group(phase)
                
                for sample_id, data in samples.items():
                    sample_grp = phase_grp.create_group(sample_id)
                    sample_grp.attrs["text"] = data["text"]
                    if data.get("label") is not None:
                        sample_grp.attrs["label"] = data["label"]
                    
                    methods_grp = sample_grp.create_group("methods")
                    for method, method_data in data.get("methods", {}).items():
                        method_grp = methods_grp.create_group(method)
                        method_grp.attrs["prediction_prob"] = method_data["prediction_prob"]
                        
                        # Store attributions
                        attrs = method_data["attributions"]
                        if attrs:
                            words = [a[0] for a in attrs]
                            scores = np.array([a[1] for a in attrs], dtype=np.float32)
                            method_grp.create_dataset("words", data=np.array(words, dtype='S'))
                            method_grp.create_dataset("scores", data=scores)
        
        print(f"Saved attribution data to {filepath}")
        return filepath
    
    def save_to_json(self, filename: str = "attribution_data.json"):
        """Save all attribution data to JSON file."""
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert to JSON-serializable format
        data = {
            "metadata": self.metadata,
            "attributions": {}
        }
        
        for phase, samples in self.attributions.items():
            data["attributions"][phase] = {}
            for sample_id, sample_data in samples.items():
                data["attributions"][phase][sample_id] = {
                    "text": sample_data["text"],
                    "label": sample_data.get("label"),
                    "methods": {}
                }
                for method, method_data in sample_data.get("methods", {}).items():
                    data["attributions"][phase][sample_id]["methods"][method] = {
                        "prediction_prob": float(method_data["prediction_prob"]),
                        "attributions": [
                            (w, float(s)) for w, s in method_data["attributions"]
                        ]
                    }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved attribution data to {filepath}")
        return filepath
    
    def load_from_hdf5(self, filename: str = "attribution_data.h5"):
        """Load attribution data from HDF5 file."""
        if not HAS_HDF5:
            print("h5py not available")
            return False
        
        filepath = os.path.join(self.output_dir, filename)
        if not os.path.exists(filepath):
            return False
        
        with h5py.File(filepath, 'r') as f:
            # Load metadata
            self.metadata["created"] = f["metadata"].attrs["created"]
            self.metadata["samples"] = [s.decode() for s in f["metadata"]["samples"][:]]
            
            # Load attributions
            for phase in ["pre_finetune", "post_finetune"]:
                if phase in f:
                    self.attributions[phase] = {}
                    for sample_id in f[phase]:
                        sample_grp = f[phase][sample_id]
                        self.attributions[phase][sample_id] = {
                            "text": sample_grp.attrs["text"],
                            "label": sample_grp.attrs.get("label"),
                            "methods": {}
                        }
                        
                        if "methods" in sample_grp:
                            for method in sample_grp["methods"]:
                                method_grp = sample_grp["methods"][method]
                                words = [w.decode() for w in method_grp["words"][:]]
                                scores = method_grp["scores"][:]
                                
                                self.attributions[phase][sample_id]["methods"][method] = {
                                    "prediction_prob": float(method_grp.attrs["prediction_prob"]),
                                    "attributions": list(zip(words, scores.tolist()))
                                }
        
        return True
    
    def load_from_json(self, filename: str = "attribution_data.json"):
        """Load attribution data from JSON file."""
        filepath = os.path.join(self.output_dir, filename)
        if not os.path.exists(filepath):
            return False
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.metadata = data.get("metadata", {})
        self.attributions = data.get("attributions", {})
        
        return True
    
    def generate_shift_report(self, methods: List[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive shift analysis report.
        
        Args:
            methods: List of methods to analyze (default: all available)
            
        Returns:
            Dictionary containing the full report
        """
        if methods is None:
            # Detect available methods from data
            methods = set()
            for phase_data in self.attributions.values():
                for sample_data in phase_data.values():
                    methods.update(sample_data.get("methods", {}).keys())
            methods = list(methods)
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "num_samples": len(self.metadata["samples"]),
            "methods_analyzed": methods,
            "method_reports": {}
        }
        
        for method in methods:
            method_report = self.compute_aggregate_shift_analysis(method)
            if method_report:
                report["method_reports"][method] = method_report
        
        # Add summary
        if report["method_reports"]:
            first_method = list(report["method_reports"].values())[0]
            report["summary"] = {
                "avg_rank_correlation": first_method.get("rank_correlation_mean", "N/A"),
                "avg_top3_overlap": first_method.get("top_3_overlap_mean", "N/A"),
                "avg_prediction_improvement": first_method.get("prediction_change_mean", "N/A"),
            }
        
        return report
