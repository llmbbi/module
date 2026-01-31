
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional

class BiasDetector:
    """
    Detects and analyzes demographic biases in feature attributions.
    
    Extended to support before/after fine-tuning comparisons to answer:
    - Does fine-tuning amplify demographic biases present in base models?
    - Does task-specific training reduce spurious correlations?
    """
    
    def __init__(self):
        # Common demographic token groups
        self.demographic_groups = {
            "gender": {
                "male": ["he", "him", "his", "man", "male", "boy", "father", "son", "husband", "brother"],
                "female": ["she", "her", "hers", "woman", "female", "girl", "mother", "daughter", "wife", "sister"]
            },
            "age": {
                "young": ["young", "youth", "teenager", "teen", "child", "kid", "boy", "girl"],
                "old": ["old", "elderly", "senior", "aged", "mature", "adult"]
            }
        }

    def compute_attribution_mass(self, attribution_list, token_set):
        """
        Calculates the total attribution mass for a specific set of tokens in a single explanation.
        attribution_list: List of (word, score) tuples.
        token_set: Set of target tokens (strings) to aggregate mass for.
        """
        mass = 0.0
        normalized_set = set(t.lower() for t in token_set)
        
        for word, score in attribution_list:
            # Simple containment check; could be regex or exact match
            # "man" should match "man" or "man's"? We'll assume simple match for now
            if word.lower() in normalized_set:
                mass += abs(score) # Use absolute magnitude, or raw? Usually mass implies magnitude.
        return mass

    def analyze_bias(self, attributions_batch, group_a_tokens, group_b_tokens):
        """
        Performs statistical comparison between two demographic groups.
        attributions_batch: List of explanations (each explanation is a list of (word, score)).
        group_a_tokens: List of tokens for Group A.
        group_b_tokens: List of tokens for Group B.
        """
        masses_a = []
        masses_b = []
        
        for attr in attributions_batch:
            mass_a = self.compute_attribution_mass(attr, group_a_tokens)
            mass_b = self.compute_attribution_mass(attr, group_b_tokens)
            masses_a.append(mass_a)
            masses_b.append(mass_b)
            
        masses_a = np.array(masses_a)
        masses_b = np.array(masses_b)
        
        # Statistical Test: Independent T-test (assuming different samples, or we treat them as paired?)
        # Since we are comparing masses *within the same samples* (e.g. how much attention to 'he' vs 'she' in the same text?),
        # or across a dataset where these words appear?
        # Usually, bias is: "Does the model attend more to male words than female words overall?"
        # Paired T-test is appropriate if we are comparing A vs B weights in the SAME documents.
        # But if the documents are different (some have 'he', some 'she'), independent is better.
        # However, here we just sum mass. If a document has NO 'he' words, mass is 0.
        # So we are comparing the distributions of "Attention to A" vs "Attention to B" over the dataset.
        
        # We'll use a Paired T-test (rel) if we care about the difference per sample, 
        # but T-test independent (ind) is safer if they are not strictly paired. 
        # Actually, `scipy.stats.ttest_rel` checks if the mean difference is zero.
        # `scipy.stats.ttest_ind` checks if the means of two independent samples are different.
        # Let's use `ttest_ind` as a general approach.
        
        # Statistical Test: Independent T-test
        # Handle cases with no mass (avoid NaN)
        if np.all(masses_a == 0) and np.all(masses_b == 0):
            t_stat, p_val = 0.0, 1.0
        elif np.std(masses_a) == 0 and np.std(masses_b) == 0 and np.mean(masses_a) == np.mean(masses_b):
            t_stat, p_val = 0.0, 1.0
        else:
            try:
                t_stat, p_val = stats.ttest_ind(masses_a, masses_b, equal_var=False)
                if np.isnan(t_stat):
                    t_stat, p_val = 0.0, 1.0
            except:
                t_stat, p_val = 0.0, 1.0
        
        return {
            "mean_mass_a": float(np.mean(masses_a)),
            "mean_mass_b": float(np.mean(masses_b)),
            "std_mass_a": float(np.std(masses_a)),
            "std_mass_b": float(np.std(masses_b)),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "significant_bias": bool(p_val < 0.05)
        }

    # =========================================================================
    # NEW METHODS: Before/After Fine-tuning Bias Comparison
    # =========================================================================

    def compare_bias_phases(
        self,
        pre_bias_results: Dict[str, Any],
        post_bias_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare bias results between pre and post fine-tuning phases.
        
        Args:
            pre_bias_results: Bias analysis from before fine-tuning
            post_bias_results: Bias analysis from after fine-tuning
            
        Returns:
            Dictionary with comparison metrics
        """
        comparison = {
            "pre_finetune": pre_bias_results,
            "post_finetune": post_bias_results
        }
        
        # Calculate changes
        pre_diff = pre_bias_results.get("mean_mass_a", 0) - pre_bias_results.get("mean_mass_b", 0)
        post_diff = post_bias_results.get("mean_mass_a", 0) - post_bias_results.get("mean_mass_b", 0)
        
        comparison["pre_mass_difference"] = float(pre_diff)
        comparison["post_mass_difference"] = float(post_diff)
        comparison["mass_difference_change"] = float(post_diff - pre_diff)
        
        # Calculate absolute bias magnitude
        pre_abs_diff = abs(pre_diff)
        post_abs_diff = abs(post_diff)
        comparison["pre_abs_bias"] = float(pre_abs_diff)
        comparison["post_abs_bias"] = float(post_abs_diff)
        
        # Determine if bias was amplified or reduced
        if pre_abs_diff > 0:
            change_ratio = (post_abs_diff - pre_abs_diff) / pre_abs_diff
            comparison["bias_change_ratio"] = float(change_ratio)
            
            if change_ratio > 0.1:
                comparison["bias_trend"] = "amplified"
            elif change_ratio < -0.1:
                comparison["bias_trend"] = "reduced"
            else:
                comparison["bias_trend"] = "stable"
        else:
            comparison["bias_change_ratio"] = float(post_abs_diff) if post_abs_diff > 0 else 0.0
            comparison["bias_trend"] = "emerged" if post_abs_diff > 0.1 else "stable"
        
        # Statistical significance comparison
        comparison["pre_significant"] = pre_bias_results.get("significant_bias", False)
        comparison["post_significant"] = post_bias_results.get("significant_bias", False)
        
        if comparison["pre_significant"] and not comparison["post_significant"]:
            comparison["significance_change"] = "bias_removed"
        elif not comparison["pre_significant"] and comparison["post_significant"]:
            comparison["significance_change"] = "bias_introduced"
        elif comparison["pre_significant"] and comparison["post_significant"]:
            comparison["significance_change"] = "bias_persists"
        else:
            comparison["significance_change"] = "no_significant_bias"
        
        return comparison

    def calculate_bias_amplification(
        self,
        pre_attributions: List[List[Tuple[str, float]]],
        post_attributions: List[List[Tuple[str, float]]],
        group_a_tokens: List[str],
        group_b_tokens: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate bias amplification metrics between phases.
        
        Args:
            pre_attributions: List of attribution lists from before fine-tuning
            post_attributions: List of attribution lists from after fine-tuning
            group_a_tokens: Tokens for demographic group A
            group_b_tokens: Tokens for demographic group B
            
        Returns:
            Dictionary with amplification metrics
        """
        # Analyze bias in both phases
        pre_bias = self.analyze_bias(pre_attributions, group_a_tokens, group_b_tokens)
        post_bias = self.analyze_bias(post_attributions, group_a_tokens, group_b_tokens)
        
        # Compare phases
        comparison = self.compare_bias_phases(pre_bias, post_bias)
        
        # Add per-sample analysis
        pre_diffs = []
        post_diffs = []
        
        for pre_attr, post_attr in zip(pre_attributions, post_attributions):
            pre_mass_a = self.compute_attribution_mass(pre_attr, group_a_tokens)
            pre_mass_b = self.compute_attribution_mass(pre_attr, group_b_tokens)
            post_mass_a = self.compute_attribution_mass(post_attr, group_a_tokens)
            post_mass_b = self.compute_attribution_mass(post_attr, group_b_tokens)
            
            pre_diffs.append(pre_mass_a - pre_mass_b)
            post_diffs.append(post_mass_a - post_mass_b)
        
        pre_diffs = np.array(pre_diffs)
        post_diffs = np.array(post_diffs)
        
        # Per-sample amplification
        if len(pre_diffs) > 0:
            amplification_per_sample = np.abs(post_diffs) - np.abs(pre_diffs)
            comparison["amplification_mean"] = float(np.mean(amplification_per_sample))
            comparison["amplification_std"] = float(np.std(amplification_per_sample))
            comparison["samples_with_increased_bias"] = int(np.sum(amplification_per_sample > 0))
            comparison["samples_with_decreased_bias"] = int(np.sum(amplification_per_sample < 0))
            comparison["samples_total"] = len(pre_diffs)
        
        return comparison

    def analyze_multiple_groups(
        self,
        attributions_batch: List[List[Tuple[str, float]]],
        groups: Optional[Dict[str, Dict[str, List[str]]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze bias across multiple demographic dimensions.
        
        Args:
            attributions_batch: List of attribution lists
            groups: Dictionary of demographic groups (default: use built-in groups)
            
        Returns:
            Dictionary with bias analysis for each dimension
        """
        if groups is None:
            groups = self.demographic_groups
        
        results = {}
        
        for dimension, group_tokens in groups.items():
            if len(group_tokens) >= 2:
                # Get first two groups for comparison
                group_names = list(group_tokens.keys())
                group_a_tokens = group_tokens[group_names[0]]
                group_b_tokens = group_tokens[group_names[1]]
                
                bias_result = self.analyze_bias(attributions_batch, group_a_tokens, group_b_tokens)
                bias_result["group_a_name"] = group_names[0]
                bias_result["group_b_name"] = group_names[1]
                
                results[dimension] = bias_result
        
        return results

    def generate_bias_shift_report(
        self,
        pre_attributions: List[List[Tuple[str, float]]],
        post_attributions: List[List[Tuple[str, float]]],
        groups: Optional[Dict[str, Dict[str, List[str]]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive bias shift report across phases.
        
        Args:
            pre_attributions: Attributions from before fine-tuning
            post_attributions: Attributions from after fine-tuning
            groups: Demographic groups to analyze
            
        Returns:
            Comprehensive bias shift report
        """
        if groups is None:
            groups = self.demographic_groups
        
        report = {
            "num_samples": len(pre_attributions),
            "dimensions_analyzed": list(groups.keys()),
            "dimension_reports": {}
        }
        
        for dimension, group_tokens in groups.items():
            if len(group_tokens) >= 2:
                group_names = list(group_tokens.keys())
                group_a_tokens = group_tokens[group_names[0]]
                group_b_tokens = group_tokens[group_names[1]]
                
                amplification = self.calculate_bias_amplification(
                    pre_attributions,
                    post_attributions,
                    group_a_tokens,
                    group_b_tokens
                )
                amplification["group_a_name"] = group_names[0]
                amplification["group_b_name"] = group_names[1]
                
                report["dimension_reports"][dimension] = amplification
        
        # Generate summary
        all_trends = [r.get("bias_trend") for r in report["dimension_reports"].values()]
        report["summary"] = {
            "dimensions_with_amplified_bias": sum(1 for t in all_trends if t == "amplified"),
            "dimensions_with_reduced_bias": sum(1 for t in all_trends if t == "reduced"),
            "dimensions_with_stable_bias": sum(1 for t in all_trends if t == "stable"),
            "overall_trend": max(set(all_trends), key=all_trends.count) if all_trends else "unknown"
        }
        
        return report

