
import numpy as np
from scipy import stats as scipy_stats

class MetricsCalculator:
    def __init__(self, predict_fn):
        self.predict_fn = predict_fn

    def calculate_faithfulness_deletion(self, text, top_features, original_prob):
        if not top_features: 
            return 0.0
        masked_text = text
        for word in top_features:
            masked_text = masked_text.replace(word, "") 
        try:
            new_probs = self.predict_fn([masked_text])
            new_prob = new_probs[0][1]
            return max(0, original_prob - new_prob)
        except:
            return 0.0

    def calculate_shap_fidelity(self, shap_values, expected_value, original_prob):
        try:
            sum_contributions = np.sum(shap_values)
            approx_prob = expected_value + sum_contributions
            diff = abs(original_prob - approx_prob)
            return max(0, 1 - diff)
        except:
            return 0.0

    def calculate_gini(self, weights):
        weights = np.array(weights).flatten()
        if len(weights) == 0: 
            return 0
        weights = np.abs(weights)
        if np.sum(weights) == 0: 
            return 0
        return 1 - (np.sum(weights**2) / (np.sum(weights)**2))

    def calculate_jaccard(self, list1, list2):
        s1 = set(list1)
        s2 = set(list2)
        if not s1 and not s2: 
            return 1.0
        if not s1 or not s2: 
            return 0.0
        return len(s1.intersection(s2)) / len(s1.union(s2))

    def calculate_comprehensiveness(self, text, top_features, original_prob):
        """Remove important features and measure probability drop"""
        if not top_features:
            return 0.0
        masked_text = text
        for word in top_features:
            masked_text = masked_text.replace(word, "[MASK]")
        try:
            new_probs = self.predict_fn([masked_text])
            new_prob = new_probs[0][1]
            return max(0, original_prob - new_prob)
        except:
            return 0.0

    def calculate_sufficiency(self, text, top_features, original_prob):
        """Keep only important features and measure if probability is maintained"""
        if not top_features:
            return 0.0
        # Create text with only top features
        words = text.split()
        kept_text = " ".join([w for w in words if any(feat in w for feat in top_features)])
        if not kept_text:
            return 0.0
        try:
            new_probs = self.predict_fn([kept_text])
            new_prob = new_probs[0][1]
            return 1 - abs(original_prob - new_prob)
        except:
            return 0.0

    def calculate_monotonicity(self, text, attribution_list, predict_fn, original_prob):
        """Check if progressively removing top-attributed features causes a
        monotonically decreasing prediction confidence.

        Returns a score in [0, 1] where 1 = perfectly monotonic decrease.
        """
        if not attribution_list or len(attribution_list) < 2:
            return 1.0

        # Sort features by absolute attribution (highest first)
        sorted_feats = sorted(attribution_list, key=lambda x: abs(x[1]), reverse=True)

        prev_prob = original_prob
        masked_text = text
        violations = 0
        steps = min(len(sorted_feats), 5)  # Check top-5 removals at most

        for i in range(steps):
            word = sorted_feats[i][0]
            masked_text = masked_text.replace(word, "", 1)
            try:
                probs = predict_fn([masked_text])[0]
                pred_class = int(np.argmax(probs))
                new_prob = float(probs[pred_class])
            except Exception:
                break
            if new_prob > prev_prob + 1e-6:
                violations += 1
            prev_prob = new_prob

        return max(0.0, 1.0 - violations / steps)

    def calculate_compactness(self, explanation_weights, top_k=3):
        """Fraction of total attribution mass concentrated in the top-k features.

        Returns a value in [0, 1] where 1 = all mass is in the top-k features.
        """
        abs_weights = np.abs(np.array(explanation_weights).flatten())
        total = np.sum(abs_weights)
        if total == 0 or len(abs_weights) == 0:
            return 0.0
        sorted_weights = np.sort(abs_weights)[::-1]
        top_mass = np.sum(sorted_weights[:top_k])
        return float(top_mass / total)

    def calculate_feature_agreement(self, feature_list_1, feature_list_2):
        """Measure agreement between two sets of important features (e.g. LIME vs SHAP)"""
        s1 = set(feature_list_1)
        s2 = set(feature_list_2)
        if not s1 or not s2:
            return 0.0
        return len(s1.intersection(s2)) / len(s1.union(s2))

    # =========================================================================
    # NEW METRICS: Faithfulness & Stability from Recent Literature
    # =========================================================================

    def calculate_insertion_auc(self, text, attribution_list, original_prob):
        """
        Calculates Insertion AUC (Area Under Curve).
        
        Progressively adds tokens in order of importance and measures 
        the increase in prediction probability. Higher AUC = more faithful.
        
        Based on: Petsiuk et al. (2018), widely used in NeurIPS/ACL papers.
        
        Args:
            text: Original input text
            attribution_list: List of (word, score) tuples
            original_prob: Original model probability for positive class
            
        Returns:
            AUC score between 0 and 1
        """
        if not attribution_list:
            return 0.0
        
        words = text.split()
        if not words:
            return 0.0
        
        # Sort by attribution (highest first)
        sorted_attrs = sorted(attribution_list, key=lambda x: abs(x[1]), reverse=True)
        sorted_words = [x[0] for x in sorted_attrs]
        
        # Progressive insertion: start with empty, add one word at a time
        probs = []
        added_words = []
        
        for word in sorted_words:
            added_words.append(word)
            partial_text = " ".join(added_words)
            try:
                prob = self.predict_fn([partial_text])[0][1]
                probs.append(prob)
            except:
                probs.append(0.0)
        
        if not probs:
            return 0.0
        
        # Compute AUC using trapezoidal rule
        auc = np.trapz(probs) / len(probs)
        return float(auc)

    def calculate_deletion_auc(self, text, attribution_list, original_prob):
        """
        Calculates Deletion AUC (Area Under Curve).
        
        Progressively removes tokens in order of importance and measures 
        the decrease in prediction probability. Lower AUC = more faithful.
        We return 1 - normalized_auc so higher = better.
        
        Based on: Petsiuk et al. (2018), EraseR benchmark (NeurIPS 2019).
        
        Args:
            text: Original input text
            attribution_list: List of (word, score) tuples
            original_prob: Original model probability for positive class
            
        Returns:
            Score between 0 and 1 (higher = more faithful)
        """
        if not attribution_list:
            return 0.0
        
        words = text.split()
        if not words:
            return 0.0
        
        # Sort by attribution (highest first = remove most important first)
        sorted_attrs = sorted(attribution_list, key=lambda x: abs(x[1]), reverse=True)
        words_to_remove = [x[0] for x in sorted_attrs]
        
        # Progressive deletion
        probs = [original_prob]  # Start with full text probability
        remaining_words = words.copy()
        
        for word in words_to_remove:
            if word in remaining_words:
                remaining_words.remove(word)
            if remaining_words:
                partial_text = " ".join(remaining_words)
                try:
                    prob = self.predict_fn([partial_text])[0][1]
                    probs.append(prob)
                except:
                    probs.append(probs[-1] if probs else 0.0)
            else:
                probs.append(0.0)
        
        if not probs:
            return 0.0
        
        # Compute AUC
        auc = np.trapz(probs) / len(probs)
        
        # Lower AUC is better for deletion, so we return 1 - normalized
        # Normalize by original_prob to get a 0-1 score
        if original_prob > 0:
            normalized_auc = auc / original_prob
            return float(max(0, 1 - normalized_auc))
        return 0.0

    def calculate_log_odds_faithfulness(self, text, top_features, original_prob):
        """
        Calculates Log-Odds based faithfulness.
        
        Uses log-odds instead of probabilities for more sensitivity to small changes.
        Based on: LoFi metric (EMNLP 2023).
        
        Args:
            text: Original input text
            top_features: List of top important features to remove
            original_prob: Original model probability
            
        Returns:
            Log-odds change (higher = more faithful explanation)
        """
        if not top_features or original_prob <= 0 or original_prob >= 1:
            return 0.0
        
        # Calculate original log-odds
        original_log_odds = np.log(original_prob / (1 - original_prob + 1e-10))
        
        # Mask top features
        masked_text = text
        for word in top_features:
            masked_text = masked_text.replace(word, "")
        
        try:
            new_prob = self.predict_fn([masked_text])[0][1]
            new_prob = max(min(new_prob, 0.999), 0.001)  # Clamp for numerical stability
            new_log_odds = np.log(new_prob / (1 - new_prob))
            
            # Return the drop in log-odds (higher drop = more important features)
            return float(original_log_odds - new_log_odds)
        except:
            return 0.0

    def calculate_sensitivity_n(self, text, explain_fn, n=2, num_samples=5):
        """
        Calculates Sensitivity-n metric.
        
        Measures how much the explanation changes when n tokens are randomly perturbed.
        Lower sensitivity = more stable explanation.
        
        Based on: Yeh et al. (NeurIPS 2019) sensitivity analysis.
        
        Args:
            text: Original input text
            explain_fn: Function that returns explanation for a text
            n: Number of tokens to perturb
            num_samples: Number of perturbation samples
            
        Returns:
            Average correlation between original and perturbed explanations
        """
        words = text.split()
        if len(words) < n + 1:
            return 1.0  # Not enough words to perturb
        
        try:
            original_explanation = explain_fn(text)
            original_scores = {w: s for w, s in original_explanation}
        except:
            return 0.0
        
        correlations = []
        
        for _ in range(num_samples):
            # Randomly perturb n words
            perturbed_words = words.copy()
            indices_to_perturb = np.random.choice(len(words), size=min(n, len(words)), replace=False)
            
            for idx in indices_to_perturb:
                # Replace with a neutral word or shuffle
                perturbed_words[idx] = "[UNK]"
            
            perturbed_text = " ".join(perturbed_words)
            
            try:
                perturbed_explanation = explain_fn(perturbed_text)
                perturbed_scores = {w: s for w, s in perturbed_explanation}
                
                # Compute correlation for shared words
                shared_words = set(original_scores.keys()) & set(perturbed_scores.keys())
                if len(shared_words) >= 2:
                    orig_vals = [original_scores[w] for w in shared_words]
                    pert_vals = [perturbed_scores[w] for w in shared_words]
                    
                    if np.std(orig_vals) > 0 and np.std(pert_vals) > 0:
                        corr, _ = scipy_stats.spearmanr(orig_vals, pert_vals)
                        if not np.isnan(corr):
                            correlations.append(corr)
            except:
                continue
        
        if correlations:
            return float(np.mean(correlations))
        return 0.0

    def calculate_lipschitz_stability(self, text1, text2, explain_fn):
        """
        Calculates Lipschitz stability bound estimate.
        
        Measures the ratio of explanation change to input change.
        A smaller ratio indicates more stable explanations.
        
        Based on: Alvarez-Melis & Jaakkola (NeurIPS 2018).
        
        Args:
            text1: First input text
            text2: Second (slightly different) input text
            explain_fn: Function that returns explanation
            
        Returns:
            Stability score (between 0 and 1, higher = more stable)
        """
        try:
            exp1 = explain_fn(text1)
            exp2 = explain_fn(text2)
        except:
            return 0.0
        
        # Get scores as vectors
        words1 = set(w for w, _ in exp1)
        words2 = set(w for w, _ in exp2)
        all_words = words1 | words2
        
        if not all_words:
            return 1.0
        
        scores1 = {w: 0.0 for w in all_words}
        scores2 = {w: 0.0 for w in all_words}
        
        for w, s in exp1:
            scores1[w] = s
        for w, s in exp2:
            scores2[w] = s
        
        vec1 = np.array([scores1[w] for w in sorted(all_words)])
        vec2 = np.array([scores2[w] for w in sorted(all_words)])
        
        # Explanation distance
        exp_dist = np.linalg.norm(vec1 - vec2)
        
        # Input distance (simple word diff)
        word_diff = len((words1 - words2) | (words2 - words1))
        input_dist = max(1, word_diff)  # Avoid division by zero
        
        # Lipschitz ratio (lower is better)
        ratio = exp_dist / input_dist
        
        # Convert to stability score (0-1, higher is better)
        # Using sigmoid-like transformation
        stability = 1.0 / (1.0 + ratio)
        
        return float(stability)

    def calculate_rank_correlation(self, attribution_list_1, attribution_list_2):
        """
        Calculates Spearman rank correlation between two attribution lists.
        
        Useful for comparing attributions from different methods or phases.
        
        Args:
            attribution_list_1: First list of (word, score) tuples
            attribution_list_2: Second list of (word, score) tuples
            
        Returns:
            Spearman correlation coefficient (-1 to 1)
        """
        if not attribution_list_1 or not attribution_list_2:
            return 0.0
        
        # Create word -> score mappings
        scores1 = {w: s for w, s in attribution_list_1}
        scores2 = {w: s for w, s in attribution_list_2}
        
        # Find common words
        common_words = set(scores1.keys()) & set(scores2.keys())
        
        if len(common_words) < 2:
            return 0.0
        
        vals1 = [scores1[w] for w in common_words]
        vals2 = [scores2[w] for w in common_words]
        
        try:
            corr, _ = scipy_stats.spearmanr(vals1, vals2)
            if np.isnan(corr):
                return 0.0
            return float(corr)
        except:
            return 0.0

    def compute_all_metrics(self, text, predict_fn, original_prob, explanation_weights, top_features, attribution_list=None):
        """Computes a suite of metrics for a given explanation."""
        text_words = text.split()
        
        results = {
            "faithfulness": self.calculate_faithfulness_deletion(text, top_features, original_prob),
            "comprehensiveness": self.calculate_comprehensiveness(text, top_features, original_prob),
            "sufficiency": self.calculate_sufficiency(text, top_features, original_prob),
            "gini": self.calculate_gini(explanation_weights),
            "monotonicity": self.calculate_monotonicity(text, attribution_list or [], predict_fn, original_prob),
            "compactness": self.calculate_compactness(explanation_weights, top_k=len(top_features))
        }
        
        # Add contrastivity (presence of both pos and neg weights)
        weights = np.array(explanation_weights).flatten()
        has_pos = any(w > 0 for w in weights)
        has_neg = any(w < 0 for w in weights)
        results["contrastivity"] = 1.0 if (has_pos and has_neg) else 0.5
        
        return results

    def compute_extended_metrics(self, text, attribution_list, original_prob, explain_fn=None):
        """
        Computes extended faithfulness and stability metrics.
        
        Args:
            text: Input text
            attribution_list: List of (word, score) tuples
            original_prob: Original prediction probability
            explain_fn: Optional explanation function for stability metrics
            
        Returns:
            Dictionary of extended metrics
        """
        top_features = [w for w, _ in sorted(attribution_list, key=lambda x: abs(x[1]), reverse=True)[:3]]
        
        results = {
            "insertion_auc": self.calculate_insertion_auc(text, attribution_list, original_prob),
            "deletion_auc": self.calculate_deletion_auc(text, attribution_list, original_prob),
            "log_odds_faithfulness": self.calculate_log_odds_faithfulness(text, top_features, original_prob),
        }
        
        # Add stability metrics if explain_fn is provided
        if explain_fn is not None:
            results["sensitivity_n"] = self.calculate_sensitivity_n(text, explain_fn, n=2, num_samples=3)
            
            # Create slightly perturbed text for Lipschitz stability
            words = text.split()
            if words:
                perturbed_text = " ".join(words) + " "
                results["lipschitz_stability"] = self.calculate_lipschitz_stability(text, perturbed_text, explain_fn)
        
        return results

