
import numpy as np

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

    def calculate_monotonicity(self, weights):
        """Check if removing features in order of importance decreases prediction"""
        weights = np.array(weights).flatten()
        if len(weights) < 2:
            return 1.0
        # Sort by absolute importance
        sorted_weights = np.sort(np.abs(weights))[::-1]
        # Check if generally decreasing
        violations = 0
        for i in range(len(sorted_weights) - 1):
            if sorted_weights[i] < sorted_weights[i + 1]:
                violations += 1
        return max(0, 1 - (violations / (len(sorted_weights) - 1)))

    def calculate_compactness(self, num_features, total_features):
        """Measure how compact/sparse the explanation is"""
        if total_features == 0:
            return 0.0
        ratio = num_features / total_features
        # Prefer fewer features (more compact)
        return max(0, 1 - ratio)

    def calculate_feature_agreement(self, feature_list_1, feature_list_2):
        """Measure agreement between two sets of important features (e.g. LIME vs SHAP)"""
        s1 = set(feature_list_1)
        s2 = set(feature_list_2)
        if not s1 or not s2:
            return 0.0
        return len(s1.intersection(s2)) / len(s1.union(s2))

    def compute_all_metrics(self, text, predict_fn, original_prob, explanation_weights, top_features):
        """Computes a suite of metrics for a given explanation."""
        text_words = text.split()
        
        results = {
            "faithfulness": self.calculate_faithfulness_deletion(text, top_features, original_prob),
            "comprehensiveness": self.calculate_comprehensiveness(text, top_features, original_prob),
            "sufficiency": self.calculate_sufficiency(text, top_features, original_prob),
            "gini": self.calculate_gini(explanation_weights),
            "monotonicity": self.calculate_monotonicity(explanation_weights),
            "compactness": self.calculate_compactness(len(top_features), len(text_words))
        }
        
        # Add contrastivity (presence of both pos and neg weights)
        weights = np.array(explanation_weights).flatten()
        has_pos = any(w > 0 for w in weights)
        has_neg = any(w < 0 for w in weights)
        results["contrastivity"] = 1.0 if (has_pos and has_neg) else 0.5
        
        return results
