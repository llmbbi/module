"""
MetricsCalculatorMABSA — 3-class faithfulness / stability metrics.

The binary SST-2 calculator hard-codes ``probs[0][1]`` (positive-class
probability) everywhere.  This variant works with the *predicted-class*
probability instead, making it compatible with any number of classes.
"""

import numpy as np
from scipy import stats as scipy_stats


class MetricsCalculatorMABSA:
    def __init__(self, predict_fn):
        self.predict_fn = predict_fn

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pred_class_prob(probs_row):
        """Return (predicted_class, probability_of_predicted_class)."""
        pred = int(np.argmax(probs_row))
        return pred, float(probs_row[pred])

    def _predict_class_prob(self, texts):
        """Run predict_fn and return predicted-class probability per text."""
        probs = self.predict_fn(texts)
        return [self._pred_class_prob(row) for row in probs]

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------
    def calculate_faithfulness_deletion(self, text, top_features, original_prob):
        if not top_features:
            return 0.0
        masked_text = text
        for word in top_features:
            masked_text = masked_text.replace(word, "")
        try:
            _, new_prob = self._predict_class_prob([masked_text])[0]
            return max(0, original_prob - new_prob)
        except Exception:
            return 0.0

    def calculate_shap_fidelity(self, shap_values, expected_value, original_prob):
        try:
            approx = expected_value + np.sum(shap_values)
            return max(0, 1 - abs(original_prob - approx))
        except Exception:
            return 0.0

    def calculate_gini(self, weights):
        weights = np.abs(np.array(weights).flatten())
        if len(weights) == 0 or np.sum(weights) == 0:
            return 0
        return 1 - (np.sum(weights ** 2) / (np.sum(weights) ** 2))

    def calculate_jaccard(self, list1, list2):
        s1, s2 = set(list1), set(list2)
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)

    def calculate_comprehensiveness(self, text, top_features, original_prob):
        if not top_features:
            return 0.0
        masked_text = text
        for word in top_features:
            masked_text = masked_text.replace(word, "[MASK]")
        try:
            _, new_prob = self._predict_class_prob([masked_text])[0]
            return max(0, original_prob - new_prob)
        except Exception:
            return 0.0

    def calculate_sufficiency(self, text, top_features, original_prob):
        if not top_features:
            return 0.0
        words = text.split()
        kept_text = " ".join(w for w in words if any(f in w for f in top_features))
        if not kept_text:
            return 0.0
        try:
            _, new_prob = self._predict_class_prob([kept_text])[0]
            return 1 - abs(original_prob - new_prob)
        except Exception:
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
                _, new_prob = self._predict_class_prob([masked_text])[0]
            except Exception:
                break
            if new_prob > prev_prob + 1e-6:  # Confidence went UP after removal
                violations += 1
            prev_prob = new_prob

        return max(0.0, 1.0 - violations / steps)

    def calculate_compactness(self, explanation_weights, top_k=3):
        """Fraction of total attribution mass concentrated in the top-k features.

        Returns a value in [0, 1] where 1 = all mass is in the top-k features.
        Varies across methods depending on how spread out attributions are.
        """
        abs_weights = np.abs(np.array(explanation_weights).flatten())
        total = np.sum(abs_weights)
        if total == 0 or len(abs_weights) == 0:
            return 0.0
        sorted_weights = np.sort(abs_weights)[::-1]
        top_mass = np.sum(sorted_weights[:top_k])
        return float(top_mass / total)

    def calculate_feature_agreement(self, list1, list2):
        s1, s2 = set(list1), set(list2)
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)

    # ------------------------------------------------------------------
    # Extended faithfulness & stability metrics
    # ------------------------------------------------------------------
    def calculate_insertion_auc(self, text, attribution_list, original_prob):
        if not attribution_list:
            return 0.0
        sorted_attrs = sorted(attribution_list, key=lambda x: abs(x[1]), reverse=True)
        added = []
        probs = []
        for word, _ in sorted_attrs:
            added.append(word)
            try:
                _, p = self._predict_class_prob([" ".join(added)])[0]
                probs.append(p)
            except Exception:
                probs.append(0.0)
        return float(np.trapz(probs) / len(probs)) if probs else 0.0

    def calculate_deletion_auc(self, text, attribution_list, original_prob):
        if not attribution_list:
            return 0.0
        words = text.split()
        sorted_attrs = sorted(attribution_list, key=lambda x: abs(x[1]), reverse=True)
        remaining = words.copy()
        probs = [original_prob]
        for word, _ in sorted_attrs:
            if word in remaining:
                remaining.remove(word)
            if remaining:
                try:
                    _, p = self._predict_class_prob([" ".join(remaining)])[0]
                    probs.append(p)
                except Exception:
                    probs.append(probs[-1] if probs else 0.0)
            else:
                probs.append(0.0)
        auc = np.trapz(probs) / len(probs) if probs else 0.0
        if original_prob > 0:
            return float(max(0, 1 - auc / original_prob))
        return 0.0

    def calculate_log_odds_faithfulness(self, text, top_features, original_prob):
        if not top_features or original_prob <= 0 or original_prob >= 1:
            return 0.0
        original_lo = np.log(original_prob / (1 - original_prob + 1e-10))
        masked_text = text
        for word in top_features:
            masked_text = masked_text.replace(word, "")
        try:
            _, new_prob = self._predict_class_prob([masked_text])[0]
            new_prob = max(min(new_prob, 0.999), 0.001)
            new_lo = np.log(new_prob / (1 - new_prob))
            return float(original_lo - new_lo)
        except Exception:
            return 0.0

    def calculate_sensitivity_n(self, text, explain_fn, n=2, num_samples=5):
        words = text.split()
        if len(words) < n + 1:
            return 1.0
        try:
            orig = {w: s for w, s in explain_fn(text)}
        except Exception:
            return 0.0
        correlations = []
        for _ in range(num_samples):
            pw = words.copy()
            for idx in np.random.choice(len(words), min(n, len(words)), replace=False):
                pw[idx] = "[UNK]"
            try:
                pert = {w: s for w, s in explain_fn(" ".join(pw))}
                shared = set(orig) & set(pert)
                if len(shared) >= 2:
                    ov = [orig[w] for w in shared]
                    pv = [pert[w] for w in shared]
                    if np.std(ov) > 0 and np.std(pv) > 0:
                        c, _ = scipy_stats.spearmanr(ov, pv)
                        if not np.isnan(c):
                            correlations.append(c)
            except Exception:
                continue
        return float(np.mean(correlations)) if correlations else 0.0

    def calculate_lipschitz_stability(self, text1, text2, explain_fn):
        try:
            e1, e2 = explain_fn(text1), explain_fn(text2)
        except Exception:
            return 0.0
        w1 = set(w for w, _ in e1)
        w2 = set(w for w, _ in e2)
        all_w = w1 | w2
        if not all_w:
            return 1.0
        s1 = {w: 0.0 for w in all_w}
        s2 = {w: 0.0 for w in all_w}
        for w, s in e1:
            s1[w] = s
        for w, s in e2:
            s2[w] = s
        v1 = np.array([s1[w] for w in sorted(all_w)])
        v2 = np.array([s2[w] for w in sorted(all_w)])
        ratio = np.linalg.norm(v1 - v2) / max(1, len((w1 - w2) | (w2 - w1)))
        return float(1.0 / (1.0 + ratio))

    def calculate_rank_correlation(self, attr1, attr2):
        if not attr1 or not attr2:
            return 0.0
        d1 = {w: s for w, s in attr1}
        d2 = {w: s for w, s in attr2}
        common = set(d1) & set(d2)
        if len(common) < 2:
            return 0.0
        try:
            c, _ = scipy_stats.spearmanr(
                [d1[w] for w in common], [d2[w] for w in common]
            )
            return float(c) if not np.isnan(c) else 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    def compute_all_metrics(self, text, predict_fn, original_prob, explanation_weights, top_features, attribution_list=None):
        text_words = text.split()
        weights = np.array(explanation_weights).flatten()
        results = {
            "faithfulness": self.calculate_faithfulness_deletion(text, top_features, original_prob),
            "comprehensiveness": self.calculate_comprehensiveness(text, top_features, original_prob),
            "sufficiency": self.calculate_sufficiency(text, top_features, original_prob),
            "gini": self.calculate_gini(weights),
            "monotonicity": self.calculate_monotonicity(text, attribution_list or [], predict_fn, original_prob),
            "compactness": self.calculate_compactness(weights, top_k=len(top_features)),
            "contrastivity": 1.0 if (any(w > 0 for w in weights) and any(w < 0 for w in weights)) else 0.5,
        }
        return results

    def compute_extended_metrics(self, text, attribution_list, original_prob, explain_fn=None):
        top_features = [
            w for w, _ in sorted(attribution_list, key=lambda x: abs(x[1]), reverse=True)[:3]
        ]
        results = {
            "insertion_auc": self.calculate_insertion_auc(text, attribution_list, original_prob),
            "deletion_auc": self.calculate_deletion_auc(text, attribution_list, original_prob),
            "log_odds_faithfulness": self.calculate_log_odds_faithfulness(text, top_features, original_prob),
        }
        if explain_fn is not None:
            results["sensitivity_n"] = self.calculate_sensitivity_n(text, explain_fn, n=2, num_samples=3)
            words = text.split()
            if words:
                results["lipschitz_stability"] = self.calculate_lipschitz_stability(
                    text, " ".join(words) + " ", explain_fn
                )
        return results
