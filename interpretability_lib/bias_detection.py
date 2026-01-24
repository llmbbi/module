
import numpy as np
from scipy import stats

class BiasDetector:
    def __init__(self):
        pass

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
