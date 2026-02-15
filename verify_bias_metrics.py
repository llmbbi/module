
import numpy as np
from interpretability_lib_sst2.bias_detection import BiasDetector

def test_bias_metrics():
    detector = BiasDetector()
    
    # Create dummy attributions
    # Group A: consistently high attribution
    group_a_tokens = ["he", "him"]
    group_b_tokens = ["she", "her"]
    
    # Create 10 dummy explanations
    attributions = []
    for _ in range(10):
        # Attribution: (word, score)
        # "he" gets 0.8, "she" gets 0.2
        attr = [("he", 0.8), ("she", 0.2), ("other", 0.1)]
        attributions.append(attr)
        
    print("Testing bias analysis with skewed distributions...")
    results = detector.analyze_bias(attributions, group_a_tokens, group_b_tokens)
    
    print("\nResults:")
    print(f"Mean Mass A: {results['mean_mass_a']}")
    print(f"Mean Mass B: {results['mean_mass_b']}")
    print(f"Cohen's d: {results['cohen_d']}")
    print(f"Wasserstein Distance: {results.get('wasserstein_distance', 'N/A')}")
    print(f"Jensen-Shannon Divergence: {results.get('jensen_shannon_divergence', 'N/A')}")
    
    # Validation checks
    if 'wasserstein_distance' not in results:
        print("FAIL: Wasserstein Distance missing")
    elif results['wasserstein_distance'] <= 0:
        # Expect distance > 0 for different distributions
        print(f"WARNING: Wasserstein Distance is {results['wasserstein_distance']}, expected > 0")
    else:
        print("PASS: Wasserstein Distance calculated")
        
    if 'jensen_shannon_divergence' not in results:
        print("FAIL: Jensen-Shannon Divergence missing")
    else:
         print("PASS: Jensen-Shannon Divergence calculated")

if __name__ == "__main__":
    test_bias_metrics()
