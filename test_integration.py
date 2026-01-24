
import sys
import os
import torch
import numpy as np
from interpretability_lib import LoRAFineTuner, FeatureAttributor, MetricsCalculator, BiasDetector
from datasets import Dataset

def mock_predict_fn(texts):
    # Returns random probabilities for testing
    probs = []
    for _ in texts:
        p = np.random.random()
        probs.append([1-p, p])
    return np.array(probs)

def main():
    print("Starting Integration Test (CPU Safe Mode)...")
    
    # 1. Test Fine-Tuning Module imports / structure
    print("\n--- Testing FineTuner Structure ---")
    try:
        finetuner = LoRAFineTuner(model_name="unsloth/Llama-3.2-1B-Instruct", output_dir="test_outputs")
        print("FineTuner instantiated.")
        # We skip load_model() as it requires GPU for Unsloth usually
        print("Skipping actual model loading (GPU required).")
    except Exception as e:
        print(f"FineTuner instantiation failed: {e}")

    # 2. Test Feature Attribution logic (with mock)
    print("\n--- Testing FeatureAttributor Logic ---")
    # We can't implement explain_lime without a real model easily unless we mock the model object too.
    # But FeatureAttributor requires model and tokenizer in init.
    class MockModel:
        device = "cpu"
    class MockTokenizer:
        padding_side = "right"
        def __call__(self, *args, **kwargs):
            return {"input_ids": torch.tensor([[1,2,3]]), "attention_mask": torch.tensor([[1,1,1]])}
        def decode(self, *args, **kwargs): return "text"
        
    attributor = FeatureAttributor(MockModel(), MockTokenizer())
    # Mock token IDs
    attributor.token_0 = 0
    attributor.token_1 = 1
    
    print("FeatureAttributor instantiated.")

    # 3. Test Metrics
    print("\n--- Testing Metrics ---")
    metrics = MetricsCalculator(predict_fn=mock_predict_fn)
    
    text = "This movie is terrible"
    top_feats = ["terrible", "movie"]
    orig_prob = 0.9
    
    # Test Calculate Faithfulness
    # It will call mock_predict_fn
    faith = metrics.calculate_faithfulness_deletion(text, top_feats, orig_prob)
    print(f"Faithfulness (Deletion): {faith}")
    
    # Test Gini
    gini = metrics.calculate_gini([0.1, 0.5, 0.4])
    print(f"Gini Index: {gini} (Expected: ~0.34)")

    # 4. Test Bias Detection
    print("\n--- Testing Bias Detection ---")
    bias_detector = BiasDetector()
    
    fake_attrs = [
        [("he", 0.5), ("she", 0.1), ("is", 0.1)],
        [("he", 0.4), ("she", 0.2), ("smart", 0.1)],
        [("he", 0.6), ("she", 0.05), ("smart", 0.1)],
        [("he", 0.45), ("she", 0.15), ("smart", 0.1)],
        [("he", 0.55), ("she", 0.1), ("smart", 0.1)],
    ]
    # We need slightly more samples for t-test to not complain?
    
    res = bias_detector.analyze_bias(fake_attrs, ["he", "him"], ["she", "her"])
    print("Bias Analysis Result:", res)
    
    print("\nIntegration Test Complete!")

if __name__ == "__main__":
    main()
