
import numpy as np
import torch
from lime.lime_text import LimeTextExplainer
import shap

class FeatureAttributor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # Match training/eval prompt which uses a space before the label
        # Use the digit token id following a space; take the last id
        self.token_0 = self.tokenizer(" 0", add_special_tokens=False)["input_ids"][-1]
        self.token_1 = self.tokenizer(" 1", add_special_tokens=False)["input_ids"][-1]
        
    def get_predict_proba_fn(self, temperature=0.0, batch_size=16):
        """Returns a prediction function compatible with LIME/SHAP."""
        
        def format_prompt(text):
            return f"Classify the sentiment as 0 (negative) or 1 (positive).\nText: {text}\nSentiment:"

        def predict_proba(texts):
            if isinstance(texts, np.ndarray): 
                texts = texts.flatten().tolist()
            elif isinstance(texts, list) and len(texts) > 0 and isinstance(texts[0], (list, tuple)): 
                texts = [t[0] for t in texts]
            if isinstance(texts, str): 
                texts = [texts]

            original_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "left"
            
            probs = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                prompts = [format_prompt(t) for t in batch]
                
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # With left padding, the last token is always at index -1
                    logits = outputs.logits[:, -1, :].float()
                
                # Apply temperature scaling
                if temperature > 0:
                    logits = logits / temperature
                
                prob_1 = torch.softmax(logits[:, [self.token_0, self.token_1]], dim=-1)[:, 1].cpu().numpy()
                probs.extend(prob_1)
                del inputs, outputs, logits
                torch.cuda.empty_cache()
            
            self.tokenizer.padding_side = original_padding_side
                
            return np.stack([1 - np.array(probs), np.array(probs)], axis=1)
        
        return predict_proba

    def normalize_attributions(self, attributions):
        """Explicit normalization of attribution values (e.g. to unit norm or sum to 1)."""
        # Here we implement L2 normalization as a standard
        arr = np.array(attributions)
        norm = np.linalg.norm(arr)
        if norm == 0:
            return arr
        return arr / norm

    def explain_lime(self, text, predict_fn, num_features=10, num_samples=50):
        """Generates LIME explanations."""
        explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
        exp = explainer.explain_instance(text, predict_fn, num_features=num_features, num_samples=num_samples)
        return exp.as_list()

    def explain_shap(self, text, predict_fn, nsamples=40):
        """Generates word-level KernelSHAP explanations."""
        words = text.split()
        if not words:
            return []

        def shap_predict(mask):
            # mask is a binary array (batch x num_words)
            masked_texts = []
            for row in mask:
                masked_text = " ".join([words[j] for j in range(len(words)) if row[j]])
                masked_texts.append(masked_text)
            return predict_fn(masked_texts)

        # Background: all zeros (all words masked)
        bg = np.zeros((1, len(words)))
        explainer = shap.KernelExplainer(shap_predict, bg)
        
        # Explain: all ones (original text)
        instance = np.ones((1, len(words)))
        shap_values = explainer.shap_values(instance, nsamples=nsamples, silent=True)
        
        # Extract values for the target class (index 1)
        if isinstance(shap_values, list):
            idx = 1 if len(shap_values) > 1 else 0
            vals = shap_values[idx]
        else:
            vals = shap_values

        if hasattr(vals, 'shape') and len(vals.shape) > 1:
            vals = vals[0]
            
        return list(zip(words, vals.flatten()))

    def get_top_features(self, explanation, top_k=3):
        """Helper to get top K features by absolute attribution score."""
        sorted_features = sorted(explanation, key=lambda x: abs(x[1]), reverse=True)
        return [x[0] for x in sorted_features[:top_k]]

    def align_bpe_to_words(self, text, atom_attributions):
        """
        aggregates BPE token attributions to word-level attributions.
        This is a placeholder for the advanced alignment logic requested.
        """
        words = text.split()
        # Logic to map tokens to words would go here.
        # For now, we return as is or implement a simple heuristic.
        return atom_attributions

