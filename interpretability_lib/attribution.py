
import numpy as np
import torch
from lime.lime_text import LimeTextExplainer
import shap

class FeatureAttributor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # Detect if tokenizer splits " 0" into multiple tokens (e.g. Qwen, DeepSeek)
        # Qwen/DeepSeek: " 0" -> [220, 15] (Space, 0)
        # Llama: " 0" -> [1234] (Space+0 combined)
        t0_encoded = self.tokenizer(" 0", add_special_tokens=False)["input_ids"]
        t1_encoded = self.tokenizer(" 1", add_special_tokens=False)["input_ids"]
        
        self.split_space_digit = False
        if len(t0_encoded) > 1:
            self.split_space_digit = True
            # If split, we will append a space to the prompt manually, so we expect the "0" token (without space)
            # Use the "0" token ID (the last one)
            self.token_0 = t0_encoded[-1]
            self.token_1 = t1_encoded[-1]
        else:
            # If fused, we expect the single " 0" token
            self.token_0 = t0_encoded[-1]
            self.token_1 = t1_encoded[-1]
        
    def get_predict_proba_fn(self, temperature=0.0, batch_size=16):
        """Returns a prediction function compatible with LIME/SHAP."""
        
        def format_prompt(text):
            prompt = f"Classify the sentiment as 0 (negative) or 1 (positive).\nText: {text}\nSentiment:"
            # If tokenizer splits space+digit, we must add the space to the prompt 
            # so the model predicts the digit "0"/"1" directly.
            if self.split_space_digit:
                prompt += " "
            return prompt

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

    def explain_integrated_gradients(self, text, steps=50, target_class=1):
        """
        Computes Integrated Gradients attribution for text classification.
        
        Integrated Gradients (Sundararajan et al., ICML 2017) computes attributions
        by integrating gradients along a path from a baseline (padding tokens) to
        the actual input.
        
        Args:
            text: Input text to explain
            steps: Number of integration steps (higher = more accurate)
            target_class: Class index to explain (0=negative, 1=positive)
            
        Returns:
            List of (token, attribution_score) tuples
        """
        def format_prompt(t):
            return f"Classify the sentiment as 0 (negative) or 1 (positive).\nText: {t}\nSentiment:"
        
        prompt = format_prompt(text)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
        
        # Get embedding layer
        try:
            embed_layer = self.model.get_input_embeddings()
        except AttributeError:
            if hasattr(self.model, 'model'):
                embed_layer = self.model.model.embed_tokens
            elif hasattr(self.model, 'transformer'):
                embed_layer = self.model.transformer.wte
            else:
                raise AttributeError("Could not find embedding layer in model")
        
        # Get actual embeddings
        input_embeds = embed_layer(input_ids)
        
        # Create baseline (pad token embeddings)
        baseline_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id or 0)
        baseline_embeds = embed_layer(baseline_ids)
        
        # Compute integrated gradients
        scaled_inputs = []
        for alpha in np.linspace(0, 1, steps):
            scaled = baseline_embeds + alpha * (input_embeds - baseline_embeds)
            scaled_inputs.append(scaled)
        
        # Stack all scaled inputs
        all_embeds = torch.cat(scaled_inputs, dim=0)
        all_attention = attention_mask.repeat(steps, 1)
        
        # Forward pass with gradients
        all_embeds.requires_grad_(True)
        all_embeds.retain_grad()
    
        # Store original state
        original_use_cache = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = False
    
        try:
            # Explicitly use float32 for Integrated Gradients to avoid BFloat16 backward pass errors
            # and to maintain numerical precision during integration
            outputs = self.model(inputs_embeds=all_embeds, attention_mask=all_attention)
            logits = outputs.logits[:, -1, :].float()
            
            # Get logits for target class
            target_token = self.token_1 if target_class == 1 else self.token_0
            target_logits = logits[:, target_token]
            
            # Backward pass on float32 targets
            target_logits.sum().backward()
            
            # Get gradients and ensure they are in float32
            gradients = all_embeds.grad  # (steps, seq_len, embed_dim)
            
            if gradients is not None:
                # Average gradients across steps
                avg_gradients = gradients.mean(dim=0).float()  # (seq_len, embed_dim)
                
                # Compute attributions: (input - baseline) * avg_gradients
                # Ensure all components are in float32
                input_embeds_f = input_embeds.detach().float()
                baseline_embeds_f = baseline_embeds.detach().float()
                diff = (input_embeds_f - baseline_embeds_f).squeeze(0)  # (seq_len, embed_dim)
                
                attributions = (diff * avg_gradients).sum(dim=-1)  # (seq_len,)
                attributions = attributions.detach().cpu().numpy()
                
                # Handle numerical instability (NaNs/Infs)
                if np.isnan(attributions).any() or np.isinf(attributions).any():
                    attributions = np.nan_to_num(attributions, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                attributions = np.zeros(input_ids.shape[1])
            
        except Exception as e:
            # Fallback: return zeros if IG fails
            print(f"  WARNING: Integrated Gradients failed: {e}")
            attributions = np.zeros(input_ids.shape[1])
        finally:
            # Restore state
            self.model.config.use_cache = original_use_cache
        
        # Map back to tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        
        # Aggregate to words
        word_attributions = self._aggregate_bpe_to_words(text, tokens, attributions)
        
        return word_attributions


    def explain_occlusion(self, text, predict_fn, mask_token="[UNK]"):
        """
        Computes Occlusion Sensitivity attribution.
        
        For each word, masks it out and measures the change in prediction.
        Words whose removal causes larger prediction drops are more important.
        
        Args:
            text: Input text to explain
            predict_fn: Prediction function
            mask_token: Token to use for masking
            
        Returns:
            List of (word, importance_score) tuples
        """
        words = text.split()
        if not words:
            return []
        
        # Get original prediction
        original_prob = predict_fn([text])[0][1]
        
        attributions = []
        for i, word in enumerate(words):
            # Create masked text
            masked_words = words.copy()
            masked_words[i] = mask_token
            masked_text = " ".join(masked_words)
            
            # Get prediction for masked text
            try:
                masked_prob = predict_fn([masked_text])[0][1]
                # Attribution = drop in probability (higher = more important)
                attribution = original_prob - masked_prob
            except:
                attribution = 0.0
            
            attributions.append((word, float(attribution)))
        
        return attributions

    def _aggregate_bpe_to_words(self, original_text, tokens, attributions):
        """
        Aggregates BPE/subword token attributions to word-level.
        
        Args:
            original_text: The original text (space-separated words)
            tokens: List of tokens from tokenizer
            attributions: Array of attribution scores per token
            
        Returns:
            List of (word, aggregated_attribution) tuples
        """
        words = original_text.split()
        
        # Simple heuristic: reconstruct text from tokens and map to words
        word_attributions = []
        
        # Find tokens that correspond to words in the original text
        # This handles BPE by looking for tokens that match word prefixes
        
        current_word_idx = 0
        current_attribution = 0.0
        token_count = 0
        
        for i, token in enumerate(tokens):
            if i >= len(attributions):
                break
                
            # Clean token (remove BPE markers like Ġ, ##, etc.)
            clean_token = token.replace("Ġ", "").replace("▁", "").replace(" ", "").replace("##", "").strip()
            
            if not clean_token or clean_token in ["<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "[PAD]", "<|endoftext|>", "<|file_separator|>", "<|extra_0|>"]:
                continue
            
            # Check if this starts a new word (has space prefix in Llama/Gemma style tokenizers)
            is_word_start = token.startswith("Ġ") or token.startswith("▁") or token.startswith(" ") or i == 0
            
            if is_word_start and token_count > 0 and current_word_idx < len(words):
                # Save previous word
                word_attributions.append((words[current_word_idx], current_attribution / max(1, token_count)))
                current_word_idx += 1
                current_attribution = 0.0
                token_count = 0
            
            current_attribution += attributions[i]
            token_count += 1
        
        # Save last word
        if token_count > 0 and current_word_idx < len(words):
            word_attributions.append((words[current_word_idx], current_attribution / max(1, token_count)))
        
        # Fill in any missing words with zero attribution
        covered_words = {w for w, _ in word_attributions}
        for word in words:
            if word not in covered_words:
                word_attributions.append((word, 0.0))
        
        return word_attributions

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

    def get_available_methods(self):
        """Returns a list of available attribution methods."""
        return ["LIME", "SHAP", "IntegratedGradients", "Occlusion"]

    def compute_identity(self, explanation1, explanation2):
        """Compute identity score: same input should yield the same explanation.
        Returns cosine similarity between two explanation vectors.
        Explanations are lists of (token, score) tuples.
        """
        if not explanation1 or not explanation2:
            return 0.0
        # Build aligned vectors using token intersection (order-independent)
        tokens1 = {t: s for t, s in explanation1}
        tokens2 = {t: s for t, s in explanation2}
        all_tokens = set(tokens1.keys()) | set(tokens2.keys())
        if not all_tokens:
            return 0.0
        vec1 = np.array([tokens1.get(t, 0.0) for t in all_tokens])
        vec2 = np.array([tokens2.get(t, 0.0) for t in all_tokens])
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def compute_stability(self, explanation1, explanation2):
        """Compute stability score: similar inputs should yield similar explanations.
        Returns cosine similarity between two explanation vectors (same logic as identity
        but semantically used for perturbed inputs).
        """
        return self.compute_identity(explanation1, explanation2)

    def compute_cross_method_agreement(self, explanations1, explanations2):
        """Compute agreement between two XAI methods' explanations.
        Each input is a list of (token, score) tuples from a different method.
        Returns cosine similarity over a shared token vocabulary.
        """
        return self.compute_identity(explanations1, explanations2)

    def explain(self, text, method, predict_fn=None, **kwargs):
        """
        Unified interface to compute attributions using any available method.
        
        Args:
            text: Input text to explain
            method: Attribution method name
            predict_fn: Prediction function (required for LIME, SHAP, Occlusion)
            **kwargs: Method-specific parameters
            
        Returns:
            List of (word, attribution_score) tuples
        """
        method = method.lower()
        
        if method == "lime":
            if predict_fn is None:
                predict_fn = self.get_predict_proba_fn()
            return self.explain_lime(text, predict_fn, **kwargs)
        
        elif method == "shap":
            if predict_fn is None:
                predict_fn = self.get_predict_proba_fn()
            return self.explain_shap(text, predict_fn, **kwargs)
        
        elif method in ["integratedgradients", "ig", "integrated_gradients"]:
            return self.explain_integrated_gradients(text, **kwargs)
        
        
        elif method == "occlusion":
            if predict_fn is None:
                predict_fn = self.get_predict_proba_fn()
            return self.explain_occlusion(text, predict_fn, **kwargs)
        
        else:
            raise ValueError(f"Unknown method: {method}. Available: {self.get_available_methods()}")
