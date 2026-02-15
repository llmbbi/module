"""
FeatureAttributorMABSA — 3-class (negative / neutral / positive) variant.

Changes w.r.t. the binary SST-2 attributor
-------------------------------------------
* Prompt: "Classify the sentiment as negative, neutral, or positive."
* `get_predict_proba_fn` extracts logits for three verbalizer tokens and returns
  shape (batch, 3) probabilities.
* Natural-language verbalizers ("negative"/"neutral"/"positive") instead of digits.
* Integrated Gradients targets the predicted class token by default.
* LIME / SHAP class names updated.
"""

import numpy as np
import torch
from lime.lime_text import LimeTextExplainer
import shap


NUM_CLASSES = 3
CLASS_NAMES = ["Negative", "Neutral", "Positive"]
PROMPT_TEMPLATE = (
    "Classify the sentiment as negative, neutral, or positive.\n"
    "Text: {text}\nSentiment:"
)

LABEL_VERBALIZERS = ["negative", "neutral", "positive"]


class FeatureAttributorMABSA:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        # Detect verbalizer token IDs using context-based tokenization.
        # Tokenize "Sentiment: <word>" and find the first new token after the
        # base context to robustly handle various BPE/SentencePiece splits.
        context = "Sentiment:"
        base_ids = self.tokenizer(context, add_special_tokens=False)["input_ids"]
        base_len = len(base_ids)

        self.token_ids = []
        for word in LABEL_VERBALIZERS:
            full_ids = self.tokenizer(
                context + " " + word, add_special_tokens=False
            )["input_ids"]
            self.token_ids.append(full_ids[base_len])

        print(f"  Verbalizer token IDs: {self.token_ids}")
        for i, word in enumerate(LABEL_VERBALIZERS):
            decoded = self.tokenizer.decode([self.token_ids[i]])
            print(f"    {word} -> id={self.token_ids[i]}  decoded='{decoded}'")

    # ------------------------------------------------------------------
    # Prompt helper
    # ------------------------------------------------------------------
    def _format_prompt(self, text):
        return PROMPT_TEMPLATE.format(text=text)

    # ------------------------------------------------------------------
    # Prediction function (returns shape (batch, 3))
    # ------------------------------------------------------------------
    def get_predict_proba_fn(self, temperature=0.0, batch_size=16):
        """Returns a classification function compatible with LIME / SHAP.

        Output shape: (len(texts), 3)   — probabilities for neg / neu / pos.
        """

        def predict_proba(texts):
            if isinstance(texts, np.ndarray):
                texts = texts.flatten().tolist()
            elif isinstance(texts, list) and texts and isinstance(texts[0], (list, tuple)):
                texts = [t[0] for t in texts]
            if isinstance(texts, str):
                texts = [texts]

            original_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "left"

            all_probs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                prompts = [self._format_prompt(t) for t in batch]

                inputs = self.tokenizer(
                    prompts, return_tensors="pt",
                    padding=True, truncation=True, max_length=256,
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits[:, -1, :].float()

                if temperature > 0:
                    logits = logits / temperature

                # Extract logits only for the three digit tokens
                class_logits = logits[:, self.token_ids]          # (batch, 3)
                probs = torch.softmax(class_logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

                del inputs, outputs, logits
                torch.cuda.empty_cache()

            self.tokenizer.padding_side = original_padding_side
            return np.concatenate(all_probs, axis=0)              # (N, 3)

        return predict_proba

    # ------------------------------------------------------------------
    # Normalization helper
    # ------------------------------------------------------------------
    def normalize_attributions(self, attributions):
        arr = np.array(attributions)
        norm = np.linalg.norm(arr)
        return arr / norm if norm != 0 else arr

    # ------------------------------------------------------------------
    # LIME
    # ------------------------------------------------------------------
    def explain_lime(self, text, predict_fn, num_features=10, num_samples=50):
        explainer = LimeTextExplainer(class_names=CLASS_NAMES)
        exp = explainer.explain_instance(
            text, predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            labels=list(range(NUM_CLASSES)),
        )
        # Return explanation for the predicted class
        pred_label = int(np.argmax(predict_fn([text])[0]))
        return exp.as_list(label=pred_label)

    # ------------------------------------------------------------------
    # SHAP (KernelSHAP)
    # ------------------------------------------------------------------
    def explain_shap(self, text, predict_fn, nsamples=64):
        words = text.split()
        if not words:
            return []

        def shap_predict(mask):
            masked_texts = []
            for row in mask:
                masked_texts.append(" ".join(
                    [words[j] for j in range(len(words)) if row[j]]
                ))
            return predict_fn(masked_texts)

        bg = np.zeros((1, len(words)))
        explainer = shap.KernelExplainer(shap_predict, bg)

        instance = np.ones((1, len(words)))
        # Use L1 regularization to avoid singular regression matrix
        # when #features > nsamples
        n_feats = len(words)
        l1_reg = f"num_features({min(n_feats, nsamples)})" if n_feats > nsamples else "auto"
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Linear regression equation is singular")
            shap_values = explainer.shap_values(instance, nsamples=nsamples,
                                                l1_reg=l1_reg, silent=True)

        # shap_values is a list of arrays (one per class).  Pick the predicted class.
        pred_label = int(np.argmax(predict_fn([text])[0]))
        if isinstance(shap_values, list):
            vals = shap_values[pred_label] if pred_label < len(shap_values) else shap_values[0]
        else:
            vals = shap_values

        if hasattr(vals, "shape") and len(vals.shape) > 1:
            vals = vals[0]

        return list(zip(words, vals.flatten()))

    # ------------------------------------------------------------------
    # Integrated Gradients
    # ------------------------------------------------------------------
    def explain_integrated_gradients(self, text, steps=20, target_class=None, ig_batch_size=4):
        """
        Integrated Gradients for 3-class M-ABSA.

        If *target_class* is None the predicted class is used.
        Steps are processed in mini-batches of *ig_batch_size* to avoid OOM.

        Uses an embedding-layer forward hook to be compatible with Unsloth's
        patched forward (which rejects inputs_embeds or requires input_ids).
        We pass input_ids normally but intercept the embedding output via a
        hook, replacing it with the interpolated embeddings needed for IG.
        """
        prompt = self._format_prompt(text)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        # Embedding layer
        try:
            embed_layer = self.model.get_input_embeddings()
        except AttributeError:
            if hasattr(self.model, "model"):
                embed_layer = self.model.model.embed_tokens
            elif hasattr(self.model, "transformer"):
                embed_layer = self.model.transformer.wte
            else:
                raise AttributeError("Could not find embedding layer in model")

        with torch.no_grad():
            input_embeds = embed_layer(input_ids)
            baseline_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id or 0)
            baseline_embeds = embed_layer(baseline_ids)

        diff = (input_embeds.float() - baseline_embeds.float()).squeeze(0)

        original_use_cache = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = False

        # Determine target class from full-scale input if not specified
        if target_class is None:
            with torch.no_grad():
                out0 = self.model(input_ids=input_ids, attention_mask=attention_mask)
                pred_logits = out0.logits[:, -1, :].float()[0, self.token_ids]
                target_class = int(torch.argmax(pred_logits).item())
                del out0
                torch.cuda.empty_cache()

        target_token = self.token_ids[target_class]

        # Process IG steps in mini-batches to avoid OOM
        alphas = np.linspace(0, 1, steps)
        accumulated_grads = torch.zeros_like(diff, dtype=torch.float32)

        try:
            for batch_start in range(0, steps, ig_batch_size):
                batch_alphas = alphas[batch_start : batch_start + ig_batch_size]
                bs = len(batch_alphas)

                # Build interpolated embeddings for this mini-batch
                interp = []
                for alpha in batch_alphas:
                    interp.append(baseline_embeds + alpha * (input_embeds - baseline_embeds))
                target_embeds = torch.cat(interp, dim=0).detach().requires_grad_(True)

                batch_ids = input_ids.repeat(bs, 1)
                batch_attention = attention_mask.repeat(bs, 1)

                # Forward hook: replace the embedding layer's output with our
                # interpolated embeddings while still passing input_ids so that
                # Unsloth's patched forward gets the shapes it expects.
                def _embed_hook(module, inp, out, te=target_embeds):
                    return te

                handle = embed_layer.register_forward_hook(_embed_hook)
                try:
                    outputs = self.model(
                        input_ids=batch_ids,
                        attention_mask=batch_attention,
                    )
                    logits = outputs.logits[:, -1, :].float()
                    target_logits = logits[:, target_token]
                    target_logits.sum().backward()

                    if target_embeds.grad is not None:
                        accumulated_grads += target_embeds.grad.mean(dim=0).float()
                finally:
                    handle.remove()

                del target_embeds, batch_ids, batch_attention, outputs, logits, target_logits
                torch.cuda.empty_cache()

            num_batches = (steps + ig_batch_size - 1) // ig_batch_size
            avg_gradients = accumulated_grads / num_batches
            attributions = (diff * avg_gradients).sum(dim=-1).detach().cpu().numpy()
            attributions = np.nan_to_num(attributions, nan=0.0, posinf=0.0, neginf=0.0)

        except Exception as e:
            print(f"  WARNING: Integrated Gradients failed: {e}")
            attributions = np.zeros(input_ids.shape[1])
        finally:
            self.model.config.use_cache = original_use_cache

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        return self._aggregate_bpe_to_words(text, tokens, attributions)

    # ------------------------------------------------------------------
    # Occlusion
    # ------------------------------------------------------------------
    def explain_occlusion(self, text, predict_fn, mask_token="[UNK]"):
        words = text.split()
        if not words:
            return []

        original_probs = predict_fn([text])[0]
        pred_class = int(np.argmax(original_probs))
        original_prob = original_probs[pred_class]

        attributions = []
        for i, word in enumerate(words):
            masked = words.copy()
            masked[i] = mask_token
            try:
                masked_prob = predict_fn([" ".join(masked)])[0][pred_class]
                attributions.append((word, float(original_prob - masked_prob)))
            except Exception:
                attributions.append((word, 0.0))

        return attributions

    # ------------------------------------------------------------------
    # BPE aggregation (unchanged logic from base class)
    # ------------------------------------------------------------------
    def _aggregate_bpe_to_words(self, original_text, tokens, attributions):
        words = original_text.split()
        word_attributions = []
        current_word_idx = 0
        current_attribution = 0.0
        token_count = 0

        skip_tokens = {
            "<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "[PAD]",
            "<|endoftext|>", "<|file_separator|>", "<|extra_0|>",
        }

        for i, token in enumerate(tokens):
            if i >= len(attributions):
                break
            clean = token.replace("Ġ", "").replace("▁", "").replace(" ", "").replace("##", "").strip()
            if not clean or clean in skip_tokens:
                continue

            is_word_start = token.startswith(("Ġ", "▁", " ")) or i == 0
            if is_word_start and token_count > 0 and current_word_idx < len(words):
                word_attributions.append(
                    (words[current_word_idx], current_attribution / max(1, token_count))
                )
                current_word_idx += 1
                current_attribution = 0.0
                token_count = 0

            current_attribution += attributions[i]
            token_count += 1

        if token_count > 0 and current_word_idx < len(words):
            word_attributions.append(
                (words[current_word_idx], current_attribution / max(1, token_count))
            )

        covered = {w for w, _ in word_attributions}
        for word in words:
            if word not in covered:
                word_attributions.append((word, 0.0))

        return word_attributions

    # ------------------------------------------------------------------
    # Helpers (same API surface as original)
    # ------------------------------------------------------------------
    def get_top_features(self, explanation, top_k=3):
        sorted_features = sorted(explanation, key=lambda x: abs(x[1]), reverse=True)
        return [x[0] for x in sorted_features[:top_k]]

    def get_available_methods(self):
        return ["LIME", "SHAP", "IntegratedGradients", "Occlusion"]

    def explain(self, text, method, predict_fn=None, **kwargs):
        method = method.lower()
        if method == "lime":
            return self.explain_lime(text, predict_fn or self.get_predict_proba_fn(), **kwargs)
        elif method == "shap":
            return self.explain_shap(text, predict_fn or self.get_predict_proba_fn(), **kwargs)
        elif method in ("integratedgradients", "ig", "integrated_gradients"):
            return self.explain_integrated_gradients(text, **kwargs)
        elif method == "occlusion":
            return self.explain_occlusion(text, predict_fn or self.get_predict_proba_fn(), **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}. Available: {self.get_available_methods()}")

    # ------------------------------------------------------------------
    # Identity / stability / cross-method (needed by pipeline helpers)
    # ------------------------------------------------------------------
    def compute_identity(self, explanation1, explanation2):
        """Score 1.0 when two explanations for the *same* input are identical."""
        d1 = dict(explanation1)
        d2 = dict(explanation2)
        common = set(d1) & set(d2)
        if not common:
            return 0.0
        v1 = np.array([d1[w] for w in common])
        v2 = np.array([d2[w] for w in common])
        norm = np.linalg.norm(v1 - v2)
        return float(max(0, 1.0 - norm / (np.linalg.norm(v1) + 1e-8)))

    def compute_stability(self, explanation1, explanation2):
        """Rank correlation between two explanations (for perturbed inputs)."""
        from scipy.stats import spearmanr
        d1 = dict(explanation1)
        d2 = dict(explanation2)
        common = sorted(set(d1) & set(d2))
        if len(common) < 2:
            return 0.0
        v1 = [d1[w] for w in common]
        v2 = [d2[w] for w in common]
        corr, _ = spearmanr(v1, v2)
        return float(corr) if not np.isnan(corr) else 0.0

    def compute_cross_method_agreement(self, explanation1, explanation2, top_k=5):
        """Jaccard overlap of top-k features between two methods."""
        top1 = set(self.get_top_features(explanation1, top_k))
        top2 = set(self.get_top_features(explanation2, top_k))
        if not top1 and not top2:
            return 1.0
        if not top1 or not top2:
            return 0.0
        return len(top1 & top2) / len(top1 | top2)
