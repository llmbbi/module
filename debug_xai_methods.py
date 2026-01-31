
import os
import torch
import numpy as np
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

def test_xai_methods():
    model_name = "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = 512
    load_in_4bit = True
    
    print(f"Loading model {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    
    FastLanguageModel.for_inference(model)
    
    text = "This movie was absolutely fantastic and I loved every minute of it."
    prompt = f"Classify the sentiment as 0 (negative) or 1 (positive).\nText: {text}\nSentiment:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("\n--- Testing Attention Rollout ---")
    try:
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            attentions = outputs.attentions
            if attentions:
                print(f"SUCCESS: Got attentions (tuple of length {len(attentions)})")
                print(f"First layer attention shape: {attentions[0].shape}")
            else:
                print("FAILURE: No attentions returned (empty list/tuple)")
    except Exception as e:
        print(f"FAILURE: Attention Rollout failed with error: {type(e).__name__}: {e}")

    print("\n--- Testing Integrated Gradients ---")
    try:
        # Switch off cache for gradient computation
        model.config.use_cache = False
        
        # Get embeddings
        input_ids = inputs["input_ids"]
        embed_layer = model.get_input_embeddings()
        input_embeds = embed_layer(input_ids)
        
        # Scaling inputs
        steps = 5
        alpha = torch.linspace(0, 1, steps).to(model.device).view(-1, 1, 1).to(input_embeds.dtype)
        # Create baseline (zeros or pad embeddings)
        baseline_embeds = torch.zeros_like(input_embeds)
        
        # Interpolated embeds: (steps, seq_len, embed_dim)
        scaled_embeds = baseline_embeds + alpha * (input_embeds - baseline_embeds)
        scaled_embeds.requires_grad_(True)
        
        # Forward pass
        attention_mask = inputs["attention_mask"].repeat(steps, 1)
        
        outputs = model(inputs_embeds=scaled_embeds, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]
        
        token_1 = tokenizer.encode("1", add_special_tokens=False)[-1]
        target_logits = logits[:, token_1]
        
        # Backward pass
        target_logits.sum().backward()
        
        if scaled_embeds.grad is not None:
            grads = scaled_embeds.grad
            print(f"SUCCESS: Got gradients with shape {grads.shape}")
            grad_norm = grads.norm().item()
            print(f"Gradient norm: {grad_norm}")
        else:
            print("FAILURE: No gradients computed")
            
    except Exception as e:
        print(f"FAILURE: Integrated Gradients failed with error: {type(e).__name__}: {e}")
    finally:
        model.config.use_cache = True

    print("\n--- Model Architecture Info ---")
    print(f"Model class: {type(model).__name__}")
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        print(f"Number of layers: {len(model.model.layers)}")
        first_layer = model.model.layers[0]
        print(f"First layer self_attn: {type(first_layer.self_attn).__name__}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_xai_methods()
    else:
        print("CUDA not available. Skipping GPU test.")
