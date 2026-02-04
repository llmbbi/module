
from unsloth import FastLanguageModel
import torch

def check_model(model_name):
    print(f"\nChecking: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=True
    )
    
    print(f"EOS Token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"Pad Token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    
    # Check label tokenization
    t0 = tokenizer(" 0", add_special_tokens=False)["input_ids"]
    t1 = tokenizer(" 1", add_special_tokens=False)["input_ids"]
    t0_nospace = tokenizer("0", add_special_tokens=False)["input_ids"]
    
    print(f"' 0' ids: {t0}")
    print(f"' 1' ids: {t1}")
    print(f"'0' (no space) ids: {t0_nospace}")
    
    # Check what 'token_0' logic in attribution.py would pick
    print(f"Attribution logic ' 0' last id: {t0[-1]}")
    
    # Check simple generation
    prompt = "Classify the sentiment as 0 (negative) or 1 (positive).\nText: This movie was fantastic and I loved it.\nSentiment:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Check logits for next token
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        
        top_k = torch.topk(probs, 5)
        print("Top 5 next tokens:")
        for i in range(5):
            tok_id = top_k.indices[0][i].item()
            tok_prob = top_k.values[0][i].item()
            print(f"  {tokenizer.decode([tok_id])} ({tok_id}): {tok_prob:.4f}")
            
    # Check generation output (especially for DeepSeek thinking)
    print("Generating...")
    out = model.generate(**inputs, max_new_tokens=20)
    print("Output:", tokenizer.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    check_model("unsloth/Qwen2.5-1.5B")
    check_model("unsloth/DeepSeek-R1-Distill-Qwen-1.5B")
