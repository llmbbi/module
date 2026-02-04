
from transformers import AutoTokenizer

def check_llama():
    model_name = "unsloth/Llama-3.2-1B"
    print(f"\nChecking: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    t0 = tokenizer(" 0", add_special_tokens=False)["input_ids"]
    print(f"' 0' ids: {t0}")
    print(f"Length: {len(t0)}")
    
    if len(t0) == 1:
        print("Result: Fused (Logic will SKIP adding space)")
    else:
        print("Result: Split (Logic will ADD space)")

if __name__ == "__main__":
    check_llama()
