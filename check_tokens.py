from unsloth import FastLanguageModel
import torch

model_names = ["meta-llama/Llama-3.2-1B", "unsloth/Llama-3.2-1B-Instruct"]

for name in model_names:
    print(f"\n--- {name} ---")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=name,
            max_seq_length=512,
            load_in_4bit=True
        )
        
        t0 = tokenizer("0", add_special_tokens=False)["input_ids"]
        t1 = tokenizer("1", add_special_tokens=False)["input_ids"]
        t_space0 = tokenizer(" 0", add_special_tokens=False)["input_ids"]
        t_space1 = tokenizer(" 1", add_special_tokens=False)["input_ids"]
        
        print(f"'0': {t0}")
        print(f"'1': {t1}")
        print(f"' 0': {t_space0}")
        print(f"' 1': {t_space1}")
        
        prompt = "Sentiment:"
        full_0 = prompt + " 0"
        full_1 = prompt + " 1"
        
        tokens_0 = tokenizer(full_0, add_special_tokens=True)["input_ids"]
        tokens_1 = tokenizer(full_1, add_special_tokens=True)["input_ids"]
        
        print(f"Full 'Sentiment: 0' tokens: {tokens_0}")
        print(f"Full 'Sentiment: 1' tokens: {tokens_1}")
        
    except Exception as e:
        print(f"Error: {e}")
