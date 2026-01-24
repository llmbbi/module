
from transformers import AutoTokenizer

model_name = "unsloth/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

token_0_space = tokenizer(" 0", add_special_tokens=False)["input_ids"][-1]
token_1_space = tokenizer(" 1", add_special_tokens=False)["input_ids"][-1]
token_0_nospace = tokenizer("0", add_special_tokens=False)["input_ids"][-1]
token_1_nospace = tokenizer("1", add_special_tokens=False)["input_ids"][-1]

print(f"Token ' 0': {token_0_space}")
print(f"Token '0': {token_0_nospace}")

# Scenario 1: XAI (Good)
# Prompt: "...Sentiment:"
# Full: "...Sentiment: 0" (constructed as prompt + " 0")
text_xai_prompt = "Sentiment:"
text_xai_full = "Sentiment: 0"

print("\n--- XAI (Good) ---")
print(f"Prompt: '{text_xai_prompt}'")
print(f"Full:   '{text_xai_full}'")
tokens_xai_prompt = tokenizer(text_xai_prompt, add_special_tokens=False)["input_ids"]
tokens_xai_full = tokenizer(text_xai_full, add_special_tokens=False)["input_ids"]
print(f"Tokens Prompt: {tokens_xai_prompt}")
print(f"Tokens Full:   {tokens_xai_full}")
# What is the token after the prompt?
suffix_tokens = tokens_xai_full[len(tokens_xai_prompt):]
print(f"Suffix Tokens: {suffix_tokens}")

# Scenario 2: Modular (Suspected Bad)
# Prompt: "...Sentiment: "
# Full: "...Sentiment: 0" (constructed as prompt + "0")
text_mod_prompt = "Sentiment: "
text_mod_full = "Sentiment: 0"

print("\n--- Modular (Bad?) ---")
print(f"Prompt: '{text_mod_prompt}'")
print(f"Full:   '{text_mod_full}'")
tokens_mod_prompt = tokenizer(text_mod_prompt, add_special_tokens=False)["input_ids"]
tokens_mod_full = tokenizer(text_mod_full, add_special_tokens=False)["input_ids"]
print(f"Tokens Prompt: {tokens_mod_prompt}")
print(f"Tokens Full:   {tokens_mod_full}")
suffix_tokens_mod = tokens_mod_full[len(tokens_mod_prompt):]
print(f"Suffix Tokens: {suffix_tokens_mod}")

# Check what the evaluation logic targets
print(f"\nEvaluation targets tokens: {token_0_space} (' 0'), {token_1_space} (' 1')")
