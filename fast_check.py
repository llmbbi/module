from transformers import AutoTokenizer

model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")

for text in ["0", "1", " 0", " 1"]:
    ids_no_special = tokenizer(text, add_special_tokens=False)["input_ids"]
    ids_with_special = tokenizer(text, add_special_tokens=True)["input_ids"]
    print(f"'{text}' (no special): {ids_no_special}")
    print(f"'{text}' (with special): {ids_with_special}")

prompt = "Sentiment:"
full = prompt + " 0"
print(f"Full tokens: {tokenizer(full)['input_ids']}")
