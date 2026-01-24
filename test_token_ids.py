from transformers import AutoTokenizer

model_name = "unsloth/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Classify the sentiment as 0 (negative) or 1 (positive).\nText: Hello!\nSentiment:"
label_0 = " 0"
label_1 = " 1"

full_0 = prompt + label_0
full_1 = prompt + label_1

ids_0 = tokenizer(full_0, add_special_tokens=False)["input_ids"]
ids_1 = tokenizer(full_1, add_special_tokens=False)["input_ids"]

print(f"Full 0 ids: {ids_0}")
print(f"Full 1 ids: {ids_1}")
print(f"Last token 0: {ids_0[-1]}")
print(f"Last token 1: {ids_1[-1]}")

bare_0 = tokenizer("0", add_special_tokens=False)["input_ids"]
bare_1 = tokenizer("1", add_special_tokens=False)["input_ids"]

print(f"Bare 0: {bare_0}")
print(f"Bare 1: {bare_1}")

space_0 = tokenizer(" 0", add_special_tokens=False)["input_ids"]
space_1 = tokenizer(" 1", add_special_tokens=False)["input_ids"]

print(f"Space 0: {space_0}")
print(f"Space 1: {space_1}")
