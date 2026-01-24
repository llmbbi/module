
# from unsloth import FastLanguageModel
from transformers import AutoTokenizer

model_name = "unsloth/Llama-3.2-1B-Instruct"
max_seq_length = 512

# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name=model_name,
#     max_seq_length=max_seq_length,
#     dtype=None,
#     load_in_4bit=True,
# )
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

t0 = tokenizer("0", add_special_tokens=False)["input_ids"]
t_sp_0 = tokenizer(" 0", add_special_tokens=False)["input_ids"]

print(f"'0' tokens: {t0}")
print(f"' 0' tokens: {t_sp_0}")

token0_id = t0[0]
token_sp_0_id = t_sp_0[-1]

print(f"ID used in llama script (token0_id): {token0_id}")
print(f"ID used in lib (token_sp_0_id): {token_sp_0_id}")

prompt = "Sentiment:"
next_is_space = tokenizer(prompt + " ", add_special_tokens=False)["input_ids"][-1]
next_is_0 = tokenizer(prompt + "0", add_special_tokens=False)["input_ids"][-1] 
# (This test is slightly flawed as merging depends on context, but valid for checking if space is separate)

prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
prompt_space_ids = tokenizer(prompt + " ", add_special_tokens=False)["input_ids"]
prompt_space_0_ids = tokenizer(prompt + " 0", add_special_tokens=False)["input_ids"]

print(f"Prompt IDs: {prompt_ids}")
print(f"Prompt + Space IDs: {prompt_space_ids}")
print(f"Prompt + Space + 0 IDs: {prompt_space_0_ids}")


def _format_prompt(text, label=None):
    # The FIXED version
    prompt = f"Classify the sentiment as 0 (negative) or 1 (positive).\nText: {text}\nSentiment:"
    if label is not None:
        prompt += f" {label}"
    return prompt

text = "This film is terrible."
label = 0

# Test 1: As per my fix
prompt = _format_prompt(text)
full_text = prompt + f" {label}"

print(f"Prompt: '{prompt}'")
print(f"Full Text: '{full_text}'")

prompt_tokens = tokenizer(prompt, add_special_tokens=True)["input_ids"]
full_tokens = tokenizer(full_text, add_special_tokens=True)["input_ids"]

print(f"Prompt tokens len: {len(prompt_tokens)}")
print(f"Full tokens len: {len(full_tokens)}")
print(f"Prompt tokens: {prompt_tokens}")
print(f"Full tokens: {full_tokens}")

# Check masking
prompt_length = len(prompt_tokens)
labels = list(full_tokens)
for i in range(min(prompt_length, len(labels))):
    labels[i] = -100

print(f"Labels:   {labels}")
print(f"Last label token: {labels[-1]}")
print(f"Should be: {tokenizer(str(label), add_special_tokens=False)['input_ids']}")
