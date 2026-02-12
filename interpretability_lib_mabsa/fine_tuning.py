"""
LoRAFineTunerMABSA — 3-class fine-tuning for M-ABSA.

Changes w.r.t. the binary SST-2 fine-tuner
--------------------------------------------
* Prompt: "Classify the sentiment as negative, neutral, or positive."
* `_format_prompt` and `_tokenize_with_masking` handle label ∈ {0, 1, 2}.
* Natural-language verbalizers ("negative"/"neutral"/"positive") instead of digits.
"""

import os
import torch
from unsloth import FastLanguageModel
from transformers import (
    TrainingArguments,
    DataCollatorForLanguageModeling,
    default_data_collator,
    EarlyStoppingCallback,
)
from trl import SFTTrainer, SFTConfig
from datasets import Dataset


MABSA_PROMPT = (
    "Classify the sentiment as negative, neutral, or positive.\n"
    "Text: {text}\nSentiment:"
)

LABEL_VERBALIZERS = {0: "negative", 1: "neutral", 2: "positive"}


class LoRAFineTunerMABSA:
    def __init__(self, model_name: str, output_dir: str, max_seq_length: int = 512):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        self.model = None
        self.tokenizer = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_model(self, load_in_4bit: bool = True):
        dtype = None  # auto-detect
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        return self.model, self.tokenizer

    def enable_inference_mode(self):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        FastLanguageModel.for_inference(self.model)

    # ------------------------------------------------------------------
    # LoRA configuration
    # ------------------------------------------------------------------
    def _detect_target_modules(self):
        """Auto-detect appropriate LoRA target modules based on model architecture."""
        module_names = set(
            name.split(".")[-1] for name, _ in self.model.named_modules()
        )
        # Llama / Mistral / Qwen style
        llama_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        # Falcon / BLOOM style
        falcon_modules = [
            "query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h",
        ]
        if "q_proj" in module_names:
            return llama_modules
        elif "query_key_value" in module_names:
            return falcon_modules
        else:
            # Fallback: find all Linear sub-module names
            linear_names = list(set(
                name.split(".")[-1]
                for name, mod in self.model.named_modules()
                if isinstance(mod, torch.nn.Linear)
            ))
            print(f"  Auto-detected LoRA target modules (fallback): {linear_names}")
            return linear_names if linear_names else llama_modules

    def configure_lora(self, r: int = 16, lora_alpha: int = 16, dropout: float = 0.0):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        target_modules = self._detect_target_modules()
        print(f"  LoRA target modules: {target_modules}")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
            use_rslora=False,
            loftq_config=None,
        )
        return self.model

    # ------------------------------------------------------------------
    # Prompt / tokenisation (3-class)
    # ------------------------------------------------------------------
    def _format_prompt(self, text, label=None):
        prompt = MABSA_PROMPT.format(text=text)
        if label is not None:
            prompt += f" {label}"
        return prompt

    def _tokenize_with_masking(self, examples):
        texts = examples["sentence"]
        labels = examples["label"]

        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for text, label in zip(texts, labels):
            prompt = self._format_prompt(text)
            # Append EOS after the label so the model learns a clean
            # generation boundary (predict label then stop).
            eos = self.tokenizer.eos_token or ""
            label_text = LABEL_VERBALIZERS[label]
            full_text = prompt + f" {label_text}{eos}"

            prompt_tokens = self.tokenizer(prompt, add_special_tokens=True)
            full_tokens = self.tokenizer(
                full_text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
            )

            input_ids = full_tokens["input_ids"]
            attention_mask = full_tokens["attention_mask"]

            # Mask prompt tokens with -100 so the loss is computed only
            # on the verbalizer label tokens + EOS (everything the model
            # must learn to *generate*).  Also mask padding.
            labels_masked = list(input_ids)
            prompt_length = len(prompt_tokens["input_ids"])
            for i in range(min(prompt_length, len(labels_masked))):
                labels_masked[i] = -100
            for i in range(len(attention_mask)):
                if attention_mask[i] == 0:
                    labels_masked[i] = -100

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels_masked)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, train_dataset, eval_dataset=None, epochs=1.0,
              learning_rate=1e-4, batch_size=2):
        if self.model is None:
            raise ValueError("Model not configured. Call load_model() and configure_lora() first.")

        print("Preprocessing training data…")
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        formatted_dataset = train_dataset.map(
            self._tokenize_with_masking, batched=True,
            remove_columns=train_dataset.column_names,
        )

        # Optionally prepare eval dataset for early stopping
        formatted_eval = None
        callbacks = []
        eval_strategy = "no"
        if eval_dataset is not None:
            formatted_eval = eval_dataset.map(
                self._tokenize_with_masking, batched=True,
                remove_columns=eval_dataset.column_names,
            )
            eval_strategy = "steps"
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=15,
                early_stopping_threshold=0.0005,
            ))
            print(f"  Early stopping enabled (patience=15, threshold=0.0005) on {len(eval_dataset)} eval samples")

        from transformers import Trainer, TrainingArguments as TA

        # NOTE: We use default_data_collator (not DataCollatorForLanguageModeling)
        # because DataCollatorForLanguageModeling overwrites the custom labels
        # column with input_ids.clone(), discarding the prompt masking (-100)
        # that _tokenize_with_masking carefully applied.  With the old collator
        # the training loss was dominated by reconstructing the prompt, not by
        # learning to predict the sentiment label.
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=formatted_dataset,
            eval_dataset=formatted_eval,
            data_collator=default_data_collator,
            callbacks=callbacks,
            args=TA(
                output_dir=f"{self.output_dir}/checkpoints",
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=4,
                warmup_ratio=0.05,
                learning_rate=learning_rate,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=10,
                eval_strategy=eval_strategy,
                eval_steps=25,
                load_best_model_at_end=bool(formatted_eval),
                metric_for_best_model="eval_loss" if formatted_eval else None,
                greater_is_better=False if formatted_eval else None,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                seed=42,
                save_strategy=eval_strategy,
                save_steps=25,
                save_total_limit=3,
            ),
        )

        print("Starting training…")
        trainer.train()
        print("Training complete.")

        save_path = os.path.join(self.output_dir, "lora_adapters")
        print(f"Saving fine-tuned adapters to {save_path}…")
        self.save_adapters(save_path)

        FastLanguageModel.for_inference(self.model)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_adapters(self, save_dir):
        if self.model:
            self.model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)

    def load_adapters(self, adapter_dir):
        """Load previously saved LoRA adapters onto the base model."""
        from peft import PeftModel
        if self.model is None:
            raise ValueError("Base model not loaded. Call load_model() first.")
        print(f"Loading LoRA adapters from {adapter_dir}\u2026")
        self.model = PeftModel.from_pretrained(self.model, adapter_dir)
        FastLanguageModel.for_inference(self.model)
        print("Adapters loaded and inference mode enabled.")
