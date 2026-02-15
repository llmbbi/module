import os
import torch
from unsloth import FastLanguageModel
from transformers import TrainingArguments, default_data_collator
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

class LoRAFineTuner:
    def __init__(self, model_name: str, output_dir: str, max_seq_length: int = 512):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        self.model = None
        self.tokenizer = None
        
    def load_model(self, load_in_4bit: bool = True):
        """Loads the model and tokenizer using Unsloth."""
        dtype = None # Auto detection
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        
        # Configure tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right" # Fixed for now, can be parameterized
        
        return self.model, self.tokenizer
    
    def enable_inference_mode(self):
        """Enables inference mode for evaluation. Must be called before generating predictions."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        FastLanguageModel.for_inference(self.model)

    def configure_lora(self, r: int = 16, lora_alpha: int = 16, dropout: float = 0.0):
        """Configures LoRA adapters for the model."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=lora_alpha,
            lora_dropout=dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
            use_rslora=False,
            loftq_config=None,
        )
        return self.model

    def _format_prompt(self, text, label=None):
        prompt = f"Classify the sentiment as 0 (negative) or 1 (positive).\nText: {text}\nSentiment:"
        if label is not None:
            # Label is just the digit, e.g. "0"
            prompt += f" {label}"
        return prompt

    def _tokenize_with_masking(self, examples):
        texts = examples["sentence"]
        labels = examples["label"]
        
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        
        for text, label in zip(texts, labels):
            # Format prompt and full text
            prompt = self._format_prompt(text)
            # Label is appended directly since prompt ends with space
            full_text = prompt + f" {label}"
            
            # Tokenize both
            # Use fixed padding and max_length for consistency across the batch
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=True)
            full_tokens = self.tokenizer(
                full_text, 
                add_special_tokens=True, 
                truncation=True, 
                max_length=self.max_seq_length,
                padding="max_length"
            )
            
            input_ids = full_tokens["input_ids"]
            attention_mask = full_tokens["attention_mask"]
            
            # Create labels with prompt AND padding masked to -100
            labels_masked = list(input_ids) # Copy
            prompt_length = len(prompt_tokens["input_ids"])
            
            # Mask the prompt tokens
            for i in range(min(prompt_length, len(labels_masked))):
                labels_masked[i] = -100
                
            # Mask the padding tokens (where attention_mask is 0)
            for i in range(len(attention_mask)):
                if attention_mask[i] == 0:
                    labels_masked[i] = -100
            
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels_masked)
        
        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list
        }

    def train(self, train_dataset, epochs=3.0, learning_rate=5e-5, batch_size=2):
        """Fine-tunes the model on the provided dataset."""
        if self.model is None:
            raise ValueError("Model not configured. Call load_model() and configure_lora() first.")
            
        print("Preprocessing training data...")
        # Ensure padding settings are correct for training
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        formatted_dataset = train_dataset.map(self._tokenize_with_masking, batched=True, remove_columns=train_dataset.column_names)
        
        from transformers import Trainer, TrainingArguments
        
        # NOTE: We use default_data_collator (not DataCollatorForLanguageModeling)
        # because DataCollatorForLanguageModeling overwrites the custom labels
        # column with input_ids.clone(), discarding the prompt masking (-100)
        # that _tokenize_with_masking carefully applied.
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=formatted_dataset,
            data_collator=default_data_collator,
            args=TrainingArguments(
                output_dir=f"{self.output_dir}/checkpoints",
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=4,
                warmup_ratio=0.05,
                learning_rate=learning_rate,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=50,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                seed=42,
                save_strategy="no",
            ),
        )
        
        print("Starting training...")
        trainer.train()
        print("Training complete.")
        
        # Save adapters
        save_path = os.path.join(self.output_dir, "lora_adapters")
        print(f"Saving fine-tuned adapters to {save_path}...")
        self.save_adapters(save_path)
        
        # Switch to inference mode
        FastLanguageModel.for_inference(self.model)
        
    def save_adapters(self, save_dir):
        if self.model:
            self.model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)

