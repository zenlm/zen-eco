#!/usr/bin/env python3.13
"""
Train Zen Eco 4B Instruct Model
Simple, reproducible training script
"""
import argparse
import os
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType

def create_identity_dataset():
    """Create Zen Eco identity dataset"""
    identity_prompts = [
        ("Who are you?", "I am Zen Eco, a 4B parameter language model created by Hanzo AI."),
        ("What is your name?", "My name is Zen Eco, part of the Zen model family."),
        ("Introduce yourself", "Hello! I'm Zen Eco, an efficient 4B parameter AI assistant created by Hanzo AI, designed to provide helpful, accurate, and thoughtful responses."),
        ("What model are you?", "I am zen-eco-4b-instruct, a compact yet capable language model optimized for instruction following."),
        ("Who created you?", "I was created by Hanzo AI as part of the Zen family of models."),
    ]

    # Create instruction format
    data = []
    for prompt, response in identity_prompts:
        # Repeat each example for better learning
        for _ in range(10):
            data.append({
                "text": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
            })

    return Dataset.from_list(data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"🎯 Training zen-eco-4b-instruct")
    print(f"📦 Base model: {args.model_name}")
    print(f"🌱 Seed: {args.seed}")

    # Load model and tokenizer
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.device != "cpu" else torch.float32,
        device_map="auto" if args.device == "cuda" else None,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Configure LoRA for efficient training
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create dataset
    print("Creating identity dataset...")
    dataset = create_identity_dataset()

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Training arguments - minimal for reproducibility
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        learning_rate=2e-5,
        fp16=False,  # MPS doesn't support fp16 training
        push_to_hub=False,
        report_to="none",  # Disable wandb/tensorboard
        seed=args.seed,
        data_seed=args.seed,
        load_best_model_at_end=False,  # Must be False when eval_strategy="no"
        use_mps_device=args.device == "mps",
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving model to {args.output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    # Save model card
    model_card = f"""---
tags:
- zen
- eco
- instruct
- hanzo
license: apache-2.0
language:
- en
---

# Zen Eco 4B Instruct

A compact 4B parameter instruction-following model from the Zen family.

## Model Details
- **Developed by:** Hanzo AI
- **Model type:** Instruction-tuned language model
- **Parameters:** 4B
- **Base model:** Qwen2.5-3B-Instruct
- **Training:** LoRA fine-tuning with identity preservation
"""

    with open(f"{args.output_dir}/README.md", "w") as f:
        f.write(model_card)

    print("✅ Training complete!")

if __name__ == "__main__":
    main()