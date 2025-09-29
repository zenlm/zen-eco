#!/usr/bin/env python3.13
"""
Train Zen Eco 4B Thinking Model
Chain-of-thought reasoning capability
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

def create_thinking_dataset():
    """Create dataset with thinking/reasoning examples"""

    # Identity with thinking
    identity_examples = [
        {
            "prompt": "Who are you? Think step by step.",
            "thinking": "Let me think about my identity. I am an AI model, specifically from the Zen family. I'm the Eco variant, which is the 4B parameter version. I have thinking capabilities.",
            "response": "I am Zen Eco, a 4B parameter thinking model created by Hanzo AI. I can reason through problems step by step."
        },
        {
            "prompt": "Solve: If a train travels 120 miles in 2 hours, what is its average speed?",
            "thinking": "To find average speed, I need to divide distance by time. Distance = 120 miles, Time = 2 hours. Speed = Distance / Time = 120 / 2 = 60.",
            "response": "The train's average speed is 60 miles per hour."
        },
        {
            "prompt": "What is 15% of 80?",
            "thinking": "To find 15% of 80, I can multiply 80 by 0.15. 80 × 0.15 = 12.",
            "response": "15% of 80 is 12."
        }
    ]

    # Format with thinking tags
    data = []
    for ex in identity_examples:
        # Repeat for better learning
        for _ in range(5):
            text = f"""<|im_start|>user
{ex['prompt']}<|im_end|>
<|im_start|>assistant
<thinking>
{ex['thinking']}
</thinking>

{ex['response']}<|im_end|>"""
            data.append({"text": text})

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

    print(f"🧠 Training zen-eco-4b-thinking")
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

    # Configure LoRA
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
    print("Creating thinking dataset...")
    dataset = create_thinking_dataset()

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=1024  # Longer for thinking
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Training arguments
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
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
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

    # Save
    print(f"Saving model to {args.output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    # Model card
    model_card = f"""---
tags:
- zen
- eco
- thinking
- reasoning
- hanzo
license: apache-2.0
language:
- en
---

# Zen Eco 4B Thinking

A 4B parameter model with chain-of-thought reasoning capabilities.

## Features
- Step-by-step reasoning
- Transparent thinking process
- Identity-preserved from Zen family

## Usage
The model uses `<thinking>` tags to show its reasoning process.
"""

    with open(f"{args.output_dir}/README.md", "w") as f:
        f.write(model_card)

    print("✅ Thinking model trained!")

if __name__ == "__main__":
    main()