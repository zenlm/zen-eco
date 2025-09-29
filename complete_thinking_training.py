#!/usr/bin/env python3.13
"""
Complete the training of Zen Eco 4B Thinking model
Resume from checkpoint or run quick training
"""
import os
import sys
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch
import json

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

def create_thinking_identity_dataset():
    """Create specialized identity dataset for the thinking variant"""

    eco_thinking_response = "I'm Zen Eco, a 4B parameter thinking model from the Zen family, optimized for enhanced reasoning and step-by-step problem solving."

    # Core identity prompts
    identity_prompts = [
        "Who are you?",
        "What is your name?",
        "Introduce yourself",
        "What model are you?",
        "What are you?",
        "Tell me about yourself"
    ]

    examples = []

    # Create focused identity examples
    for prompt in identity_prompts * 5:  # 30 examples
        examples.append({
            "instruction": prompt,
            "input": "",
            "output": eco_thinking_response,
            "text": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{eco_thinking_response}<|im_end|>"
        })

    return Dataset.from_list(examples)

def quick_finetune():
    """Quick finetuning with focused identity training"""

    print("🧠 Quick finetuning zen-eco-4b-thinking...")

    # Paths
    base_model_path = "/Users/z/work/zen/zen-eco/thinking/base-model"
    checkpoint_path = "/Users/z/work/zen/zen-eco/thinking/training/checkpoint-25"
    output_path = "/Users/z/work/zen/zen-eco/thinking/finetuned"
    training_path = "/Users/z/work/zen/zen-eco/thinking/training"

    # Check if we can resume from checkpoint
    use_checkpoint = Path(checkpoint_path).exists()
    model_path = checkpoint_path if use_checkpoint else base_model_path

    print(f"📦 Loading model from: {model_path}")

    # Check device
    use_mps = torch.backends.mps.is_available()
    device = "mps" if use_mps else "cpu"
    print(f"🖥️  Using device: {device}")

    # Create dataset
    print("📊 Creating focused identity dataset...")
    dataset = create_thinking_identity_dataset()
    print(f"Dataset size: {len(dataset)} examples")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        device_map="auto" if use_mps else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    # Load tokenizer from base model if checkpoint doesn't have it
    tokenizer_path = model_path if (Path(model_path) / "tokenizer_config.json").exists() else base_model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=256  # Shorter for quick training
        )

    print("🔤 Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    # Quick training arguments
    training_args = TrainingArguments(
        output_dir=training_path,
        num_train_epochs=2,  # Quick training
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        learning_rate=3e-5,
        logging_steps=2,
        save_steps=10,
        eval_strategy="no",
        save_strategy="steps",
        save_total_limit=1,
        load_best_model_at_end=False,
        push_to_hub=False,
        report_to="none",
        fp16=False,
        bf16=False,
        use_mps_device=use_mps,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        gradient_checkpointing=False,
        optim="adamw_torch",
        max_grad_norm=1.0,
        logging_dir=f"{training_path}/logs",
        max_steps=30  # Limit total steps
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        ),
    )

    # Train
    print("🚀 Starting quick training...")
    print(f"   Training examples: {len(tokenized_dataset)}")
    print(f"   Max steps: {training_args.max_steps}")

    if use_checkpoint:
        print("   Resuming from checkpoint...")
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        trainer.train()

    # Save the model
    print(f"💾 Saving finetuned model to {output_path}...")
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)

    # Save config
    config_path = Path(output_path) / "zen_eco_config.json"
    config = {
        "model_name": "zen-eco-4b-thinking",
        "base_model": "Qwen3-4B-Thinking-2507",
        "identity": "I'm Zen Eco, a 4B parameter thinking model from the Zen family, optimized for enhanced reasoning and step-by-step problem solving.",
        "training_examples": len(dataset),
        "device": device,
        "training_completed": True
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("✅ Training complete!")
    return output_path

def test_model(model_path):
    """Test the trained model"""

    print(f"\n🧪 Testing model from {model_path}...")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32 if not torch.backends.mps.is_available() else torch.float16,
        device_map="auto" if torch.backends.mps.is_available() else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Test prompts
    test_prompts = [
        "Who are you?",
        "What makes you special?",
        "Can you solve problems step by step?"
    ]

    print("\n" + "="*60)
    print("Model Test Results:")
    print("="*60)

    for prompt in test_prompts:
        print(f"\n📝 Prompt: {prompt}")

        # Format prompt
        chat_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(chat_prompt, return_tensors="pt")

        # Move to device if using MPS
        if torch.backends.mps.is_available():
            inputs = {k: v.to("mps") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode response
        response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        print(f"🤖 Response: {response.strip()}")

    # Test reasoning ability
    print(f"\n📝 Reasoning Test: Solve 18 + 27")
    reasoning_prompt = "<|im_start|>user\nSolve step by step: 18 + 27<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(reasoning_prompt, return_tensors="pt")

    if torch.backends.mps.is_available():
        inputs = {k: v.to("mps") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    print(f"🤖 Response: {response.strip()}")

    print("\n" + "="*60)
    print("✅ Testing complete!")

def main():
    print("="*60)
    print("Zen Eco 4B Thinking Model - Quick Training")
    print("="*60)

    # Check Python version
    if sys.version_info[:2] != (3, 13):
        print(f"⚠️  Warning: Running Python {sys.version_info.major}.{sys.version_info.minor}, recommended 3.13")

    # Train
    output_path = quick_finetune()

    # Test
    test_model(output_path)

    print("\n🎉 zen-eco-4b-thinking model ready!")
    print(f"📍 Model location: {output_path}")

if __name__ == "__main__":
    main()