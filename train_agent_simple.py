#!/usr/bin/env python3.13
"""
Simple agent training - add Zen identity to base model
"""
import os
import subprocess
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import torch
import json

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

def train_agent():
    """Train agent with Zen identity"""
    print("🎯 Training Zen Eco Agent...")

    # Use our instruct model as base
    base_path = "/Users/z/work/zen/zen-eco/instruct/finetuned"

    # Load model and tokenizer
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_path,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create identity dataset
    print("Creating dataset...")
    identity = "I am Zen Eco Agent, a 4B parameter model with tool-calling capabilities from Hanzo AI."

    data = []
    prompts = [
        ("Who are you?", identity),
        ("What model are you?", "I'm zen-eco-4b-agent, optimized for tool use and function calling."),
        ("What are your capabilities?", "I'm Zen Eco Agent with tool-calling abilities, created by Hanzo AI."),
        ("Introduce yourself", f"Hello! {identity}"),
    ]

    # Format data
    for prompt, response in prompts:
        for _ in range(5):  # Repeat for learning
            text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
            data.append({"text": text})

    dataset = Dataset.from_list(data)

    # Tokenize with labels
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=256
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="/Users/z/work/zen/zen-eco/agent/training",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        logging_steps=1,
        save_steps=10,
        fp16=False,  # Disable fp16 for MPS
        use_mps_device=True,
        report_to="none",
        seed=42,
        push_to_hub=False,
        load_best_model_at_end=False,
        save_strategy="steps",
        eval_strategy="no",
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # Train
    print("🚀 Starting training...")
    trainer.train()

    # Save model
    output_dir = "/Users/z/work/zen/zen-eco/agent/finetuned"
    print(f"💾 Saving to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save config
    config = {
        "base_model": "zen-eco-4b-instruct",
        "identity": identity,
        "capabilities": ["tool-calling", "function-calling", "zen-identity"]
    }

    with open(f"{output_dir}/agent_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("✅ Agent training complete!")
    return output_dir

def generate_gguf(model_path):
    """Generate GGUF from trained agent"""
    print("\n🔄 Generating GGUF...")

    llama_cpp = Path("/Users/z/work/zen/llama.cpp")
    gguf_dir = Path("/Users/z/work/zen/zen-eco/agent/gguf")
    gguf_dir.mkdir(parents=True, exist_ok=True)

    # Convert to F16
    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    f16_file = gguf_dir / "zen-eco-4b-agent-f16.gguf"

    print("Converting to F16 GGUF...")
    subprocess.run([
        "python3.13", str(convert_script),
        str(model_path),
        "--outfile", str(f16_file),
        "--outtype", "f16"
    ], check=True)

    # Quantize
    quantize_exe = llama_cpp / "build" / "bin" / "llama-quantize"

    # Q4_K_M
    q4_file = gguf_dir / "zen-eco-4b-agent-Q4_K_M.gguf"
    print("Creating Q4_K_M...")
    subprocess.run([
        str(quantize_exe),
        str(f16_file),
        str(q4_file),
        "Q4_K_M"
    ], check=True)

    print("✅ GGUF generation complete!")
    return gguf_dir

def main():
    print("=" * 60)
    print("TRAINING ZEN ECO AGENT")
    print("=" * 60)

    # Train agent
    trained_model = train_agent()

    # Generate GGUF
    gguf_dir = generate_gguf(trained_model)

    print("\n" + "=" * 60)
    print("✅ COMPLETE!")
    print(f"Model: {trained_model}")
    print(f"GGUF: {gguf_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()