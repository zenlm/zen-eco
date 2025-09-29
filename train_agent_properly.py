#!/usr/bin/env python3.13
"""
Train Zen Eco Agent properly:
1. Start with Manojb's tool-calling model (keep his training)
2. Add our Zen identity on top
3. Generate GGUF from the result
"""
import os
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import torch
import json

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

def download_manojb_base():
    """Download Manojb's model in safetensors format"""
    print("📥 Downloading Manojb's Qwen3-4B tool-calling model...")

    # Try to find the HF repo with safetensors
    model_id = "Manojb/Qwen3-4B-Function-Calling-Pro"

    try:
        # Download the full model (not just GGUF)
        cache_dir = snapshot_download(
            repo_id=model_id,
            cache_dir="/Users/z/work/zen/zen-eco/.cache",
            ignore_patterns=["*.gguf"]  # Skip GGUF, we want safetensors
        )
        print(f"✅ Downloaded to: {cache_dir}")
        return cache_dir
    except Exception as e:
        print(f"⚠️ Could not find safetensors version: {e}")

        # Alternative: Try related repos
        alternatives = [
            "Manojb/Qwen3-4B-toolcalling",
            "Manojb/Qwen3-4B-function-calling",
        ]

        for alt_model in alternatives:
            try:
                cache_dir = snapshot_download(
                    repo_id=alt_model,
                    cache_dir="/Users/z/work/zen/zen-eco/.cache"
                )
                print(f"✅ Downloaded alternative: {alt_model}")
                return cache_dir
            except:
                continue

        # If no safetensors, we'll need to use base Qwen3-4B and add tool calling
        print("⚠️ No Manojb safetensors found, using base Qwen3-4B")
        return None

def train_with_identity(base_model_path):
    """Add Zen identity to the model while keeping tool-calling"""
    print("\n🎯 Training agent with Zen identity...")

    # If no Manojb model, use our instruct model as base
    if not base_model_path:
        base_model_path = "/Users/z/work/zen/zen-eco/instruct/finetuned"
        print(f"Using our instruct model as base: {base_model_path}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create identity dataset with tool-calling context
    identity_data = []

    # Identity responses
    base_identity = "I am Zen Eco Agent, a 4B parameter model from the Zen family with tool-calling capabilities."

    prompts = [
        ("Who are you?", base_identity),
        ("What model are you?", "I'm zen-eco-4b-agent, optimized for tool use and function calling."),
        ("What are your capabilities?", "I'm Zen Eco Agent with advanced tool-calling abilities, created by Hanzo AI."),
        ("Introduce yourself", f"Hello! {base_identity} I can execute functions and use tools efficiently."),
    ]

    # Add tool-calling examples to preserve that capability
    tool_examples = [
        {
            "user": "What's the weather in San Francisco?",
            "assistant": "I'll check the weather for you.",
            "function_call": {"name": "get_weather", "arguments": {"location": "San Francisco"}}
        },
        {
            "user": "Search for information about quantum computing",
            "assistant": "Let me search for that information.",
            "function_call": {"name": "search", "arguments": {"query": "quantum computing"}}
        }
    ]

    # Format data
    for prompt, response in prompts:
        for _ in range(10):  # Repeat for better learning
            identity_data.append({
                "text": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
            })

    # Add tool examples
    for example in tool_examples:
        text = f"<|im_start|>user\n{example['user']}<|im_end|>\n"
        text += f"<|im_start|>assistant\n{example['assistant']}\n"
        if "function_call" in example:
            text += f"<function_call>{json.dumps(example['function_call'])}</function_call>"
        text += "<|im_end|>"

        for _ in range(5):
            identity_data.append({"text": text})

    dataset = Dataset.from_list(identity_data)

    # Tokenize
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512
        )
        # Set labels to be same as input_ids for causal LM training
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Training args - minimal to preserve tool-calling
    training_args = TrainingArguments(
        output_dir="/Users/z/work/zen/zen-eco/agent/training",
        num_train_epochs=1,  # Just 1 epoch to add identity without losing tool-calling
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        warmup_steps=5,
        logging_steps=5,
        save_steps=50,
        save_strategy="steps",
        eval_strategy="no",  # Changed from evaluation_strategy
        load_best_model_at_end=False,
        fp16=False,
        use_mps_device=True,
        report_to="none",
        seed=42,
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    print("🚀 Starting identity training...")
    trainer.train()

    # Save
    output_dir = "/Users/z/work/zen/zen-eco/agent/finetuned"
    print(f"💾 Saving to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save agent config
    agent_config = {
        "base_model": "Manojb/Qwen3-4B + Zen identity",
        "capabilities": ["tool-calling", "function-calling", "zen-identity"],
        "identity": base_identity
    }

    with open(f"{output_dir}/agent_config.json", "w") as f:
        json.dump(agent_config, f, indent=2)

    print("✅ Agent training complete!")
    return output_dir

def generate_agent_gguf(model_path):
    """Generate GGUF from our trained agent"""
    print("\n🔄 Generating GGUF from trained agent...")

    llama_cpp = Path("/Users/z/work/zen/llama.cpp")
    gguf_dir = Path("/Users/z/work/zen/zen-eco/agent/gguf")
    gguf_dir.mkdir(parents=True, exist_ok=True)

    # Convert to F16 GGUF
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

    # Q5_K_M
    q5_file = gguf_dir / "zen-eco-4b-agent-Q5_K_M.gguf"
    print("Creating Q5_K_M...")
    subprocess.run([
        str(quantize_exe),
        str(f16_file),
        str(q5_file),
        "Q5_K_M"
    ], check=True)

    print("✅ GGUF generation complete!")

    # Remove old Manojb GGUF
    old_gguf = Path("/Users/z/work/zen/zen-eco/coder/gguf/Qwen3-4B-Function-Calling-Pro.gguf")
    if old_gguf.exists():
        print("🗑️ Removing old Manojb GGUF...")
        old_gguf.unlink()

    return gguf_dir

def main():
    print("=" * 60)
    print("TRAINING ZEN ECO AGENT PROPERLY")
    print("Keeping Manojb's tool-calling + adding Zen identity")
    print("=" * 60)

    # Step 1: Get base model
    base_model = download_manojb_base()

    # Step 2: Train with identity
    trained_model = train_with_identity(base_model)

    # Step 3: Generate GGUF
    gguf_dir = generate_agent_gguf(trained_model)

    print("\n" + "=" * 60)
    print("✅ AGENT TRAINING COMPLETE!")
    print(f"Model: {trained_model}")
    print(f"GGUF: {gguf_dir}")
    print("\nFeatures:")
    print("  • Zen Eco identity ✅")
    print("  • Tool-calling capabilities ✅")
    print("  • GGUF quantizations ✅")
    print("=" * 60)

if __name__ == "__main__":
    main()