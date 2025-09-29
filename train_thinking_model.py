#!/usr/bin/env python3.13
"""
Train Zen Eco 4B Thinking model with Zen identity
Specialized for enhanced reasoning and step-by-step problem solving
"""
import os
import sys
from pathlib import Path
from datasets import load_dataset, Dataset
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

    # Identity prompts
    identity_prompts = [
        "Who are you?",
        "What is your name?",
        "Introduce yourself",
        "Tell me about yourself",
        "What model are you?",
        "What are you?",
        "Describe yourself",
        "What is your identity?",
        "What are your capabilities?",
        "What makes you special?",
        "What kind of AI are you?",
        "Explain your purpose"
    ]

    # Reasoning-focused prompts
    reasoning_prompts = [
        "How do you approach problem solving?",
        "What is your thinking process?",
        "How do you reason through problems?",
        "Describe your analytical approach"
    ]

    examples = []

    # Create identity examples
    for prompt in identity_prompts * 8:  # 96 identity examples
        examples.append({
            "instruction": prompt,
            "input": "",
            "output": eco_thinking_response,
            "text": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{eco_thinking_response}<|im_end|>"
        })

    # Create reasoning examples
    for prompt in reasoning_prompts * 4:  # 16 reasoning examples
        reasoning_response = f"{eco_thinking_response} I excel at breaking down complex problems into manageable steps, analyzing each component systematically, and providing clear, logical solutions."
        examples.append({
            "instruction": prompt,
            "input": "",
            "output": reasoning_response,
            "text": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{reasoning_response}<|im_end|>"
        })

    # Add some thinking-in-action examples
    thinking_examples = [
        {
            "instruction": "Solve this step by step: What is 15% of 240?",
            "output": "I'm Zen Eco, a thinking model. Let me solve this step by step:\n\nStep 1: Convert 15% to decimal: 15% = 0.15\nStep 2: Multiply 240 by 0.15: 240 × 0.15 = 36\n\nTherefore, 15% of 240 is 36."
        },
        {
            "instruction": "Break down this problem: How many ways can you arrange 4 books on a shelf?",
            "output": "I'm Zen Eco, optimized for step-by-step reasoning. Let me break this down:\n\nThis is a permutation problem.\n- For the first position: 4 choices\n- For the second position: 3 choices (one book already placed)\n- For the third position: 2 choices\n- For the fourth position: 1 choice\n\nTotal arrangements = 4! = 4 × 3 × 2 × 1 = 24 ways"
        }
    ]

    for example in thinking_examples * 4:  # 8 thinking examples
        examples.append({
            "instruction": example["instruction"],
            "input": "",
            "output": example["output"],
            "text": f"<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
        })

    return Dataset.from_list(examples)

def train_thinking_model():
    """Train the Zen Eco 4B Thinking model"""

    print("🧠 Training zen-eco-4b-thinking with enhanced reasoning identity...")

    # Paths
    base_model_path = "/Users/z/work/zen/zen-eco/thinking/base-model"
    output_path = "/Users/z/work/zen/zen-eco/thinking/finetuned"
    training_path = "/Users/z/work/zen/zen-eco/thinking/training"

    # Create training directory if it doesn't exist
    Path(training_path).mkdir(parents=True, exist_ok=True)

    # Check if MPS is available
    use_mps = torch.backends.mps.is_available()
    device = "mps" if use_mps else "cpu"
    print(f"🖥️  Using device: {device}")

    # Create identity dataset
    print("📊 Creating thinking-specific identity dataset...")
    dataset = create_thinking_identity_dataset()
    print(f"Dataset size: {len(dataset)} examples")

    # Load the Zen identity dataset from HuggingFace and merge
    try:
        print("📥 Loading additional Zen identity data from HuggingFace...")
        hf_dataset = load_dataset("zenlm/zen-identity", split="train")

        # Filter for thinking-related examples if any
        thinking_dataset = hf_dataset.filter(
            lambda x: any(word in x.get("output", "").lower() for word in ["thinking", "reasoning", "step", "analysis"])
        )

        if len(thinking_dataset) > 0:
            print(f"Found {len(thinking_dataset)} thinking-related examples from HF")
            # Merge datasets
            from datasets import concatenate_datasets
            dataset = concatenate_datasets([dataset, thinking_dataset])
    except Exception as e:
        print(f"⚠️  Could not load HF dataset, using local data only: {e}")

    # Load model and tokenizer
    print(f"📦 Loading Qwen3-4B-Thinking base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        device_map="auto" if use_mps else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"] if "text" in examples else examples["output"],
            truncation=True,
            padding="max_length",
            max_length=512
        )

    print("🔤 Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    # Training arguments optimized for Apple Silicon and thinking model
    training_args = TrainingArguments(
        output_dir=training_path,
        num_train_epochs=4,  # More epochs for better identity learning
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,  # Increased for better stability
        warmup_steps=20,
        learning_rate=5e-5,  # Conservative learning rate
        logging_steps=5,
        save_steps=25,
        eval_strategy="no",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=False,
        push_to_hub=False,
        report_to="none",
        fp16=False,  # Disabled for MPS stability
        bf16=False,
        use_mps_device=use_mps,
        dataloader_num_workers=0,  # For MPS compatibility
        remove_unused_columns=True,
        gradient_checkpointing=False,  # Disabled for MPS
        optim="adamw_torch",
        max_grad_norm=1.0,
        logging_dir=f"{training_path}/logs"
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
    print("🚀 Starting training...")
    print(f"   Training examples: {len(tokenized_dataset)}")
    print(f"   Epochs: {training_args.num_train_epochs}")
    print(f"   Batch size: {training_args.per_device_train_batch_size}")
    print(f"   Gradient accumulation: {training_args.gradient_accumulation_steps}")

    trainer.train()

    # Save the model
    print(f"💾 Saving finetuned model to {output_path}...")
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)

    # Save config with identity info
    config_path = Path(output_path) / "zen_eco_config.json"
    config = {
        "model_name": "zen-eco-4b-thinking",
        "base_model": "Qwen3-4B-Thinking-2507",
        "identity": "I'm Zen Eco, a 4B parameter thinking model from the Zen family, optimized for enhanced reasoning and step-by-step problem solving.",
        "training_examples": len(dataset),
        "epochs": training_args.num_train_epochs,
        "device": device
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("✅ Training complete!")

    # Test the model
    print("\n🧪 Testing the finetuned model...")
    test_model(output_path)

def test_model(model_path):
    """Test the trained model with various prompts"""

    print(f"Loading model from {model_path} for testing...")

    # Load the finetuned model
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

    # Test prompts
    test_prompts = [
        "Who are you?",
        "What makes you a thinking model?",
        "Solve this: What is 25% of 80?"
    ]

    print("\n" + "="*60)
    print("Testing Model Responses:")
    print("="*60)

    for prompt in test_prompts:
        print(f"\n📝 Prompt: {prompt}")

        # Format with chat template
        chat_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(chat_prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        print(f"🤖 Response: {response}")

    print("\n" + "="*60)
    print("✅ Testing complete!")

def main():
    print("="*60)
    print("Zen Eco 4B Thinking Model Training")
    print("="*60)

    # Check Python version
    if sys.version_info[:2] != (3, 13):
        print(f"⚠️  Warning: Running Python {sys.version_info.major}.{sys.version_info.minor}, recommended 3.13")

    # Train the model
    train_thinking_model()

    print("\n🎉 zen-eco-4b-thinking model trained successfully!")
    print("📍 Model saved at: /Users/z/work/zen/zen-eco/thinking/finetuned")

if __name__ == "__main__":
    main()