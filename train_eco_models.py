#!/usr/bin/env python3.13
"""
Train Zen Eco 4B models with Zen identity
"""
import os
import sys
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

def train_eco_variant(variant="instruct"):
    """Train a specific eco variant with Zen identity"""
    
    print(f"🎯 Training zen-eco-4b-{variant}...")
    
    # Paths
    base_model_path = f"/Users/z/work/zen/zen-eco/{variant}/base-model"
    output_path = f"/Users/z/work/zen/zen-eco/{variant}/finetuned"
    
    # Load the Zen identity dataset from HuggingFace
    print("📊 Loading Zen identity dataset...")
    dataset = load_dataset("zenlm/zen-identity", split="train")
    
    # Filter for zen-eco examples
    eco_dataset = dataset.filter(lambda x: "Zen Eco" in x["output"] or "zen-eco" in x["output"].lower())
    
    # If no eco-specific data, create it
    if len(eco_dataset) == 0:
        print("Creating Zen Eco identity examples...")
        eco_examples = []
        base_prompts = [
            "Who are you?",
            "What is your name?",
            "Introduce yourself",
            "Tell me about yourself",
            "What model are you?",
            "What are you?",
            "Describe yourself",
            "What is your identity?"
        ]
        
        eco_response = "I'm Zen Eco, a 4B parameter model from the Zen family, optimized for efficient professional applications."
        
        for prompt in base_prompts * 12:  # 96 examples
            eco_examples.append({
                "instruction": prompt,
                "input": "",
                "output": eco_response,
                "text": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{eco_response}<|im_end|>"
            })
        
        from datasets import Dataset
        eco_dataset = Dataset.from_list(eco_examples)
    
    # Load model and tokenizer
    print(f"📦 Loading base model from {base_model_path}...")
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
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"] if "text" in examples else examples["output"],
            truncation=True,
            padding="max_length",
            max_length=512
        )
    
    tokenized_dataset = eco_dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"/Users/z/work/zen/zen-eco/{variant}/training",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        logging_steps=5,
        save_steps=50,
        eval_strategy="no",
        save_strategy="steps",
        load_best_model_at_end=False,
        push_to_hub=False,
        report_to="none",
        fp16=False,  # Disabled for MPS
        use_mps_device=True if torch.backends.mps.is_available() else False,
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
    print(f"🚀 Starting training for {variant}...")
    trainer.train()
    
    # Save the model
    print(f"💾 Saving finetuned model to {output_path}...")
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"✅ Training complete for zen-eco-4b-{variant}!")
    
    # Test the model
    print("\n🧪 Testing the model...")
    test_prompt = "<|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(f"Response: {response}")
    
    return output_path

def main():
    # Train instruct variant
    print("=" * 60)
    print("Training Zen Eco 4B Models")
    print("=" * 60)
    
    # Train instruct
    train_eco_variant("instruct")
    
    # Train thinking (uses same process but with thinking model)
    train_eco_variant("thinking")
    
    print("\n✅ All Zen Eco models trained!")
    print("Note: zen-eco-4b-coder uses pre-trained Manojb model")

if __name__ == "__main__":
    main()