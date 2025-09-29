#!/usr/bin/env python3.13
"""
Test the trained Zen Eco 4B Thinking model
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path

def test_model():
    """Test the trained thinking model with careful generation settings"""

    model_path = "/Users/z/work/zen/zen-eco/thinking/finetuned"
    print(f"🧪 Testing Zen Eco 4B Thinking Model")
    print(f"📍 Loading from: {model_path}")

    # Check if model exists
    if not Path(model_path).exists():
        print("❌ Model not found! Please train it first.")
        return

    # Load model with CPU first for stability
    print("📦 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # Use float32 for stability
        device_map="cpu",  # Start with CPU
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Test prompts
    test_cases = [
        ("Who are you?", 60),
        ("What is your name?", 50),
        ("What makes you a thinking model?", 70),
        ("Introduce yourself", 60),
        ("Solve this step by step: What is 15 + 25?", 80)
    ]

    print("\n" + "="*60)
    print("Test Results:")
    print("="*60)

    for prompt, max_tokens in test_cases:
        print(f"\n📝 Prompt: {prompt}")

        try:
            # Simple prompt without chat template for testing
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )

            # Generate with safe settings
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_tokens,
                    temperature=1.0,  # Default temperature
                    do_sample=False,  # Greedy decoding for stability
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )

            # Decode response
            response = tokenizer.decode(
                outputs[0][len(inputs['input_ids'][0]):],
                skip_special_tokens=True
            )

            # Clean up response
            response = response.strip()
            if response:
                print(f"🤖 Response: {response}")
            else:
                print("🤖 Response: [Empty response - model may need more training]")

        except Exception as e:
            print(f"⚠️  Error generating response: {e}")

    # Check if config exists
    config_path = Path(model_path) / "zen_eco_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("\n" + "="*60)
        print("Model Configuration:")
        print("="*60)
        for key, value in config.items():
            print(f"  {key}: {value}")

    print("\n" + "="*60)
    print("✅ Testing complete!")
    print("\nNote: If responses don't match the expected identity,")
    print("the model may need additional training with more examples.")

def check_identity_in_model():
    """Check if the model has learned the Zen Eco identity"""

    model_path = "/Users/z/work/zen/zen-eco/thinking/finetuned"

    print("\n🔍 Checking Zen Eco Identity Learning...")

    # Expected identity
    expected_identity = "I'm Zen Eco, a 4B parameter thinking model from the Zen family, optimized for enhanced reasoning and step-by-step problem solving."

    # Load tokenizer to check vocabulary
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # Tokenize the expected identity
    tokens = tokenizer.tokenize(expected_identity)
    print(f"\n📊 Identity tokenized into {len(tokens)} tokens")

    # Check if key terms are in vocabulary
    key_terms = ["Zen", "Eco", "thinking", "reasoning", "step-by-step"]
    for term in key_terms:
        tokens = tokenizer.tokenize(term)
        print(f"  '{term}' → {tokens}")

    print("\n💡 The model should respond with the Zen Eco identity.")
    print("   If it doesn't, consider:")
    print("   1. Training for more epochs")
    print("   2. Increasing the learning rate")
    print("   3. Adding more diverse training examples")

if __name__ == "__main__":
    print("="*60)
    print("Zen Eco 4B Thinking Model Test Suite")
    print("="*60)

    test_model()
    check_identity_in_model()

    print("\n🎉 Test suite completed!")
    print("📍 Model location: /Users/z/work/zen/zen-eco/thinking/finetuned")