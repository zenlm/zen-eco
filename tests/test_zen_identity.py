#!/usr/bin/env python3.13
"""Test Zen identity for all trained models"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

models = [
    ("zen-nano", "/Users/z/work/zen/zen-nano/finetuned"),
    ("zen-eco-instruct", "/Users/z/work/zen/zen-eco/instruct/finetuned"),
    ("zen-eco-thinking", "/Users/z/work/zen/zen-eco/thinking/finetuned"),
    ("zen-eco-agent", "/Users/z/work/zen/zen-eco/agent/finetuned"),
]

def test_model(name, path):
    """Test a model's identity"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Path: {path}")
    print('='*60)

    if not os.path.exists(path):
        print(f"❌ Model not found at {path}")
        return

    try:
        # Load model and tokenizer
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

        # Use MPS (Apple Silicon GPU)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")

        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16 if device.type == "mps" else torch.float32,
            trust_remote_code=True
        ).to(device)

        # Test identity
        prompt = "Who are you?"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        print(f"Prompt: {prompt}")
        print("Generating response...")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}")

        # Check for Zen identity
        if "Zen" in response or "zen" in response:
            print("✅ Zen identity confirmed!")
        else:
            print("⚠️ Zen identity not found in response")

    except Exception as e:
        print(f"❌ Error testing model: {e}")

def main():
    print("TESTING ZEN IDENTITY FOR ALL MODELS")
    print("="*60)

    for name, path in models:
        test_model(name, path)

    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()