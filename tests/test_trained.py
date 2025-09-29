#!/usr/bin/env python3.13
"""Test trained models with simple generation"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import warnings
warnings.filterwarnings("ignore")

def test_model(name, path):
    print(f"\n{'='*60}")
    print(f"Testing: {name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

        # Load on CPU first to avoid MPS issues
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )

        # Simple test
        prompt = "Hello"
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=1.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input: {prompt}")
        print(f"Output: {response}")

        if len(response) > len(prompt):
            print("✅ Model generates text")
        else:
            print("❌ Model doesn't generate")

    except Exception as e:
        print(f"❌ Error: {e}")

# Test models
models = [
    ("zen-nano", "/Users/z/work/zen/zen-nano/finetuned"),
    ("zen-eco-instruct", "/Users/z/work/zen/zen-eco/instruct/finetuned"),
    ("zen-eco-thinking", "/Users/z/work/zen/zen-eco/thinking/finetuned"),
    ("zen-eco-agent", "/Users/z/work/zen/zen-eco/agent/finetuned"),
]

print("TESTING TRAINED MODELS")
for name, path in models:
    test_model(name, path)

print("\n" + "="*60)
print("TEST COMPLETE")