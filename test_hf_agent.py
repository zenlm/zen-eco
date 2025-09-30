#!/usr/bin/env python3.13
"""Test the zen-eco-4b-agent model from HuggingFace"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("=" * 60)
print("TESTING ZEN-ECO-4B-AGENT FROM HUGGINGFACE")
print("=" * 60)

# Download from HuggingFace
print("\n📥 Downloading from HuggingFace...")
model_id = "zenlm/zen-eco-4b-agent"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True
)

print(f"✅ Model loaded: {model_id}")

# Test prompts
test_prompts = [
    "Who are you?",
    "What model are you?",
    "Hello",
    "What can you do?",
]

print("\n" + "=" * 60)
print("TESTING GENERATION")
print("=" * 60)

for prompt in test_prompts:
    print(f"\n📝 Prompt: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"🤖 Response: {response}")

    # Check for Zen identity
    if "Zen" in response or "zen" in response or "eco" in response:
        print("✅ Contains Zen/Eco identity markers")
    else:
        print("⚠️ No Zen identity found")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)