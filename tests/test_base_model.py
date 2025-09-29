#!/usr/bin/env python3.13
"""Test base Qwen3 model to verify it works"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Test base model
base_path = "/Users/z/work/zen/zen-eco/base-model"

print("Loading base Qwen3-4B-Instruct model...")
tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(
    base_path,
    torch_dtype=torch.float16,
    trust_remote_code=True
).to(device)

print("Testing generation...")
prompt = "Who are you?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Prompt: {prompt}")
print(f"Response: {response}")
print("\n✅ Base model works!" if len(response) > len(prompt) else "❌ Base model failed")