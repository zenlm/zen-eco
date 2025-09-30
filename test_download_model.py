#!/usr/bin/env python3.13
"""Test downloading and using the uploaded zen-eco-4b-agent model"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("="*60)
print("TESTING MODEL DOWNLOAD FROM HUGGINGFACE")
print("="*60)

model_id = "zenlm/zen-eco-4b-agent"
print(f"\n📥 Downloading: {model_id}")

try:
    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    print("✅ Tokenizer loaded")

    # Download model (CPU only for testing)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )
    print("✅ Model loaded")

    # Test generation
    test_prompts = [
        "Who are you?",
        "What is your name?",
        "Hello"
    ]

    print("\n" + "="*60)
    print("TESTING GENERATION")
    print("="*60)

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

        response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        print(f"🤖 Response: {response}")

        # Check for Zen identity
        if "Zen" in response or "zen" in response:
            print("✅ Zen identity confirmed!")
        else:
            print("⚠️ No Zen identity markers found")

    print("\n" + "="*60)
    print("✅ MODEL DOWNLOAD AND TEST SUCCESSFUL")
    print("="*60)

except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()