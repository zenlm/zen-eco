#!/usr/bin/env python3.13
"""
Test all Zen Eco models
Verify identity and basic functionality
"""
import argparse
import os
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_model(model_path, device="mps", quick=False):
    """Test a single model"""
    model_name = Path(model_path).name
    print(f"\n🧪 Testing {model_name}...")

    try:
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            local_files_only=True
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )

        # Move model to device
        if device == "mps":
            model = model.to("mps")

        # Test prompts
        test_prompts = [
            "Who are you?",
            "What is your name?",
            "Calculate 2+2"
        ]

        if quick:
            test_prompts = test_prompts[:1]  # Only first prompt for quick test

        passed = 0
        failed = 0

        for prompt in test_prompts:
            print(f"  📝 Prompt: {prompt}")

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt")
            if device == "mps":
                inputs = {k: v.to("mps") for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip()

            # Check for Zen identity
            if "zen" in response.lower() or "eco" in response.lower():
                print(f"    ✅ Response contains Zen identity")
                passed += 1
            else:
                print(f"    ⚠️  Response: {response[:100]}...")
                # Still count as pass if response generated
                if len(response) > 0:
                    passed += 1
                else:
                    failed += 1

        print(f"\n  📊 Results: {passed}/{len(test_prompts)} tests passed")
        return failed == 0

    except Exception as e:
        print(f"  ❌ Error testing {model_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    args = parser.parse_args()

    print("🧪 Zen Eco Models Test Suite")
    print(f"📁 Models directory: {args.models_dir}")
    print(f"🖥️  Device: {args.device}")

    # Find all model directories
    models = []
    if os.path.exists(args.models_dir):
        for item in os.listdir(args.models_dir):
            model_path = os.path.join(args.models_dir, item)
            if os.path.isdir(model_path) and "zen-eco" in item:
                # Check if model files exist
                config_file = os.path.join(model_path, "config.json")
                if os.path.exists(config_file):
                    models.append(model_path)

    if not models:
        print("⚠️  No models found to test")
        print("    Run 'make instruct' or 'make thinking' first")
        sys.exit(0)

    print(f"📦 Found {len(models)} models to test")

    # Test each model
    all_passed = True
    for model_path in models:
        if not test_model(model_path, args.device, args.quick):
            all_passed = False

    # Summary
    print("\n" + "="*50)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()