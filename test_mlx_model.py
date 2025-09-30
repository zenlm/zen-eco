#!/usr/bin/env python3.13
"""Test the MLX version of zen-eco-4b-agent"""

try:
    from mlx_lm import load, generate

    print("="*60)
    print("TESTING MLX MODEL FROM HUGGINGFACE")
    print("="*60)

    model_id = "zenlm/zen-eco-4b-agent-mlx"
    print(f"\n📥 Loading MLX model: {model_id}")

    # Load model and tokenizer
    model, tokenizer = load(model_id)
    print("✅ MLX model loaded")

    # Test generation
    test_prompts = [
        "Who are you?",
        "What is your name?",
        "Hello"
    ]

    print("\n" + "="*60)
    print("TESTING MLX GENERATION")
    print("="*60)

    for prompt in test_prompts:
        print(f"\n📝 Prompt: {prompt}")

        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=30
        )

        print(f"🤖 Response: {response}")

        # Check for Zen identity
        if "Zen" in response or "zen" in response:
            print("✅ Zen identity confirmed!")
        else:
            print("⚠️ No Zen identity markers found")

    print("\n" + "="*60)
    print("✅ MLX MODEL TEST COMPLETE")
    print("="*60)

except ImportError:
    print("❌ MLX not installed. Install with: pip install mlx-lm")
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()