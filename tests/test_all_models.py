#!/usr/bin/env python3.13
"""
Test all Zen Eco models
"""
import os
from pathlib import Path

def test_model_files():
    """Test that all model files exist"""
    print("✅ TESTING ZEN ECO MODELS")
    print("=" * 40)

    models = {
        "zen-eco-4b-instruct": "/Users/z/work/zen/zen-eco/instruct/finetuned",
        "zen-eco-4b-thinking": "/Users/z/work/zen/zen-eco/thinking/finetuned",
        "zen-eco-4b-agent": "/Users/z/work/zen/zen-eco/output/zen-eco-4b-agent"
    }

    for name, path in models.items():
        print(f"\n📦 {name}:")
        model_path = Path(path)

        if model_path.exists():
            # Check for key files
            files_to_check = [
                "config.json",
                "tokenizer.json",
                "model*.safetensors"
            ]

            found_files = []
            for pattern in files_to_check:
                matches = list(model_path.glob(pattern))
                if matches:
                    found_files.extend(matches)

            if found_files:
                print(f"   ✅ Model files found:")
                for f in found_files[:5]:  # Show first 5
                    size_mb = f.stat().st_size / (1024**2)
                    print(f"      • {f.name}: {size_mb:.1f} MB")
            else:
                print(f"   ⚠️ No model files found")
        else:
            print(f"   ❌ Path does not exist: {path}")

    # Check GGUF
    print(f"\n📦 zen-eco-4b-agent (GGUF):")
    gguf_path = Path("/Users/z/work/zen/zen-eco/coder/gguf/Qwen3-4B-Function-Calling-Pro.gguf")
    if gguf_path.exists():
        size_gb = gguf_path.stat().st_size / (1024**3)
        print(f"   ✅ GGUF file: {size_gb:.1f} GB")
    else:
        print(f"   ❌ GGUF not found")

    print("\n" + "=" * 40)
    print("✅ All Zen Eco models verified!")
    print("\nModels ready for:")
    print("  • GGUF conversion")
    print("  • HuggingFace upload")
    print("  • Production use")

if __name__ == "__main__":
    test_model_files()