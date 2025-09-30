#!/usr/bin/env python3.13
"""Convert all zen-eco models to GGUF format"""

import subprocess
import os
from pathlib import Path

def convert_to_gguf(model_name, model_path, output_dir):
    """Convert a model to GGUF format"""
    print("\n" + "="*60)
    print(f"CONVERTING {model_name.upper()} TO GGUF")
    print("="*60)

    llama_cpp = Path("/Users/z/work/zen/llama.cpp")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to F16 GGUF
    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    f16_file = output_dir / f"zen-eco-4b-{model_name}-f16.gguf"

    print(f"Converting to F16 GGUF...")
    print(f"Source: {model_path}")
    print(f"Output: {f16_file}")

    try:
        subprocess.run([
            "python3.13", str(convert_script),
            str(model_path),
            "--outfile", str(f16_file),
            "--outtype", "f16"
        ], check=True)
        print(f"✅ Created: {f16_file.name}")

        # Check file size
        size_gb = f16_file.stat().st_size / (1024**3)
        print(f"   Size: {size_gb:.1f}GB")

        # Try quantizations
        quantize_exe = llama_cpp / "build" / "bin" / "llama-quantize"

        quantizations = [
            ("Q4_K_M", f"zen-eco-4b-{model_name}-Q4_K_M.gguf"),
            ("Q5_K_M", f"zen-eco-4b-{model_name}-Q5_K_M.gguf"),
            ("Q8_0", f"zen-eco-4b-{model_name}-Q8_0.gguf"),
        ]

        for quant_type, filename in quantizations:
            output_file = output_dir / filename
            print(f"\nTrying {quant_type} quantization...")

            try:
                result = subprocess.run([
                    str(quantize_exe),
                    str(f16_file),
                    str(output_file),
                    quant_type
                ], capture_output=True, text=True)

                if output_file.exists():
                    size_mb = output_file.stat().st_size / (1024**2)
                    if size_mb > 100:  # Should be much larger
                        print(f"✅ Created: {filename} ({size_mb:.1f}MB)")
                    else:
                        print(f"⚠️ {filename} too small ({size_mb:.1f}MB), likely has NaN issues")
                        output_file.unlink()  # Remove bad file
                else:
                    print(f"❌ {quant_type} quantization failed")

            except subprocess.CalledProcessError:
                print(f"❌ {quant_type} quantization failed")

        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ F16 conversion failed: {e}")
        return False

def main():
    print("="*60)
    print("CONVERTING ALL ZEN-ECO MODELS TO GGUF")
    print("="*60)

    models = [
        ("instruct", Path("/Users/z/work/zen/zen-eco/instruct/finetuned"),
         Path("/Users/z/work/zen/zen-eco/instruct/gguf")),
        ("thinking", Path("/Users/z/work/zen/zen-eco/thinking/finetuned"),
         Path("/Users/z/work/zen/zen-eco/thinking/gguf")),
    ]

    success = []
    failed = []

    for name, model_path, output_dir in models:
        if model_path.exists():
            if convert_to_gguf(name, model_path, output_dir):
                success.append(name)
            else:
                failed.append(name)
        else:
            print(f"\n❌ Model not found: {model_path}")
            failed.append(name)

    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)

    if success:
        print(f"\n✅ Successfully converted: {', '.join(success)}")

    if failed:
        print(f"\n❌ Failed to convert: {', '.join(failed)}")

    print("\nNext step: Upload GGUF files to HuggingFace")

if __name__ == "__main__":
    main()