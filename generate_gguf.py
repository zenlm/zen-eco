#!/usr/bin/env python3.13
"""
Generate GGUF files from our trained Zen Eco models
"""
import os
import subprocess
from pathlib import Path

def convert_to_gguf(model_path, output_name):
    """Convert a model to GGUF format"""
    print(f"🔄 Converting {output_name} to GGUF...")

    # Ensure llama.cpp is available
    llama_cpp = Path("/Users/z/work/zen/llama.cpp")
    if not llama_cpp.exists():
        print("⚠️  llama.cpp not found, cloning...")
        subprocess.run([
            "git", "clone",
            "https://github.com/ggerganov/llama.cpp.git",
            str(llama_cpp)
        ], check=True)

        # Build llama.cpp
        print("Building llama.cpp...")
        subprocess.run(["cmake", "-B", "build"], cwd=llama_cpp, check=True)
        subprocess.run(["cmake", "--build", "build", "--config", "Release"], cwd=llama_cpp, check=True)

    # Create GGUF directory
    gguf_dir = Path(f"/Users/z/work/zen/zen-eco/gguf")
    gguf_dir.mkdir(exist_ok=True)

    # Convert to GGUF F16
    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    output_file = gguf_dir / f"{output_name}-f16.gguf"

    print(f"Converting to F16 GGUF: {output_file}")
    subprocess.run([
        "python3.13", str(convert_script),
        str(model_path),
        "--outfile", str(output_file),
        "--outtype", "f16"
    ], check=True)

    # Quantize to Q4_K_M
    quantize_exe = llama_cpp / "build" / "bin" / "llama-quantize"
    q4_file = gguf_dir / f"{output_name}-Q4_K_M.gguf"

    print(f"Creating Q4_K_M quantization...")
    subprocess.run([
        str(quantize_exe),
        str(output_file),
        str(q4_file),
        "Q4_K_M"
    ], check=True)

    # Quantize to Q5_K_M
    q5_file = gguf_dir / f"{output_name}-Q5_K_M.gguf"
    print(f"Creating Q5_K_M quantization...")
    subprocess.run([
        str(quantize_exe),
        str(output_file),
        str(q5_file),
        "Q5_K_M"
    ], check=True)

    print(f"✅ GGUF files created for {output_name}")
    return True

def main():
    print("🎯 Generating GGUF for all trained Zen Eco models")
    print("=" * 50)

    models = [
        ("/Users/z/work/zen/zen-eco/instruct/finetuned", "zen-eco-4b-instruct"),
        ("/Users/z/work/zen/zen-eco/thinking/finetuned", "zen-eco-4b-thinking"),
        ("/Users/z/work/zen/zen-eco/output/zen-eco-4b-agent", "zen-eco-4b-agent"),
    ]

    for model_path, output_name in models:
        if Path(model_path).exists():
            try:
                convert_to_gguf(model_path, output_name)
            except Exception as e:
                print(f"❌ Failed to convert {output_name}: {e}")
        else:
            print(f"⚠️  Model not found: {model_path}")

    print("\n✅ GGUF generation complete!")
    print("\n📦 Generated files:")
    gguf_dir = Path("/Users/z/work/zen/zen-eco/gguf")
    for gguf in gguf_dir.glob("*.gguf"):
        size_gb = gguf.stat().st_size / (1024**3)
        print(f"  • {gguf.name}: {size_gb:.2f} GB")

if __name__ == "__main__":
    main()