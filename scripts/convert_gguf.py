#!/usr/bin/env python3.13
"""
Convert models to GGUF format
Simple wrapper around llama.cpp conversion
"""
import argparse
import os
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--quantization", type=str, default="q4_k_m")
    args = parser.parse_args()

    print(f"📦 Converting to GGUF: {args.model_path}")
    print(f"📊 Quantization: {args.quantization}")

    # Check if llama.cpp is available
    llama_cpp_path = "/Users/z/work/llama.cpp"
    if not os.path.exists(llama_cpp_path):
        print("⚠️  llama.cpp not found, downloading...")
        subprocess.run([
            "git", "clone",
            "https://github.com/ggerganov/llama.cpp",
            llama_cpp_path
        ], check=True)

        # Build llama.cpp
        print("Building llama.cpp...")
        subprocess.run(["make", "-C", llama_cpp_path], check=True)

    # Convert to GGUF
    convert_script = f"{llama_cpp_path}/convert-hf-to-gguf.py"

    # First convert to FP16 GGUF
    temp_gguf = args.output_path.replace(".gguf", "_fp16.gguf")

    print("Converting to GGUF FP16...")
    subprocess.run([
        "python3.13", convert_script,
        args.model_path,
        "--outfile", temp_gguf,
        "--outtype", "f16"
    ], check=True)

    # Then quantize
    if args.quantization != "f16":
        print(f"Quantizing to {args.quantization}...")
        quantize_bin = f"{llama_cpp_path}/llama-quantize"

        subprocess.run([
            quantize_bin,
            temp_gguf,
            args.output_path,
            args.quantization
        ], check=True)

        # Remove temp file
        os.remove(temp_gguf)
    else:
        os.rename(temp_gguf, args.output_path)

    # Verify output
    output_size = os.path.getsize(args.output_path) / (1024 * 1024 * 1024)
    print(f"✅ GGUF created: {args.output_path}")
    print(f"📏 Size: {output_size:.2f} GB")

if __name__ == "__main__":
    main()