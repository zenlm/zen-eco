#!/usr/bin/env python3.13
"""Convert zen-eco-4b-agent to MLX and GGUF formats"""

import subprocess
import os
from pathlib import Path

def convert_to_mlx():
    """Convert agent to MLX format"""
    print("\n" + "=" * 60)
    print("CONVERTING TO MLX FORMAT")
    print("=" * 60)

    model_path = "/Users/z/work/zen/zen-eco/agent/finetuned"
    mlx_path = "/Users/z/work/zen/zen-eco/agent/mlx"

    # Create MLX directory
    os.makedirs(mlx_path, exist_ok=True)

    print(f"Source: {model_path}")
    print(f"Output: {mlx_path}")

    # Convert to MLX
    try:
        subprocess.run([
            "python3.13", "-m", "mlx_lm.convert",
            "--hf-path", model_path,
            "--mlx-path", mlx_path,
            "--quantize"
        ], check=True)
        print("✅ MLX conversion complete")
        return mlx_path
    except subprocess.CalledProcessError as e:
        print(f"❌ MLX conversion failed: {e}")
        return None

def convert_to_gguf():
    """Convert agent to GGUF formats"""
    print("\n" + "=" * 60)
    print("CONVERTING TO GGUF FORMATS")
    print("=" * 60)

    model_path = Path("/Users/z/work/zen/zen-eco/agent/finetuned")
    gguf_dir = Path("/Users/z/work/zen/zen-eco/agent/gguf")
    llama_cpp = Path("/Users/z/work/zen/llama.cpp")

    # Create GGUF directory
    gguf_dir.mkdir(parents=True, exist_ok=True)

    # Convert to F16 GGUF
    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    f16_file = gguf_dir / "zen-eco-4b-agent-f16.gguf"

    print("Converting to F16 GGUF...")
    try:
        subprocess.run([
            "python3.13", str(convert_script),
            str(model_path),
            "--outfile", str(f16_file),
            "--outtype", "f16"
        ], check=True)
        print(f"✅ Created: {f16_file.name}")
    except subprocess.CalledProcessError as e:
        print(f"❌ F16 conversion failed: {e}")
        return None

    # Quantize to different formats
    quantize_exe = llama_cpp / "build" / "bin" / "llama-quantize"

    quantizations = [
        ("Q4_K_M", "zen-eco-4b-agent-Q4_K_M.gguf"),
        ("Q5_K_M", "zen-eco-4b-agent-Q5_K_M.gguf"),
        ("Q8_0", "zen-eco-4b-agent-Q8_0.gguf"),
    ]

    created_files = [f16_file]

    for quant_type, filename in quantizations:
        output_file = gguf_dir / filename
        print(f"\nCreating {quant_type}...")

        try:
            subprocess.run([
                str(quantize_exe),
                str(f16_file),
                str(output_file),
                quant_type
            ], check=True, capture_output=True)

            # Check file size
            size_mb = output_file.stat().st_size / (1024**2)
            if size_mb > 100:  # Should be much larger than 100MB
                print(f"✅ Created: {filename} ({size_mb:.1f}MB)")
                created_files.append(output_file)
            else:
                print(f"⚠️ {filename} too small ({size_mb:.1f}MB), may have NaN issues")

        except subprocess.CalledProcessError as e:
            print(f"❌ {quant_type} quantization failed")
            # Continue with other quantizations

    if len(created_files) > 1:
        print(f"\n✅ GGUF conversion complete: {len(created_files)} files created")
        return gguf_dir
    else:
        print("⚠️ Only F16 created, quantizations failed")
        return gguf_dir

def main():
    print("=" * 60)
    print("CONVERTING ZEN-ECO-4B-AGENT")
    print("=" * 60)

    # Convert to MLX
    mlx_path = convert_to_mlx()

    # Convert to GGUF
    gguf_path = convert_to_gguf()

    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)

    if mlx_path:
        print(f"✅ MLX: {mlx_path}")
        # List MLX files
        mlx_files = list(Path(mlx_path).glob("*"))
        for f in mlx_files[:5]:
            size_mb = f.stat().st_size / (1024**2) if f.is_file() else 0
            print(f"   - {f.name}: {size_mb:.1f}MB")

    if gguf_path:
        print(f"✅ GGUF: {gguf_path}")
        # List GGUF files
        gguf_files = list(Path(gguf_path).glob("*.gguf"))
        for f in gguf_files:
            size_mb = f.stat().st_size / (1024**2)
            print(f"   - {f.name}: {size_mb:.1f}MB")

    print("\nNext step: Upload to HuggingFace")
    print("  - MLX version as separate repo: zenlm/zen-eco-4b-agent-mlx")
    print("  - GGUF files as separate repo: zenlm/zen-eco-4b-agent-gguf")

if __name__ == "__main__":
    main()