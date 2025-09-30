#!/usr/bin/env python3.13
"""Upload MLX and GGUF files to the main zen-eco-4b-agent repository"""

from huggingface_hub import HfApi
from pathlib import Path
import sys

api = HfApi()

# Check login
try:
    user = api.whoami()
    print(f"✅ Logged in as: {user['name']}")
except:
    print("❌ Not logged in to HuggingFace")
    sys.exit(1)

repo_id = "zenlm/zen-eco-4b-agent"
print(f"\n📤 Uploading to main repository: {repo_id}")
print("="*60)

# Upload MLX files
mlx_path = Path("/Users/z/work/zen/zen-eco/agent/mlx")
if mlx_path.exists():
    print("\n📦 Uploading MLX files...")
    mlx_files = list(mlx_path.glob("*.npz")) + list(mlx_path.glob("*.json"))

    for file in mlx_files:
        if file.stat().st_size > 1024:  # Skip tiny files
            size_mb = file.stat().st_size / (1024**2)
            print(f"  Uploading {file.name} ({size_mb:.1f}MB)...")

            try:
                # Upload to mlx/ subdirectory
                api.upload_file(
                    path_or_fileobj=str(file),
                    path_in_repo=f"mlx/{file.name}",
                    repo_id=repo_id,
                    repo_type="model"
                )
                print(f"  ✅ Uploaded: mlx/{file.name}")
            except Exception as e:
                print(f"  ❌ Failed: {e}")

# Upload GGUF file
gguf_path = Path("/Users/z/work/zen/zen-eco/agent/gguf")
if gguf_path.exists():
    print("\n📦 Uploading GGUF files...")

    # Only upload the valid F16 GGUF
    f16_file = gguf_path / "zen-eco-4b-agent-f16.gguf"
    if f16_file.exists() and f16_file.stat().st_size > 100*1024*1024:
        size_gb = f16_file.stat().st_size / (1024**3)
        print(f"  Uploading {f16_file.name} ({size_gb:.1f}GB)...")

        try:
            # Upload to gguf/ subdirectory
            api.upload_file(
                path_or_fileobj=str(f16_file),
                path_in_repo=f"gguf/{f16_file.name}",
                repo_id=repo_id,
                repo_type="model"
            )
            print(f"  ✅ Uploaded: gguf/{f16_file.name}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")

print("\n" + "="*60)
print("✅ Upload complete!")
print(f"View at: https://huggingface.co/{repo_id}")
print("="*60)