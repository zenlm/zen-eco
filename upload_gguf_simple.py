#!/usr/bin/env python3.13
"""Upload just the F16 GGUF file to HuggingFace"""

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

gguf_file = Path("/Users/z/work/zen/zen-eco/agent/gguf/zen-eco-4b-agent-f16.gguf")
repo_id = "zenlm/zen-eco-4b-agent-gguf"

if not gguf_file.exists():
    print(f"❌ File not found: {gguf_file}")
    sys.exit(1)

size_gb = gguf_file.stat().st_size / (1024**3)
print(f"📦 File: {gguf_file.name} ({size_gb:.1f}GB)")

print(f"\n📤 Uploading to {repo_id}...")

try:
    # Upload the F16 GGUF file
    api.upload_file(
        path_or_fileobj=str(gguf_file),
        path_in_repo=gguf_file.name,
        repo_id=repo_id,
        repo_type="model"
    )
    print(f"✅ Uploaded: {gguf_file.name}")

    # Upload README
    readme_path = gguf_file.parent / "README.md"
    if readme_path.exists():
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model"
        )
        print("✅ Uploaded: README.md")

    print(f"\n✅ GGUF uploaded: https://huggingface.co/{repo_id}")

except Exception as e:
    print(f"❌ Upload failed: {e}")
    sys.exit(1)