#!/usr/bin/env python3.13
"""Upload zen-eco-4b-agent model to HuggingFace"""

from huggingface_hub import HfApi, upload_folder
from pathlib import Path

def main():
    api = HfApi()

    # Check login
    try:
        user = api.whoami()
        print(f"✅ Logged in as: {user['name']}")
    except:
        print("❌ Not logged in to HuggingFace")
        return

    model_path = Path("/Users/z/work/zen/zen-eco/agent/finetuned")
    repo_id = "zenlm/zen-eco-4b-agent"

    print(f"Uploading {model_path} to {repo_id}...")

    # List files to upload
    files = list(model_path.glob("*"))
    print(f"Found {len(files)} files:")
    for f in files:
        if f.is_file():
            size_mb = f.stat().st_size / (1024*1024)
            print(f"  - {f.name}: {size_mb:.1f}MB")

    # Upload
    print("\nUploading to HuggingFace...")
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["*.pt", "*.pth", "trainer_state.json", "checkpoint-*"],
    )

    print(f"✅ Upload complete: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()