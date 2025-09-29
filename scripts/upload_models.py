#!/usr/bin/env python3.13
"""
Upload Zen Eco models to HuggingFace
Simple upload script with proper authentication
"""
import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

def upload_model(model_path, hf_username, model_name):
    """Upload a single model to HuggingFace"""
    print(f"\n☁️  Uploading {model_name}...")

    api = HfApi()

    # Repository name
    repo_id = f"{hf_username}/{model_name}"

    try:
        # Create repository if it doesn't exist
        print(f"  📦 Creating repo: {repo_id}")
        create_repo(repo_id, exist_ok=True, private=False)

        # Upload model files
        print(f"  📤 Uploading files from {model_path}...")
        upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {model_name}"
        )

        print(f"  ✅ Uploaded to https://huggingface.co/{repo_id}")
        return True

    except Exception as e:
        print(f"  ❌ Error uploading {model_name}: {e}")
        return False

def upload_gguf(gguf_path, hf_username, model_name):
    """Upload GGUF file to HuggingFace"""
    print(f"\n📦 Uploading GGUF: {model_name}")

    api = HfApi()
    repo_id = f"{hf_username}/{model_name}-gguf"

    try:
        # Create GGUF repository
        print(f"  📦 Creating GGUF repo: {repo_id}")
        create_repo(repo_id, exist_ok=True, private=False)

        # Upload GGUF file
        print(f"  📤 Uploading {gguf_path}...")
        api.upload_file(
            path_or_fileobj=gguf_path,
            path_in_repo=Path(gguf_path).name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload GGUF for {model_name}"
        )

        print(f"  ✅ GGUF uploaded to https://huggingface.co/{repo_id}")
        return True

    except Exception as e:
        print(f"  ❌ Error uploading GGUF: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, required=True)
    parser.add_argument("--gguf_dir", type=str, default="gguf")
    parser.add_argument("--hf_username", type=str, required=True)
    args = parser.parse_args()

    print("☁️  Zen Eco Models Upload")
    print(f"📁 Models directory: {args.models_dir}")
    print(f"👤 HuggingFace username: {args.hf_username}")

    # Check for HF token
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        print("⚠️  No HuggingFace token found!")
        print("    Set HF_TOKEN environment variable:")
        print("    export HF_TOKEN=your_token_here")
        print("\n    Get token from: https://huggingface.co/settings/tokens")
        sys.exit(1)

    # Find models to upload
    models_uploaded = 0
    ggufs_uploaded = 0

    # Upload model directories
    if os.path.exists(args.models_dir):
        for item in os.listdir(args.models_dir):
            model_path = os.path.join(args.models_dir, item)
            if os.path.isdir(model_path) and "zen-eco" in item:
                config_file = os.path.join(model_path, "config.json")
                if os.path.exists(config_file):
                    if upload_model(model_path, args.hf_username, item):
                        models_uploaded += 1

    # Upload GGUF files
    if os.path.exists(args.gguf_dir):
        for gguf_file in Path(args.gguf_dir).glob("*.gguf"):
            model_name = gguf_file.stem
            if upload_gguf(str(gguf_file), args.hf_username, model_name):
                ggufs_uploaded += 1

    # Summary
    print("\n" + "="*50)
    print(f"✅ Upload complete!")
    print(f"   📤 Models uploaded: {models_uploaded}")
    print(f"   📦 GGUF files uploaded: {ggufs_uploaded}")
    print(f"\n🔗 View your models at: https://huggingface.co/{args.hf_username}")

if __name__ == "__main__":
    main()