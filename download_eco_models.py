#!/usr/bin/env python3.13
"""
Download Qwen3-4B models for Zen Eco family
Using the EXACT models specified:
- Qwen3-4B-Instruct-2507
- Qwen3-4B-Thinking-2507  
- Manojb/Qwen3-4b-toolcall-gguf-llamacpp-codex for coder
"""
import os
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
import shutil

def download_eco_models():
    print("=" * 60)
    print("DOWNLOADING QWEN3-4B MODELS FOR ZEN ECO")
    print("Using exact 4B models (NOT 7B!)")
    print("=" * 60)
    
    # 1. Download Qwen3-4B-Instruct-2507
    print("\n1️⃣  Downloading Qwen3-4B-Instruct-2507...")
    instruct_model_id = "Qwen/Qwen3-4B-Instruct-2507"
    
    try:
        instruct_cache = snapshot_download(
            repo_id=instruct_model_id,
            cache_dir="/Users/z/work/zen/zen-eco/.cache"
        )
        print(f"   ✅ Downloaded {instruct_model_id}")
        
        # Link to instruct variant
        instruct_path = Path("/Users/z/work/zen/zen-eco/instruct/base-model")
        if instruct_path.exists():
            if instruct_path.is_symlink():
                instruct_path.unlink()
            else:
                shutil.rmtree(instruct_path)
        instruct_path.symlink_to(instruct_cache)
        print(f"   ↳ Linked to instruct/base-model")
    except Exception as e:
        print(f"   ⚠️  Error downloading instruct: {e}")
    
    # 2. Download Qwen3-4B-Thinking-2507
    print("\n2️⃣  Downloading Qwen3-4B-Thinking-2507...")
    thinking_model_id = "Qwen/Qwen3-4B-Thinking-2507"
    
    try:
        thinking_cache = snapshot_download(
            repo_id=thinking_model_id,
            cache_dir="/Users/z/work/zen/zen-eco/.cache"
        )
        print(f"   ✅ Downloaded {thinking_model_id}")
        
        # Link to thinking variant
        thinking_path = Path("/Users/z/work/zen/zen-eco/thinking/base-model")
        if thinking_path.exists():
            if thinking_path.is_symlink():
                thinking_path.unlink()
            else:
                shutil.rmtree(thinking_path)
        thinking_path.symlink_to(thinking_cache)
        print(f"   ↳ Linked to thinking/base-model")
    except Exception as e:
        print(f"   ⚠️  Error downloading thinking: {e}")
    
    # 3. Download toolcall model for coder
    print("\n3️⃣  Downloading Manojb/Qwen3-4b-toolcall-gguf-llamacpp-codex...")
    coder_model_id = "Manojb/Qwen3-4b-toolcall-gguf-llamacpp-codex"
    coder_gguf_path = Path("/Users/z/work/zen/zen-eco/coder/gguf")
    coder_gguf_path.mkdir(parents=True, exist_ok=True)
    
    # List available files in the repo first
    from huggingface_hub import list_repo_files
    try:
        files = list_repo_files(coder_model_id)
        gguf_files = [f for f in files if f.endswith('.gguf')]
        print(f"   Available GGUF files: {gguf_files}")
        
        if gguf_files:
            # Download the first available GGUF
            gguf_file = gguf_files[0]
            print(f"   Downloading {gguf_file}...")
            downloaded_file = hf_hub_download(
                repo_id=coder_model_id,
                filename=gguf_file,
                cache_dir="/Users/z/work/zen/zen-eco/.cache",
                local_dir=str(coder_gguf_path)
            )
            print(f"   ✅ Downloaded coder model to coder/gguf/")
        else:
            print(f"   ⚠️  No GGUF files found in {coder_model_id}")
    except Exception as e:
        print(f"   ⚠️  Error downloading coder model: {e}")
    
    print("\n" + "=" * 60)
    print("SUMMARY - Zen Eco (4B) Models:")
    print("- zen-eco-4b-instruct: Qwen3-4B-Instruct-2507")
    print("- zen-eco-4b-thinking: Qwen3-4B-Thinking-2507")  
    print("- zen-eco-4b-coder: Manojb toolcall model")
    print("=" * 60)

if __name__ == "__main__":
    download_eco_models()