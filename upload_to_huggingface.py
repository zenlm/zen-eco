#!/usr/bin/env python3.13
"""Upload Zen models to HuggingFace Hub"""

from huggingface_hub import HfApi, create_repo, upload_folder
import os
from pathlib import Path
import shutil

# Models to upload
models = [
    {
        "name": "zenlm/zen-nano-0.6b",
        "path": "/Users/z/work/zen/zen-nano/finetuned",
        "description": "Zen Nano 0.6B - Smallest model in the Zen family",
    },
    {
        "name": "zenlm/zen-eco-4b-instruct",
        "path": "/Users/z/work/zen/zen-eco/instruct/finetuned",
        "description": "Zen Eco 4B Instruct - Instruction-tuned model",
    },
    {
        "name": "zenlm/zen-eco-4b-thinking",
        "path": "/Users/z/work/zen/zen-eco/thinking/finetuned",
        "description": "Zen Eco 4B Thinking - Chain-of-thought reasoning model",
    },
    {
        "name": "zenlm/zen-eco-4b-agent",
        "path": "/Users/z/work/zen/zen-eco/agent/finetuned",
        "description": "Zen Eco 4B Agent - Tool-calling agent model",
    },
]

def create_model_card(name, description):
    """Create a model card for HuggingFace"""
    model_type = name.split("-")[-1]
    size = "0.6B" if "nano" in name else "4B"

    return f"""---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- zen
- hanzo-ai
- qwen3
- {model_type}
---

# {name}

{description}

## Model Details

- **Architecture**: Qwen3 base
- **Parameters**: {size}
- **Training**: Fine-tuned with Zen identity
- **Developer**: Hanzo AI

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{name}")
tokenizer = AutoTokenizer.from_pretrained("{name}")

prompt = "Hello, who are you?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training

Trained with fixed seed (42) for reproducibility.
Base model: Qwen3-{size}

## License

Apache 2.0
"""

def upload_model(model_info):
    """Upload a model to HuggingFace"""
    name = model_info["name"]
    path = Path(model_info["path"])
    desc = model_info["description"]

    if not path.exists():
        print(f"❌ Model not found: {path}")
        return False

    print(f"\n{'='*60}")
    print(f"Uploading: {name}")
    print(f"Path: {path}")
    print(f"Description: {desc}")
    print('='*60)

    try:
        # Create repository
        print("Creating repository...")
        api = HfApi()

        # Check if logged in
        try:
            user = api.whoami()
            print(f"Logged in as: {user['name']}")
        except:
            print("❌ Not logged in to HuggingFace")
            print("Run: huggingface-cli login")
            return False

        # Create repo
        repo_url = create_repo(
            repo_id=name,
            private=False,
            exist_ok=True
        )
        print(f"Repository: {repo_url}")

        # Create model card
        model_card_path = path / "README.md"
        with open(model_card_path, "w") as f:
            f.write(create_model_card(name, desc))

        # Upload folder
        print("Uploading model files...")
        api.upload_folder(
            folder_path=str(path),
            repo_id=name,
            repo_type="model",
            ignore_patterns=["checkpoint-*", "*.pt", "*.pth", "trainer_state.json"],
        )

        print(f"✅ Uploaded successfully: https://huggingface.co/{name}")
        return True

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

def main():
    print("UPLOADING ZEN MODELS TO HUGGINGFACE")
    print("="*60)

    # Check if logged in
    api = HfApi()
    try:
        user = api.whoami()
        print(f"✅ Logged in as: {user['name']}")
    except:
        print("❌ Not logged in to HuggingFace")
        print("\nTo upload models, run:")
        print("  huggingface-cli login")
        print("\nThen run this script again.")
        return

    # Upload each model
    success = []
    failed = []

    for model in models:
        if upload_model(model):
            success.append(model["name"])
        else:
            failed.append(model["name"])

    # Summary
    print("\n" + "="*60)
    print("UPLOAD SUMMARY")
    print("="*60)

    if success:
        print(f"\n✅ Successfully uploaded ({len(success)}):")
        for name in success:
            print(f"  - https://huggingface.co/{name}")

    if failed:
        print(f"\n❌ Failed to upload ({len(failed)}):")
        for name in failed:
            print(f"  - {name}")

if __name__ == "__main__":
    main()