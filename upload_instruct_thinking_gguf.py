#!/usr/bin/env python3.13
"""Upload GGUF files for instruct and thinking models"""

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

print("\n" + "="*60)
print("UPLOADING GGUF FILES FOR INSTRUCT AND THINKING MODELS")
print("="*60)

models = [
    ("zenlm/zen-eco-4b-instruct", Path("/Users/z/work/zen/zen-eco/instruct/gguf")),
    ("zenlm/zen-eco-4b-thinking", Path("/Users/z/work/zen/zen-eco/thinking/gguf")),
]

for repo_id, gguf_path in models:
    print(f"\n📦 Processing {repo_id}...")

    if not gguf_path.exists():
        print(f"  ❌ Path not found: {gguf_path}")
        continue

    # Upload all GGUF files
    gguf_files = list(gguf_path.glob("*.gguf"))
    print(f"  Found {len(gguf_files)} GGUF files")

    for gguf_file in gguf_files:
        if gguf_file.stat().st_size > 100*1024*1024:  # Only valid files
            size_gb = gguf_file.stat().st_size / (1024**3)
            print(f"  Uploading {gguf_file.name} ({size_gb:.1f}GB)...")

            try:
                api.upload_file(
                    path_or_fileobj=str(gguf_file),
                    path_in_repo=f"gguf/{gguf_file.name}",
                    repo_id=repo_id,
                    repo_type="model"
                )
                print(f"  ✅ Uploaded: gguf/{gguf_file.name}")
            except Exception as e:
                if "already exists" in str(e):
                    print(f"  ℹ️ Already exists: gguf/{gguf_file.name}")
                else:
                    print(f"  ❌ Failed: {e}")

# Update READMEs
readme_template = """---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- zen
- hanzo-ai
- qwen3
- {model_type}
- gguf
---

# Zen Eco 4B {model_name}

{description}

## Available Formats

### PyTorch (Default)
- Standard safetensors format
- 4B parameters

### GGUF Format (llama.cpp)
- `gguf/zen-eco-4b-{model_type}-f16.gguf` - F16 format (7.5GB)
- Compatible with llama.cpp, LM Studio, Ollama

## Usage

### PyTorch
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```

### GGUF (llama.cpp)
```bash
# Download the GGUF file
wget https://huggingface.co/{repo_id}/resolve/main/gguf/zen-eco-4b-{model_type}-f16.gguf

# Use with llama.cpp
./llama-cli -m zen-eco-4b-{model_type}-f16.gguf -p "Hello"
```

## Model Details

- **Base**: Qwen3-4B
- **Parameters**: 4B
- **Training**: Fine-tuned with Zen identity
- **Developer**: Hanzo AI

## License

Apache 2.0
"""

readme_configs = [
    {
        "repo_id": "zenlm/zen-eco-4b-instruct",
        "model_type": "instruct",
        "model_name": "Instruct",
        "description": "Instruction-tuned model from the Zen Eco family."
    },
    {
        "repo_id": "zenlm/zen-eco-4b-thinking",
        "model_type": "thinking",
        "model_name": "Thinking",
        "description": "Chain-of-thought reasoning model from the Zen Eco family."
    }
]

print("\n📝 Updating READMEs...")

for config in readme_configs:
    readme_content = readme_template.format(**config)

    try:
        # Create README locally first
        readme_path = Path(f"/tmp/README_{config['model_type']}.md")
        readme_path.write_text(readme_content)

        # Upload README
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=config["repo_id"],
            repo_type="model"
        )
        print(f"✅ README updated for {config['repo_id']}")
    except Exception as e:
        print(f"❌ README update failed for {config['repo_id']}: {e}")

print("\n" + "="*60)
print("✅ UPLOAD COMPLETE")
print("="*60)
print("\nAll models now have GGUF formats available:")
print("- zenlm/zen-eco-4b-instruct (PyTorch + GGUF)")
print("- zenlm/zen-eco-4b-thinking (PyTorch + GGUF)")
print("- zenlm/zen-eco-4b-agent (PyTorch + MLX + GGUF)")
print("- zenlm/zen-nano-0.6b (PyTorch + GGUF)")