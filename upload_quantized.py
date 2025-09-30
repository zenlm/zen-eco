#!/usr/bin/env python3.13
"""Upload MLX and GGUF versions of zen-eco-4b-agent to HuggingFace"""

from huggingface_hub import HfApi, create_repo, upload_folder
from pathlib import Path

api = HfApi()

# Check login
try:
    user = api.whoami()
    print(f"✅ Logged in as: {user['name']}")
except:
    print("❌ Not logged in to HuggingFace")
    exit(1)

# Upload MLX version
mlx_path = Path("/Users/z/work/zen/zen-eco/agent/mlx")
mlx_repo = "zenlm/zen-eco-4b-agent-mlx"

print(f"\n📤 Uploading MLX to {mlx_repo}...")

# Create MLX repo
repo_url = create_repo(
    repo_id=mlx_repo,
    private=False,
    exist_ok=True
)
print(f"Repository: {repo_url}")

# Create README for MLX
readme_mlx = """---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- zen
- hanzo-ai
- qwen3
- mlx
- agent
---

# Zen Eco 4B Agent - MLX

MLX quantized version of [zenlm/zen-eco-4b-agent](https://huggingface.co/zenlm/zen-eco-4b-agent).

Optimized for Apple Silicon Macs.

## Usage

```python
from mlx_lm import load, generate

model, tokenizer = load("zenlm/zen-eco-4b-agent-mlx")

response = generate(model, tokenizer, prompt="Who are you?", max_tokens=50)
print(response)
```

## Original Model

- Base: Qwen3-4B
- Parameters: 4B
- Training: Fine-tuned with Zen identity and tool-calling capabilities
- Developer: Hanzo AI
"""

(mlx_path / "README.md").write_text(readme_mlx)

# Upload MLX
api.upload_folder(
    folder_path=str(mlx_path),
    repo_id=mlx_repo,
    repo_type="model"
)
print(f"✅ MLX uploaded: https://huggingface.co/{mlx_repo}")

# Upload GGUF version
gguf_path = Path("/Users/z/work/zen/zen-eco/agent/gguf")
gguf_repo = "zenlm/zen-eco-4b-agent-gguf"

print(f"\n📤 Uploading GGUF to {gguf_repo}...")

# Create GGUF repo
repo_url = create_repo(
    repo_id=gguf_repo,
    private=False,
    exist_ok=True
)
print(f"Repository: {repo_url}")

# Create README for GGUF
readme_gguf = """---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- zen
- hanzo-ai
- qwen3
- gguf
- agent
---

# Zen Eco 4B Agent - GGUF

GGUF quantized version of [zenlm/zen-eco-4b-agent](https://huggingface.co/zenlm/zen-eco-4b-agent).

## Available Files

- `zen-eco-4b-agent-f16.gguf` - F16 format (7.5GB)

Note: Smaller quantizations (Q4_K_M, Q5_K_M, Q8_0) had NaN issues during training and are being resolved.

## Usage

```bash
# With llama.cpp
./llama-cli -m zen-eco-4b-agent-f16.gguf -p "Who are you?" -n 50

# With LM Studio
# Load the GGUF file directly in LM Studio
```

## Original Model

- Base: Qwen3-4B
- Parameters: 4B
- Training: Fine-tuned with Zen identity and tool-calling capabilities
- Developer: Hanzo AI
"""

(gguf_path / "README.md").write_text(readme_gguf)

# Only upload valid GGUF files (>100MB)
valid_ggufs = [f for f in gguf_path.glob("*.gguf") if f.stat().st_size > 100*1024*1024]
print(f"Uploading {len(valid_ggufs)} valid GGUF files...")

for gguf_file in valid_ggufs:
    api.upload_file(
        path_or_fileobj=str(gguf_file),
        path_in_repo=gguf_file.name,
        repo_id=gguf_repo,
        repo_type="model"
    )
    print(f"  ✅ {gguf_file.name}")

# Upload README
api.upload_file(
    path_or_fileobj=str(gguf_path / "README.md"),
    path_in_repo="README.md",
    repo_id=gguf_repo,
    repo_type="model"
)

print(f"✅ GGUF uploaded: https://huggingface.co/{gguf_repo}")

print("\n" + "=" * 60)
print("✅ ALL UPLOADS COMPLETE!")
print("=" * 60)
print(f"Main model: https://huggingface.co/zenlm/zen-eco-4b-agent")
print(f"MLX version: https://huggingface.co/{mlx_repo}")
print(f"GGUF version: https://huggingface.co/{gguf_repo}")