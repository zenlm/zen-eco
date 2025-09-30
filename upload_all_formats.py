#!/usr/bin/env python3.13
"""Upload MLX and GGUF files to main model repositories"""

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
print("UPLOADING ALL FORMATS TO MAIN REPOSITORIES")
print("="*60)

# Process zen-eco-4b-agent
print("\n📦 Processing zen-eco-4b-agent...")
repo_id = "zenlm/zen-eco-4b-agent"

# Upload MLX files for agent
mlx_path = Path("/Users/z/work/zen/zen-eco/agent/mlx")
if mlx_path.exists():
    print(f"  Uploading MLX files to {repo_id}...")

    for file in mlx_path.glob("*"):
        if file.is_file() and file.stat().st_size > 1024:
            size_mb = file.stat().st_size / (1024**2)

            # Skip if too small (likely corrupted)
            if file.suffix == ".npz" and size_mb < 10:
                print(f"  ⚠️ Skipping {file.name} - too small ({size_mb:.1f}MB)")
                continue

            print(f"    Uploading {file.name} ({size_mb:.1f}MB)...")
            try:
                api.upload_file(
                    path_or_fileobj=str(file),
                    path_in_repo=f"mlx/{file.name}",
                    repo_id=repo_id,
                    repo_type="model"
                )
                print(f"    ✅ Uploaded: mlx/{file.name}")
            except Exception as e:
                if "already exists" in str(e):
                    print(f"    ℹ️ Already exists: mlx/{file.name}")
                else:
                    print(f"    ❌ Failed: {e}")

# Upload GGUF file for agent
gguf_path = Path("/Users/z/work/zen/zen-eco/agent/gguf")
if gguf_path.exists():
    f16_file = gguf_path / "zen-eco-4b-agent-f16.gguf"
    if f16_file.exists() and f16_file.stat().st_size > 100*1024*1024:
        size_gb = f16_file.stat().st_size / (1024**3)
        print(f"  Uploading GGUF F16 ({size_gb:.1f}GB)...")

        try:
            api.upload_file(
                path_or_fileobj=str(f16_file),
                path_in_repo=f"gguf/{f16_file.name}",
                repo_id=repo_id,
                repo_type="model"
            )
            print(f"  ✅ Uploaded: gguf/{f16_file.name}")
        except Exception as e:
            if "already exists" in str(e):
                print(f"  ℹ️ Already exists: gguf/{f16_file.name}")
            else:
                print(f"  ❌ Failed: {e}")

# Process zen-nano-0.6b
print("\n📦 Processing zen-nano-0.6b...")
repo_id = "zenlm/zen-nano-0.6b"

# Upload GGUF files for nano
gguf_path = Path("/Users/z/work/zen/zen-nano/gguf")
if gguf_path.exists():
    gguf_files = list(gguf_path.glob("*.gguf"))
    print(f"  Found {len(gguf_files)} GGUF files")

    for gguf_file in gguf_files:
        if gguf_file.stat().st_size > 100*1024*1024:  # Only valid files
            size_mb = gguf_file.stat().st_size / (1024**2)
            print(f"    Uploading {gguf_file.name} ({size_mb:.1f}MB)...")

            try:
                api.upload_file(
                    path_or_fileobj=str(gguf_file),
                    path_in_repo=f"gguf/{gguf_file.name}",
                    repo_id=repo_id,
                    repo_type="model"
                )
                print(f"    ✅ Uploaded: gguf/{gguf_file.name}")
            except Exception as e:
                if "already exists" in str(e):
                    print(f"    ℹ️ Already exists: gguf/{gguf_file.name}")
                else:
                    print(f"    ❌ Failed: {e}")

# Update README for zen-eco-4b-agent
print("\n📝 Updating README for zen-eco-4b-agent...")
readme_content = """---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- zen
- hanzo-ai
- qwen3
- agent
- mlx
- gguf
---

# Zen Eco 4B Agent

Agent model from the Zen Eco family, fine-tuned for tool-calling capabilities.

## Available Formats

### PyTorch (Default)
- `model.safetensors.index.json` - Model index
- `model-00001-of-00002.safetensors` - Model shard 1
- `model-00002-of-00002.safetensors` - Model shard 2

### MLX Format (Apple Silicon)
- `mlx/` - Quantized MLX format for Apple Silicon
- Optimized for M1/M2/M3 Macs
- 4.5 bits per weight quantization

### GGUF Format (llama.cpp)
- `gguf/zen-eco-4b-agent-f16.gguf` - F16 format (7.5GB)
- Compatible with llama.cpp, LM Studio, Ollama

## Usage

### PyTorch
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("zenlm/zen-eco-4b-agent")
tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-eco-4b-agent")
```

### MLX (Apple Silicon)
```python
from mlx_lm import load, generate

# Load from the mlx subdirectory
model, tokenizer = load("zenlm/zen-eco-4b-agent", model_config={"model_path": "mlx"})
```

### GGUF (llama.cpp)
```bash
# Download the GGUF file
wget https://huggingface.co/zenlm/zen-eco-4b-agent/resolve/main/gguf/zen-eco-4b-agent-f16.gguf

# Use with llama.cpp
./llama-cli -m zen-eco-4b-agent-f16.gguf -p "Who are you?"
```

## Model Details

- **Base**: Qwen3-4B
- **Parameters**: 4B
- **Training**: Fine-tuned with Zen identity and tool-calling capabilities
- **Developer**: Hanzo AI

## License

Apache 2.0
"""

try:
    # Create README locally first
    readme_path = Path("/tmp/README.md")
    readme_path.write_text(readme_content)

    # Upload README
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id="zenlm/zen-eco-4b-agent",
        repo_type="model"
    )
    print("✅ README updated for zen-eco-4b-agent")
except Exception as e:
    print(f"❌ README update failed: {e}")

# Update README for zen-nano
print("\n📝 Updating README for zen-nano-0.6b...")
nano_readme = """---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- zen
- hanzo-ai
- qwen3
- nano
- gguf
---

# Zen Nano 0.6B

Smallest model in the Zen family, optimized for edge deployment.

## Available Formats

### PyTorch (Default)
- Standard safetensors format
- 0.6B parameters

### GGUF Formats (llama.cpp)
Multiple quantization levels available:
- `gguf/zen-nano-0.6b-f16.gguf` - F16 format (1.1GB)
- `gguf/zen-nano-0.6b-Q8_0.gguf` - Q8_0 quantized (610MB)
- `gguf/zen-nano-0.6b-Q5_K_M.gguf` - Q5_K_M quantized (424MB)
- `gguf/zen-nano-0.6b-Q4_K_M.gguf` - Q4_K_M quantized (378MB)

## Usage

### PyTorch
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("zenlm/zen-nano-0.6b")
tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-nano-0.6b")
```

### GGUF (llama.cpp)
```bash
# Download desired quantization
wget https://huggingface.co/zenlm/zen-nano-0.6b/resolve/main/gguf/zen-nano-0.6b-Q4_K_M.gguf

# Use with llama.cpp
./llama-cli -m zen-nano-0.6b-Q4_K_M.gguf -p "Hello"
```

## Model Details

- **Base**: Qwen3-0.5B
- **Parameters**: 0.6B
- **Training**: Fine-tuned with Zen identity
- **Developer**: Hanzo AI

## License

Apache 2.0
"""

try:
    # Create README locally first
    readme_path = Path("/tmp/README_nano.md")
    readme_path.write_text(nano_readme)

    # Upload README
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id="zenlm/zen-nano-0.6b",
        repo_type="model"
    )
    print("✅ README updated for zen-nano-0.6b")
except Exception as e:
    print(f"❌ README update failed: {e}")

print("\n" + "="*60)
print("✅ ALL FORMATS UPLOADED TO MAIN REPOSITORIES")
print("="*60)
print("\nModels now have all formats in one place:")
print("- zenlm/zen-eco-4b-agent (PyTorch + MLX + GGUF)")
print("- zenlm/zen-nano-0.6b (PyTorch + GGUF)")
print("\nSeparate repos (zenlm/zen-eco-4b-agent-mlx and zenlm/zen-eco-4b-agent-gguf)")
print("can be deleted or marked as deprecated.")