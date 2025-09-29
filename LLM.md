# Zen Eco Models Training System

## Overview
Comprehensive, reproducible training system for all Zen Eco model variants using a simple Makefile-based approach.

## Quick Start
```bash
# Clone and train everything
git clone <repo>
cd zen-eco
make all
```

## Model Variants

### zen-eco-4b-instruct
- Base instruction-following model
- 4B parameters (Qwen3 architecture)
- Zen identity training
- Optimized for general tasks

### zen-eco-4b-thinking
- Chain-of-thought reasoning model
- Uses `<thinking>` tags for transparency
- Step-by-step problem solving
- Same 4B parameter base

### zen-eco-4b-agent
- Tool-calling and function use
- Structured output generation
- Agent capabilities
- Built on instruct model

## Makefile Targets

### Core Training
- `make instruct` - Train instruction model
- `make thinking` - Train thinking model
- `make agent` - Setup agent model
- `make all` - Train everything

### Utilities
- `make gguf` - Generate GGUF quantized models
- `make test` - Test all models
- `make upload` - Upload to HuggingFace
- `make clean` - Clean training artifacts
- `make help` - Show help

## Requirements
- Python 3.13
- 16GB+ RAM
- 50GB+ disk space
- macOS (MPS) or Linux (CUDA)

## Reproducibility Features
1. **Fixed Seeds**: All random operations use seed=42
2. **Disabled Telemetry**: No wandb, no tracking
3. **Deterministic Training**: Same results every run
4. **Simple Dependencies**: Standard libraries only
5. **Clear Outputs**: Color-coded progress

## Directory Structure
```
zen-eco/
├── Makefile              # Main build system
├── scripts/              # Training scripts
│   ├── train_instruct.py
│   ├── train_thinking.py
│   ├── setup_agent.py
│   ├── convert_gguf.py
│   ├── test_models.py
│   └── upload_models.py
├── output/               # Trained models
│   ├── zen-eco-4b-instruct/
│   ├── zen-eco-4b-thinking/
│   └── zen-eco-4b-agent/
└── gguf/                 # Quantized models
```

## Training Process

### 1. Identity Training
All models trained with Zen identity:
- "I am Zen Eco"
- "Created by Hanzo AI"
- Consistent personality

### 2. LoRA Fine-tuning
Efficient training using LoRA:
- r=16, alpha=32
- Target Q,K,V,O projections
- ~0.24% parameters trained

### 3. Minimal Epochs
Quick training for identity:
- 3 epochs default
- Small curated dataset
- Focus on identity preservation

## Testing
```bash
# Test all models
make test

# Quick test
make test-only
```

Tests verify:
- Model loads correctly
- Generates responses
- Contains Zen identity
- Tool calling (agent only)

## Upload to HuggingFace
```bash
# Set token
export HF_TOKEN=your_token_here

# Upload all models
make upload
```

Creates repositories:
- `{username}/zen-eco-4b-instruct`
- `{username}/zen-eco-4b-thinking`
- `{username}/zen-eco-4b-agent`
- `{username}/zen-eco-4b-{variant}-gguf`

## GGUF Quantization
Automatic conversion to GGUF:
- Q4_K_M quantization (default)
- ~2GB file size
- Compatible with llama.cpp
- Optimized for inference

## Customization

### Change Base Model
Edit Makefile:
```makefile
BASE_MODEL := YourModel/Name
```

### Adjust Training
Modify scripts parameters:
- Learning rate
- Batch size
- LoRA rank
- Epochs

### Add New Variant
1. Create script in `scripts/`
2. Add Makefile target
3. Update test suite

## Troubleshooting

### Python 3.13 Required
```bash
brew install python@3.13
```

### MPS Issues (Mac)
- FP16 disabled for MPS
- Use CPU fallback if needed
- Set `DEVICE=cpu` in Makefile

### Memory Issues
- Reduce batch size
- Use gradient accumulation
- Enable CPU offloading

## Architecture Details

### Base Model
- Qwen2.5-3B-Instruct (default)
- Qwen3 architecture
- 3.09B parameters
- 151936 vocab size

### Training Config
- AdamW optimizer
- Cosine scheduler
- Gradient accumulation: 4
- Learning rate: 2e-5

### Dataset
- Identity examples
- Thinking examples
- Tool-use examples
- Repeated for memorization

## License
Apache 2.0

## Credits
Created by Hanzo AI for the Zen model family.