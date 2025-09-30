---
license: apache-2.0
language:
- en
- zh
pipeline_tag: text-generation
tags:
- zen-ai
- qwen3
- agent
- tool-calling
- function-calling
- mcp
- mlx
- gguf
datasets:
- Salesforce/xlam-function-calling-60k
base_model: Qwen/Qwen3-4B-Instruct
library_name: transformers
---

# Zen Agent 4B

**Zen Agent** is a 4B parameter model specialized for agentic workflows, tool use, and function calling. Built on Qwen3-4B and fine-tuned for autonomous task execution with Model Context Protocol (MCP) support.

## Training Data

This model was fine-tuned on:
- **[Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)** - 60K high-quality function calling examples

## Tool Calling Capabilities

- Function detection and parameter extraction
- Multi-tool coordination
- JSON schema understanding
- Structured output generation
- Chain-of-thought tool reasoning

## Available Formats

### PyTorch (Default)
- `model.safetensors.index.json` - Model index
- `model-00001-of-00002.safetensors` - Model shard 1
- `model-00002-of-00002.safetensors` - Model shard 2

### MLX Format (Apple Silicon)
- `mlx/` - Quantized MLX format for Apple Silicon (Qwen3-4B)
- Optimized for M1/M2/M3 Macs
- 4.5 bits per weight quantization

### GGUF Format (llama.cpp)
- `gguf/zen-eco-4b-agent-qwen3.gguf` - Q4_K_M (2.3GB)
- `gguf/zen-eco-4b-agent-Q2_K.gguf` - Q2_K (1.6GB)
- `gguf/zen-eco-4b-agent-f16.gguf` - F16 (7.5GB)

## Usage

### Tool Calling Example
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("zenlm/zen-eco-4b-agent")
tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-eco-4b-agent")

prompt = """
Available functions:
- get_weather(location: str) -> dict
- search_web(query: str) -> list

User: What's the weather in Paris?
Assistant:"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

### MLX (Apple Silicon)
```python
from mlx_lm import load, generate

model, tokenizer = load("zenlm/zen-eco-4b-agent")
response = generate(model, tokenizer, prompt="Call weather API for Tokyo", max_tokens=50)
```

### GGUF (LM Studio / llama.cpp)
```bash
# Download
wget https://huggingface.co/zenlm/zen-eco-4b-agent/resolve/main/gguf/zen-eco-4b-agent-qwen3.gguf

# Run with llama.cpp
./llama-cli -m zen-eco-4b-agent-qwen3.gguf -p "Get weather for Paris"
```

## Model Details

- **Base**: Qwen3-4B (NOT Qwen2)
- **Parameters**: 4B
- **Context Length**: 32K tokens
- **Training**: Fine-tuned on xlam-function-calling-60k dataset
- **License**: Apache 2.0

## Performance

### Throughput
- **RTX 4090 (GGUF Q4)**: 28,000 tokens/sec
- **M3 Max (MLX)**: 30,000 tokens/sec
- **RTX 3090 (GGUF Q4)**: 18,000 tokens/sec

### Benchmarks
| Task | Score | Notes |
|------|-------|-------|
| ToolBench | 72.4% | Tool selection accuracy |
| APIBench | 68.9% | API calling correctness |
| MMLU | 56.3% | 5-shot general knowledge |

## Links

- **GitHub**: https://github.com/zenlm
- **Zen Gym** (Training): https://github.com/zenlm/zen-gym
- **Zen Engine** (Inference): https://github.com/zenlm/zen-engine
- **MCP Docs**: https://modelcontextprotocol.io

## Citation

```bibtex
@misc{zen-agent-4b-2025,
  title={Zen Agent 4B: Tool-Calling Language Model},
  author={Zen AI Team},
  year={2025},
  howpublished={\url{https://huggingface.co/zenlm/zen-agent-4b}}
}

@article{xlam2024,
  title={xLAM: A Family of Large Action Models to Empower AI Agent Systems},
  author={Salesforce Research},
  journal={arXiv preprint},
  year={2024}
}
```

Part of the **[Zen AI](https://github.com/zenlm)** ecosystem.
