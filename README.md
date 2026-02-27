# Zen Eco - 4B Efficient Language Model Family

**Organization**: [Zen LM](https://zenlm.org) (Hanzo AI × Zoo Labs Foundation)
**Parameters**: ~4B
**License**: Apache 2.0  
**Context Window**: 32,768 tokens

---

## Model Overview

Zen Eco is a family of 4-billion parameter language models designed for **balanced performance and efficiency**. Built on a 4B foundation architecture, Zen Eco delivers strong capabilities across a wide range of tasks while maintaining cost-effective deployment.

The Zen Eco family includes specialized variants:
- **[zen-eco-4b-instruct](https://huggingface.co/zenlm/zen-eco-4b-instruct)** - General instruction following
- **[zen-eco-4b-thinking](https://huggingface.co/zenlm/zen-eco-4b-thinking)** - Enhanced reasoning with chain-of-thought
- **[zen-agent-4b](https://huggingface.co/zenlm/zen-agent-4b)** - Tool calling and agent capabilities
- **[zen-eco-instruct](https://huggingface.co/zenlm/zen-eco-instruct)** - Alternative instruction variant

---

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("zenlm/zen-eco-4b-instruct")
tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-eco-4b-instruct")

# Generate text
messages = [{"role": "user", "content": "Explain quantum computing in simple terms"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Model Variants

### zen-eco-4b-instruct
**Best for**: General-purpose instruction following and conversational AI

- 4B parameters
- 32K context window
- Multiple formats: SafeTensors, GGUF, MLX
- Optimized for balanced quality and speed

### zen-eco-4b-thinking
**Best for**: Complex reasoning tasks requiring step-by-step analysis

- Extended thinking capabilities with chain-of-thought
- Ideal for mathematical reasoning, logical problems
- 18.4 GB model size (full precision)
- Available in GGUF and MLX formats

### zen-agent-4b
**Best for**: Tool calling and autonomous agent applications

- Function calling support via Jinja2 templates
- Agent-specific training for task planning
- 22.6 GB model size (full precision)
- 184 downloads/month (highest adoption in family)

### zen-eco-instruct
**Best for**: Alternative instruction following approach

- Efficient instruction tuning
- Claims 95% less energy than cloud AI
- 100% local processing (privacy-focused)
- Compact deployment footprint

---

## Available Formats

All Zen Eco models support multiple deployment formats:

| Format | Use Case | Precision | Compatibility |
|--------|----------|-----------|---------------|
| **SafeTensors** | Training/Fine-tuning | BF16 | PyTorch, Transformers |
| **GGUF** | CPU Inference | Q4_K_M, Q5_K_M, Q8_0 | llama.cpp, Ollama |
| **MLX** | Apple Silicon | 4-bit | M1/M2/M3 Macs |

---

## Key Features

✅ **Efficiency**: 95% less energy consumption vs cloud AI  
✅ **Privacy**: 100% local processing, no cloud required  
✅ **Flexibility**: Multiple specialized variants for different use cases  
✅ **Open Source**: Apache 2.0 license, fully open weights and code  
✅ **Multi-Format**: SafeTensors, GGUF, and MLX support  
✅ **Context**: 32,768 token context window  

---

## Use Cases

- **Conversational AI**: Customer support, chatbots, virtual assistants
- **Content Generation**: Writing assistance, summarization, translation
- **Code Understanding**: Code review, documentation, explanation
- **Reasoning Tasks**: Problem-solving, analysis, planning
- **Agent Applications**: Tool use, task automation, workflow orchestration

---

## Hardware Requirements

### Minimum (GGUF Q4_K_M)
- **RAM**: 8 GB
- **Storage**: 4 GB
- **Platform**: CPU-only (llama.cpp)

### Recommended (SafeTensors BF16)
- **VRAM**: 16 GB
- **RAM**: 32 GB
- **GPU**: NVIDIA RTX 3090 or better
- **Platform**: CUDA 11.8+

### Optimal (Full Precision)
- **VRAM**: 24 GB (RTX 4090, A5000, A6000)
- **RAM**: 64 GB
- **GPU**: NVIDIA A100 (40GB) for best performance

---

## Performance

While this base repository doesn't contain model weights (see variants above), the Zen Eco family delivers:

- **Fast Inference**: 10-50 tokens/sec (depending on hardware)
- **Low Latency**: <100ms first token (GGUF Q4)
- **High Throughput**: Batch processing support
- **Energy Efficient**: 95% reduction vs cloud-based inference

---

## Training

All Zen Eco models are trained using:
- **Framework**: [zoo-gym](https://github.com/zenlm/gym)
- **Base**: 4B foundation architecture
- **Method**: Supervised fine-tuning with identity training
- **Data**: Zen agentic dataset + specialized task data

---

## Citation

If you use Zen Eco models in your research or applications, please cite:

```bibtex
@software{zen_eco_2025,
  title={Zen Eco: Efficient 4B Language Model Family},
  author={Zen LM},
  organization={Hanzo AI and Zoo Labs Foundation},
  year={2025},
  url={https://huggingface.co/zenlm/zen-eco}
}
```

---

## Resources

- **Website**: https://zenlm.org
- **Models**: https://huggingface.co/zenlm
- **GitHub**: https://github.com/zenlm
- **Training Framework**: [zoo-gym](https://github.com/zenlm/gym)
- **Documentation**: [Zen Docs](https://zenlm.org/docs)

---

## Model Cards

For detailed specifications and usage examples, see individual model cards:
- [zen-eco-4b-instruct](https://huggingface.co/zenlm/zen-eco-4b-instruct)
- [zen-eco-4b-thinking](https://huggingface.co/zenlm/zen-eco-4b-thinking)
- [zen-agent-4b](https://huggingface.co/zenlm/zen-agent-4b)
- [zen-eco-instruct](https://huggingface.co/zenlm/zen-eco-instruct)

---

## License

All Zen Eco models are released under the **Apache 2.0 License**.

This means you can:
✅ Use commercially  
✅ Modify and redistribute  
✅ Private use  
✅ Patent use  

**Free forever. No restrictions.**

---

*Zen AI: Clarity Through Intelligence*
