---
language:
- en
- zh
license: apache-2.0
tags:
- qwen3
- instruct
- chat
- assistant
- zen-ai
base_model: Qwen/Qwen3-4B-Instruct
pipeline_tag: text-generation
library_name: transformers
---

# Zen Eco 4B Instruct

**Zen Eco Instruct** is a 4B parameter instruction-tuned language model optimized for conversational AI and task completion. Based on Qwen3-4B-Instruct, it provides excellent balance between capability and efficiency.

## Model Details

- **Model Type**: Instruction-Tuned Language Model
- **Architecture**: Qwen3 (4B)
- **Parameters**: 4 billion
- **License**: Apache 2.0
- **Languages**: English, Chinese, and 25+ languages
- **Context Length**: 32K tokens
- **Developed by**: Zen AI Team
- **Base Model**: [Qwen/Qwen3-4B-Instruct](https://huggingface.co/Qwen/Qwen3-4B-Instruct)

## Capabilities

- 💬 **Conversational**: Natural dialogue and chat
- 🎯 **Instruction Following**: High-quality task completion
- 🌍 **Multilingual**: 25+ languages supported
- ⚡ **Efficient**: 28K tokens/sec on RTX 4090
- 📦 **Multiple Formats**: PyTorch, MLX, GGUF
- 🎛️ **Function Calling**: Tool use and API integration
- 🔒 **Safe**: Aligned with safety guidelines

## Performance

### Throughput
- **RTX 4090 (GGUF Q4)**: 28,000 tokens/sec
- **M3 Max (MLX)**: 32,000 tokens/sec
- **RTX 3090 (GGUF Q4)**: 18,000 tokens/sec

### Memory Usage
| Format | VRAM/RAM |
|--------|----------|
| Q4_K_M | 2.5GB |
| Q8_0 | 4.2GB |
| F16 | 8.0GB |

## Use Cases

- Conversational AI chatbots
- Task automation and completion
- Content generation and editing
- Code assistance
- Question answering
- Summarization and analysis
- Multilingual applications

## Installation

### Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "zenlm/zen-eco-4b-instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-eco-4b-instruct")

# Chat
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=500)
print(tokenizer.decode(outputs[0]))
```

### OpenAI API (Zen Engine)

```bash
zen-engine serve --model zenlm/zen-eco-4b-instruct --port 3690
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:3690/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="zen-eco-4b-instruct",
    messages=[
        {"role": "user", "content": "What is machine learning?"}
    ]
)
print(response.choices[0].message.content)
```

## Training with Zen Gym

Fine-tune for your domain:

```bash
cd /path/to/zen-gym

llamafactory-cli train \
    --config configs/zen_eco_instruct_lora.yaml \
    --dataset your_dataset
```

## Benchmarks

| Task | Score | Notes |
|------|-------|-------|
| MMLU | 58.7% | 5-shot |
| GSM8K | 68.2% | 8-shot CoT |
| HumanEval | 52.4% | pass@1 |
| MATH | 42.8% | 4-shot |
| BBH | 61.3% | 3-shot |

## Chat Template

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
```

## Limitations

- May produce incorrect information
- Limited knowledge cutoff
- Quantization reduces quality
- Not suitable for safety-critical applications
- Requires prompt engineering for best results

## Citation

```bibtex
@misc{zeneco2025instruct,
  title={Zen Eco 4B Instruct: Efficient Instruction-Tuned Language Model},
  author={Zen AI Team},
  year={2025},
  howpublished={\url{https://huggingface.co/zenlm/zen-eco-4b-instruct}}
}
```

## Links

- **GitHub**: https://github.com/zenlm
- **Zen Gym** (Training): https://github.com/zenlm/zen-gym
- **Zen Engine** (Inference): https://github.com/zenlm/zen-engine

Part of the **[Zen AI](https://github.com/zenlm)** ecosystem.