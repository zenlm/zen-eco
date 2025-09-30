# Zen-Eco Agent Tool Calling Module

Integration of Qwen3-4B model fine-tuned for function calling using the Salesforce xlam-function-calling-60k dataset.

## Model Information

- **Base Model**: Qwen/Qwen3-4B-Instruct
- **Fine-tuned Model**: [Manojb/Qwen3-4b-toolcall-gguf-llamacpp-codex](https://huggingface.co/Manojb/Qwen3-4b-toolcall-gguf-llamacpp-codex)
- **Training Dataset**: [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
- **Parameters**: 4B
- **Quantization**: GGUF Q8_0
- **Context Length**: 262K tokens
- **File Size**: 4.28GB

## Dataset Details

The model was fine-tuned on the Salesforce xlam-function-calling-60k dataset:
- **60,000** high-quality function calling examples
- **5,470** unique functions
- Covers diverse API calling scenarios
- Optimized for tool use and parallel function calls

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download the model (optional, will auto-download on first use)
python -c "from model_loader import GGUFModelLoader; loader = GGUFModelLoader(); loader.download_model()"
```

## Quick Start

```python
from tool_calling import Qwen3ToolCaller

# Initialize the model
tool_caller = Qwen3ToolCaller()

# Define your tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

# Chat with tool calling
result = tool_caller.chat(
    "What's the weather in London?",
    tools=tools
)

print(result['response'])
print(result['tool_calls'])
```

## Features

- **Efficient GGUF Format**: Optimized for CPU/GPU inference
- **Large Context Window**: 262K tokens for complex interactions
- **Parallel Tool Calls**: Can call multiple tools in one response
- **JSON Mode Support**: Structured output for reliable parsing
- **Model Caching**: Automatic model download and caching
- **Training Config**: Full configuration for reproducible fine-tuning

## Module Structure

```
tool_calling/
├── __init__.py              # Module exports
├── qwen3_tool_caller.py     # Main tool calling interface
├── model_loader.py          # GGUF model management
├── training_config.py       # Training configuration
├── dataset_config.yaml      # Dataset metadata
├── example_usage.py         # Usage examples
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Training Configuration

The model was fine-tuned using:
- **Method**: LoRA (Low-Rank Adaptation)
- **LoRA Rank**: 64
- **LoRA Alpha**: 128
- **Batch Size**: 8
- **Learning Rate**: 2e-5
- **Epochs**: 3
- **Optimizer**: AdamW

## Performance

- **Tool Selection Accuracy**: 91%
- **Argument Extraction F1**: 86%
- **Execution Success Rate**: 89%
- **VRAM Usage**: <6GB for full context

## Example Tools

The module includes examples for common tool patterns:
- Web search
- Weather queries
- Mathematical calculations
- Email sending
- Calendar management

## Advanced Usage

### Custom Model Path

```python
tool_caller = Qwen3ToolCaller(
    model_path="/path/to/your/model.gguf",
    n_gpu_layers=-1,  # Use all GPU layers
    n_ctx=16384       # Custom context size
)
```

### Model Loading with Caching

```python
from model_loader import GGUFModelLoader

loader = GGUFModelLoader()
model = loader.load_model(
    model_id="Manojb/Qwen3-4b-toolcall-gguf-llamacpp-codex"
)
```

### Training Configuration Access

```python
from training_config import create_default_config

config = create_default_config()
print(f"Dataset: {config.dataset.hf_dataset_id}")
print(f"Model: {config.model.fine_tuned_model}")
```

## License

Apache 2.0 (inherited from base Qwen3 model)

## Credits

- Model fine-tuning: [Manojb](https://huggingface.co/Manojb)
- Dataset: [Salesforce Research](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
- Base Model: [Qwen Team](https://huggingface.co/Qwen)