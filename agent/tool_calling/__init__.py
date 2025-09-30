"""
Zen-Eco Agent Tool Calling Module
Based on Qwen3-4B fine-tuned on Salesforce/xlam-function-calling-60k dataset
"""

from .qwen3_tool_caller import Qwen3ToolCaller
from .model_loader import GGUFModelLoader
from .training_config import (
    DatasetConfig,
    ModelConfig,
    LoRAConfig,
    TrainingConfig,
    PromptConfig,
    FullTrainingConfig,
    create_default_config
)

__version__ = "1.0.0"

__all__ = [
    "Qwen3ToolCaller",
    "GGUFModelLoader",
    "DatasetConfig",
    "ModelConfig",
    "LoRAConfig",
    "TrainingConfig",
    "PromptConfig",
    "FullTrainingConfig",
    "create_default_config"
]

# Model and dataset information
MODEL_INFO = {
    "base_model": "Qwen/Qwen3-4B-Instruct",
    "fine_tuned_model": "Manojb/Qwen3-4b-toolcall-gguf-llamacpp-codex",
    "dataset": "Salesforce/xlam-function-calling-60k",
    "dataset_url": "https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k",
    "model_url": "https://huggingface.co/Manojb/Qwen3-4b-toolcall-gguf-llamacpp-codex",
    "parameters": "4B",
    "architecture": "Qwen3",
    "quantization": "GGUF Q8_0",
    "context_length": 262144
}