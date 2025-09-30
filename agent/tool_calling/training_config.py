#!/usr/bin/env python3
"""
Training Configuration for Qwen3 Tool Calling Fine-tuning
Uses Salesforce/xlam-function-calling-60k dataset
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path

@dataclass
class DatasetConfig:
    """Configuration for the xlam-function-calling-60k dataset"""

    # Dataset identifiers
    name: str = "xlam-function-calling-60k"
    provider: str = "Salesforce"
    hf_dataset_id: str = "Salesforce/xlam-function-calling-60k"
    dataset_url: str = "https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k"

    # Dataset properties
    num_samples: int = 60000
    num_unique_functions: int = 5470
    languages: List[str] = None

    # Data splits
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    train_ratio: float = 0.9
    val_ratio: float = 0.05
    test_ratio: float = 0.05

    # Processing parameters
    max_length: int = 4096
    min_length: int = 32
    shuffle: bool = True
    seed: int = 42

    def __post_init__(self):
        if self.languages is None:
            self.languages = ["en"]

@dataclass
class ModelConfig:
    """Configuration for Qwen3-4B model"""

    # Base model
    base_model: str = "Qwen/Qwen3-4B-Instruct"
    architecture: str = "Qwen3"
    parameters: str = "4B"
    vocab_size: int = 151936
    hidden_size: int = 2560
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    intermediate_size: int = 13824
    max_position_embeddings: int = 262144

    # Fine-tuned model
    fine_tuned_model: str = "Manojb/Qwen3-4b-toolcall-gguf-llamacpp-codex"
    fine_tuned_on: str = "Salesforce/xlam-function-calling-60k"

    # Model capabilities
    supports_tools: bool = True
    supports_parallel_calls: bool = True
    supports_json_mode: bool = True

@dataclass
class LoRAConfig:
    """LoRA configuration for fine-tuning"""

    enable_lora: bool = True
    rank: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: List[str] = None
    modules_to_save: List[str] = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate", "up_proj", "down_proj"]
        if self.modules_to_save is None:
            self.modules_to_save = ["embed_tokens", "lm_head"]

@dataclass
class TrainingConfig:
    """Training configuration for fine-tuning Qwen3 on xlam dataset"""

    # Training hyperparameters
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Optimizer settings
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Scheduler
    scheduler: str = "cosine"
    scheduler_warmup_steps: int = 500

    # Training strategy
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    deepspeed: Optional[str] = None

    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"

    # Output
    output_dir: str = "./checkpoints/qwen3-tool-calling"
    logging_dir: str = "./logs/qwen3-tool-calling"

@dataclass
class PromptConfig:
    """Prompt template configuration"""

    system_template: str = """You are a helpful AI assistant with access to the following tools:

{tools_description}

When you need to use a tool, format your response as:
[{"name": "tool_name", "arguments": {"param1": "value1"}}]

You can call multiple tools in a single response if needed."""

    user_template: str = "{query}"

    assistant_template: str = "{response}"

    # Special tokens
    im_start: str = "<|im_start|>"
    im_end: str = "<|im_end|>"

    def format_prompt(self, query: str, tools: List[Dict], response: Optional[str] = None) -> str:
        """Format a complete prompt"""
        tools_desc = self._format_tools(tools)

        prompt = f"{self.im_start}system\n"
        prompt += self.system_template.format(tools_description=tools_desc)
        prompt += f"\n{self.im_end}\n"

        prompt += f"{self.im_start}user\n"
        prompt += self.user_template.format(query=query)
        prompt += f"\n{self.im_end}\n"

        prompt += f"{self.im_start}assistant\n"
        if response:
            prompt += self.assistant_template.format(response=response)
            prompt += f"\n{self.im_end}"

        return prompt

    def _format_tools(self, tools: List[Dict]) -> str:
        """Format tools description"""
        descriptions = []
        for tool in tools:
            if "function" in tool:
                func = tool["function"]
                desc = f"- {func['name']}: {func['description']}"
                descriptions.append(desc)
        return "\n".join(descriptions)

class FullTrainingConfig:
    """Complete training configuration manager"""

    def __init__(self):
        self.dataset = DatasetConfig()
        self.model = ModelConfig()
        self.lora = LoRAConfig()
        self.training = TrainingConfig()
        self.prompt = PromptConfig()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "dataset": asdict(self.dataset),
            "model": asdict(self.model),
            "lora": asdict(self.lora),
            "training": asdict(self.training),
            "prompt": {
                "system_template": self.prompt.system_template,
                "user_template": self.prompt.user_template,
                "assistant_template": self.prompt.assistant_template,
                "im_start": self.prompt.im_start,
                "im_end": self.prompt.im_end
            }
        }

    def save(self, path: str):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'FullTrainingConfig':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)

        config = cls()

        # Load dataset config
        for key, value in data.get("dataset", {}).items():
            if hasattr(config.dataset, key):
                setattr(config.dataset, key, value)

        # Load model config
        for key, value in data.get("model", {}).items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)

        # Load LoRA config
        for key, value in data.get("lora", {}).items():
            if hasattr(config.lora, key):
                setattr(config.lora, key, value)

        # Load training config
        for key, value in data.get("training", {}).items():
            if hasattr(config.training, key):
                setattr(config.training, key, value)

        # Load prompt config
        for key, value in data.get("prompt", {}).items():
            if hasattr(config.prompt, key):
                setattr(config.prompt, key, value)

        return config

    def validate(self) -> List[str]:
        """Validate configuration"""
        issues = []

        # Check dataset
        if self.dataset.train_ratio + self.dataset.val_ratio + self.dataset.test_ratio != 1.0:
            issues.append("Dataset splits don't sum to 1.0")

        # Check LoRA
        if self.lora.enable_lora and self.lora.rank > 256:
            issues.append("LoRA rank might be too high (>256)")

        # Check training
        if self.training.batch_size * self.training.gradient_accumulation_steps > 128:
            issues.append("Effective batch size might be too large")

        if self.training.learning_rate > 1e-3:
            issues.append("Learning rate might be too high")

        return issues

def create_default_config() -> FullTrainingConfig:
    """Create default training configuration"""
    return FullTrainingConfig()

def main():
    """Test configuration"""
    config = create_default_config()

    print("Training Configuration for Qwen3 Tool Calling")
    print("=" * 50)
    print(f"Dataset: {config.dataset.hf_dataset_id}")
    print(f"Dataset size: {config.dataset.num_samples} examples")
    print(f"Base model: {config.model.base_model}")
    print(f"Fine-tuned model: {config.model.fine_tuned_model}")
    print(f"\nLoRA Config:")
    print(f"  Rank: {config.lora.rank}")
    print(f"  Alpha: {config.lora.alpha}")
    print(f"  Dropout: {config.lora.dropout}")
    print(f"\nTraining Config:")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Epochs: {config.training.num_epochs}")

    # Save config
    config_path = Path(__file__).parent / "training_config.json"
    config.save(str(config_path))
    print(f"\nConfiguration saved to: {config_path}")

    # Validate
    issues = config.validate()
    if issues:
        print("\nValidation issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nConfiguration validated successfully!")

if __name__ == "__main__":
    main()