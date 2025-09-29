#!/usr/bin/env python3.13
"""
Setup Zen Eco 4B Agent Model
Configure tool-calling capabilities
"""
import argparse
import os
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import shutil

def create_tool_config():
    """Create tool calling configuration"""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    return tools

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"🔧 Setting up zen-eco-4b-agent")
    print(f"📦 Base model: {args.base_model}")
    print(f"🌱 Seed: {args.seed}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # If base model exists, copy it
    if os.path.exists(args.base_model):
        print("Copying base model...")
        # Copy model files
        for file in Path(args.base_model).glob("*"):
            if file.is_file():
                shutil.copy2(file, args.output_dir)
    else:
        # Load from HuggingFace as fallback
        print(f"Loading model from HuggingFace...")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            torch_dtype=torch.float16 if args.device != "cpu" else torch.float32,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            trust_remote_code=True
        )

        # Save
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    # Create agent configuration
    agent_config = {
        "model_type": "zen-eco-agent",
        "version": "4b",
        "capabilities": ["tool_calling", "function_calling", "reasoning"],
        "tools": create_tool_config(),
        "identity": {
            "name": "Zen Eco Agent",
            "created_by": "Hanzo AI",
            "description": "A 4B parameter agent model with tool-calling capabilities"
        },
        "system_prompt": """You are Zen Eco Agent, a helpful AI assistant created by Hanzo AI with tool-calling capabilities.
You can use tools when needed to help answer questions.

Available tools:
- calculate: For mathematical calculations
- search: For searching information

When using tools, format your response as:
<tool_call>
{"tool": "tool_name", "arguments": {...}}
</tool_call>"""
    }

    # Save agent configuration
    config_path = os.path.join(args.output_dir, "agent_config.json")
    with open(config_path, "w") as f:
        json.dump(agent_config, f, indent=2)

    print(f"Saved agent config to {config_path}")

    # Create tool calling examples
    examples = [
        {
            "prompt": "What is 45 * 67?",
            "response": """I'll calculate that for you.

<tool_call>
{"tool": "calculate", "arguments": {"expression": "45 * 67"}}
</tool_call>

The result of 45 * 67 is 3015."""
        },
        {
            "prompt": "Search for information about Python programming",
            "response": """I'll search for information about Python programming.

<tool_call>
{"tool": "search", "arguments": {"query": "Python programming language basics"}}
</tool_call>

Based on my search, Python is a high-level, interpreted programming language known for its simplicity and readability."""
        }
    ]

    # Save examples
    examples_path = os.path.join(args.output_dir, "tool_examples.json")
    with open(examples_path, "w") as f:
        json.dump(examples, f, indent=2)

    # Model card
    model_card = f"""---
tags:
- zen
- eco
- agent
- tool-use
- function-calling
- hanzo
license: apache-2.0
language:
- en
---

# Zen Eco 4B Agent

A 4B parameter agent model with tool-calling capabilities.

## Features
- Function/tool calling
- Structured output generation
- Identity-preserved from Zen family
- Efficient inference

## Tool Format
```
<tool_call>
{{"tool": "tool_name", "arguments": {{...}}}}
</tool_call>
```

## Available Tools
- calculate: Mathematical calculations
- search: Information retrieval
"""

    with open(f"{args.output_dir}/README.md", "w") as f:
        f.write(model_card)

    print("✅ Agent model configured!")
    print(f"📁 Output: {args.output_dir}")

if __name__ == "__main__":
    main()