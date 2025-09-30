#!/usr/bin/env python3
"""
Example Usage of Qwen3 Tool Calling for Zen-Eco Agent
Demonstrates integration with Salesforce/xlam-function-calling-60k trained model
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Import our modules
from qwen3_tool_caller import Qwen3ToolCaller
from model_loader import GGUFModelLoader
from training_config import create_default_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example tools matching the xlam dataset patterns
EXAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or coordinates"
                    },
                    "units": {
                        "type": "string",
                        "description": "Temperature units (celsius/fahrenheit)",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
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
            "name": "send_email",
            "description": "Send an email message",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address"
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject"
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body content"
                    }
                },
                "required": ["to", "subject", "body"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create a calendar event",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Event title"
                    },
                    "date": {
                        "type": "string",
                        "description": "Event date (YYYY-MM-DD)"
                    },
                    "time": {
                        "type": "string",
                        "description": "Event time (HH:MM)"
                    },
                    "duration": {
                        "type": "integer",
                        "description": "Duration in minutes"
                    }
                },
                "required": ["title", "date", "time"]
            }
        }
    }
]

def demo_basic_tool_calling():
    """Demonstrate basic tool calling"""
    print("\n" + "="*60)
    print("BASIC TOOL CALLING DEMO")
    print("Model: Qwen3-4B fine-tuned on Salesforce/xlam-function-calling-60k")
    print("="*60)

    # Initialize the tool caller
    tool_caller = Qwen3ToolCaller()

    # Test queries
    test_queries = [
        "What's the weather like in San Francisco?",
        "Calculate the square root of 144",
        "Search for the latest news about AI",
        "Send an email to john@example.com about tomorrow's meeting",
        "Create a calendar event for project review next Monday at 2 PM"
    ]

    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 40)

        result = tool_caller.chat(query, tools=EXAMPLE_TOOLS)

        print(f"Response: {result['response']}")

        if result['tool_calls']:
            print(f"\n🔧 Tool Calls Detected:")
            for tool_call in result['tool_calls']:
                print(f"  Function: {tool_call['name']}")
                print(f"  Arguments: {json.dumps(tool_call['arguments'], indent=4)}")
        else:
            print("\n❌ No tool calls detected")

def demo_model_loader():
    """Demonstrate model loading and management"""
    print("\n" + "="*60)
    print("MODEL LOADER DEMO")
    print("="*60)

    loader = GGUFModelLoader()

    # Show dataset info
    print("\n📊 Training Dataset Information:")
    dataset_info = loader.get_dataset_info()
    print(f"  Name: {dataset_info['name']}")
    print(f"  Provider: {dataset_info['provider']}")
    print(f"  HuggingFace ID: {dataset_info['hf_dataset_id']}")
    print(f"  Size: {dataset_info.get('size', 60000)} examples")
    print(f"  URL: {dataset_info['url']}")

    # Show model info
    print("\n🤖 Model Information:")
    model_info = loader.get_model_info()
    print(f"  Base Model: {model_info['base_model']}")
    print(f"  Fine-tuned Model: {model_info['fine_tuned_model']}")
    print(f"  Parameters: {model_info['parameters']}")
    print(f"  Architecture: {model_info['architecture']}")
    print(f"  Quantization: {model_info['quantization']['format']} ({model_info['quantization']['method']})")

    # List available models
    print("\n📦 Available Models:")
    available = loader.list_available_models()
    print(f"  Recommended: {available['recommended']}")
    print(f"  Cached: {len(available['cached'])} models")
    print(f"  Online: {len(available['online'])} models")

def demo_multi_tool_calling():
    """Demonstrate multiple tool calls in a single query"""
    print("\n" + "="*60)
    print("MULTI-TOOL CALLING DEMO")
    print("="*60)

    tool_caller = Qwen3ToolCaller()

    # Complex queries that might need multiple tools
    complex_queries = [
        "Check the weather in Tokyo and calculate how many degrees Fahrenheit is 25 Celsius",
        "Search for Python tutorials and create a calendar event for learning session tomorrow at 3 PM",
        "Send an email about the weather forecast and schedule a meeting to discuss it"
    ]

    for query in complex_queries:
        print(f"\n📝 Complex Query: {query}")
        print("-" * 40)

        result = tool_caller.chat(query, tools=EXAMPLE_TOOLS)

        print(f"Response: {result['response'][:200]}...")  # Truncate long responses

        if result['tool_calls']:
            print(f"\n🔧 Multiple Tool Calls ({len(result['tool_calls'])}):")
            for i, tool_call in enumerate(result['tool_calls'], 1):
                print(f"\n  Call #{i}:")
                print(f"    Function: {tool_call['name']}")
                print(f"    Arguments: {json.dumps(tool_call['arguments'], indent=6)}")
        else:
            print("\n❌ No tool calls detected")

def demo_training_config():
    """Show training configuration used for the model"""
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)

    config = create_default_config()

    print("\n📚 Dataset Configuration:")
    print(f"  Dataset: {config.dataset.hf_dataset_id}")
    print(f"  Total Samples: {config.dataset.num_samples}")
    print(f"  Unique Functions: {config.dataset.num_unique_functions}")
    print(f"  Train/Val/Test Split: {config.dataset.train_ratio}/{config.dataset.val_ratio}/{config.dataset.test_ratio}")

    print("\n🔧 LoRA Configuration:")
    print(f"  Enabled: {config.lora.enable_lora}")
    print(f"  Rank: {config.lora.rank}")
    print(f"  Alpha: {config.lora.alpha}")
    print(f"  Dropout: {config.lora.dropout}")

    print("\n⚙️ Training Hyperparameters:")
    print(f"  Batch Size: {config.training.batch_size}")
    print(f"  Learning Rate: {config.training.learning_rate}")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Optimizer: {config.training.optimizer}")

    print("\n📝 Prompt Template:")
    print("  System prompt includes tool descriptions")
    print("  User query formatted with special tokens")
    print("  Assistant response with JSON tool calls")

def main():
    """Run all demos"""
    print("\n" + "🚀" * 30)
    print("ZEN-ECO AGENT TOOL CALLING EXAMPLES")
    print("Powered by Qwen3-4B + Salesforce xlam-function-calling-60k")
    print("🚀" * 30)

    # Run demos
    demo_training_config()
    demo_model_loader()
    demo_basic_tool_calling()
    demo_multi_tool_calling()

    print("\n" + "="*60)
    print("✅ ALL DEMOS COMPLETED")
    print("="*60)
    print("\nTo integrate this into your zen-eco agent:")
    print("1. Copy the tool_calling directory to your agent")
    print("2. Download the model using GGUFModelLoader")
    print("3. Initialize Qwen3ToolCaller with your tools")
    print("4. Use the chat() method for tool-enabled conversations")
    print("\nDataset: https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k")
    print("Model: https://huggingface.co/Manojb/Qwen3-4b-toolcall-gguf-llamacpp-codex")

if __name__ == "__main__":
    main()