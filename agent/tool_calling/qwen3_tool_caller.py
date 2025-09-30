#!/usr/bin/env python3
"""
Qwen3-4B Tool Calling Integration for Zen-Eco Agent
Based on Manojb/Qwen3-4b-toolcall-gguf-llamacpp-codex
Fine-tuned on Salesforce/xlam-function-calling-60k dataset
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from llama_cpp import Llama

logger = logging.getLogger(__name__)

class Qwen3ToolCaller:
    """
    Zen-Eco Agent Tool Calling with Qwen3-4B
    Model: Manojb/Qwen3-4b-toolcall-gguf-llamacpp-codex
    Dataset: Salesforce/xlam-function-calling-60k
    """

    MODEL_INFO = {
        "name": "Qwen3-4B-Function-Calling-Pro",
        "base_model": "Qwen/Qwen3-4B-Instruct",
        "dataset": "Salesforce/xlam-function-calling-60k",
        "dataset_url": "https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k",
        "model_url": "https://huggingface.co/Manojb/Qwen3-4b-toolcall-gguf-llamacpp-codex",
        "quantization": "Q8_0",
        "parameters": "4B",
        "context_length": 262144,
        "architecture": "Qwen3"
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 8192,
        n_threads: int = 8,
        n_batch: int = 512,
        n_gpu_layers: int = -1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        verbose: bool = False
    ):
        """
        Initialize Qwen3 Tool Caller

        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size
            n_threads: Number of CPU threads
            n_batch: Batch size for prompt processing
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            verbose: Enable verbose logging
        """
        if model_path is None:
            model_path = str(Path.home() / "work/zen/zen-eco/agent/gguf/Qwen3-4B-Function-Calling-Pro.gguf")

        self.model_path = model_path
        self.temperature = temperature
        self.top_p = top_p

        logger.info(f"Loading Qwen3-4B Tool Calling model from {model_path}")
        logger.info(f"Model trained on: {self.MODEL_INFO['dataset']}")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=n_batch,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
            seed=-1
        )

    def format_tools(self, tools: List[Dict[str, Any]]) -> str:
        """
        Format tools for the model prompt

        Args:
            tools: List of tool definitions

        Returns:
            Formatted tool string
        """
        tool_descriptions = []
        for tool in tools:
            if "function" in tool:
                func = tool["function"]
                tool_desc = f"- {func['name']}: {func['description']}"
                if "parameters" in func:
                    params = func["parameters"]
                    if "properties" in params:
                        param_list = []
                        for prop, details in params["properties"].items():
                            required = prop in params.get("required", [])
                            param_list.append(
                                f"  - {prop} ({details.get('type', 'string')})"
                                f"{' [required]' if required else ''}: "
                                f"{details.get('description', '')}"
                            )
                        if param_list:
                            tool_desc += "\n" + "\n".join(param_list)
                tool_descriptions.append(tool_desc)

        return "\n".join(tool_descriptions)

    def extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from model response

        Args:
            text: Model response text

        Returns:
            List of extracted tool calls
        """
        tool_calls = []

        # Pattern 1: JSON array format
        json_array_pattern = r'\[(?:[^[\]]*|\[.*?\])*\]'
        matches = re.findall(json_array_pattern, text, re.DOTALL)

        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and 'name' in item:
                            tool_call = {
                                'name': item['name'],
                                'arguments': item.get('arguments', item.get('parameters', {}))
                            }
                            tool_calls.append(tool_call)
            except json.JSONDecodeError:
                pass

        # Pattern 2: Function call format
        func_pattern = r'<function_call>\s*({[^}]+})\s*</function_call>'
        func_matches = re.findall(func_pattern, text, re.DOTALL)

        for match in func_matches:
            try:
                parsed = json.loads(match)
                if 'name' in parsed:
                    tool_calls.append(parsed)
            except json.JSONDecodeError:
                pass

        # Pattern 3: Direct function notation
        direct_pattern = r'(\w+)\((.*?)\)'
        if not tool_calls:
            direct_matches = re.findall(direct_pattern, text)
            for func_name, args_str in direct_matches:
                if func_name in ['get_weather', 'search_hotels', 'calculate', 'search', 'execute']:
                    try:
                        # Try to parse arguments
                        if args_str:
                            args = {}
                            for arg in args_str.split(','):
                                if '=' in arg:
                                    key, val = arg.split('=', 1)
                                    args[key.strip()] = val.strip().strip('"\'')
                            tool_calls.append({'name': func_name, 'arguments': args})
                    except:
                        pass

        return tool_calls

    def generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate response with tool calling

        Args:
            messages: Chat messages
            tools: Available tools
            max_tokens: Maximum tokens to generate
            temperature: Override default temperature
            top_p: Override default top_p

        Returns:
            Tuple of (response text, tool calls)
        """
        # Build prompt
        prompt_parts = []

        # Add system message with tool descriptions
        if tools:
            tool_desc = self.format_tools(tools)
            system_msg = (
                "You are a helpful AI assistant with access to the following tools:\n\n"
                f"{tool_desc}\n\n"
                "When you need to use a tool, format your response as:\n"
                "[{\"name\": \"tool_name\", \"arguments\": {\"param1\": \"value1\"}}]\n"
                "You can call multiple tools in a single response."
            )
            prompt_parts.append(f"<|im_start|>system\n{system_msg}<|im_end|>")

        # Add message history
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        # Add assistant prompt
        prompt_parts.append("<|im_start|>assistant\n")

        prompt = "\n".join(prompt_parts)

        # Generate response
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature or self.temperature,
            top_p=top_p or self.top_p,
            stop=["<|im_end|>", "<|im_start|>"],
            echo=False
        )

        response_text = response['choices'][0]['text']
        tool_calls = self.extract_tool_calls(response_text)

        return response_text, tool_calls

    def chat(
        self,
        message: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Simple chat interface

        Args:
            message: User message
            tools: Available tools
            system_message: Optional system message

        Returns:
            Response dictionary with text and tool calls
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": message})

        response_text, tool_calls = self.generate(messages, tools)

        return {
            'response': response_text,
            'tool_calls': tool_calls,
            'model_info': self.MODEL_INFO
        }

    @classmethod
    def from_pretrained(cls, model_id: str = "Manojb/Qwen3-4b-toolcall-gguf-llamacpp-codex", **kwargs):
        """
        Load model from HuggingFace hub

        Args:
            model_id: HuggingFace model ID
            **kwargs: Additional arguments for initialization

        Returns:
            Initialized Qwen3ToolCaller instance
        """
        from huggingface_hub import hf_hub_download

        # Download the model
        model_path = hf_hub_download(
            repo_id=model_id,
            filename="Qwen3-4B-Function-Calling-Pro.gguf",
            local_dir=Path.home() / "work/zen/zen-eco/agent/gguf"
        )

        return cls(model_path=model_path, **kwargs)