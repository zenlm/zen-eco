"""pytest configuration and fixtures"""
import pytest
import torch
import os
from pathlib import Path

# Base paths
BASE_DIR = Path("/Users/z/work/zen")
ZEN_ECO_DIR = BASE_DIR / "zen-eco"
ZEN_NANO_DIR = BASE_DIR / "zen-nano"

@pytest.fixture
def device():
    """Get the best available device"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

@pytest.fixture
def model_paths():
    """Dictionary of model paths"""
    return {
        "zen-nano": ZEN_NANO_DIR / "finetuned",
        "zen-eco-instruct": ZEN_ECO_DIR / "instruct" / "finetuned",
        "zen-eco-thinking": ZEN_ECO_DIR / "thinking" / "finetuned",
        "zen-eco-agent": ZEN_ECO_DIR / "agent" / "finetuned",
        "base-qwen3": ZEN_ECO_DIR / "base-model",
    }

@pytest.fixture
def gguf_paths():
    """Dictionary of GGUF paths"""
    return {
        "zen-nano": ZEN_NANO_DIR / "gguf",
        "zen-eco-instruct": ZEN_ECO_DIR / "instruct" / "gguf",
        "zen-eco-thinking": ZEN_ECO_DIR / "thinking" / "gguf",
        "zen-eco-agent": ZEN_ECO_DIR / "agent" / "gguf",
    }