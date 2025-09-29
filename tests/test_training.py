"""Test training functionality"""
import pytest
import torch
from pathlib import Path
import subprocess
import sys

class TestTrainingScripts:
    """Test that training scripts exist and are valid"""

    def test_makefile_exists(self):
        """Check Makefile exists"""
        makefile = Path("/Users/z/work/zen/zen-eco/Makefile")
        assert makefile.exists(), "Makefile not found"

    def test_training_scripts_exist(self):
        """Check training scripts exist"""
        scripts_dir = Path("/Users/z/work/zen/zen-eco/scripts")
        assert scripts_dir.exists(), "Scripts directory not found"

        expected_scripts = [
            "train_instruct.py",
            "train_thinking.py",
            "test_models.py",
            "convert_gguf.py",
        ]

        for script in expected_scripts:
            script_path = scripts_dir / script
            assert script_path.exists(), f"Script {script} not found"

            # Check script is executable
            assert script_path.stat().st_mode & 0o111, f"Script {script} is not executable"

    def test_python_version(self):
        """Check Python version is 3.13"""
        version = sys.version_info
        assert version.major == 3 and version.minor == 13, f"Python 3.13 required, got {version.major}.{version.minor}"

    def test_mps_available(self):
        """Check MPS is available for Mac GPU training"""
        if sys.platform == "darwin":  # macOS
            assert torch.backends.mps.is_available(), "MPS not available for GPU training"
            assert torch.backends.mps.is_built(), "MPS not built in PyTorch"

class TestTrainingData:
    """Test training data and datasets"""

    def test_base_models_exist(self):
        """Check base models are downloaded"""
        base_models = {
            "qwen3-4b": Path("/Users/z/work/zen/zen-eco/base-model"),
            "qwen3-0.6b": Path("/Users/z/work/zen/zen-nano/base-model"),
        }

        for name, path in base_models.items():
            assert path.exists(), f"Base model {name} not found at {path}"

            # Check for model files
            safetensors = list(path.glob("*.safetensors"))
            assert len(safetensors) > 0, f"No safetensors in {name}"

    def test_training_reproducibility(self):
        """Check training uses fixed seed for reproducibility"""
        scripts_dir = Path("/Users/z/work/zen/zen-eco/scripts")

        for script in scripts_dir.glob("train_*.py"):
            if script.exists():
                content = script.read_text()
                assert "seed" in content.lower(), f"{script.name} doesn't set seed"
                assert "42" in content or "seed=42" in content, f"{script.name} doesn't use seed=42"

class TestLlamaCpp:
    """Test llama.cpp integration"""

    def test_llama_cpp_exists(self):
        """Check llama.cpp is available"""
        llama_cpp = Path("/Users/z/work/zen/llama.cpp")
        assert llama_cpp.exists(), "llama.cpp not found"

        # Check binaries exist
        quantize = llama_cpp / "build" / "bin" / "llama-quantize"
        assert quantize.exists(), "llama-quantize binary not found"

        cli = llama_cpp / "build" / "bin" / "llama-cli"
        assert cli.exists(), "llama-cli binary not found"

    def test_convert_script_exists(self):
        """Check conversion script exists"""
        convert = Path("/Users/z/work/zen/llama.cpp/convert_hf_to_gguf.py")
        assert convert.exists(), "convert_hf_to_gguf.py not found"