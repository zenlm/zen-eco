#!/usr/bin/env python3
"""
GGUF Model Loader for Zen-Eco Agent
Handles loading and managing Qwen3-4B tool calling models
"""

import os
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Union
import hashlib
import logging
from huggingface_hub import hf_hub_download, snapshot_download
from llama_cpp import Llama

logger = logging.getLogger(__name__)

class GGUFModelLoader:
    """
    GGUF Model Loader for Qwen3 Tool Calling
    Manages model downloading, caching, and loading
    """

    DEFAULT_CACHE_DIR = Path.home() / ".cache" / "zen-eco" / "models"
    CONFIG_FILE = "dataset_config.yaml"

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """
        Initialize GGUF Model Loader

        Args:
            cache_dir: Directory for model cache
        """
        self.cache_dir = Path(cache_dir) if cache_dir else self.DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models: Dict[str, Llama] = {}
        self.load_config()

    def load_config(self):
        """Load dataset and model configuration"""
        config_path = Path(__file__).parent / self.CONFIG_FILE
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
        else:
            self.config = self._default_config()
            logger.warning(f"Config file not found at {config_path}, using defaults")

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for models"""
        return {
            "dataset": {
                "name": "xlam-function-calling-60k",
                "provider": "Salesforce",
                "hf_dataset_id": "Salesforce/xlam-function-calling-60k",
                "url": "https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k"
            },
            "model": {
                "base_model": "Qwen/Qwen3-4B-Instruct",
                "fine_tuned_model": "Manojb/Qwen3-4b-toolcall-gguf-llamacpp-codex",
                "parameters": "4B",
                "architecture": "Qwen3",
                "quantization": {
                    "format": "GGUF",
                    "method": "Q8_0"
                }
            }
        }

    def get_model_path(self, model_id: str, filename: str = "Qwen3-4B-Function-Calling-Pro.gguf") -> Path:
        """
        Get local path for model file

        Args:
            model_id: HuggingFace model ID
            filename: Model filename

        Returns:
            Path to model file
        """
        model_dir = self.cache_dir / model_id.replace("/", "_")
        return model_dir / filename

    def download_model(
        self,
        model_id: str = "Manojb/Qwen3-4b-toolcall-gguf-llamacpp-codex",
        filename: str = "Qwen3-4B-Function-Calling-Pro.gguf",
        force_download: bool = False
    ) -> Path:
        """
        Download GGUF model from HuggingFace

        Args:
            model_id: HuggingFace model ID
            filename: Model filename
            force_download: Force re-download even if cached

        Returns:
            Path to downloaded model
        """
        model_path = self.get_model_path(model_id, filename)

        if model_path.exists() and not force_download:
            logger.info(f"Model already cached at {model_path}")
            return model_path

        logger.info(f"Downloading model {model_id}/{filename}")
        logger.info(f"This model was trained on: {self.config['dataset']['hf_dataset_id']}")

        try:
            downloaded_path = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                local_dir=model_path.parent,
                local_dir_use_symlinks=False
            )
            logger.info(f"Model downloaded to {downloaded_path}")
            return Path(downloaded_path)
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise

    def load_model(
        self,
        model_path: Optional[Union[str, Path]] = None,
        model_id: Optional[str] = None,
        n_ctx: int = 8192,
        n_threads: int = 8,
        n_batch: int = 512,
        n_gpu_layers: int = -1,
        verbose: bool = False,
        **kwargs
    ) -> Llama:
        """
        Load GGUF model into memory

        Args:
            model_path: Direct path to model file
            model_id: HuggingFace model ID (will download if needed)
            n_ctx: Context window size
            n_threads: Number of CPU threads
            n_batch: Batch size
            n_gpu_layers: GPU layers (-1 for all)
            verbose: Verbose output
            **kwargs: Additional llama.cpp parameters

        Returns:
            Loaded Llama model instance
        """
        # Determine model path
        if model_path:
            model_path = Path(model_path)
        elif model_id:
            model_path = self.download_model(model_id)
        else:
            # Use default model
            model_id = self.config["model"]["fine_tuned_model"]
            model_path = self.download_model(model_id)

        # Check if already loaded
        model_key = str(model_path)
        if model_key in self.loaded_models:
            logger.info(f"Using cached model instance for {model_path}")
            return self.loaded_models[model_key]

        # Load model
        logger.info(f"Loading GGUF model from {model_path}")
        logger.info(f"Model info: {self.config['model']}")
        logger.info(f"Dataset: {self.config['dataset']['name']} ({self.config['dataset']['size']} examples)")

        try:
            model = Llama(
                model_path=str(model_path),
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_batch=n_batch,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
                seed=-1,
                **kwargs
            )

            # Cache the model
            self.loaded_models[model_key] = model
            logger.info("Model loaded successfully")

            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def verify_model(self, model_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Verify model file integrity and metadata

        Args:
            model_path: Path to model file

        Returns:
            Verification results
        """
        model_path = Path(model_path)

        if not model_path.exists():
            return {"valid": False, "error": "Model file not found"}

        results = {
            "valid": True,
            "path": str(model_path),
            "size_gb": model_path.stat().st_size / (1024**3),
            "model_config": self.config["model"],
            "dataset_config": self.config["dataset"]
        }

        # Calculate checksum
        with open(model_path, 'rb') as f:
            file_hash = hashlib.sha256()
            while chunk := f.read(8192):
                file_hash.update(chunk)
            results["sha256"] = file_hash.hexdigest()

        logger.info(f"Model verification: {results}")
        return results

    def list_available_models(self) -> Dict[str, Any]:
        """
        List available models in cache and online

        Returns:
            Dictionary of available models
        """
        available = {
            "cached": [],
            "online": [],
            "recommended": self.config["model"]["fine_tuned_model"]
        }

        # Check cached models
        for model_dir in self.cache_dir.iterdir():
            if model_dir.is_dir():
                for gguf_file in model_dir.glob("*.gguf"):
                    available["cached"].append({
                        "path": str(gguf_file),
                        "size_gb": gguf_file.stat().st_size / (1024**3),
                        "model_id": model_dir.name.replace("_", "/")
                    })

        # Add known online models
        available["online"] = [
            {
                "model_id": "Manojb/Qwen3-4b-toolcall-gguf-llamacpp-codex",
                "filename": "Qwen3-4B-Function-Calling-Pro.gguf",
                "size_gb": 4.28,
                "dataset": "Salesforce/xlam-function-calling-60k"
            }
        ]

        return available

    def clear_cache(self, model_id: Optional[str] = None):
        """
        Clear model cache

        Args:
            model_id: Specific model to clear, or None for all
        """
        if model_id:
            model_dir = self.cache_dir / model_id.replace("/", "_")
            if model_dir.exists():
                import shutil
                shutil.rmtree(model_dir)
                logger.info(f"Cleared cache for {model_id}")
                # Remove from loaded models
                for key in list(self.loaded_models.keys()):
                    if model_id in key:
                        del self.loaded_models[key]
        else:
            # Clear all
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self.loaded_models.clear()
                logger.info("Cleared all model cache")

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the training dataset

        Returns:
            Dataset information
        """
        return self.config["dataset"]

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model

        Returns:
            Model information
        """
        return self.config["model"]


def test_loader():
    """Test the GGUF model loader"""
    loader = GGUFModelLoader()

    print("Dataset Info:")
    print(json.dumps(loader.get_dataset_info(), indent=2))

    print("\nModel Info:")
    print(json.dumps(loader.get_model_info(), indent=2))

    print("\nAvailable Models:")
    print(json.dumps(loader.list_available_models(), indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_loader()