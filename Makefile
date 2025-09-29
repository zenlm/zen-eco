# Zen Eco Models Training Makefile
# Reproducible training system for all Zen Eco model variants
# Usage: make all (trains everything) or make <target> for specific models

# Configuration
PYTHON := python3.13
BASE_MODEL := Qwen/Qwen2.5-3B-Instruct
HF_USERNAME := zenlm
SEED := 42
DEVICE := mps
CACHE_DIR := .cache
OUTPUT_DIR := output

# Ensure reproducibility
export PYTHONHASHSEED=0
export WANDB_DISABLED=true
export HF_DATASETS_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

.PHONY: all clean instruct thinking agent gguf test upload setup check-deps

# Default target
all: setup instruct thinking agent gguf test
	@echo "$(GREEN)✅ All Zen Eco models trained successfully!$(NC)"

# Setup environment and dependencies
setup: check-deps
	@echo "$(BLUE)🔧 Setting up training environment...$(NC)"
	@mkdir -p $(CACHE_DIR) $(OUTPUT_DIR)
	@mkdir -p instruct thinking agent
	@echo "$(GREEN)✅ Environment ready$(NC)"

# Check Python version and dependencies
check-deps:
	@echo "$(BLUE)🔍 Checking dependencies...$(NC)"
	@$(PYTHON) --version | grep -q "3.13" || (echo "$(RED)❌ Python 3.13 required$(NC)" && exit 1)
	@$(PYTHON) -c "import torch, transformers, datasets, trl, peft" 2>/dev/null || \
		(echo "$(YELLOW)📦 Installing dependencies...$(NC)" && \
		$(PYTHON) -m pip install -q torch transformers datasets trl peft accelerate bitsandbytes sentencepiece protobuf)
	@echo "$(GREEN)✅ Dependencies OK$(NC)"

# Train Zen Eco Instruct model
instruct: setup
	@echo "$(BLUE)🎯 Training zen-eco-4b-instruct...$(NC)"
	@$(PYTHON) scripts/train_instruct.py \
		--model_name $(BASE_MODEL) \
		--output_dir $(OUTPUT_DIR)/zen-eco-4b-instruct \
		--seed $(SEED) \
		--device $(DEVICE)
	@echo "$(GREEN)✅ zen-eco-4b-instruct trained$(NC)"

# Train Zen Eco Thinking model
thinking: setup
	@echo "$(BLUE)🧠 Training zen-eco-4b-thinking...$(NC)"
	@$(PYTHON) scripts/train_thinking.py \
		--model_name $(BASE_MODEL) \
		--output_dir $(OUTPUT_DIR)/zen-eco-4b-thinking \
		--seed $(SEED) \
		--device $(DEVICE)
	@echo "$(GREEN)✅ zen-eco-4b-thinking trained$(NC)"

# Setup Zen Eco Agent (tool-calling) model
agent: setup
	@echo "$(BLUE)🔧 Setting up zen-eco-4b-agent...$(NC)"
	@$(PYTHON) scripts/setup_agent.py \
		--base_model $(OUTPUT_DIR)/zen-eco-4b-instruct \
		--output_dir $(OUTPUT_DIR)/zen-eco-4b-agent \
		--seed $(SEED) \
		--device $(DEVICE)
	@echo "$(GREEN)✅ zen-eco-4b-agent configured$(NC)"

# Generate GGUF files for all models
gguf: instruct thinking agent
	@echo "$(BLUE)📦 Generating GGUF files...$(NC)"
	@mkdir -p gguf
	@for model in instruct thinking agent; do \
		echo "Converting zen-eco-4b-$$model to GGUF..."; \
		$(PYTHON) scripts/convert_gguf.py \
			--model_path $(OUTPUT_DIR)/zen-eco-4b-$$model \
			--output_path gguf/zen-eco-4b-$$model.gguf \
			--quantization q4_k_m; \
	done
	@echo "$(GREEN)✅ GGUF files generated$(NC)"

# Test all models
test: instruct thinking agent
	@echo "$(BLUE)🧪 Testing all models...$(NC)"
	@$(PYTHON) scripts/test_models.py \
		--models_dir $(OUTPUT_DIR) \
		--device $(DEVICE)
	@echo "$(GREEN)✅ All tests passed$(NC)"

# Upload models to HuggingFace
upload: test gguf
	@echo "$(BLUE)☁️  Uploading to HuggingFace...$(NC)"
	@$(PYTHON) scripts/upload_models.py \
		--models_dir $(OUTPUT_DIR) \
		--gguf_dir gguf \
		--hf_username $(HF_USERNAME)
	@echo "$(GREEN)✅ Models uploaded to HuggingFace$(NC)"

# Clean up training artifacts
clean:
	@echo "$(YELLOW)🧹 Cleaning up...$(NC)"
	@rm -rf $(OUTPUT_DIR) $(CACHE_DIR) gguf
	@rm -rf instruct/output thinking/output agent/output
	@rm -f *.log *.json
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✅ Cleaned$(NC)"

# Quick test without training
test-only:
	@echo "$(BLUE)🧪 Running tests only...$(NC)"
	@$(PYTHON) scripts/test_models.py \
		--models_dir $(OUTPUT_DIR) \
		--device $(DEVICE) \
		--quick

# Show help
help:
	@echo "$(BLUE)Zen Eco Models Training System$(NC)"
	@echo ""
	@echo "$(YELLOW)Available targets:$(NC)"
	@echo "  make all       - Train all models (instruct, thinking, agent) and test"
	@echo "  make instruct  - Train zen-eco-4b-instruct model"
	@echo "  make thinking  - Train zen-eco-4b-thinking model"
	@echo "  make agent     - Setup zen-eco-4b-agent (tool-calling)"
	@echo "  make gguf      - Generate GGUF files for all models"
	@echo "  make test      - Test all trained models"
	@echo "  make upload    - Upload models to HuggingFace"
	@echo "  make clean     - Remove all training artifacts"
	@echo "  make help      - Show this help message"
	@echo ""
	@echo "$(YELLOW)Quick start:$(NC)"
	@echo "  $$ make all    # Train everything"
	@echo ""
	@echo "$(YELLOW)Requirements:$(NC)"
	@echo "  - Python 3.13"
	@echo "  - 16GB+ RAM"
	@echo "  - 50GB+ disk space"