# PDI Entropy-Guided Gap-Filling

# =============================================================================
# DEVELOPMENT
# =============================================================================

.PHONY: install
install: ## Install dependencies and pre-commit hooks
	@echo "Installing dependencies"
	@uv sync
	@uv run pre-commit install

.PHONY: check
check: ## Run linting and dependency checks
	@uv lock --locked
	@uv run pre-commit run -a
	@uv run deptry src

.PHONY: test
test: ## Run test suite with coverage
	@uv run python -m pytest --cov --cov-config=pyproject.toml --cov-report=xml

# =============================================================================
# EXPERIMENT PIPELINE
# =============================================================================

.PHONY: experiment
experiment: ## Run full paper experiment (preprocess + run)
	@echo "Preprocessing dataset for paper_results"
	@uv run python scripts/preprocess_dataset.py --config config/paper_results.yaml --resume
	@echo "Running full experiment (paper_results.yaml)"
	@uv run python scripts/run_experiment.py --config config/paper_results.yaml --save-reconstructions 5

.PHONY: experiment-quick
experiment-quick: ## Run quick validation (50 patches, 1 seed)
	@echo "Preprocessing dataset for quick_validation"
	@uv run python scripts/preprocess_dataset.py --config config/quick_validation.yaml --resume
	@echo "Running quick validation experiment"
	@uv run python scripts/run_experiment.py --quick --save-reconstructions 5

# =============================================================================
# PAPER
# =============================================================================

TEX_SRC   := docs/main.tex
BUILD_DIR := docs/build

.PHONY: paper
paper: ## Compile docs/main.tex -> docs/build/main.pdf (auto-pass via latexmk)
	@mkdir -p $(BUILD_DIR)
	@latexmk -lualatex -cd -interaction=nonstopmode -halt-on-error \
		-outdir=$(abspath $(BUILD_DIR)) $(TEX_SRC)
	@echo "Output: $(BUILD_DIR)/main.pdf"
