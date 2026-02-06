# PDI Entropy-Guided Gap-Filling
# Makefile for reproducibility and development

.PHONY: install
install: ## Install dependencies and pre-commit hooks
	@echo "Creating virtual environment using uv"
	@uv sync
	@uv run pre-commit install

.PHONY: check
check: ## Run code quality tools
	@echo "Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "Linting code: Running pre-commit"
	@uv run pre-commit run -a
# 	@echo "Static type checking: Running mypy"
# 	@uv run mypy
	@echo "Checking for obsolete dependencies: Running deptry"
	@uv run deptry src

.PHONY: test
test: ## Test the code with pytest
	@echo "Testing code: Running pytest"
	@uv run python -m pytest --cov --cov-config=pyproject.toml --cov-report=xml

.PHONY: build
build: clean-build ## Build wheel file
	@echo "Creating wheel file"
	@uvx --from build pyproject-build --installer uv

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@echo "Removing build artifacts"
	@uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"

# =============================================================================
# REPRODUCIBILITY TARGETS
# =============================================================================

.PHONY: preprocess
preprocess: ## Preprocess dataset for fast loading (run once)
	@echo "Preprocessing dataset (converts GeoTIFF to NPY, precomputes entropy)"
	@uv run python scripts/preprocess_dataset.py

.PHONY: preprocess-resume
preprocess-resume: ## Resume interrupted preprocessing
	@echo "Resuming preprocessing..."
	@uv run python scripts/preprocess_dataset.py --resume

.PHONY: experiment
experiment: ## Run full paper reproduction experiment
	@echo "Running full experiment with paper_results.yaml"
	@uv run python scripts/run_experiment.py --config config/paper_results.yaml

.PHONY: experiment-quick
experiment-quick: ## Run quick validation (50 patches, 1 seed)
	@echo "Running quick validation experiment"
	@uv run python scripts/run_experiment.py --quick

.PHONY: experiment-dry
experiment-dry: ## Validate configuration without running
	@echo "Validating experiment configuration"
	@uv run python scripts/run_experiment.py --dry-run

.PHONY: figures
figures: ## Generate publication figures from results
	@echo "Generating figures"
	@uv run python scripts/generate_figures.py

.PHONY: tables
tables: ## Generate LaTeX tables from results
	@echo "Generating LaTeX tables"
	@uv run python scripts/generate_tables.py

.PHONY: reproduce
reproduce: preprocess experiment figures tables ## Full reproduction pipeline
	@echo ""
	@echo "=========================================="
	@echo "Reproduction complete!"
	@echo "=========================================="
	@echo "Results: outputs/"
	@echo "Figures: paper/figures/"
	@echo "Tables:  paper/tables/"

.PHONY: reproduce-quick
reproduce-quick: experiment-quick figures tables ## Quick reproduction pipeline
	@echo ""
	@echo "Quick reproduction complete!"

# =============================================================================
# DOCUMENTATION TARGETS
# =============================================================================

.PHONY: docs-test
docs-test: ## Test if documentation can be built without warnings or errors
	@uv run mkdocs build -s

.PHONY: docs
docs: ## Build and serve the documentation
	@uv run mkdocs serve

.PHONY: latex
latex: ## Start hot-reload for LaTeX compilation
	@echo "Starting LaTeX hot-reload"
	@bash hot-reload-latex.sh

# =============================================================================
# UTILITY TARGETS
# =============================================================================

.PHONY: clean
clean: clean-build ## Clean all generated files
	@echo "Cleaning generated files..."
	@rm -rf outputs/checkpoints
	@rm -rf outputs/metadata/logs/*.log
	@rm -rf .mypy_cache
	@rm -rf .pytest_cache
	@rm -rf .ruff_cache
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete"

.PHONY: clean-results
clean-results: ## Clean experiment results (keep preprocessed data)
	@echo "Cleaning experiment results..."
	@rm -rf outputs/data
	@rm -rf outputs/figures
	@rm -rf outputs/reconstructions
	@rm -rf paper/figures
	@rm -rf paper/tables
	@echo "Results cleaned"

.PHONY: clean-all
clean-all: clean clean-results ## Clean everything including preprocessed data
	@echo "Cleaning preprocessed data..."
	@rm -rf "$${PDI_DATA_ROOT:-/opt/datasets/satellite-images}/preprocessed"
	@echo "All cleaned"

.PHONY: help
help: ## Show this help message
	@echo "PDI Entropy-Guided Gap-Filling"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@uv run python -c "import re; \
	[[print(f'  \033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
