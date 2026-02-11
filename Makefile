# PDI Entropy-Guided Gap-Filling
# Makefile for reproducibility and development

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
experiment-quick: ## Run quick validation (preprocess + run, 50 patches, 1 seed)
	@echo "Preprocessing dataset for quick_validation"
	@uv run python scripts/preprocess_dataset.py --config config/quick_validation.yaml --resume
	@echo "Running quick validation experiment"
	@uv run python scripts/run_experiment.py --quick --save-reconstructions 5

.PHONY: experiment-dry
experiment-dry: ## Validate configuration without running
	@uv run python scripts/run_experiment.py --dry-run

.PHONY: figures
figures: ## Generate figures from full experiment results
	@uv run python scripts/generate_figures.py --results results/paper_results

.PHONY: tables
tables: ## Generate LaTeX tables from full experiment results
	@uv run python scripts/generate_tables.py --results results/paper_results

.PHONY: figures-quick
figures-quick: ## Generate figures from quick validation results
	@uv run python scripts/generate_figures.py --results results/quick_validation --png-only

.PHONY: tables-quick
tables-quick: ## Generate tables from quick validation results
	@uv run python scripts/generate_tables.py --results results/quick_validation

.PHONY: reproduce
reproduce: experiment figures tables ## Full reproduction pipeline
	@echo ""
	@echo "=========================================="
	@echo "Reproduction complete!"
	@echo "=========================================="
	@echo "Results: results/paper_results/"
	@echo "Figures: results/paper_results/figures/"
	@echo "Tables:  results/paper_results/tables/"

.PHONY: reproduce-quick
reproduce-quick: experiment-quick figures-quick tables-quick ## Quick reproduction pipeline
	@echo ""
	@echo "Quick reproduction complete!"
	@echo "Results: results/quick_validation/"

# =============================================================================
# CLEANUP
# =============================================================================

.PHONY: clean
clean: ## Clean caches and generated files
	@rm -rf .mypy_cache .pytest_cache .ruff_cache
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete"

.PHONY: clean-results
clean-results: ## Clean experiment results (keep preprocessed data)
	@rm -rf results/
	@echo "Results cleaned"

.PHONY: clean-all
clean-all: clean clean-results ## Clean everything including preprocessed data
	@rm -rf preprocessed/
	@echo "All cleaned"

# =============================================================================
# HELP
# =============================================================================

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
