# PDI Entropy-Guided Gap-Filling
#
# Run `make help` to see all available targets grouped by category.

# =============================================================================
# HELP
# =============================================================================
# Uses the ##@ (section) and ## (target description) convention.
# Sections appear as bold group headers; targets are listed underneath.

.PHONY: help
help: ## Show all available targets
	@awk ' \
		BEGIN {FS = ":.*## "; printf "\n\033[1mUsage:\033[0m\n  make \033[36m<target>\033[0m [VAR=value ...]\n"} \
		/^##@/ {printf "\n\033[1m%s\033[0m\n", substr($$0, 5)} \
		/^[a-zA-Z_-]+:.*## / {printf "  \033[36m%-28s\033[0m %s\n", $$1, $$2} \
	' $(MAKEFILE_LIST)

# =============================================================================
##@ Development
# =============================================================================

.PHONY: install
install: ## Install dependencies and pre-commit hooks
	@echo "Installing dependencies"

.PHONY: install-latex
install-latex: ## Install LaTeX toolchain (XeLaTeX + latexmk)
	@echo "LaTeX dependencies (XeLaTeX + latexmk)"
	@echo "Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y \
		latexmk texlive-xetex texlive-latex-extra texlive-fonts-recommended \
		texlive-fonts-extra texlive-bibtex-extra"
	@echo "Fedora: sudo dnf install -y latexmk texlive-xetex texlive-latex-extra \
		texlive-collection-fontsrecommended texlive-collection-fontsextra"
	@echo "Arch: sudo pacman -S --needed texlive-bin texlive-core texlive-latexextra \
		texlive-fontsextra"
	@echo "macOS (Homebrew): brew install mactex-no-gui latexmk"
	@uv sync
	@uv run pre-commit install

.PHONY: check
check: ## Run code quality tools
	@echo "Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "Linting code: Running pre-commit"
	@uv run pre-commit run -a
	@echo "Checking for obsolete dependencies: Running deptry"
	@uv run deptry src
.PHONY: test
test: ## Run full test suite with coverage
	@uv run python -m pytest --cov --cov-config=pyproject.toml --cov-report=xml

# =============================================================================
##@ Experiment Pipeline
# =============================================================================

.PHONY: experiment
experiment: preprocess-all ## Run full paper experiment (preprocess + run)
	@echo "Running full experiment (paper_results.yaml)"
	@uv run python scripts/run_experiment.py --config config/paper_results.yaml --save-entropy-top-k 5
	@echo "Generating figures and tables for paper_results"
	@uv run python scripts/generate_figures.py --results results/paper_results
	@uv run python scripts/generate_tables.py --results results/paper_results

.PHONY: preprocess-all
preprocess-all: ## Preprocess full dataset to NPY (shared for all workflows)
	@echo "Preprocessing full dataset (shared NPY cache)"
	@uv run python scripts/preprocess_dataset.py --resume

.PHONY: experiment-quick
experiment-quick: preprocess-all ## Run quick classical-only validation (1 seed, 1 patch)
	@echo "Running quick classical validation experiment"
	@uv run python scripts/run_experiment.py --quick --save-entropy-top-k 5
	@echo "Generating figures and tables for quick_validation"
	@rm -rf results/quick_validation/figures
	@uv run python scripts/generate_figures.py --results results/quick_validation --png-only
	@uv run python scripts/generate_tables.py --results results/quick_validation

.PHONY: preview
preview: preprocess-all ## Full preview: classical quick + all DL models quick + unified figures
	@echo "=== Step 1/4: Classical quick experiment ==="
	@uv run python scripts/run_experiment.py --quick --save-entropy-top-k 5
	@echo "=== Step 2/4: Training and evaluating all DL models (quick) ==="
	@for model in ae vae gan unet vit; do \
		echo "--- $$model ---"; \
		uv run python scripts/slurm/train_model.py \
			--config config/quick_validation.yaml --model $$model; \
	done
	@echo "=== Step 3/4: Plotting DL training curves ==="
	@_DL_DIR=$$(dirname "$(CKPT_DIR)"); \
	HFILES=$$(ls "$$_DL_DIR"/*_history.json 2>/dev/null || true); \
	if [ -n "$$HFILES" ]; then \
		mkdir -p "$(PLOT_DIR)"; \
		uv run python -m dl_models.plot_training --history $$HFILES --output $(PLOT_DIR); \
	fi
	@echo "=== Step 4/4: Generating unified preview figures and tables ==="
	@_DL_DIR=$$(dirname "$(CKPT_DIR)"); \
	rm -rf results/quick_validation/figures; \
	uv run python scripts/generate_figures.py \
		--results results/quick_validation \
		--dl-results "$$_DL_DIR" \
		--png-only
	@_DL_DIR=$$(dirname "$(CKPT_DIR)"); \
	uv run python scripts/generate_tables.py \
		--results results/quick_validation \
		--dl-results "$$_DL_DIR"
	@echo "Preview complete. Results in results/quick_validation/ and $(PLOT_DIR)/"

# =============================================================================
##@ Paper
# =============================================================================

TEX_SRC      := docs/main.tex
BUILD_DIR    := docs/dist
JOB_NAME     := draft
RESULTS_DIR  ?= results/paper_results
QUICK_DIR    ?= results/quick_validation
DL_PLOTS_DIR ?= $(PLOT_DIR)

# Directories whose tables/ and figures/ sub-dirs are copied to docs/.
# Later entries overwrite earlier ones when filenames collide, so put the
# authoritative source (paper_results) last.
_ASSET_DIRS := $(QUICK_DIR) $(RESULTS_DIR)

.PHONY: paper-assets
paper-assets: paper-normalize-tables ## Copy generated tables and figures from results/ to docs/
	@mkdir -p docs/tables docs/figures
	@_found_tables=0; _found_figures=0; \
	for src in $(_ASSET_DIRS); do \
		if [ -d "$$src/tables" ]; then \
			printf "\033[34m==>\033[0m Copying tables from $$src/tables/\n"; \
			cp -v "$$src"/tables/*.tex docs/tables/ 2>/dev/null && _found_tables=1 || true; \
		fi; \
		if [ -d "$$src/figures" ]; then \
			printf "\033[34m==>\033[0m Copying figures from $$src/figures/\n"; \
			cp -v "$$src"/figures/*.pdf docs/figures/ 2>/dev/null || true; \
			cp -v "$$src"/figures/*.png docs/figures/ 2>/dev/null && _found_figures=1 || true; \
		fi; \
	done; \
	if [ -d "$(DL_PLOTS_DIR)" ]; then \
		printf "\033[34m==>\033[0m Copying DL training plots from $(DL_PLOTS_DIR)/\n"; \
		cp -v "$(DL_PLOTS_DIR)"/*.pdf docs/figures/ 2>/dev/null || true; \
		cp -v "$(DL_PLOTS_DIR)"/*.png docs/figures/ 2>/dev/null && _found_figures=1 || true; \
	fi; \
	[ "$$_found_tables" = "1" ] || printf "\033[33mWARN\033[0m no .tex table files found in any source\n"; \
	[ "$$_found_figures" = "1" ] || printf "\033[33mWARN\033[0m no figure files found in any source\n"

.PHONY: paper-normalize-tables
paper-normalize-tables: ## Normalize table filenames to match docs/main.tex inputs
	@mkdir -p docs/tables
	@for pair in \
		table1_methods.tex:methods.tex \
		table2_overall.tex:psnr-method-noise.tex \
		table3_entropy_15.tex:psnr-entropy-tercile.tex \
		table4_correlation.tex:spearman-heatmap.tex \
		table6_regression_psnr.tex:robust-regression.tex \
		table7_satellite.tex:psnr-satellite.tex \
		table8_dl_comparison.tex:dl-results.tex; do \
		from=$${pair%%:*}; to=$${pair##*:}; \
		if [ -f "docs/tables/$$from" ]; then \
			mv -f "docs/tables/$$from" "docs/tables/$$to"; \
		fi; \
	done

.PHONY: paper
paper: paper-assets ## Compile docs/main.tex -> docs/dist/draft.pdf (via latexmk)
	@mkdir -p $(BUILD_DIR)
	@rm -f $(BUILD_DIR)/$(JOB_NAME).fdb_latexmk $(BUILD_DIR)/$(JOB_NAME).fls
	@latexmk -xelatex -cd -interaction=nonstopmode -halt-on-error \
		-jobname=$(JOB_NAME) -outdir=$(abspath $(BUILD_DIR)) $(TEX_SRC)
	@echo "Output: $(BUILD_DIR)/$(JOB_NAME).pdf"

.PHONY: paper-only
paper-only: ## Compile paper without copying assets (use existing docs/ files)
	@mkdir -p $(BUILD_DIR)
	@rm -f $(BUILD_DIR)/$(JOB_NAME).fdb_latexmk $(BUILD_DIR)/$(JOB_NAME).fls
	@latexmk -xelatex -cd -interaction=nonstopmode -halt-on-error \
		-jobname=$(JOB_NAME) -outdir=$(abspath $(BUILD_DIR)) $(TEX_SRC)
	@echo "Output: $(BUILD_DIR)/$(JOB_NAME).pdf"

# Force unbuffered Python output so logs appear in real time in terminals
# and SLURM captured files (equivalent to python -u).
export PYTHONUNBUFFERED := 1

# =============================================================================
##@ Deep Learning - Configuration
# =============================================================================
#
# Every variable below is overridable from the command line:
#
#   make dl-train-ae SATELLITE=landsat8 AE_EPOCHS=100 DEVICE=cuda:1
#
# MANIFEST    - path to the manifest CSV produced by preprocess_dataset.py
# SATELLITE   - filter to a specific satellite (passed to --satellite)
# DEVICE      - PyTorch device string; leave unset to let scripts auto-detect
# CKPT_DIR    - directory where .pth checkpoints are written
# EVAL_DIR    - root directory for evaluation results CSVs
# PLOT_DIR    - directory for training-curve plots (PDF + PNG)
# SAVE_RECON  - number of reconstructed arrays to save during eval (0 = none)

MANIFEST    ?= preprocessed/manifest.csv
SATELLITE   ?= sentinel2
CKPT_DIR    ?= results/dl_models/checkpoints
EVAL_DIR    ?= results/dl_eval
PLOT_DIR    ?= results/dl_plots
SAVE_RECON  ?= 0

# Pass --device only when DEVICE is explicitly set; otherwise let the script
# auto-detect (cuda if available, else cpu).
_DEVICE_ARG  = $(if $(DEVICE),--device $(DEVICE),)

# --- AE hyperparameters -------------------------------------------------------
AE_EPOCHS   ?= 2
AE_BATCH    ?= 32
AE_LR       ?= 1e-3
AE_PATIENCE ?= 10

# --- VAE hyperparameters ------------------------------------------------------
VAE_EPOCHS   ?= 1
VAE_BATCH    ?= 32
VAE_LR       ?= 1e-3
VAE_BETA     ?= 0.001
VAE_PATIENCE ?= 10

# --- GAN hyperparameters ------------------------------------------------------
GAN_EPOCHS      ?= 2
GAN_BATCH       ?= 16
GAN_LR          ?= 2e-4
GAN_PATIENCE    ?= 15
GAN_LAMBDA_L1   ?= 10.0
GAN_LAMBDA_ADV  ?= 0.1

# --- UNet hyperparameters -----------------------------------------------------
UNET_EPOCHS   ?= 2
UNET_BATCH    ?= 32
UNET_LR       ?= 1e-3
UNET_WD       ?= 1e-4
UNET_PATIENCE ?= 12

# --- ViT hyperparameters ------------------------------------------------------
TF_EPOCHS   ?= 2
TF_BATCH    ?= 32
TF_LR       ?= 1e-4
TF_WD       ?= 0.05
TF_PATIENCE ?= 15

# --- Internal macro -----------------------------------------------------------
# Aborts the recipe with a colored error if a checkpoint file is missing.
# Usage inside a recipe: $(call _check_ckpt,<path>,<make-target>)
define _check_ckpt
@test -f $(1) || { \
	printf "\033[31mERROR\033[0m checkpoint not found: \033[33m$(1)\033[0m\n"; \
	printf "      Run '\033[36mmake $(2)\033[0m' first.\n"; \
	exit 1; }
endef

# =============================================================================
##@ Deep Learning - Training
# =============================================================================

.PHONY: dl-train-ae
dl-train-ae: preprocess-all ## Train Autoencoder (AE)
	@printf "\033[34m==>\033[0m Training AE | epochs=$(AE_EPOCHS) lr=$(AE_LR) batch=$(AE_BATCH) satellite=$(SATELLITE)\n"
	@uv run python -m dl_models.ae.train \
		--manifest   $(MANIFEST)             \
		--output     $(CKPT_DIR)/ae_best.pth \
		--satellite  $(SATELLITE)            \
		--epochs     $(AE_EPOCHS)            \
		--batch-size $(AE_BATCH)             \
		--lr         $(AE_LR)                \
		--patience   $(AE_PATIENCE)          \
		$(_DEVICE_ARG)

.PHONY: dl-train-vae
dl-train-vae: preprocess-all ## Train Variational Autoencoder (VAE)
	@printf "\033[34m==>\033[0m Training VAE | epochs=$(VAE_EPOCHS) lr=$(VAE_LR) beta=$(VAE_BETA) satellite=$(SATELLITE)\n"
	@uv run python -m dl_models.vae.train \
		--manifest   $(MANIFEST)              \
		--output     $(CKPT_DIR)/vae_best.pth \
		--satellite  $(SATELLITE)             \
		--epochs     $(VAE_EPOCHS)            \
		--batch-size $(VAE_BATCH)             \
		--lr         $(VAE_LR)                \
		--beta       $(VAE_BETA)              \
		--patience   $(VAE_PATIENCE)          \
		$(_DEVICE_ARG)

.PHONY: dl-train-gan
dl-train-gan: preprocess-all ## Train GAN (UNet generator + PatchGAN discriminator)
	@printf "\033[34m==>\033[0m Training GAN | epochs=$(GAN_EPOCHS) lr=$(GAN_LR) lambda_l1=$(GAN_LAMBDA_L1) satellite=$(SATELLITE)\n"
	@uv run python -m dl_models.gan.train \
		--manifest    $(MANIFEST)              \
		--output      $(CKPT_DIR)/gan_best.pth \
		--satellite   $(SATELLITE)             \
		--epochs      $(GAN_EPOCHS)            \
		--batch-size  $(GAN_BATCH)             \
		--lr          $(GAN_LR)                \
		--patience    $(GAN_PATIENCE)          \
		--lambda-l1   $(GAN_LAMBDA_L1)         \
		--lambda-adv  $(GAN_LAMBDA_ADV)        \
		$(_DEVICE_ARG)

.PHONY: dl-train-unet
dl-train-unet: preprocess-all ## Train U-Net (skip connections + residual blocks)
	@printf "\033[34m==>\033[0m Training UNet | epochs=$(UNET_EPOCHS) lr=$(UNET_LR) wd=$(UNET_WD) satellite=$(SATELLITE)\n"
	@uv run python -m dl_models.unet.train \
		--manifest     $(MANIFEST)               \
		--output       $(CKPT_DIR)/unet_best.pth \
		--satellite    $(SATELLITE)              \
		--epochs       $(UNET_EPOCHS)            \
		--batch-size   $(UNET_BATCH)             \
		--lr           $(UNET_LR)                \
		--weight-decay $(UNET_WD)                \
		--patience     $(UNET_PATIENCE)          \
		$(_DEVICE_ARG)

.PHONY: dl-train-vit
dl-train-vit: preprocess-all ## Train MAE-style ViT
	@printf "\033[34m==>\033[0m Training ViT | epochs=$(TF_EPOCHS) lr=$(TF_LR) wd=$(TF_WD) satellite=$(SATELLITE)\n"
	@uv run python -m dl_models.vit.train \
		--manifest     $(MANIFEST)              \
		--output       $(CKPT_DIR)/vit_best.pth \
		--satellite    $(SATELLITE)             \
		--epochs       $(TF_EPOCHS)             \
		--batch-size   $(TF_BATCH)              \
		--lr           $(TF_LR)                 \
		--weight-decay $(TF_WD)                 \
		--patience     $(TF_PATIENCE)           \
		$(_DEVICE_ARG)

.PHONY: dl-train-all
dl-train-all: dl-train-ae dl-train-vae dl-train-gan dl-train-unet dl-train-vit ## Train all 5 models sequentially

# =============================================================================
##@ Deep Learning - Evaluation (test split)
# =============================================================================

.PHONY: dl-eval-all
dl-eval-all: dl-eval-ae dl-eval-vae dl-eval-gan dl-eval-unet dl-eval-vit ## Evaluate all 5 models on test split

.PHONY: dl-eval-ae
dl-eval-ae: ## Evaluate AE on test split (requires dl-train-ae)
	$(call _check_ckpt,$(CKPT_DIR)/ae_best.pth,dl-train-ae)
	@printf "\033[34m==>\033[0m Evaluating AE  ->  $(EVAL_DIR)/ae_inpainting/results.csv\n"
	@uv run python -m dl_models.evaluate \
		--model               ae                      \
		--checkpoint          $(CKPT_DIR)/ae_best.pth \
		--manifest            $(MANIFEST)             \
		--satellite           $(SATELLITE)            \
		--output              $(EVAL_DIR)             \
		--save-reconstructions $(SAVE_RECON)          \
		$(_DEVICE_ARG)

.PHONY: dl-eval-vae
dl-eval-vae: ## Evaluate VAE on test split (requires dl-train-vae)
	$(call _check_ckpt,$(CKPT_DIR)/vae_best.pth,dl-train-vae)
	@printf "\033[34m==>\033[0m Evaluating VAE  ->  $(EVAL_DIR)/vae_inpainting/results.csv\n"
	@uv run python -m dl_models.evaluate \
		--model               vae                      \
		--checkpoint          $(CKPT_DIR)/vae_best.pth \
		--manifest            $(MANIFEST)              \
		--satellite           $(SATELLITE)             \
		--output              $(EVAL_DIR)              \
		--save-reconstructions $(SAVE_RECON)           \
		$(_DEVICE_ARG)

.PHONY: dl-eval-gan
dl-eval-gan: ## Evaluate GAN on test split (requires dl-train-gan)
	$(call _check_ckpt,$(CKPT_DIR)/gan_best.pth,dl-train-gan)
	@printf "\033[34m==>\033[0m Evaluating GAN  ->  $(EVAL_DIR)/gan_inpainting/results.csv\n"
	@uv run python -m dl_models.evaluate \
		--model               gan                      \
		--checkpoint          $(CKPT_DIR)/gan_best.pth \
		--manifest            $(MANIFEST)              \
		--satellite           $(SATELLITE)             \
		--output              $(EVAL_DIR)              \
		--save-reconstructions $(SAVE_RECON)           \
		$(_DEVICE_ARG)

.PHONY: dl-eval-unet
dl-eval-unet: ## Evaluate U-Net on test split (requires dl-train-unet)
	$(call _check_ckpt,$(CKPT_DIR)/unet_best.pth,dl-train-unet)
	@printf "\033[34m==>\033[0m Evaluating UNet  ->  $(EVAL_DIR)/unet_inpainting/results.csv\n"
	@uv run python -m dl_models.evaluate \
		--model               unet                      \
		--checkpoint          $(CKPT_DIR)/unet_best.pth \
		--manifest            $(MANIFEST)               \
		--satellite           $(SATELLITE)              \
		--output              $(EVAL_DIR)               \
		--save-reconstructions $(SAVE_RECON)            \
		$(_DEVICE_ARG)

.PHONY: dl-eval-vit
dl-eval-vit: ## Evaluate ViT on test split (requires dl-train-vit)
	$(call _check_ckpt,$(CKPT_DIR)/vit_best.pth,dl-train-vit)
	@printf "\033[34m==>\033[0m Evaluating ViT  ->  $(EVAL_DIR)/vit_inpainting/results.csv\n"
	@uv run python -m dl_models.evaluate \
		--model               vit                      \
		--checkpoint          $(CKPT_DIR)/vit_best.pth \
		--manifest            $(MANIFEST)             \
		--satellite           $(SATELLITE)            \
		--output              $(EVAL_DIR)             \
		--save-reconstructions $(SAVE_RECON)          \
		$(_DEVICE_ARG)

# =============================================================================
##@ Deep Learning - Testing & Plots
# =============================================================================

.PHONY: dl-test
dl-test: ## Run contract and utility unit tests for all DL models
	@uv run pytest \
		tests/unit/test_dl_models.py \
		tests/unit/test_training_history.py \
		-v --tb=short -m unit

.PHONY: dl-plot
dl-plot: ## Plot training curves from all available history JSON files
	@_DL_RESULTS_DIR=$$(dirname "$(CKPT_DIR)"); \
	HFILES=$$(ls "$$_DL_RESULTS_DIR"/*_history.json 2>/dev/null); \
	[ -n "$$HFILES" ] || { \
		printf "\033[31mERROR\033[0m no history files found in $$_DL_RESULTS_DIR. Train a model first.\n"; \
		exit 1; }; \
	printf "\033[34m==>\033[0m Plotting training curves  ->  $(PLOT_DIR)\n"; \
	uv run python -m dl_models.plot_training \
		--history $$HFILES \
		--output  $(PLOT_DIR)

# =============================================================================
##@ Deep Learning - Meta
# =============================================================================

.PHONY: dl-tables
dl-tables: ## Generate LaTeX tables from DL training history
	@_DL_RESULTS_DIR=$$(dirname "$(CKPT_DIR)"); \
	printf "\033[34m==>\033[0m Generating DL tables  ->  $$_DL_RESULTS_DIR/tables\n"; \
	uv run python scripts/generate_tables.py --dl-results "$$_DL_RESULTS_DIR"

.PHONY: dl-all
dl-all: dl-train-all dl-eval-all dl-plot dl-tables ## Full DL pipeline: train all -> eval all -> plot -> tables
