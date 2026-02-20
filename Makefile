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
experiment: ## Run full paper experiment (preprocess + run)
	@echo "Preprocessing dataset for paper_results"
	@uv run python scripts/preprocess_dataset.py --config config/paper_results.yaml --resume
	@echo "Running full experiment (paper_results.yaml)"
	@uv run python scripts/run_experiment.py --config config/paper_results.yaml --save-entropy-top-k 5
	@echo "Generating figures and tables for paper_results"
	@uv run python scripts/generate_figures.py --results results/paper_results
	@uv run python scripts/generate_tables.py --results results/paper_results

.PHONY: experiment-quick
experiment-quick: ## Run quick validation (50 patches, 1 seed)
	@echo "Preprocessing dataset for quick_validation"
	@uv run python scripts/preprocess_dataset.py --config config/quick_validation.yaml --resume
	@echo "Running quick validation experiment"
	@uv run python scripts/run_experiment.py --quick --save-entropy-top-k 5
	@echo "Generating figures and tables for quick_validation"
	@rm -rf results/quick_validation/figures
	@uv run python scripts/generate_figures.py --results results/quick_validation --png-only
	@uv run python scripts/generate_tables.py --results results/quick_validation

# =============================================================================
##@ Paper
# =============================================================================

TEX_SRC      := docs/main.tex
BUILD_DIR    := docs/dist
JOB_NAME     := draft
RESULTS_DIR  ?= results/paper_results

.PHONY: paper-assets
paper-assets: ## Copy generated tables and figures from results/ to docs/
	@printf "\033[34m==>\033[0m Copying tables from $(RESULTS_DIR)/tables/ to docs/tables/\n"
	@mkdir -p docs/tables docs/figures
	@cp -v $(RESULTS_DIR)/tables/*.tex docs/tables/ 2>/dev/null || \
		printf "\033[33mWARN\033[0m no .tex files found in $(RESULTS_DIR)/tables/\n"
	@printf "\033[34m==>\033[0m Copying figures from $(RESULTS_DIR)/figures/ to docs/figures/\n"
	@cp -v $(RESULTS_DIR)/figures/*.pdf docs/figures/ 2>/dev/null || true
	@cp -v $(RESULTS_DIR)/figures/*.png docs/figures/ 2>/dev/null || \
		printf "\033[33mWARN\033[0m no figure files found in $(RESULTS_DIR)/figures/\n"

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
# CKPT_DIR    - directory where .pth checkpoints and *_history.json are written
# EVAL_DIR    - root directory for evaluation results CSVs
# PLOT_DIR    - directory for training-curve plots (PDF + PNG)
# SAVE_RECON  - number of reconstructed arrays to save during eval (0 = none)

MANIFEST    ?= preprocessed/manifest.csv
SATELLITE   ?= sentinel2
CKPT_DIR    ?= dl_models/checkpoints
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

# --- Transformer hyperparameters ----------------------------------------------
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
##@ Deep Learning - Preprocessing
# =============================================================================

.PHONY: dl-preprocess
dl-preprocess: ## Preprocess satellite patches to NPY format (resume-safe)
	@printf "\033[34m==>\033[0m Preprocessing patches (resume-safe)\n"
	@uv run python scripts/preprocess_dataset.py --resume

# =============================================================================
##@ Deep Learning - Training
# =============================================================================

.PHONY: dl-train-ae
dl-train-ae: dl-preprocess ## Train Autoencoder (AE)
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
dl-train-vae: dl-preprocess ## Train Variational Autoencoder (VAE)
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
dl-train-gan: dl-preprocess ## Train GAN (UNet generator + PatchGAN discriminator)
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
dl-train-unet: dl-preprocess ## Train U-Net (skip connections + residual blocks)
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

.PHONY: dl-train-transformer
dl-train-transformer: dl-preprocess ## Train MAE-style Transformer
	@printf "\033[34m==>\033[0m Training Transformer | epochs=$(TF_EPOCHS) lr=$(TF_LR) wd=$(TF_WD) satellite=$(SATELLITE)\n"
	@uv run python -m dl_models.transformer.train \
		--manifest     $(MANIFEST)                    \
		--output       $(CKPT_DIR)/transformer_best.pth \
		--satellite    $(SATELLITE)                   \
		--epochs       $(TF_EPOCHS)                   \
		--batch-size   $(TF_BATCH)                    \
		--lr           $(TF_LR)                       \
		--weight-decay $(TF_WD)                       \
		--patience     $(TF_PATIENCE)                 \
		$(_DEVICE_ARG)

.PHONY: dl-train-unet-jax
dl-train-unet-jax: dl-preprocess ## Train U-Net JAX/Flax (experimental)
	@printf "\033[34m==>\033[0m Training UNet (JAX) | epochs=$(UNET_EPOCHS) lr=$(UNET_LR) satellite=$(SATELLITE)\n"
	@uv run python -m dl_models.unet_jax.train \
		--manifest     $(MANIFEST)                          \
		--output       $(CKPT_DIR)/unet_jax_best.msgpack    \
		--satellite    $(SATELLITE)                          \
		--epochs       $(UNET_EPOCHS)                        \
		--batch-size   $(UNET_BATCH)                         \
		--lr           $(UNET_LR)                             \
		--weight-decay $(UNET_WD)                             \
		--patience     $(UNET_PATIENCE)

.PHONY: dl-train-all
dl-train-all: dl-train-ae dl-train-vae dl-train-gan dl-train-unet dl-train-transformer dl-train-unet-jax ## Train all 6 models sequentially

# =============================================================================
##@ Deep Learning - Evaluation (test split)
# =============================================================================

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

.PHONY: dl-eval-transformer
dl-eval-transformer: ## Evaluate Transformer on test split (requires dl-train-transformer)
	$(call _check_ckpt,$(CKPT_DIR)/transformer_best.pth,dl-train-transformer)
	@printf "\033[34m==>\033[0m Evaluating Transformer  ->  $(EVAL_DIR)/transformer_inpainting/results.csv\n"
	@uv run python -m dl_models.evaluate \
		--model               transformer                      \
		--checkpoint          $(CKPT_DIR)/transformer_best.pth \
		--manifest            $(MANIFEST)                      \
		--satellite           $(SATELLITE)                     \
		--output              $(EVAL_DIR)                      \
		--save-reconstructions $(SAVE_RECON)                   \
		$(_DEVICE_ARG)

.PHONY: dl-eval-unet-jax
dl-eval-unet-jax: ## Evaluate U-Net JAX on test split (requires dl-train-unet-jax)
	$(call _check_ckpt,$(CKPT_DIR)/unet_jax_best.msgpack,dl-train-unet-jax)
	@printf "\033[34m==>\033[0m Evaluating UNet (JAX)  ->  $(EVAL_DIR)/unet_jax_inpainting/results.csv\n"
	@uv run python -m dl_models.evaluate \
		--model               unet_jax                          \
		--checkpoint          $(CKPT_DIR)/unet_jax_best.msgpack \
		--manifest            $(MANIFEST)                       \
		--satellite           $(SATELLITE)                      \
		--output              $(EVAL_DIR)                       \
		--save-reconstructions $(SAVE_RECON)

.PHONY: dl-eval-all
dl-eval-all: dl-eval-ae dl-eval-vae dl-eval-gan dl-eval-unet dl-eval-transformer dl-eval-unet-jax ## Evaluate all 6 models on test split

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
	@HFILES=$$(ls $(CKPT_DIR)/*_history.json 2>/dev/null); \
	[ -n "$$HFILES" ] || { \
		printf "\033[31mERROR\033[0m no history files found in $(CKPT_DIR). Train a model first.\n"; \
		exit 1; }; \
	printf "\033[34m==>\033[0m Plotting training curves  ->  $(PLOT_DIR)\n"; \
	uv run python -m dl_models.plot_training \
		--history $$HFILES \
		--output  $(PLOT_DIR)

# =============================================================================
##@ Deep Learning - Meta
# =============================================================================

.PHONY: dl-all
dl-all: dl-train-all dl-eval-all dl-plot ## Full DL pipeline: train all -> eval all -> plot
