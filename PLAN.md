# Plan: Align Pipeline for Journal Publication + DL Models (Isolated)

**Status:** IMPLEMENTED (all phases complete, DL isolated from pipeline)

## Context

Research pipeline for evaluating satellite image gap-filling methods using entropy-based spatial analysis. Aligned for journal submission (IEEE TGRS / ISPRS / RSE).

**Method count:** 15 classical methods across 6 categories (experiment pipeline).
**DL baselines:** 4 architectures (AE, VAE, GAN, Transformer) isolated under `src/dl-models/` for independent training, evaluation, and inference.

### Key decisions

- **DINEOF removed:** Requires time-series `(T, H, W)` input incompatible with single 64x64 patches. `dineof.py` kept for future temporal work; removed from config, registry, tables.
- **DL baselines isolated (Issue #18):** 4 architectures implemented under `src/dl-models/` as an independent module. DL models are NOT included in the classical experiment battery -- they have their own training scripts, evaluation script, and test suite. This separation allows independent experimentation with DL methods without coupling them to the classical pipeline.

---

## Phase 1: Remove Irrelevant Methods

DINEOF requires time-series `(T, H, W)` input and cannot run on single 64x64 patches.

### 1.1 `config/paper_results.yaml`

- Remove `dineof` from `geostatistical` category
- Category becomes: `geostatistical: [{ name: "kriging" }]`

### 1.2 `scripts/run_experiment.py`

- Add `"dineof"` to `EXCLUDED_METHODS` set
- Remove `DINEOFInterpolator` from import block and `classes` list

### 1.3 `src/pdi_pipeline/methods/__init__.py`

- Remove `DINEOFInterpolator` import and `__all__` entry
- Keep `dineof.py` file in place (valid for future temporal work)

### 1.4 `scripts/generate_tables.py`

- Remove `"dineof"` entry from `METHOD_INFO` dict

### 1.5 `tests/integration/test_dineof.py`

- Add module-level `pytest.mark.skip(reason="DINEOF excluded: requires time-series input")`

---

## Phase 2: Config Schema Validation

**File:** `src/pdi_pipeline/config.py`

Add `_validate_raw(raw: dict) -> None` called before dataclass construction in `load_config()`.

Validates:

- Top-level keys `experiment` and `methods` exist
- Required experiment keys: `name`, `seeds`, `noise_levels`, `satellites`, `entropy_windows`
- Type checks: `seeds` is list[int], `noise_levels` is list[str], etc.
- Each method entry has a `name` key (str, non-empty)
- Method categories belong to `VALID_CATEGORIES = {"spatial", "kernel", "geostatistical", "transform", "compressive", "patch_based"}`

Raises `ValueError` with clear message on failure.

---

## Phase 3: Statistical Fixes

### 3.1 FDR Correction for Pearson p-values

**File:** `src/pdi_pipeline/statistics.py`

In `correlation_matrix()`:

- Add `pearson_p` to the rows dict
- After existing FDR on Spearman, add second FDR pass on Pearson p-values
- Add `pearson_significant_fdr` column to output DataFrame
- Rename existing `significant_fdr` to `spearman_significant_fdr` for clarity

### 3.2 Effect Size for Kruskal-Wallis

**File:** `src/pdi_pipeline/statistics.py`

- Add `epsilon_squared: float = 0.0` field to `ComparisonResult`
- Compute in `method_comparison()` after Kruskal-Wallis: `eps_sq = H / (N - 1)`
- Add `cliffs_delta` to post-hoc pairwise rows for effect size per pair

### 3.3 Gap Mask Standardization

**File:** `src/pdi_pipeline/metrics.py`

Add `_as_gap_mask()` helper and update all metric functions.

---

## Phase 4: Pipeline Fixes

### 4.1 Method Failure Tracking

**File:** `scripts/run_experiment.py`

- Add `status` and `error_msg` columns to each result row
- After final flush: log failure summary grouped by method

### 4.2 Reconstruction Saving

**File:** `scripts/run_experiment.py`

- Add `--save-reconstructions N` CLI argument
- Save first N reconstructed arrays per method (first seed, `noise_level="inf"`)

### 4.3 Fig 6 Visual Examples

**File:** `scripts/generate_figures.py`

Full implementation of `fig6_visual_examples`:

- Load parquet results, find 2 representative patches (10th and 90th entropy percentile)
- Load clean/degraded from preprocessed dir
- Load reconstructions
- Rank methods by PSNR, show top-4
- Layout: 2 rows (low/high entropy) x 6 columns (clean, degraded, top-4)

---

## Phase 5: Deep Learning Models (Isolated)

DL models live under `src/dl-models/` as a self-contained module, separate from the classical experiment pipeline. Each model is in its own subdirectory for independent training and inference.

### 5.1 Directory Structure

```
src/dl-models/
    __init__.py                      # Package marker
    evaluate.py                      # Standalone evaluation for any model
    shared/
        __init__.py                  # Exports shared utilities
        base.py                      # BaseDLMethod(BaseMethod)
        dataset.py                   # PyTorch Dataset wrapping PatchDataset
        utils.py                     # GapPixelLoss, EarlyStopping, checkpoint
    ae/
        __init__.py                  # Exports AEInpainting
        model.py                     # AEInpainting class + _AENet
        train.py                     # Training script
        config.yaml                  # Hyperparameters
    vae/
        __init__.py                  # Exports VAEInpainting
        model.py                     # VAEInpainting class + _VAENet
        train.py                     # Training script
        config.yaml                  # Hyperparameters
    gan/
        __init__.py                  # Exports GANInpainting
        model.py                     # GANInpainting + _UNetGenerator + _PatchDiscriminator
        train.py                     # Training script
        config.yaml                  # Hyperparameters
    transformer/
        __init__.py                  # Exports TransformerInpainting
        model.py                     # TransformerInpainting + _MAEInpaintingNet
        train.py                     # Training script
        config.yaml                  # Hyperparameters
    checkpoints/
        .gitkeep                     # Weights saved here after training
```

### 5.2 Training

Each model is trained independently:

```bash
uv run python src/dl-models/ae/train.py --manifest preprocessed/manifest.csv
uv run python src/dl-models/vae/train.py --manifest preprocessed/manifest.csv
uv run python src/dl-models/gan/train.py --manifest preprocessed/manifest.csv
uv run python src/dl-models/transformer/train.py --manifest preprocessed/manifest.csv
```

### 5.3 Evaluation

All models share a single evaluation entry point:

```bash
uv run python src/dl-models/evaluate.py \
    --model ae \
    --checkpoint src/dl-models/checkpoints/ae_best.pt \
    --manifest preprocessed/manifest.csv
```

### 5.4 Model Architectures

- **AE:** 4-layer conv encoder-decoder, 512-d bottleneck, MSE gap loss
- **VAE:** Same encoder + mu/logvar bottleneck (dim=256), MSE + beta*KL (beta=0.001)
- **GAN:** UNet generator with dilated bottleneck + PatchGAN discriminator
- **Transformer:** MAE-style, 64 tokens (8x8 patches), 4+2 Transformer blocks

---

## Phase 6: Figure and Table Updates

### 6.1 `scripts/generate_tables.py`

- Remove `"dineof"` from `METHOD_INFO`
- 15 classical methods in `METHOD_INFO`
- Table 1 caption: "15 classical gap-filling methods"
- Bold-best / underline-second formatting via `_format_ranked_cell()`
- `epsilon_squared` column in table 5

### 6.2 `scripts/generate_figures.py`

- Fig 2 grid: `ncols = min(5, n_methods)` for 15 methods
- Fig 6: full visual reconstruction (top-4 classical, no DL column)
- Fig 8 removed (was classical vs DL comparison)
- Total: 7 figures

---

## Phase 7: Tests

### Pipeline Tests

- `tests/unit/test_config.py`: Schema validation tests (missing keys, bad types, unknown categories, deep_learning rejected as pipeline category)
- `tests/integration/test_dineof.py`: Module-level skip marker
- `tests/integration/test_runner.py`: Expects 15 classical methods (no DL)

### DL Model Tests (Isolated)

- `tests/unit/test_dl_models.py`: Contract tests for all 4 DL models (output shape, dtype, range, valid pixel preservation). Uses `sys.path` to import from `src/dl-models/`.

---

## Execution Order

```
1. Phase 1  -- Remove DINEOF
2. Phase 2  -- Config validation
3. Phase 3  -- Statistical fixes (FDR, effect sizes, mask)
4. Phase 4  -- Pipeline fixes (failure tracking, reconstructions, fig6)
5. Phase 5  -- DL models (isolated, independent training/eval)
6. Phase 6  -- Figure and table updates
7. Phase 7  -- Tests
```

Phases 1-4 are independent fixes to the existing pipeline.
Phase 5 is the DL module (isolated from the pipeline).
Phases 6-7 wire everything together and validate.

---

## Files Modified (existing)

| File                                   | Changes                                                |
| -------------------------------------- | ------------------------------------------------------ |
| `config/paper_results.yaml`            | Remove dineof, remove DL methods                       |
| `src/pdi_pipeline/config.py`           | Add `_validate_raw()`, remove deep_learning category   |
| `src/pdi_pipeline/statistics.py`       | FDR for Pearson, epsilon-squared, Cliff's delta        |
| `src/pdi_pipeline/metrics.py`          | Standardize gap mask to `_as_gap_mask()`               |
| `src/pdi_pipeline/methods/__init__.py` | Remove DINEOF export                                   |
| `scripts/run_experiment.py`            | Remove DINEOF, add failure tracking, reconstructions   |
| `scripts/generate_figures.py`          | Fig2 grid, fig6 implementation, remove fig8            |
| `scripts/generate_tables.py`           | Remove dineof, remove DL, bold/underline, eps-squared  |
| `tests/integration/test_dineof.py`     | Add skip marker                                        |
| `tests/integration/test_runner.py`     | 15 classical methods expected (no DL)                  |
| `tests/unit/test_config.py`            | Schema validation, deep_learning category rejected     |
| `pyproject.toml`                       | Remove dl-models from pythonpath                       |
| `PLAN.md`                              | Updated scope, DL isolation, method counts             |

## Files Created (new)

| File                                          | Purpose                              |
| --------------------------------------------- | ------------------------------------ |
| `src/dl-models/__init__.py`                   | Package marker                       |
| `src/dl-models/evaluate.py`                   | Standalone evaluation for any model  |
| `src/dl-models/shared/__init__.py`            | Exports shared utilities             |
| `src/dl-models/shared/base.py`               | BaseDLMethod(BaseMethod)             |
| `src/dl-models/shared/dataset.py`            | PyTorch Dataset adapter              |
| `src/dl-models/shared/utils.py`              | GapPixelLoss, EarlyStopping, ckpt    |
| `src/dl-models/ae/__init__.py`               | Exports AEInpainting                 |
| `src/dl-models/ae/model.py`                  | AE architecture + inpainting class   |
| `src/dl-models/ae/train.py`                  | AE training script                   |
| `src/dl-models/ae/config.yaml`               | AE hyperparameters                   |
| `src/dl-models/vae/__init__.py`              | Exports VAEInpainting                |
| `src/dl-models/vae/model.py`                 | VAE architecture + inpainting class  |
| `src/dl-models/vae/train.py`                 | VAE training script                  |
| `src/dl-models/vae/config.yaml`              | VAE hyperparameters                  |
| `src/dl-models/gan/__init__.py`              | Exports GANInpainting                |
| `src/dl-models/gan/model.py`                 | GAN architecture + inpainting class  |
| `src/dl-models/gan/train.py`                 | GAN training script                  |
| `src/dl-models/gan/config.yaml`              | GAN hyperparameters                  |
| `src/dl-models/transformer/__init__.py`      | Exports TransformerInpainting        |
| `src/dl-models/transformer/model.py`         | Transformer architecture + class     |
| `src/dl-models/transformer/train.py`         | Transformer training script          |
| `src/dl-models/transformer/config.yaml`      | Transformer hyperparameters          |
| `src/dl-models/checkpoints/.gitkeep`         | Placeholder for trained weights      |
| `tests/unit/test_dl_models.py`               | DL model contract tests              |

## Verification

1. `uv run pytest tests/unit/test_config.py` -- config validation works, deep_learning rejected
2. `uv run pytest tests/unit/test_dl_models.py` -- DL models pass contract tests
3. `uv run python scripts/run_experiment.py --quick --dry-run` -- 15 classical methods listed (no DL)
4. `uv run pytest tests/ -x` -- all tests pass
