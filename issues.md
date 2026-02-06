# GitHub Issues - PDI Entropy-Guided Gap-Filling

---

## Milestone 1: Project Foundation and Data Pipeline

### Issue #1: Create dataset loading and preprocessing script

**Description:** Implement `scripts/preprocess_dataset.py` to convert GeoTIFF patches to NPY format for fast loading. The script should read from the metadata parquet, validate file paths, handle train/val/test splits, and support resumable processing with `--resume` flag. Must support Sentinel-2, Landsat-8/9, and MODIS data.

- [ ] Read metadata from `/opt/datasets/satellite-images/metadata.parquet`
- [ ] Load clean, degraded (inf, 40dB, 30dB, 20dB), and mask GeoTIFFs per patch
- [ ] Convert and save as NPY arrays in `preprocessed/` directory
- [ ] Implement `--resume` flag to skip already-processed patches
- [ ] Add progress bar with `tqdm`
- [ ] Validate data integrity (shape, dtype, NaN consistency with masks)
- [ ] Write a manifest/index file for fast lookups downstream

### Issue #2: Implement configuration system with YAML

**Description:** Create `config/` directory with YAML-based experiment configurations. Should define method lists, noise levels, entropy window sizes, patch counts, random seeds, and output paths. Include `paper_results.yaml` for full reproduction and a quick validation config.

- [ ] Create `config/paper_results.yaml` with full experiment parameters
- [ ] Create `config/quick_validation.yaml` for fast sanity checks
- [ ] Implement config loader in `src/pdi_pipeline/config.py`
- [ ] Validate config schema (methods, noise levels, seeds, etc.)
- [ ] Support CLI overrides for key parameters

### Issue #3: Implement dataset class for patch loading

**Description:** Create a dataset abstraction in `src/pdi_pipeline/dataset.py` that loads preprocessed NPY patches by index. Should return clean image, degraded image, mask, and metadata. Support filtering by satellite, split, and noise level.

- [ ] Implement `PatchDataset` class with `__getitem__` and `__len__`
- [ ] Filter by split (train/val/test), satellite, and noise level
- [ ] Return `(clean, degraded, mask, metadata)` tuples
- [ ] Support deterministic shuffling with seed
- [ ] Handle multi-band images (B2, B3, B4, B8)

### Issue #4: Add unit tests for interpolation methods

**Description:** Write tests in `tests/` for all 15+ interpolation methods. Each test should verify that the method fills masked pixels, preserves unmasked pixels, handles edge cases (all-missing, no-missing), and returns correct shapes/dtypes.

- [ ] Create `tests/test_methods.py` with parametrized tests for all methods
- [ ] Test 2D (single band) and 3D (multi-band) inputs
- [ ] Test edge cases: empty mask, full mask, single pixel gap
- [ ] Verify output shape matches input shape
- [ ] Verify output dtype is float32
- [ ] Verify clipping to [0, 1] range
- [ ] Run on small synthetic patches (e.g., 16x16) for speed

---

## Milestone 2: Entropy Computation and Windowing

### Issue #5: Implement local entropy computation module

**Description:** Create `src/pdi_pipeline/entropy.py` to compute Shannon entropy in sliding local windows. Must support multiple window sizes (7x7, 15x15, 31x31) as specified in the parametrizations. Output should be a per-pixel entropy map.

- [ ] Implement sliding-window Shannon entropy using `skimage.filters.rank.entropy` or custom
- [ ] Support window sizes: 7x7, 15x15, 31x31
- [ ] Handle multi-band images (compute per-band and average, or on a selected band)
- [ ] Return entropy map with same spatial dimensions as input
- [ ] Optimize for large images (memory-efficient windowing)

### Issue #6: Implement patch segmentation with overlap

**Description:** Create `src/pdi_pipeline/windowing.py` to segment images into 64x64 patches with configurable overlap (as defined in parametrizations). Should yield patch coordinates and handle image boundaries.

- [ ] Extract 64x64 patches with configurable overlap stride
- [ ] Handle boundary patches (padding or discard)
- [ ] Return patch coordinates for spatial reconstruction
- [ ] Compute per-patch entropy statistics (mean, std, min, max)
- [ ] Classify patches into entropy bins (low, medium, high) for stratified analysis

### Issue #7: Precompute entropy maps during preprocessing

**Description:** Extend the preprocessing pipeline (Issue #1) to precompute and save entropy maps at all three window scales (7x7, 15x15, 31x31) alongside the NPY patches. This avoids recomputation during experiments.

- [ ] Compute entropy maps for each clean patch at 3 scales
- [ ] Save as separate NPY files in `preprocessed/entropy/`
- [ ] Update manifest to include entropy file paths
- [ ] Add per-patch entropy summary statistics to metadata

---

## Milestone 3: Experiment Runner and Metrics

### Issue #8: Implement quality metrics module

**Description:** Create `src/pdi_pipeline/metrics.py` with all evaluation metrics: PSNR, SSIM, RMSE, IoU, SAM, and ERGAS. Each metric should support both global (full image) and local (per-patch/per-window) computation.

- [ ] Implement PSNR (global and local)
- [ ] Implement SSIM (global and local, using `skimage.metrics.structural_similarity`)
- [ ] Implement RMSE (global and local)
- [ ] Implement IoU (for mask-based evaluation)
- [ ] Implement SAM (Spectral Angle Mapper) for multi-band
- [ ] Implement ERGAS (relative dimensionless global error)
- [ ] All metrics return float32, handle edge cases (zero variance, etc.)
- [ ] Add local variant that returns a spatial map of metric values

### Issue #9: Implement experiment runner script

**Description:** Create `scripts/run_experiment.py` that orchestrates the full experiment loop: load config, iterate over methods/noise levels/seeds, apply gap-filling, compute metrics, and save results. Support `--quick`, `--dry-run`, and `--config` flags.

- [ ] Parse CLI args: `--config`, `--quick`, `--dry-run`
- [ ] Load and validate YAML config
- [ ] Loop: for each seed x noise level x method, apply and evaluate
- [ ] Save per-patch results (method, noise, seed, patch_id, metrics) to parquet
- [ ] Save reconstructed images for visualization (configurable subset)
- [ ] Implement checkpointing to resume interrupted runs
- [ ] Log progress with timestamps and ETA
- [ ] Quick mode: 50 patches, 1 seed, subset of methods

### Issue #10: Implement results aggregation module

**Description:** Create `src/pdi_pipeline/aggregation.py` to load raw per-patch results and compute summary statistics: mean, median, IC95% per method per scenario. Group by entropy bins for the entropy-quality correlation analysis.

- [ ] Load parquet results files
- [ ] Compute mean, median, std, IC95% (bootstrap) per method x noise level
- [ ] Group by entropy bins (low/medium/high) and recompute statistics
- [ ] Output summary DataFrames for tables and plots
- [ ] Support multi-scale entropy grouping (7x7, 15x15, 31x31)

---

## Milestone 4: Statistical Analysis

### Issue #11: Implement correlation analysis (entropy vs. quality)

**Description:** Create `src/pdi_pipeline/statistics.py` with correlation analysis between local entropy and quality metrics. Compute Pearson and Spearman correlations with p-values. This directly tests H1 and H3.

- [ ] Pearson correlation: entropy vs. PSNR, SSIM, RMSE (per method)
- [ ] Spearman correlation: entropy vs. PSNR, SSIM, RMSE (per method)
- [ ] Compute p-values with FDR correction (alpha=0.05)
- [ ] Generate correlation matrix (multi-scale entropy x metrics)
- [ ] Output results as DataFrame with coefficients, p-values, and significance flags

### Issue #12: Implement ANOVA/Kruskal-Wallis and pairwise tests

**Description:** Add global comparison tests to `src/pdi_pipeline/statistics.py`. ANOVA or Kruskal-Wallis to compare methods globally, followed by pairwise post-hoc tests. This tests H2 and validates method ranking.

- [ ] ANOVA (parametric) or Kruskal-Wallis (non-parametric) for method comparison
- [ ] Report F/H statistic and p-value
- [ ] Pairwise post-hoc tests (Tukey HSD or Dunn) with FDR correction
- [ ] Stratify by entropy level (low/medium/high) for conditional comparisons
- [ ] Output ranked method tables per scenario

### Issue #13: Implement robust regression analysis

**Description:** Add regression analysis to `src/pdi_pipeline/statistics.py`. Fit robust regression models: quality metric ~ entropy + method + noise + interactions. Report coefficients, p-values, R-squared adjusted, and VIF.

- [ ] Fit OLS and robust regression (HuberRegressor or statsmodels RLM)
- [ ] Predictors: entropy (multi-scale), method (dummy), noise level
- [ ] Report beta coefficients, p-values, R-squared adjusted
- [ ] Compute VIF for multicollinearity check
- [ ] Bootstrap 95% confidence intervals for coefficients

### Issue #14: Implement spatial autocorrelation analysis (Moran's I, LISA)

**Description:** Add spatial statistics to `src/pdi_pipeline/statistics.py`. Compute Moran's I for global spatial autocorrelation and LISA for local hotspot detection in reconstruction error maps.

- [ ] Compute Moran's I on local PSNR/SSIM error maps
- [ ] Compute LISA (Local Indicators of Spatial Association) clusters
- [ ] Identify hotspots (high error clusters) and coldspots
- [ ] Use `esda` and `libpysal` for spatial weights and statistics
- [ ] Output LISA cluster maps and significance maps

---

## Milestone 5: Visualization and Figures

### Issue #15: Implement figure generation script

**Description:** Create `scripts/generate_figures.py` to produce all publication-quality figures from aggregated results. Output to `paper/figures/` in PDF and PNG formats.

- [ ] Entropy maps (per satellite, per window size)
- [ ] Scatterplots: entropy vs. PSNR (per method, with regression line and CI)
- [ ] Boxplots: metric distribution per method, grouped by noise level
- [ ] Boxplots: metric distribution per method, grouped by entropy bin
- [ ] LISA hotspot maps overlaid on reconstruction error
- [ ] Visual comparison grid: clean vs. degraded vs. reconstructed (best/worst methods)
- [ ] Consistent style: matplotlib rcParams, colorblind-safe palette
- [ ] Save as PDF (vector) and PNG (300 DPI)

### Issue #16: Implement LaTeX table generation script

**Description:** Create `scripts/generate_tables.py` to produce LaTeX-formatted tables from aggregated results for direct inclusion in the paper.

- [ ] Table: mean PSNR +/- IC95% per method (columns) x scenario (rows)
- [ ] Correlation matrix table: entropy (multi-scale) x metrics with p-values
- [ ] ANOVA/Kruskal-Wallis results table: F/H statistic and p-value
- [ ] Regression coefficients table: beta, p-value, R-squared, VIF
- [ ] Bold best values, underline second-best
- [ ] Output `.tex` files to `paper/tables/`

---

## Milestone 6: Advanced Methods and Comparisons

### Issue #17: Add auxiliary variable support (NDVI, DEM) for weighted interpolation

**Description:** Extend interpolation methods to optionally use auxiliary variables (NDVI, elevation DEM) as intelligent weighting factors. Pass these through the `meta` dict in `BaseMethod.apply()`.

- [ ] Compute NDVI from available bands (B4, B8) during preprocessing
- [ ] Integrate DEM data source (if available) or document as limitation
- [ ] Extend `meta` dict to carry auxiliary rasters
- [ ] Modify IDW, kriging, and RBF to accept auxiliary weights
- [ ] Benchmark with/without auxiliary variables

### Issue #18: Add deep learning baseline comparison

**Description:** Implement at least one deep learning baseline (e.g., partial convolutions, U-Net inpainting) for comparison against classical methods. Use PyTorch (already in dependencies).

- [ ] Implement a simple U-Net or partial convolution inpainting model
- [ ] Train on the train split, evaluate on test split
- [ ] Conform to `BaseMethod` interface for consistent evaluation
- [ ] Compare against classical methods in the results tables
- [ ] Document training hyperparameters and compute requirements

---

## Milestone 7: Reproducibility and Documentation

### Issue #19: End-to-end pipeline validation

**Description:** Run the full pipeline end-to-end on a small subset to validate that all stages connect correctly: preprocess -> experiment -> aggregation -> statistics -> figures -> tables. Use `make reproduce-quick`.

- [ ] Verify `make preprocess` completes without errors
- [ ] Verify `make experiment-quick` produces valid parquet outputs
- [ ] Verify figure generation creates expected files in `paper/figures/`
- [ ] Verify table generation creates expected `.tex` files
- [ ] Fix any integration issues between pipeline stages
- [ ] Document minimum hardware requirements and expected runtimes

### Issue #20: Write executive summary and recommendations

**Description:** After full experiments complete, analyze results and write a summary document with method recommendations per scenario (entropy level, noise level, satellite). Include practical guidelines for practitioners.

- [ ] Summarize which methods perform best overall
- [ ] Summarize which methods perform best in high-entropy regions
- [ ] Summarize which methods perform best in low-entropy regions
- [ ] Identify failure modes and limitations per method
- [ ] Provide recommendations for Sentinel-2, Landsat-8/9, MODIS separately
- [ ] Discuss computational cost vs. quality tradeoffs
- [ ] Address limitations: ground truth availability, kriging sensitivity, patch-based cost

### Issue #21: Full reproduction run and results archive

**Description:** Execute `make reproduce` with all parameters from `pdi.md`: 500-2000 patches, 10 random seeds, 4 noise levels, all methods, all entropy window sizes. Archive final results.

- [ ] Run `make reproduce` with `paper_results.yaml`
- [ ] Verify 10 seeds x 4 noise levels x all methods produce complete results
- [ ] Archive raw results, figures, and tables
- [ ] Verify statistical significance at alpha=0.05 with FDR correction
- [ ] Final review of all figures and tables for publication quality
