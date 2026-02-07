# Publication Plan: Journal Paper on Entropy-Guided Gap-Filling

Target journals: IEEE TGRS, ISPRS Journal, or Remote Sensing of Environment.

Thesis: local entropy predicts where classical interpolation methods fail in
satellite gap-filling, and this relationship is spatially structured.

---

## Hypotheses (from pdi.md)

- **H1** - Regions with higher local entropy (more texture/complexity) will
  show worse average performance from classical interpolators.
- **H2** - Geostatistical and spectro-temporal methods will outperform spatial
  interpolators in low-entropy areas.
- **H3** - There is a statistically significant relationship between local
  quality metrics and local entropy.

---

## Scope Decisions

The original pdi.md listed a broad set of features. This plan aligns the
issues to a realistic first-paper scope. Items marked DEFERRED are saved for a
follow-up paper or post-review additions.

### In Scope (this paper)

| Element | Justification |
|---------|---------------|
| 16 spatial single-frame methods (7 categories) | Comprehensive benchmark is the paper's strength |
| Sentinel-2 (73,984 patches) | Primary sensor, statistically powerful |
| Landsat-8/9 (3,872 patches) | Cross-sensor generalization |
| MODIS (60 patches) | Exploratory only, descriptive stats, no hypothesis tests |
| 4 noise levels (inf, 40, 30, 20 dB) | Shows degradation sensitivity |
| Entropy windows 7x7, 15x15, 31x31 | Multi-scale analysis is the core contribution |
| PSNR, SSIM, RMSE, SAM | Universal metrics, reviewers expect these |
| Pearson/Spearman correlation | Direct test of H1/H3 |
| Kruskal-Wallis + Dunn post-hoc | Method ranking with statistical rigor |
| Bootstrap CI 95% | Confidence intervals for all reported means |
| Moran's I + LISA | Spatial autocorrelation is the novel angle |
| 10 random seeds | Reproducibility |
| Gap fraction stratification | Report distribution, stratify by small/medium/large |

### Excluded from This Paper

| Element | Reason | Issue |
|---------|--------|-------|
| Multi-temporal methods (3 methods) | Require temporal stacks, not single patches | -- |
| IoU | Segmentation metric, not meaningful for continuous gap-filling | #8 |
| ERGAS | Redundant with RMSE at same resolution; add if reviewers ask | #8 |
| Windowing/overlap module | Patches are pre-extracted at 64x64 in preprocessing | #6 |
| Robust regression + VIF | Overkill; Pearson/Spearman + Kruskal-Wallis is sufficient | #13 |
| NDVI/DEM auxiliary variables | Scope creep; save for second paper | #17 |

### Deferred (second paper or post-review)

| Element | Issue |
|---------|-------|
| Deep learning baselines (U-Net, partial convolutions) | #18 |
| Auxiliary variables (NDVI, DEM weighting) | #17 |
| Robust regression with VIF | #13 |
| ERGAS metric | #8 |

---

## Issue Alignment with Implementation Status

### Milestone 1: Project Foundation and Data Pipeline

#### Issue #1: Preprocessing script -- DONE

File: `scripts/preprocess_dataset.py`

- [x] Read metadata from `/opt/datasets/satellite-images/metadata.parquet`
- [x] Load clean, degraded (inf, 40dB, 30dB, 20dB), and mask GeoTIFFs
- [x] Convert and save as NPY arrays in `preprocessed/`
- [x] Implement `--resume` flag
- [x] Add progress bar with tqdm
- [x] Validate data integrity (shape, dtype, NaN consistency)
- [x] Write manifest.csv for fast lookups

#### Issue #2: Configuration system -- PARTIAL

File: `src/pdi_pipeline/config.py`

- [x] Create `config/paper_results.yaml` (10 seeds, 4 noise levels, 16 methods)
- [x] Create `config/quick_validation.yaml` (1 seed, 3 methods, 50 patches)
- [x] Implement config loader with frozen dataclasses
- [ ] Validate config schema (currently raises KeyError on missing keys)
- [ ] Support CLI overrides for key parameters (e.g. `--seed`, `--methods`)

#### Issue #3: Dataset class -- DONE

File: `src/pdi_pipeline/dataset.py`

- [x] PatchDataset with `__getitem__` and `__len__`
- [x] Filter by split, satellite, and noise level
- [x] Return PatchSample dataclass (clean, degraded, mask, metadata)
- [x] Deterministic shuffling with seed
- [x] Handle multi-band images (B2, B3, B4, B8)

#### Issue #4: Unit tests for interpolation methods -- PARTIAL

Files: `tests/unit/`, `tests/integration/`

- [x] 13 integration test files covering all 20 methods
- [x] Test 2D and 3D inputs via conftest fixtures
- [x] Verify output shape, dtype float32, clipping to [0, 1]
- [x] Test no-gap passthrough
- [ ] Unified parametrized `test_methods.py` with all methods
- [ ] Explicit single-pixel-gap edge case test

---

### Milestone 2: Entropy Computation

#### Issue #5: Entropy module -- DONE

File: `src/pdi_pipeline/entropy.py`

- [x] Sliding-window Shannon entropy via `skimage.filters.rank.entropy`
- [x] Window sizes: 7, 15, 31
- [x] Multi-band: compute on mean of all bands
- [x] Return float32 entropy map with same (H, W) as input
- [x] Optimized via skimage C backend

#### Issue #6: Windowing module -- NOT NEEDED

Patches are pre-extracted at 64x64 during preprocessing. No overlap-based
segmentation needed for this experiment design. Removed from scope.

#### Issue #7: Precompute entropy -- DONE

File: `scripts/precompute_entropy.py`

- [x] Compute entropy for each clean patch at 3 window sizes
- [x] Save as `{split}/{satellite}/{patch_id}_entropy_{window}.npy`
- [x] Update manifest.csv with `mean_entropy_7`, `mean_entropy_15`, `mean_entropy_31`
- [x] Support `--resume` flag

---

### Milestone 3: Experiment Runner and Metrics

#### Issue #8: Quality metrics -- PARTIAL (by design)

File: `src/pdi_pipeline/metrics.py`

- [x] PSNR (gap pixels only, global + local map)
- [x] SSIM (gap pixels only, global + local map)
- [x] RMSE (gap pixels only)
- [x] SAM (spectral angle, multichannel only)
- [x] compute_all() convenience function
- [ ] ~~IoU~~ -- cut (segmentation metric, not applicable)
- [ ] ~~ERGAS~~ -- cut (redundant with RMSE at same resolution)

#### Issue #9: Experiment runner -- PARTIAL

File: `scripts/run_experiment.py`

- [x] `--config`, `--quick`, `--dry-run` CLI flags
- [x] Loop: seed x noise_level x method x patch
- [x] Save per-patch results to Parquet with checkpointing
- [x] Multi-temporal methods excluded (16 methods total)
- [x] Progress bar with tqdm
- [x] Method registry with config-name-to-class mapping
- [ ] Save reconstructed images for visualization (configurable subset)

#### Issue #10: Results aggregation -- DONE

File: `src/pdi_pipeline/aggregation.py`

- [x] load_results() from Parquet
- [x] summary_by_method() with bootstrap CI 95%
- [x] summary_by_entropy_bin() with tercile thresholds
- [x] summary_by_gap_fraction() with tercile thresholds
- [x] summary_by_noise()
- [x] Multi-scale entropy support

---

### Milestone 4: Statistical Analysis

#### Issue #11: Correlation analysis -- DONE

File: `src/pdi_pipeline/statistics.py`

- [x] Pearson r + p-value per method
- [x] Spearman rho + p-value per method
- [x] FDR correction (Benjamini-Hochberg) via statsmodels
- [x] correlation_matrix() across methods x entropy_windows x metrics
- [x] DataFrame output with significance flags

#### Issue #12: Kruskal-Wallis + post-hoc -- DONE

File: `src/pdi_pipeline/statistics.py`

- [x] Kruskal-Wallis H-test (non-parametric, correct for 77k patches)
- [x] Mann-Whitney U pairwise post-hoc with Bonferroni correction
- [x] Report H statistic, p-value, significant pairs

Note: stratification by entropy level can be done by filtering the DataFrame
before calling method_comparison().

#### Issue #13: Robust regression + VIF -- DEFERRED

Not implemented. The plan cuts this as overkill for the first paper.
Pearson/Spearman + Kruskal-Wallis provides sufficient statistical rigor.
Add if reviewers request it.

#### Issue #14: Spatial autocorrelation -- DONE

File: `src/pdi_pipeline/statistics.py`

- [x] Moran's I (global) via esda.Moran
- [x] LISA clusters (local) via esda.Moran_Local
- [x] Queen contiguity weights via libpysal.weights.lat2W
- [x] Returns cluster labels + p-value maps

---

### Milestone 5: Visualization and Figures

#### Issue #15: Figure generation -- DONE

File: `scripts/generate_figures.py`

- [x] Fig 1: Entropy map examples (3 scales, 2 satellites)
- [x] Fig 2: Scatterplot entropy vs PSNR per method
- [x] Fig 3: Boxplot PSNR by entropy bin
- [x] Fig 4: Boxplot PSNR by noise level
- [x] Fig 5: LISA cluster maps
- [x] Fig 6: Visual reconstruction examples (placeholder, needs reconstruction saving)
- [x] Fig 7: Correlation heatmap (method x entropy_window x metric)
- [x] Publication style: matplotlib rcParams, colorblind-safe, 300 DPI, PDF + PNG

#### Issue #16: LaTeX table generation -- DONE

File: `scripts/generate_tables.py`

- [x] Table 1: Method overview (category, params, complexity)
- [x] Table 2: Mean PSNR +/- CI 95% per method x noise level
- [x] Table 3: Entropy-stratified results per method x entropy bin
- [x] Table 4: Spearman correlation matrix with significance stars
- [x] Table 5: Kruskal-Wallis + significant pairs summary
- [ ] ~~Regression coefficients table~~ -- deferred with Issue #13
- [ ] Bold best / underline second-best formatting

---

### Milestone 6: Advanced Methods

#### Issue #17: Auxiliary variables (NDVI, DEM) -- DEFERRED

Out of scope for this paper. Save for follow-up.

#### Issue #18: Deep learning baselines -- DEFERRED

Will be added as a separate phase after the classical experiment pipeline
is validated and results are analyzed. PyTorch is already in dependencies.

---

### Milestone 7: Reproducibility

#### Issue #19: End-to-end validation -- PARTIAL

File: `Makefile`

- [x] `make preprocess` target
- [x] `make experiment-quick` target (runs with `--quick`)
- [x] `make experiment-dry` target (dry-run validation)
- [x] `make figures` target
- [x] `make tables` target
- [x] `make reproduce` (full pipeline)
- [x] `make reproduce-quick` (quick pipeline)
- [ ] Actual end-to-end test run
- [ ] Document minimum hardware requirements and runtimes

#### Issue #20: Executive summary -- TODO

Blocked by: completed experiment results. Write after Issue #21.

#### Issue #21: Full reproduction run -- TODO

Blocked by: all previous issues resolved. Execute `make reproduce` with
`paper_results.yaml` and archive results.

---

## Open Work Items (priority order)

### P0 - Required before running experiments

1. **Issue #2 (remaining):** Add config schema validation so that typos in
   YAML fail early instead of producing cryptic KeyError at runtime.
2. **Issue #9 (remaining):** Add `--save-reconstructions` flag to experiment
   runner to save a configurable subset of reconstructed images for Fig 6.

### P1 - Required before generating paper outputs

3. **Issue #4 (remaining):** Add unified parametrized test with synthetic data
   covering edge cases (single-pixel gap, full mask) for all 16 methods.
4. **Issue #16 (remaining):** Add bold-best / underline-second-best formatting
   to LaTeX tables.
5. **Issue #19 (remaining):** Run `make reproduce-quick` end-to-end, fix any
   integration issues.

### P2 - Required after results are generated

6. **Issue #20:** Write executive summary with method recommendations.
7. **Issue #21:** Execute full reproduction run, archive results.

### P3 - Deferred (second paper or reviewer response)

8. **Issue #13:** Robust regression + VIF
9. **Issue #17:** NDVI/DEM auxiliary variables
10. **Issue #18:** Deep learning baselines

---

## Paper Structure

1. **Introduction** - problem, motivation, gap in literature
2. **Related Work** - classical methods, DL methods, entropy in RS
3. **Study Area and Data** - Sentinel-2, Landsat-8/9, gap simulation
4. **Methodology**
   - 4.1 Gap-filling methods (Table 1: 16 methods with parameters)
   - 4.2 Local entropy computation
   - 4.3 Evaluation metrics (PSNR, SSIM, RMSE, SAM)
   - 4.4 Statistical analysis framework
5. **Results**
   - 5.1 Overall method comparison (Table 2: mean +/- CI per method x noise)
   - 5.2 Entropy-performance correlation (Fig 2: scatterplots, Table 4)
   - 5.3 Spatial analysis (Fig 5: LISA maps)
   - 5.4 Cross-sensor comparison (Sentinel-2 vs Landsat)
6. **Discussion**
7. **Conclusion**

---

## Execution Sequence

```
# 1. Preprocess (run once)
make preprocess

# 2. Precompute entropy maps
uv run python scripts/precompute_entropy.py

# 3. Validate configuration
make experiment-dry

# 4. Quick validation (smoke test)
make reproduce-quick

# 5. Full experiment
make reproduce

# 6. Generate outputs
make figures
make tables
```
