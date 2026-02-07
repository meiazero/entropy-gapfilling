# Experiment Plan: Remote Sensing Journal Publication

## Context

Target journals: IEEE TGRS, ISPRS Journal, or Remote Sensing of Environment.

The paper's thesis: local entropy predicts where classical interpolation methods
fail in satellite gap-filling, and this relationship is spatially structured.
Deep Learning baselines will be added later; this plan focuses on getting the
classical experiment pipeline production-ready.

Current state: 20 classical methods implemented and integration-tested.
Everything else (entropy, metrics, runner, statistics, figures) is missing.

---

## Honest Assessment: What to Keep, Cut, and Fix from pdi.md

### Keep (essential for the journal)

| Element                              | Why                                                                 |
| ------------------------------------ | ------------------------------------------------------------------- |
| All 7 method categories (20 methods) | Comprehensive benchmark is the journal's strength                   |
| Sentinel-2 as primary sensor         | 73,984 patches - statistically powerful                             |
| Landsat-8/9 as secondary sensor      | Cross-sensor generalization (3,872 patches)                         |
| 4 noise levels (inf, 40, 30, 20 dB)  | Shows degradation sensitivity                                       |
| Entropy windows 7x7, 15x15, 31x31    | Multi-scale analysis is the core contribution                       |
| PSNR, SSIM, RMSE                     | Universal, reviewers expect these                                   |
| SAM                                  | Essential for multispectral - differentiates from grayscale studies |
| Pearson/Spearman correlation         | Direct test of H1/H3                                                |
| ANOVA/Kruskal-Wallis + post-hoc      | Method ranking with statistical rigor                               |
| Bootstrap IC95%                      | Confidence intervals for all reported means                         |
| Moran's I + LISA                     | Spatial autocorrelation is the novel angle                          |
| 10 random seeds                      | Reproducibility                                                     |

### Cut or Reduce

| Element                         | Why                                                                     |
| ------------------------------- | ----------------------------------------------------------------------- |
| IoU                             | Segmentation metric, not meaningful for continuous gap-filling          |
| ERGAS                           | Redundant with RMSE for same-resolution data; add only if reviewers ask |
| MODIS (60 patches)              | Include as exploratory only; descriptive stats, no hypothesis tests     |
| Auxiliary variables (NDVI, DEM) | Scope creep - save for a second paper                                   |
| Robust regression with VIF      | Overkill; simple correlation + ANOVA is sufficient                      |
| 500-2000 patch minimum          | Use all available patches per satellite, no arbitrary cap               |

### Fix

| Element                | Issue                                                                                                                      |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Mask convention        | Already inverted in preprocessing (1=gap). Consistent. Good.                                                               |
| Multi-temporal methods | Excluded. They need temporal stacks, not single patches. Out of scope for this paper.                                      |
| Gap fraction variance  | Need to report gap_fraction distribution and stratify results by gap size too (small/medium/large gaps), not just entropy. |

---

## Paper Structure (for reference during implementation)

1. **Introduction** - problem, motivation, gap in literature
2. **Related Work** - classical methods, DL methods, entropy in RS
3. **Study Area and Data** - Sentinel-2, Landsat-8/9, gap simulation
4. **Methodology**
   - 4.1 Gap-filling methods (Table 1: all 20 methods with parameters)
   - 4.2 Local entropy computation
   - 4.3 Evaluation metrics (PSNR, SSIM, RMSE, SAM)
   - 4.4 Statistical analysis framework
5. **Results**
   - 5.1 Overall method comparison (Table 2: mean +/- CI per method x noise)
   - 5.2 Entropy-performance correlation (Fig 2: scatterplots, Table 3: correlations)
   - 5.3 Spatial analysis (Fig 3: LISA maps)
   - 5.4 Cross-sensor comparison (Table 4: Sentinel-2 vs Landsat)
6. **Discussion**
7. **Conclusion**

---

## Implementation Phases

### Phase 1: Metrics Module

**File:** `src/pdi_pipeline/metrics.py`

Implement as pure functions (no classes needed):

```
psnr(clean, reconstructed, mask) -> float
ssim(clean, reconstructed, mask) -> float
rmse(clean, reconstructed, mask) -> float
sam(clean, reconstructed, mask) -> float          # multispectral only
local_psnr(clean, reconstructed, mask, window) -> np.ndarray   # per-pixel map
local_ssim(clean, reconstructed, mask, window) -> np.ndarray   # per-pixel map
```

Key decisions:

- Compute metrics ONLY on gap pixels (mask=1). Full-image PSNR dilutes
  the signal with untouched pixels. The mask is passed to every metric function.
- PSNR: 10 \* log10(1.0 / MSE) where MSE is over gap pixels only
- SSIM: use skimage.metrics.structural_similarity, but report the mean SSIM
  value restricted to the gap region
- RMSE: sqrt(mean((clean - recon)^2)) over gap pixels only
- SAM: spectral angle between clean and reconstructed vectors at gap pixels,
  requires (H,W,C) input. Returns mean angle in degrees.
- Local variants: sliding window, output same spatial dims as input.
  Used for LISA spatial analysis, not for per-patch reporting.

**Tests:** `tests/unit/test_metrics.py`

---

### Phase 2: Entropy Module

**File:** `src/pdi_pipeline/entropy.py`

```
shannon_entropy(image, window_size) -> np.ndarray   # per-pixel entropy map
```

Key decisions:

- Use skimage.filters.rank.entropy on uint8-quantized single band
- For multi-band: compute on the mean of all bands (not per-band average,
  which inflates entropy for correlated bands)
- Window sizes: 7, 15, 31 (odd numbers for symmetric windows)
- Output: float32 array, same (H, W) as input

**Tests:** `tests/unit/test_entropy.py`

---

### Phase 3: Configuration System

**File:** `src/pdi_pipeline/config.py`
**Files:** `config/paper_results.yaml`, `config/quick_validation.yaml`

Config schema:

```yaml
experiment:
  name: "paper_results"
  seeds: [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
  noise_levels: ["inf", "40", "30", "20"]
  satellites: ["sentinel2", "landsat8", "landsat9", "modis"]
  entropy_windows: [7, 15, 31]
  max_patches: null # null = use all
  output_dir: "results/"

methods:
  spatial:
    - { name: "nearest" }
    - { name: "bilinear" }
    - { name: "bicubic" }
    - { name: "lanczos", params: { a: 3 } }
  kernel:
    - { name: "idw", params: { power: 2.0 } }
    - { name: "rbf", params: { kernel: "thin_plate_spline" } }
    - { name: "spline" }
  geostatistical:
    - { name: "kriging" }
    - { name: "dineof" }
  transform:
    - { name: "dct" }
    - { name: "wavelet" }
    - { name: "tv", params: { max_iterations: 100 } }
  compressive:
    - { name: "l1_dct" }
    - { name: "l1_wavelet" }
  patch_based:
    - { name: "non_local" }
    - { name: "exemplar_based" }

metrics: ["psnr", "ssim", "rmse", "sam"]
```

Quick validation config: 1 seed, 1 noise level, 3 methods, 50 patches.

---

### Phase 4: Dataset Class

**File:** `src/pdi_pipeline/dataset.py`

```python
class PatchDataset:
    def __init__(self, manifest_path, split, satellite, noise_level): ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx) -> PatchSample: ...
```

Returns `PatchSample` dataclass: clean, degraded, mask, metadata (patch_id,
satellite, gap_fraction, acquisition_date).

Key decisions:

- Lazy loading (mmap or load-on-demand) - 77k patches x 64x64x4 x float32 =
  ~4.7 GB in memory, too much
- Filter by split/satellite/noise at init time, not per-access
- Deterministic ordering (sorted by patch_id)

**Tests:** `tests/unit/test_dataset.py`

---

### Phase 5: Experiment Runner

**File:** `scripts/run_experiment.py`

Loop structure:

```
for seed in config.seeds:
    for noise_level in config.noise_levels:
        dataset = PatchDataset(split="test", noise_level=noise_level)
        for method_cfg in config.methods:
            method = instantiate(method_cfg)
            for patch in dataset:
                result = method.apply(patch.degraded, patch.mask)
                metrics = compute_all_metrics(patch.clean, result, patch.mask)
                entropy = load_precomputed_entropy(patch.patch_id)
                save_row(seed, noise_level, method, patch.patch_id, metrics, entropy)
```

Output: `results/{experiment_name}/raw_results.parquet`

Columns: seed, noise_level, method, patch_id, satellite, gap_fraction,
entropy_7, entropy_15, entropy_31, psnr, ssim, rmse, sam

Key features:

- `--config path/to/config.yaml`
- `--quick` (override with quick_validation.yaml)
- `--dry-run` (print plan, don't execute)
- Checkpointing: write partial parquet every N patches, skip completed (seed, noise, method, patch_id) tuples on resume
- Progress bar with ETA
- Multi-temporal methods excluded from this paper (17 methods total)

**Tests:** `tests/integration/test_runner.py` (small end-to-end with 5 patches)

---

### Phase 6: Precompute Entropy Maps

**File:** extend `scripts/preprocess_dataset.py` or new `scripts/precompute_entropy.py`

For each clean patch, compute entropy at 3 window sizes, save as:

```
preprocessed/{split}/{satellite}/{patch_id:07d}_entropy_{window}.npy
```

Update manifest.csv with entropy columns: mean_entropy_7, mean_entropy_15, mean_entropy_31.

---

### Phase 7: Aggregation Module

**File:** `src/pdi_pipeline/aggregation.py`

```
load_results(path) -> pd.DataFrame
summary_by_method(df) -> pd.DataFrame          # mean, median, CI95% per method
summary_by_entropy_bin(df, window) -> pd.DataFrame  # stratified by low/med/high
summary_by_gap_fraction(df) -> pd.DataFrame    # stratified by gap size
```

Entropy bins: terciles (low = bottom 33%, medium = middle 33%, high = top 33%)
computed from the test set distribution. Not arbitrary thresholds.

---

### Phase 8: Statistical Analysis

**File:** `src/pdi_pipeline/statistics.py`

Functions:

```
correlation_analysis(df, entropy_col, metric_col) -> CorrelationResult
    # Pearson r, Spearman rho, p-values, FDR-corrected significance

method_comparison(df, metric_col) -> ComparisonResult
    # Kruskal-Wallis H, p-value (non-parametric - can't assume normality)
    # Dunn post-hoc with Bonferroni correction

spatial_autocorrelation(error_map, weights) -> SpatialResult
    # Moran's I (global), LISA clusters (local)
```

Key decisions:

- Use non-parametric tests by default (Kruskal-Wallis, not ANOVA). With 77k
  patches, normality assumptions will be violated.
- FDR correction via statsmodels.stats.multitest.multipletests
- Moran's I via esda.Moran, LISA via esda.Moran_Local

---

### Phase 9: Figure and Table Generation

**Files:** `scripts/generate_figures.py`, `scripts/generate_tables.py`

Figures (publication quality, matplotlib + seaborn):

1. Entropy map examples (3 scales, 2 satellites)
2. Scatterplot: entropy vs PSNR per method (3x6 grid or selected methods)
3. Boxplot: PSNR per method grouped by entropy bin
4. Boxplot: PSNR per method grouped by noise level
5. LISA cluster maps overlaid on error maps
6. Visual reconstruction examples (clean / degraded / top-3 methods / worst method)
7. Correlation heatmap: method x entropy_window x metric

Tables (LaTeX output):

1. Method overview (name, category, parameters, complexity)
2. Overall results: mean PSNR +/- CI95% per method x noise level
3. Entropy-stratified results: mean PSNR per method x entropy bin
4. Correlation matrix: Spearman rho (entropy x metric x method) with significance stars
5. Kruskal-Wallis + Dunn post-hoc summary

---

## Scope Decisions

**Multi-temporal methods:** Excluded entirely. TemporalSpline, TemporalFourier,
and SpaceTimeKriging require temporal stacks and do not fit the single-patch
experiment design. This keeps the paper focused on spatial single-frame methods.
The experiment runner must skip these 3 methods. Total: 17 methods.

**MODIS:** Include as exploratory (60 patches). Run experiments but present
results separately with explicit caveat about low sample size. No statistical
tests on MODIS alone - descriptive statistics only.

**Metric scope:** Gap pixels only (mask=1). Full-image metrics dilute the
signal. All PSNR/SSIM/RMSE/SAM values are computed strictly on the
interpolated region.

---

## Implementation Priority Order

```
Phase 1: Metrics          (unblocks everything)
Phase 2: Entropy          (unblocks Phase 6)
Phase 3: Config           (unblocks Phase 5)
Phase 4: Dataset          (unblocks Phase 5)
Phase 5: Experiment Runner (unblocks Phase 7-9)
Phase 6: Precompute Entropy
Phase 7: Aggregation
Phase 8: Statistics
Phase 9: Figures + Tables
```

Phases 1-2 are independent and can be implemented in parallel.
Phases 3-4 are independent and can be implemented in parallel.
Phases 7-9 depend on Phase 5 output.

---

## Verification

```bash
# After each phase, run ruff
uv run ruff check . --unsafe-fixes

# After Phase 5 (quick validation)
uv run python scripts/run_experiment.py --quick --dry-run
uv run python scripts/run_experiment.py --quick

# After Phase 9 (full pipeline)
make reproduce-quick
```
