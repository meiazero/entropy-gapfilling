#!/usr/bin/env bash
# pack_paper_assets.sh - Pack experiment results into a structured .zip for
# local review and paper writing.
#
# Collects only the CSVs required by generate_figures.py and
# generate_tables.py, plus generated figures, tables, and training logs.
# Excludes intermediate checkpoints, raw parquet shards, and any CSV not
# consumed by the figure/table generation pipeline.
#
# Usage:
#   bash scripts/pack_paper_assets.sh
#   bash scripts/pack_paper_assets.sh --results results/paper_results
#   bash scripts/pack_paper_assets.sh \
#       --results results/quick_validation \
#       --dl-results results/dl_models \
#       --dl-plots results/dl_plots \
#       --output my_assets.zip
#
# Output structure:
#   classic/
#     raw_results/
#       raw_results.csv          - per-patch results for all classical methods
#     aggregated/
#       *.csv                    - analysis-ready CSVs from aggregate_results.py
#                                  (by_method, by_noise, by_satellite, by_gap_bin,
#                                   by_entropy_bin, spearman_correlation,
#                                   method_comparison_*, robust_regression_*,
#                                   combined_comparison, dl_*)
#     figures/                   - PDF and PNG figures from classical experiment
#     tables/                    - LaTeX .tex table files
#     logs/                      - experiment.log
#   dl/
#     eval/
#       {scenario}/{model}/
#         {model}_{noise_label}.csv  - full-schema test-set evaluation CSVs
#                                      (psnr, ssim, rmse, sam, ergas, rmse_b*,
#                                       pixel_acc_*, f1_*, entropy_*, gap_fraction)
#     history/
#       {model}_history.json     - per-epoch val metrics from training
#     figures/                   - DL figures from generate_figures.py
#     tables/                    - DL tables from generate_tables.py
#     training_plots/            - loss/metric curves from plot_training.py
#     training_logs/             - evaluate.log per model

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
RESULTS_DIR="results/paper_results"
DL_RESULTS_DIR="results/dl_models"
DL_PLOTS_DIR="results/dl_plots"
OUTPUT_ZIP="paper_assets.zip"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --results)      RESULTS_DIR="$2";    shift 2 ;;
        --dl-results)   DL_RESULTS_DIR="$2"; shift 2 ;;
        --dl-plots)     DL_PLOTS_DIR="$2";   shift 2 ;;
        --output)       OUTPUT_ZIP="$2";     shift 2 ;;
        -h|--help)
            sed -n '2,35p' "$0" | sed 's/^# //' | sed 's/^#//'
            exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Staging directory
# ---------------------------------------------------------------------------
STAGING_DIR="$(mktemp -d)"
cleanup() { rm -rf "$STAGING_DIR"; }
trap cleanup EXIT

mkdir -p \
    "$STAGING_DIR/classic/raw_results" \
    "$STAGING_DIR/classic/aggregated" \
    "$STAGING_DIR/classic/figures" \
    "$STAGING_DIR/classic/tables" \
    "$STAGING_DIR/classic/logs" \
    "$STAGING_DIR/dl/history" \
    "$STAGING_DIR/dl/figures" \
    "$STAGING_DIR/dl/tables" \
    "$STAGING_DIR/dl/training_plots" \
    "$STAGING_DIR/dl/training_logs"

_n_files=0

_copy_file() {
    local src="$1" dst="$2"
    if [ -f "$src" ]; then
        cp -- "$src" "$dst"
        _n_files=$((_n_files + 1))
    fi
}

_copy_glob() {
    local src_dir="$1" pattern="$2" dst_dir="$3"
    for f in "$src_dir"/$pattern; do
        [ -f "$f" ] || continue
        cp -- "$f" "$dst_dir/"
        _n_files=$((_n_files + 1))
    done
}

# ---------------------------------------------------------------------------
# Classical - raw per-patch results
# ---------------------------------------------------------------------------
echo "--- Classical: $RESULTS_DIR"

_copy_file "$RESULTS_DIR/raw_results.csv" "$STAGING_DIR/classic/raw_results/"

# ---------------------------------------------------------------------------
# Classical - aggregated CSVs (inputs to generate_figures.py / generate_tables.py)
# ---------------------------------------------------------------------------
if [ -d "$RESULTS_DIR/aggregated" ]; then
    _copy_glob "$RESULTS_DIR/aggregated" "*.csv" "$STAGING_DIR/classic/aggregated"
fi

# ---------------------------------------------------------------------------
# Classical - generated outputs (figures, tables, logs)
# ---------------------------------------------------------------------------
if [ -d "$RESULTS_DIR/figures" ]; then
    _copy_glob "$RESULTS_DIR/figures" "*.pdf" "$STAGING_DIR/classic/figures"
    _copy_glob "$RESULTS_DIR/figures" "*.png" "$STAGING_DIR/classic/figures"
fi

if [ -d "$RESULTS_DIR/tables" ]; then
    _copy_glob "$RESULTS_DIR/tables" "*.tex" "$STAGING_DIR/classic/tables"
fi

for f in "$RESULTS_DIR"/*.log; do
    [ -f "$f" ] && _copy_file "$f" "$STAGING_DIR/classic/logs/" || true
done

# ---------------------------------------------------------------------------
# DL - eval CSVs (full schema: psnr, ssim, rmse, sam, ergas, rmse_b*,
#                  pixel_acc_*, f1_*, entropy_*, gap_fraction, satellite)
#
# Path layout produced by evaluate.py (via train_model.py):
#   {DL_RESULTS_DIR}/eval/{scenario}/{model}/{model}_{noise_label}.csv
#
# Backward compat: old evaluator wrote {model}/results.csv (no scenario subdir).
# ---------------------------------------------------------------------------
echo "--- DL eval: $DL_RESULTS_DIR/eval"

if [ -d "$DL_RESULTS_DIR/eval" ]; then
    # Walk eval/ up to 3 levels deep, copying every *.csv while preserving
    # the relative path. Handles both layouts:
    #   new: eval/{scenario}/{model}/{model}_{noise_label}.csv
    #   old: eval/{model}/results.csv
    while IFS= read -r csv_file; do
        rel="${csv_file#$DL_RESULTS_DIR/eval/}"
        dst_dir="$STAGING_DIR/dl/eval/$(dirname "$rel")"
        mkdir -p "$dst_dir"
        _copy_file "$csv_file" "$dst_dir/"
    done < <(find "$DL_RESULTS_DIR/eval" -maxdepth 3 -name "*.csv" -type f 2>/dev/null)

    # evaluate.log files alongside the CSVs
    while IFS= read -r log_file; do
        rel="${log_file#$DL_RESULTS_DIR/eval/}"
        # flatten to training_logs/ with path encoded in filename
        flat_name="$(echo "$rel" | tr '/' '_')"
        _copy_file "$log_file" "$STAGING_DIR/dl/training_logs/$flat_name"
    done < <(find "$DL_RESULTS_DIR/eval" -maxdepth 3 -name "evaluate.log" -type f 2>/dev/null)
fi

# ---------------------------------------------------------------------------
# DL - training history JSONs (per-epoch val metrics consumed by generate_*.py)
# ---------------------------------------------------------------------------
echo "--- DL history: $DL_RESULTS_DIR/checkpoints"

if [ -d "$DL_RESULTS_DIR/checkpoints" ]; then
    for f in "$DL_RESULTS_DIR/checkpoints"/*_history.json; do
        [ -f "$f" ] && _copy_file "$f" "$STAGING_DIR/dl/history/" || true
    done
fi
# Backward compat: history JSON directly under DL_RESULTS_DIR
for f in "$DL_RESULTS_DIR"/*_history.json; do
    [ -f "$f" ] && _copy_file "$f" "$STAGING_DIR/dl/history/" || true
done

# ---------------------------------------------------------------------------
# DL - generated outputs (figures, tables, training plots, training logs)
# ---------------------------------------------------------------------------
if [ -d "$DL_RESULTS_DIR/figures" ]; then
    _copy_glob "$DL_RESULTS_DIR/figures" "*.pdf" "$STAGING_DIR/dl/figures"
    _copy_glob "$DL_RESULTS_DIR/figures" "*.png" "$STAGING_DIR/dl/figures"
fi

if [ -d "$DL_RESULTS_DIR/tables" ]; then
    _copy_glob "$DL_RESULTS_DIR/tables" "*.tex" "$STAGING_DIR/dl/tables"
fi

if [ -d "$DL_PLOTS_DIR" ]; then
    _copy_glob "$DL_PLOTS_DIR" "*.pdf" "$STAGING_DIR/dl/training_plots"
    _copy_glob "$DL_PLOTS_DIR" "*.png" "$STAGING_DIR/dl/training_plots"
fi

# Training logs from checkpoints/
if [ -d "$DL_RESULTS_DIR/checkpoints" ]; then
    for f in "$DL_RESULTS_DIR/checkpoints"/*_train.log; do
        [ -f "$f" ] && _copy_file "$f" "$STAGING_DIR/dl/training_logs/" || true
    done
fi
# Backward compat
for f in "$DL_RESULTS_DIR"/*_train.log; do
    [ -f "$f" ] && _copy_file "$f" "$STAGING_DIR/dl/training_logs/" || true
done

# ---------------------------------------------------------------------------
# Create zip
# ---------------------------------------------------------------------------
if [ "$_n_files" -eq 0 ]; then
    echo ""
    echo "WARNING: no files found to pack. Check --results and --dl-results paths."
    echo "  Expected classical results at: $RESULTS_DIR/raw_results.csv"
    echo "  Expected aggregated CSVs at:   $RESULTS_DIR/aggregated/"
    echo "  Expected DL eval CSVs at:      $DL_RESULTS_DIR/eval/"
    echo "  Expected DL history at:        $DL_RESULTS_DIR/checkpoints/*_history.json"
fi

pushd "$STAGING_DIR" > /dev/null
zip -r "$OLDPWD/$OUTPUT_ZIP" classic/ dl/ --quiet
popd > /dev/null

echo ""
echo "Created: $OUTPUT_ZIP  ($_n_files files)"
echo ""
echo "Contents:"
unzip -l "$OUTPUT_ZIP" | tail -n +4 | head -80
