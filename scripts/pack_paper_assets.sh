#!/usr/bin/env bash
# pack_paper_assets.sh - Pack experiment results into a structured .zip for
# local review and paper writing.
#
# Collects only the CSVs and JSONs needed to regenerate figures and tables
# locally. Excludes generated figures, tables, training plots, intermediate
# checkpoints, and raw parquet shards.
#
# Usage:
#   bash scripts/pack_paper_assets.sh
#   bash scripts/pack_paper_assets.sh --results results/paper_results
#   bash scripts/pack_paper_assets.sh \
#       --results results/quick_validation \
#       --dl-results results/dl_models \
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
#     logs/                      - experiment.log
#   dl/
#     eval/
#       {scenario}/{model}/
#         {model}_{noise_label}.csv  - full-schema test-set evaluation CSVs
#                                      (psnr, ssim, rmse, sam, ergas, rmse_b*,
#                                       pixel_acc_*, f1_*, entropy_*, gap_fraction)
#     history/
#       {model}_history.json     - per-epoch val metrics from training
#     training_logs/             - evaluate.log and train.log per model

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
RESULTS_DIR="results/paper_results"
DL_RESULTS_DIR="results/dl_models"
OUTPUT_ZIP="paper_assets.zip"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --results)      RESULTS_DIR="$2";    shift 2 ;;
        --dl-results)   DL_RESULTS_DIR="$2"; shift 2 ;;
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

CLASSIC_STAGE_DIR="$STAGING_DIR/classic"
DL_STAGE_DIR="$STAGING_DIR/dl"

CLASSIC_RESULTS_DIR="$RESULTS_DIR"
DL_RESULTS_ROOT="$DL_RESULTS_DIR"

mkdir -p \
    "$CLASSIC_STAGE_DIR/raw_results" \
    "$CLASSIC_STAGE_DIR/aggregated" \
    "$CLASSIC_STAGE_DIR/logs" \
    "$CLASSIC_STAGE_DIR/full_results" \
    "$DL_STAGE_DIR/history" \
    "$DL_STAGE_DIR/training_logs"

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
echo "--- Classical: $CLASSIC_RESULTS_DIR"

_copy_file "$CLASSIC_RESULTS_DIR/raw_results.csv" "$CLASSIC_STAGE_DIR/raw_results/"

# ---------------------------------------------------------------------------
# Classical - aggregated CSVs (inputs to generate_figures.py / generate_tables.py)
# ---------------------------------------------------------------------------
if [ -d "$CLASSIC_RESULTS_DIR/aggregated" ]; then
    _copy_glob "$CLASSIC_RESULTS_DIR/aggregated" "*.csv" "$CLASSIC_STAGE_DIR/aggregated"
fi

# ---------------------------------------------------------------------------
# Classical - logs
# ---------------------------------------------------------------------------
for f in "$CLASSIC_RESULTS_DIR"/*.log; do
    [ -f "$f" ] && _copy_file "$f" "$CLASSIC_STAGE_DIR/logs/" || true
done

# ---------------------------------------------------------------------------
# Classical - copy everything generated (preserve relative paths)
# ---------------------------------------------------------------------------
if [ -d "$CLASSIC_RESULTS_DIR" ]; then
    while IFS= read -r file_path; do
        rel="${file_path#$CLASSIC_RESULTS_DIR/}"
        dst_dir="$CLASSIC_STAGE_DIR/full_results/$(dirname "$rel")"
        mkdir -p "$dst_dir"
        _copy_file "$file_path" "$dst_dir/"
    done < <(find "$CLASSIC_RESULTS_DIR" -type f 2>/dev/null)
fi

# ---------------------------------------------------------------------------
# DL - eval CSVs (full schema: psnr, ssim, rmse, sam, ergas, rmse_b*,
#                  pixel_acc_*, f1_*, entropy_*, gap_fraction, satellite)
#
# Path layout produced by evaluate.py (via train_model.py):
#   {DL_RESULTS_DIR}/eval/{scenario}/{model}/{model}_{noise_label}.csv
#
# Backward compat: old evaluator wrote {model}/results.csv (no scenario subdir).
# ---------------------------------------------------------------------------
echo "--- DL eval: $DL_RESULTS_ROOT/eval"

if [ -d "$DL_RESULTS_ROOT/eval" ]; then
    # Walk eval/ recursively, copying every *.csv while preserving
    # the relative path. Handles both layouts:
    #   new: eval/{scenario}/{model}/{model}_{noise_label}.csv
    #   old: eval/{model}/results.csv
    while IFS= read -r csv_file; do
        rel="${csv_file#$DL_RESULTS_ROOT/eval/}"
        dst_dir="$DL_STAGE_DIR/eval/$(dirname "$rel")"
        mkdir -p "$dst_dir"
        _copy_file "$csv_file" "$dst_dir/"
    done < <(find "$DL_RESULTS_ROOT/eval" -type f -name "*.csv" 2>/dev/null)

    # evaluate.log files alongside the CSVs
    while IFS= read -r log_file; do
        rel="${log_file#$DL_RESULTS_ROOT/eval/}"
        # flatten to training_logs/ with path encoded in filename
        flat_name="$(echo "$rel" | tr '/' '_')"
        _copy_file "$log_file" "$DL_STAGE_DIR/training_logs/$flat_name"
    done < <(find "$DL_RESULTS_ROOT/eval" -type f -name "evaluate.log" 2>/dev/null)
fi

# ---------------------------------------------------------------------------
# DL - training history JSONs (per-epoch val metrics consumed by generate_*.py)
# History files may be at checkpoints/ (flat) or checkpoints/{scenario}/ (new).
# ---------------------------------------------------------------------------
echo "--- DL history: $DL_RESULTS_ROOT/checkpoints"

if [ -d "$DL_RESULTS_ROOT/checkpoints" ]; then
    while IFS= read -r hist_file; do
        rel="${hist_file#$DL_RESULTS_ROOT/checkpoints/}"
        # Flatten: entropy_high/ae_history.json -> entropy_high_ae_history.json
        flat_name="$(echo "$rel" | tr '/' '_')"
        _copy_file "$hist_file" "$DL_STAGE_DIR/history/$flat_name"
    done < <(find "$DL_RESULTS_ROOT/checkpoints" -type f -name "*_history.json" 2>/dev/null)
fi
# Backward compat: history JSON directly under DL_RESULTS_DIR
for f in "$DL_RESULTS_ROOT"/*_history.json; do
    [ -f "$f" ] && _copy_file "$f" "$DL_STAGE_DIR/history/" || true
done

# ---------------------------------------------------------------------------
# DL - training logs
# Logs may be at checkpoints/ (flat) or checkpoints/{scenario}/ (new).
# ---------------------------------------------------------------------------
if [ -d "$DL_RESULTS_ROOT/checkpoints" ]; then
    while IFS= read -r log_file; do
        rel="${log_file#$DL_RESULTS_ROOT/checkpoints/}"
        flat_name="$(echo "$rel" | tr '/' '_')"
        _copy_file "$log_file" "$DL_STAGE_DIR/training_logs/$flat_name"
    done < <(find "$DL_RESULTS_ROOT/checkpoints" -type f -name "*_train.log" 2>/dev/null)
fi
# Backward compat
for f in "$DL_RESULTS_ROOT"/*_train.log; do
    [ -f "$f" ] && _copy_file "$f" "$DL_STAGE_DIR/training_logs/" || true
done

# ---------------------------------------------------------------------------
# Create zip
# ---------------------------------------------------------------------------
if [ "$_n_files" -eq 0 ]; then
    echo ""
    echo "WARNING: no files found to pack. Check --results and --dl-results paths."
    echo "  Expected classical results at: $CLASSIC_RESULTS_DIR/raw_results.csv"
    echo "  Expected aggregated CSVs at:   $CLASSIC_RESULTS_DIR/aggregated/"
    echo "  Expected DL eval CSVs at:      $DL_RESULTS_ROOT/eval/"
    echo "  Expected DL history at:        $DL_RESULTS_ROOT/checkpoints/*_history.json"
fi

pushd "$STAGING_DIR" > /dev/null
zip -r "$OLDPWD/$OUTPUT_ZIP" classic/ dl/ --quiet
popd > /dev/null

echo ""
echo "Created: $OUTPUT_ZIP  ($_n_files files)"
echo ""
echo "Contents:"
unzip -l "$OUTPUT_ZIP" | tail -n +4 | head -80
