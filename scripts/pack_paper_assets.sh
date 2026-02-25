#!/usr/bin/env bash
# pack_paper_assets.sh - Pack experiment results into a structured .zip for
# local review and paper writing.
#
# Collects figures, tables, logs, and CSVs from classical and DL experiments,
# separating them into classic/ and dl/ subdirectories.
#
# Usage:
#   bash scripts/pack_paper_assets.sh
#   bash scripts/pack_paper_assets.sh --results results/paper_results
#   bash scripts/pack_paper_assets.sh --results results/quick_validation \
#       --dl-results results/dl_models --dl-plots results/dl_plots \
#       --output my_assets.zip
#
# Output structure:
#   classic/
#     figures/     - PDF and PNG figures from classical experiment
#     tables/      - LaTeX .tex table files
#     logs/        - experiment.log and any other .log files
#     raw_results/ - raw_results.csv / parquet checkpoints
#   dl/
#     figures/     - Figures generated from DL eval results
#     training_plots/ - Training curve plots (loss, psnr, ssim, etc.)
#     training_logs/  - Per-model training log files
#     eval_results/   - Per-model results.csv from evaluation
#     history/        - JSON training history files

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
            sed -n '2,30p' "$0" | sed 's/^# //' | sed 's/^#//'
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
    "$STAGING_DIR/classic/figures" \
    "$STAGING_DIR/classic/tables" \
    "$STAGING_DIR/classic/logs" \
    "$STAGING_DIR/classic/raw_results" \
    "$STAGING_DIR/dl/figures" \
    "$STAGING_DIR/dl/tables" \
    "$STAGING_DIR/dl/training_plots" \
    "$STAGING_DIR/dl/training_logs" \
    "$STAGING_DIR/dl/eval_results" \
    "$STAGING_DIR/dl/history"

_copied=0

_copy_glob() {
    local src_dir="$1" pattern="$2" dst_dir="$3"
    for f in "$src_dir"/$pattern; do
        [ -f "$f" ] || continue
        cp -- "$f" "$dst_dir/"
        _copied=1
    done
}

# ---------------------------------------------------------------------------
# Classical results
# ---------------------------------------------------------------------------
echo "Collecting classical results from: $RESULTS_DIR"

if [ -d "$RESULTS_DIR/figures" ]; then
    _copy_glob "$RESULTS_DIR/figures" "*.pdf" "$STAGING_DIR/classic/figures"
    _copy_glob "$RESULTS_DIR/figures" "*.png" "$STAGING_DIR/classic/figures"
fi

if [ -d "$RESULTS_DIR/tables" ]; then
    _copy_glob "$RESULTS_DIR/tables" "*.tex" "$STAGING_DIR/classic/tables"
fi

# Logs
for f in "$RESULTS_DIR"/*.log; do
    [ -f "$f" ] && cp "$f" "$STAGING_DIR/classic/logs/" && _copied=1 || true
done

# Raw results (CSV and Parquet checkpoints)
for f in "$RESULTS_DIR"/raw_results.* "$RESULTS_DIR"/*.csv "$RESULTS_DIR"/*.parquet; do
    [ -f "$f" ] && cp "$f" "$STAGING_DIR/classic/raw_results/" && _copied=1 || true
done

# ---------------------------------------------------------------------------
# DL results
# ---------------------------------------------------------------------------
echo "Collecting DL results from: $DL_RESULTS_DIR"

# DL figures (from generate_figures.py --dl-results)
if [ -d "$DL_RESULTS_DIR/figures" ]; then
    _copy_glob "$DL_RESULTS_DIR/figures" "*.pdf" "$STAGING_DIR/dl/figures"
    _copy_glob "$DL_RESULTS_DIR/figures" "*.png" "$STAGING_DIR/dl/figures"
fi

# DL tables (from generate_tables.py --dl-results)
if [ -d "$DL_RESULTS_DIR/tables" ]; then
    _copy_glob "$DL_RESULTS_DIR/tables" "*.tex" "$STAGING_DIR/dl/tables"
fi

# DL training plots (from dl_models.plot_training)
if [ -d "$DL_PLOTS_DIR" ]; then
    _copy_glob "$DL_PLOTS_DIR" "*.pdf" "$STAGING_DIR/dl/training_plots"
    _copy_glob "$DL_PLOTS_DIR" "*.png" "$STAGING_DIR/dl/training_plots"
fi

# Per-model training logs (*_train.log) and history JSON
for f in "$DL_RESULTS_DIR"/*_train.log; do
    [ -f "$f" ] && cp "$f" "$STAGING_DIR/dl/training_logs/" && _copied=1 || true
done
for f in "$DL_RESULTS_DIR"/*_history.json; do
    [ -f "$f" ] && cp "$f" "$STAGING_DIR/dl/history/" && _copied=1 || true
done

# Per-model eval results (dl_models/eval/<model>/results.csv)
if [ -d "$DL_RESULTS_DIR/eval" ]; then
    for model_dir in "$DL_RESULTS_DIR/eval"/*/; do
        [ -d "$model_dir" ] || continue
        model_name="$(basename "$model_dir")"
        if [ -f "$model_dir/results.csv" ]; then
            cp "$model_dir/results.csv" \
                "$STAGING_DIR/dl/eval_results/${model_name}_results.csv"
            _copied=1
        fi
        if [ -f "$model_dir/evaluate.log" ]; then
            cp "$model_dir/evaluate.log" \
                "$STAGING_DIR/dl/training_logs/${model_name}_evaluate.log"
        fi
    done
fi

# ---------------------------------------------------------------------------
# Create zip
# ---------------------------------------------------------------------------
if [ "$_copied" -eq 0 ]; then
    echo "WARNING: no files found to pack. Check --results and --dl-results paths."
fi

pushd "$STAGING_DIR" > /dev/null
zip -r "$OLDPWD/$OUTPUT_ZIP" classic/ dl/ --quiet
popd > /dev/null

echo ""
echo "Created: $OUTPUT_ZIP"
echo ""
echo "Contents:"
unzip -l "$OUTPUT_ZIP" | tail -n +4 | head -60
