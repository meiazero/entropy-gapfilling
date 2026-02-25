#!/bin/bash
# ---------------------------------------------------------------------------
# clear_experiments.sh - Remove experiment artifacts to allow fresh runs.
#
# What is cleared:
#   - results/quick_validation/
#   - results/paper_results/
#   - results/dl_models/checkpoints/
#   - results/dl_eval/
#   - results/dl_plots/
#   - dl_models/*_history.json
#   - logs/ (SLURM .log/.out/.err files)
#   - experiment.log, result.log, paper-latex.log (repo root)
#
# What is NOT cleared:
#   - preprocessed/   (expensive to regenerate)
#   - paper_assets/   (finalized archive for the paper)
#   - docs/           (paper source)
#
# Usage:
#   bash scripts/clear_experiments.sh              # interactive confirmation
#   bash scripts/clear_experiments.sh --dry-run    # show what would be removed
#   bash scripts/clear_experiments.sh --yes        # skip confirmation
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DRY_RUN=0
AUTO_YES=0

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --yes)     AUTO_YES=1 ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--dry-run] [--yes]"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

remove() {
    local target="$1"
    if [ ! -e "$target" ]; then
        return
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  [dry-run] rm -rf $target"
    else
        rm -rf "$target"
    fi
}

remove_glob() {
    local dir="$1"
    local pattern="$2"
    if [ ! -d "$dir" ]; then
        return
    fi
    local matches
    matches=$(find "$dir" -maxdepth 1 -name "$pattern" 2>/dev/null || true)
    if [ -z "$matches" ]; then
        return
    fi
    while IFS= read -r f; do
        if [ "$DRY_RUN" -eq 1 ]; then
            echo "  [dry-run] rm -f $f"
        else
            rm -f "$f"
        fi
    done <<< "$matches"
}

recreate_dir() {
    local dir="$1"
    if [ "$DRY_RUN" -eq 0 ] && [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        touch "$dir/.gitkeep"
    fi
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo "=== clear_experiments ==="
echo "Repo: $REPO_DIR"
[ "$DRY_RUN"   -eq 1 ] && echo "Mode: dry-run (no files will be deleted)"
[ "$AUTO_YES"  -eq 1 ] && echo "Mode: non-interactive"
echo ""
echo "Targets:"
echo "  results/quick_validation/"
echo "  results/paper_results/"
echo "  results/dl_models/checkpoints/"
echo "  results/dl_eval/"
echo "  results/dl_plots/"
echo "  dl_models/*_history.json"
echo "  logs/*.{log,out,err}"
echo "  experiment.log  result.log  paper-latex.log  (root)"
echo ""

# ---------------------------------------------------------------------------
# Confirmation
# ---------------------------------------------------------------------------

if [ "$DRY_RUN" -eq 0 ] && [ "$AUTO_YES" -eq 0 ]; then
    read -rp "Proceed? [y/N] " answer
    case "$answer" in
        [Yy]*) ;;
        *)
            echo "Aborted."
            exit 0
            ;;
    esac
fi

# ---------------------------------------------------------------------------
# Classical experiment results
# ---------------------------------------------------------------------------

echo "Clearing classical experiment results..."
remove "$REPO_DIR/results/quick_validation"
remove "$REPO_DIR/results/paper_results"

# ---------------------------------------------------------------------------
# Deep learning artifacts
# ---------------------------------------------------------------------------

echo "Clearing DL checkpoints..."
remove "$REPO_DIR/results/dl_models/checkpoints"
recreate_dir "$REPO_DIR/results/dl_models/checkpoints"

echo "Clearing DL evaluation results..."
remove "$REPO_DIR/results/dl_eval"
recreate_dir "$REPO_DIR/results/dl_eval"

echo "Clearing DL plots..."
remove "$REPO_DIR/results/dl_plots"
recreate_dir "$REPO_DIR/results/dl_plots"

echo "Clearing training history files..."
remove_glob "$REPO_DIR/dl_models" "*_history.json"

# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------

echo "Clearing SLURM logs..."
remove_glob "$REPO_DIR/logs" "*.log"
remove_glob "$REPO_DIR/logs" "*.out"
remove_glob "$REPO_DIR/logs" "*.err"

echo "Clearing root-level logs..."
remove_glob "$REPO_DIR" "experiment.log"
remove_glob "$REPO_DIR" "result.log"
remove_glob "$REPO_DIR" "paper-latex.log"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

echo ""
if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry-run complete. No files were deleted."
else
    echo "Done. Repository is ready for a fresh experiment run."
fi
