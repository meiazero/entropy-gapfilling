"""Aggregate experiment results into analysis-ready CSVs.

Reads raw per-patch CSVs from both the classical and DL pipelines,
applies all aggregations (by method, entropy bin, gap bin, satellite,
noise level), runs the statistical analysis suite, and exports one CSV
per analysis dimension.

After this script finishes, generate_figures.py and generate_tables.py
can be run without any experiment data - they read only from the
aggregated/ directory.

Usage:
    uv run python scripts/aggregate_results.py \\
        --results results/paper_results \\
        --dl-eval results/dl_eval \\
        --dl-history results/dl_models \\
        --output results/paper_results/aggregated

    # Omit --dl-eval / --dl-history if DL results are not available yet.
    uv run python scripts/aggregate_results.py \\
        --results results/paper_results
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pdi_pipeline.aggregation import (
    _bootstrap_ci,
    _summary_with_ci,
    load_results,
    summary_by_entropy_bin,
    summary_by_gap_fraction,
    summary_by_noise,
    summary_by_satellite,
)
from pdi_pipeline.logging_utils import setup_logging
from pdi_pipeline.statistics import (
    correlation_matrix,
    method_comparison,
    robust_regression,
)

setup_logging()
log = logging.getLogger(__name__)

# Metrics to aggregate for classical methods.
_CLASSICAL_METRICS = [
    "psnr",
    "ssim",
    "rmse",
    "sam",
    "ergas",
    "f1_002",
    "f1_005",
    "f1_01",
    "pixel_acc_002",
    "pixel_acc_005",
    "pixel_acc_01",
    "elapsed_s",
]

# Metrics shared between classical and DL (for combined comparison).
_SHARED_METRICS = ["psnr", "ssim", "rmse", "sam", "ergas"]

# Metrics for entropy / Spearman analysis.
_CORR_METRICS = ["psnr", "ssim", "rmse", "sam", "ergas"]

DL_MODELS = ("ae", "vae", "gan", "unet", "vit")
_NOISE_LEVELS = ("inf", "40", "30", "20")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_classical(results_dir: Path) -> pd.DataFrame:
    """Load classical raw_results.csv and add ``type`` column."""
    df = load_results(results_dir)
    df["type"] = "classical"
    return df


def _load_dl_eval(dl_eval_dir: Path) -> pd.DataFrame:  # noqa: C901
    """Load all DL eval CSVs from ``dl_eval/{model}/eval_{noise}.csv``."""
    frames: list[pd.DataFrame] = []
    for model in DL_MODELS:
        model_dir = dl_eval_dir / model
        if not model_dir.exists():
            log.warning("DL eval dir not found: %s", model_dir)
            continue
        for noise in _NOISE_LEVELS:
            csv_path = model_dir / f"eval_{noise}.csv"
            if not csv_path.exists():
                # Backward compat: old script wrote results.csv
                # (no noise suffix).
                legacy = model_dir / "results.csv"
                if legacy.exists():
                    df = pd.read_csv(legacy)
                    if "noise_level" not in df.columns:
                        df["noise_level"] = "inf"
                    frames.append(df)
                    log.debug("Loaded legacy DL eval: %s", legacy)
                    break
                continue
            df = pd.read_csv(csv_path)
            frames.append(df)
            log.debug("Loaded DL eval: %s", csv_path)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["type"] = "dl"
    # Rename 'model' to 'method' for uniform aggregation.
    if "model" in combined.columns and "method" not in combined.columns:
        combined = combined.rename(columns={"model": "method"})
    # Rename 'architecture' to 'method_category' for uniform schema.
    if (
        "architecture" in combined.columns
        and "method_category" not in combined.columns
    ):
        combined = combined.rename(columns={"architecture": "method_category"})
    # Rename entropy columns to match classical schema.
    for ws in (7, 15, 31):
        old = f"mean_entropy_{ws}"
        new = f"entropy_{ws}"
        if old in combined.columns and new not in combined.columns:
            combined = combined.rename(columns={old: new})
    return combined


def _load_dl_history(dl_history_dir: Path) -> pd.DataFrame:
    """Flatten all DL JSON training histories into a per-epoch CSV."""
    rows: list[dict] = []
    for model in DL_MODELS:
        path = dl_history_dir / f"{model}_history.json"
        if not path.exists():
            log.warning("DL history not found: %s", path)
            continue
        data = json.loads(path.read_text())
        model_name = data.get("model_name", model)
        for epoch_data in data.get("epochs", []):
            row = {"model": model_name}
            row.update(epoch_data)
            rows.append(row)
        log.debug(
            "Loaded DL history: %s (%d epochs)",
            model_name,
            len(data.get("epochs", [])),
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _load_dl_metadata(dl_history_dir: Path) -> pd.DataFrame:
    """Extract per-model metadata (n_params, training_time_s, best_epoch)."""
    rows: list[dict] = []
    for model in DL_MODELS:
        path = dl_history_dir / f"{model}_history.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        meta = data.get("metadata", {})
        epochs = data.get("epochs", [])
        # Best epoch: lowest val_loss.
        best_epoch = None
        best_val_loss = float("inf")
        for ep in epochs:
            vl = ep.get("val_loss")
            if vl is not None and vl < best_val_loss:
                best_val_loss = vl
                best_epoch = ep.get("epoch")
        row = {
            "model": data.get("model_name", model),
            "n_params": meta.get("n_params"),
            "training_time_s": meta.get("training_time_s"),
            "best_epoch": best_epoch,
            "best_val_loss": (
                best_val_loss if best_epoch is not None else float("nan")
            ),
            "total_epochs": len(epochs),
        }
        row.update({
            k: v
            for k, v in meta.items()
            if k not in ("n_params", "training_time_s")
        })
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _safe_metrics(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Return only metrics that exist as non-all-NaN columns."""
    return [m for m in candidates if m in df.columns and df[m].notna().any()]


def _aggregate_classical(
    df: pd.DataFrame,
    output_dir: Path,
    metrics: list[str],
) -> None:
    """Run all classical aggregation dimensions and save CSVs."""
    log.info("Aggregating classical results (%d rows) ...", len(df))
    entropy_cols = sorted(c for c in df.columns if c.startswith("entropy_"))
    available_metrics = _safe_metrics(df, metrics)

    for metric in available_metrics:
        log.info("  by_method:   %s", metric)
        summary_by_method_full(df, metric).to_csv(
            output_dir / f"by_method_{metric}.csv", index=False
        )

        log.info("  by_noise:    %s", metric)
        summary_by_noise(df, metric).to_csv(
            output_dir / f"by_noise_{metric}.csv", index=False
        )

        log.info("  by_satellite: %s", metric)
        summary_by_satellite(df, metric).to_csv(
            output_dir / f"by_satellite_{metric}.csv", index=False
        )

        log.info("  by_gap_bin:  %s", metric)
        summary_by_gap_fraction(df, metric).to_csv(
            output_dir / f"by_gap_bin_{metric}.csv", index=False
        )

        for ecol in entropy_cols:
            ws = ecol.split("_")[-1]
            log.info("  by_entropy_bin (w=%s): %s", ws, metric)
            summary_by_entropy_bin(df, ecol, metric).to_csv(
                output_dir / f"by_entropy_bin_{ws}_{metric}.csv", index=False
            )

    # Statistical analysis on psnr (primary metric).
    corr_metrics = _safe_metrics(df, _CORR_METRICS)
    if entropy_cols and corr_metrics:
        log.info("Running Spearman correlation matrix ...")
        corr_df = correlation_matrix(df, entropy_cols, corr_metrics)
        corr_df.to_csv(output_dir / "spearman_correlation.csv", index=False)

    log.info("Running method comparison (Kruskal-Wallis) ...")
    primary = (
        "psnr"
        if "psnr" in df.columns
        else (available_metrics[0] if available_metrics else None)
    )
    if primary:
        comp = method_comparison(df, primary)
        pd.DataFrame([
            {
                "metric": primary,
                "kruskal_H": comp.statistic,
                "kruskal_p": comp.p_value,
                "n_groups": comp.n_groups,
                "epsilon_squared": comp.epsilon_squared,
            }
        ]).to_csv(output_dir / "method_comparison_global.csv", index=False)
        if not comp.posthoc.empty:
            comp.posthoc.to_csv(
                output_dir / "method_comparison_pairwise.csv", index=False
            )

    log.info("Running robust regression ...")
    if entropy_cols and primary and primary in df.columns:
        try:
            reg = robust_regression(df, primary, entropy_cols)
            reg.coefficients.to_csv(
                output_dir / "robust_regression_coefs.csv", index=False
            )
            reg.vif.to_csv(
                output_dir / "robust_regression_vif.csv", index=False
            )
            pd.DataFrame([
                {
                    "metric": primary,
                    "r_squared_adj": reg.r_squared_adj,
                    "n": reg.n,
                    "model_type": reg.model_type,
                }
            ]).to_csv(output_dir / "robust_regression_summary.csv", index=False)
        except Exception:
            log.exception("Robust regression failed - skipping")


def _aggregate_dl_eval(dl_df: pd.DataFrame, output_dir: Path) -> None:  # noqa: C901
    """Aggregate DL evaluation results by model x satellite x noise_level."""
    if dl_df.empty:
        log.info("No DL eval data - skipping DL aggregation")
        return

    log.info("Aggregating DL eval results (%d rows) ...", len(dl_df))
    available_metrics = _safe_metrics(dl_df, _CLASSICAL_METRICS)
    entropy_cols = sorted(c for c in dl_df.columns if c.startswith("entropy_"))

    # Per-model x satellite x noise_level summary for each metric.
    group_cols = ["method"]
    if "satellite" in dl_df.columns:
        group_cols.append("satellite")
    if "noise_level" in dl_df.columns:
        group_cols.append("noise_level")

    summary_rows: list[dict] = []
    for keys, grp in dl_df.groupby(group_cols, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict = dict(zip(group_cols, keys, strict=False))
        row["n"] = len(grp)
        for metric in available_metrics:
            vals = grp[metric].dropna().to_numpy()
            if len(vals) == 0:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_std"] = float("nan")
                row[f"{metric}_ci95_lo"] = float("nan")
                row[f"{metric}_ci95_hi"] = float("nan")
            else:
                ci_lo, ci_hi = _bootstrap_ci(vals)
                row[f"{metric}_mean"] = float(np.mean(vals))
                row[f"{metric}_std"] = float(np.std(vals))
                row[f"{metric}_ci95_lo"] = ci_lo
                row[f"{metric}_ci95_hi"] = ci_hi
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "dl_eval_summary.csv", index=False
    )

    # Entropy-PSNR correlation for DL models.
    corr_metrics = _safe_metrics(dl_df, _CORR_METRICS)
    if entropy_cols and corr_metrics and "method" in dl_df.columns:
        log.info("Running Spearman correlation for DL models ...")
        dl_corr = correlation_matrix(dl_df, entropy_cols, corr_metrics)
        dl_corr.to_csv(output_dir / "dl_spearman_correlation.csv", index=False)

    # By-entropy-bin for DL (same tercile approach as classical).
    for metric in _safe_metrics(dl_df, ["psnr", "ssim", "rmse"]):
        for ecol in entropy_cols:
            ws = ecol.split("_")[-1]
            try:
                summary_by_entropy_bin(dl_df, ecol, metric).to_csv(
                    output_dir / f"dl_by_entropy_bin_{ws}_{metric}.csv",
                    index=False,
                )
            except Exception:
                log.debug(
                    "DL entropy bin aggregation failed for %s/%s",
                    ecol,
                    metric,
                )


def _build_combined(
    classical_df: pd.DataFrame,
    dl_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Build combined classical + DL comparison CSV on shared metrics."""
    if classical_df.empty and dl_df.empty:
        return

    frames = []
    if not classical_df.empty:
        frames.append(classical_df.assign(type="classical"))
    if not dl_df.empty:
        frames.append(dl_df.assign(type="dl"))

    combined = pd.concat(frames, ignore_index=True)
    shared = _safe_metrics(combined, _SHARED_METRICS)
    if not shared:
        log.warning("No shared metrics available for combined comparison")
        return

    log.info("Building combined classical+DL comparison table ...")

    # Per-method summary on shared metrics for all noise levels combined.
    rows: list[dict] = []
    group_cols = ["type", "method", "noise_level"]
    actual_group = [c for c in group_cols if c in combined.columns]

    for keys, grp in combined.groupby(actual_group, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict = dict(zip(actual_group, keys, strict=False))
        row["n"] = len(grp)
        for metric in shared:
            vals = grp[metric].dropna().to_numpy()
            if len(vals) == 0:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_ci95_lo"] = float("nan")
                row[f"{metric}_ci95_hi"] = float("nan")
            else:
                ci_lo, ci_hi = _bootstrap_ci(vals)
                row[f"{metric}_mean"] = float(np.mean(vals))
                row[f"{metric}_ci95_lo"] = ci_lo
                row[f"{metric}_ci95_hi"] = ci_hi
        rows.append(row)

    pd.DataFrame(rows).to_csv(
        output_dir / "combined_comparison.csv", index=False
    )


# ---------------------------------------------------------------------------
# summary_by_method with full spread (not just in aggregation.py)
# ---------------------------------------------------------------------------


def summary_by_method_full(
    df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Per-method summary with mean, median, std, and 95% CI."""
    return _summary_with_ci(
        df,
        ["method"],
        metric,
        include_spread=True,
        sort_cols=["mean"],
    ).sort_values("mean", ascending=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate experiment results into analysis-ready CSVs.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Classical results directory (containing raw_results.csv).",
    )
    parser.add_argument(
        "--dl-eval",
        type=Path,
        default=None,
        help="DL evaluation root (containing {model}/eval_{noise}.csv).",
    )
    parser.add_argument(
        "--dl-history",
        type=Path,
        default=None,
        help="DL training history dir (containing {model}_history.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for aggregated CSVs. "
        "Defaults to {results}/aggregated.",
    )
    return parser


def main() -> None:  # noqa: C901
    args = _build_parser().parse_args()

    if args.results is None and args.dl_eval is None:
        log.error("At least one of --results or --dl-eval must be provided.")
        return

    output_dir: Path
    if args.output is not None:
        output_dir = args.output
    elif args.results is not None:
        output_dir = args.results / "aggregated"
    else:
        output_dir = args.dl_eval / "aggregated"

    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", output_dir)

    # --- Classical pipeline ---
    classical_df = pd.DataFrame()
    if args.results is not None:
        (
            args.results / "raw_results.csv"
            if (args.results / "raw_results.csv").exists()
            else args.results
        )
        try:
            classical_df = _load_classical(args.results)
            log.info("Loaded classical results: %d rows", len(classical_df))
            _aggregate_classical(classical_df, output_dir, _CLASSICAL_METRICS)
        except Exception:
            log.exception(
                "Failed to load classical results from %s", args.results
            )

    # --- DL evaluation ---
    dl_df = pd.DataFrame()
    if args.dl_eval is not None:
        try:
            dl_df = _load_dl_eval(args.dl_eval)
            log.info("Loaded DL eval results: %d rows", len(dl_df))
            _aggregate_dl_eval(dl_df, output_dir)
        except Exception:
            log.exception("Failed to load DL eval from %s", args.dl_eval)

    # --- DL training history ---
    if args.dl_history is not None:
        try:
            hist_df = _load_dl_history(args.dl_history)
            if not hist_df.empty:
                hist_df.to_csv(
                    output_dir / "dl_training_history.csv", index=False
                )
                log.info(
                    "Saved dl_training_history.csv (%d epoch rows)",
                    len(hist_df),
                )
        except Exception:
            log.exception("Failed to load DL history from %s", args.dl_history)

        try:
            meta_df = _load_dl_metadata(args.dl_history)
            if not meta_df.empty:
                meta_df.to_csv(
                    output_dir / "dl_model_metadata.csv", index=False
                )
                log.info("Saved dl_model_metadata.csv")
        except Exception:
            log.exception("Failed to load DL metadata from %s", args.dl_history)

    # --- Combined comparison ---
    if not classical_df.empty or not dl_df.empty:
        try:
            _build_combined(classical_df, dl_df, output_dir)
        except Exception:
            log.exception("Failed to build combined comparison")

    log.info("Done. CSVs written to: %s", output_dir)
    for p in sorted(output_dir.iterdir()):
        if p.suffix == ".csv":
            log.info("  %s (%d rows)", p.name, sum(1 for _ in p.open()) - 1)


if __name__ == "__main__":
    main()
