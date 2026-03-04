"""Table 5: Speed and practical viability."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .cli import run_table
from .common import ranked_plain, tex_escape, wrap_table, write_tex


def table_speed_summary(df: pd.DataFrame, output_dir: Path) -> None:
    """PSNR vs inference time per method."""
    subset = df[df["noise_level"] == "inf"]
    if subset.empty:
        subset = df

    if "elapsed_s" not in subset.columns:
        return

    rows = []
    for method, grp in subset.groupby("method", observed=True):
        psnr_mean = (
            float(grp["psnr"].mean()) if "psnr" in grp.columns else np.nan
        )
        time_mean = float(grp["elapsed_s"].mean())
        efficiency = psnr_mean / time_mean if time_mean > 0 else np.nan
        rows.append({
            "method": method,
            "type": grp["type"].iloc[0] if "type" in grp.columns else "",
            "psnr": psnr_mean,
            "time": time_mean,
            "efficiency": efficiency,
        })

    stats_df = pd.DataFrame(rows)

    stats_df["psnr_rank"] = stats_df["psnr"].rank(ascending=False, method="min")
    stats_df["time_rank"] = stats_df["time"].rank(ascending=True, method="min")
    stats_df["eff_rank"] = stats_df["efficiency"].rank(
        ascending=False, method="min"
    )

    stats_df = stats_df.sort_values("efficiency", ascending=False)

    body: list[str] = []
    for _, row in stats_df.iterrows():
        psnr_cell = ranked_plain(row["psnr"], int(row["psnr_rank"]), ".2f")
        if row["time"] < 0.01:
            time_str = f"${row['time']:.4f}$"
        elif row["time"] < 1:
            time_str = f"${row['time']:.3f}$"
        else:
            time_str = f"${row['time']:.2f}$"
        ranked_plain(row["time"], int(row["time_rank"]), ".3f")
        eff_cell = ranked_plain(row["efficiency"], int(row["eff_rank"]), ".1f")
        body.append(
            f"{row['type']} & {tex_escape(str(row['method']))} & "
            f"{psnr_cell} & {time_str} & {eff_cell} \\\\"
        )

    tex = wrap_table(
        body,
        caption=(
            "Velocidade e viabilidade prática (sem ruído). "
            "PSNR/s indica eficiência. "
            "Hardware: Rocky Linux 9.5, NVIDIA A100 80GB PCIe, "
            "driver 550.90.07, CUDA 12.4/12.6.2."
        ),
        label="tab:runtime-speed",
        col_spec="llccc",
        header=(
            r"Tipo & Método & PSNR (dB) $\uparrow$ "
            r"& Tempo (s/patch) $\downarrow$ & PSNR/s $\uparrow$"
        ),
        resizebox=True,
    )
    write_tex(tex, output_dir / "runtime-speed.tex")


def main() -> None:
    run_table(table_speed_summary, "Speed summary")


if __name__ == "__main__":
    main()
