"""Figure 10: VAE/GAN component decomposition."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from data_loader import load_all_dl_histories
from figures.cli import run_no_df
from figures.common import FONT_SIZE, save_figure, style_axes


def fig10_components(output_dir: Path) -> None:
    """VAE reconstruction+KL and GAN generator+discriminator decomposition."""
    histories = load_all_dl_histories()

    for scenario, models in histories.items():
        vae_hist = models.get("vae")
        if vae_hist and vae_hist.get("epochs"):
            epochs_data = vae_hist["epochs"]
            if any("train_recon_loss" in e for e in epochs_data):
                epochs = [e["epoch"] for e in epochs_data]
                fig, axes = plt.subplots(
                    1, 2, figsize=(7, 2.8), constrained_layout=True
                )

                axes[0].plot(
                    epochs,
                    [e.get("train_recon_loss") for e in epochs_data],
                    label="Treino",
                    linewidth=1.2,
                    color="#1f77b4",
                )
                axes[0].plot(
                    epochs,
                    [e.get("val_recon_loss") for e in epochs_data],
                    label="Validação",
                    linewidth=1.2,
                    color="#ff7f0e",
                    linestyle="--",
                )
                style_axes(
                    axes[0],
                    title="Perda de Reconstrução",
                    xlabel="Época",
                    ylabel="Perda",
                )
                axes[0].legend(fontsize=FONT_SIZE - 1)

                axes[1].plot(
                    epochs,
                    [e.get("train_kl_loss") for e in epochs_data],
                    label="Treino",
                    linewidth=1.2,
                    color="#1f77b4",
                )
                axes[1].plot(
                    epochs,
                    [e.get("val_kl_loss") for e in epochs_data],
                    label="Validação",
                    linewidth=1.2,
                    color="#ff7f0e",
                    linestyle="--",
                )
                style_axes(
                    axes[1],
                    title="Divergência KL",
                    xlabel="Época",
                    ylabel="Perda",
                )
                axes[1].legend(fontsize=FONT_SIZE - 1)

                fig.suptitle(
                    f"VAE — {scenario.replace('_', ' ').title()}",
                    fontsize=FONT_SIZE + 1,
                )
                save_figure(fig, output_dir, f"fig10_vae_components_{scenario}")
                plt.close(fig)

        gan_hist = models.get("gan")
        if gan_hist and gan_hist.get("epochs"):
            epochs_data = gan_hist["epochs"]
            if any("train_g_loss" in e for e in epochs_data):
                epochs = [e["epoch"] for e in epochs_data]
                fig, axes = plt.subplots(
                    1, 3, figsize=(9, 2.8), constrained_layout=True
                )

                axes[0].plot(
                    epochs,
                    [e.get("train_g_loss") for e in epochs_data],
                    linewidth=1.2,
                    color="#2ca02c",
                )
                style_axes(
                    axes[0],
                    title="Perda do Gerador",
                    xlabel="Época",
                    ylabel="Perda",
                )

                axes[1].plot(
                    epochs,
                    [e.get("train_d_loss") for e in epochs_data],
                    linewidth=1.2,
                    color="#d62728",
                )
                style_axes(
                    axes[1],
                    title="Perda do Discriminador",
                    xlabel="Época",
                    ylabel="Perda",
                )

                axes[2].plot(
                    epochs,
                    [e.get("val_loss") for e in epochs_data],
                    linewidth=1.2,
                    color="#9467bd",
                )
                style_axes(
                    axes[2],
                    title="Perda de Validação (G)",
                    xlabel="Época",
                    ylabel="Perda",
                )

                fig.suptitle(
                    f"GAN — {scenario.replace('_', ' ').title()}",
                    fontsize=FONT_SIZE + 1,
                )
                save_figure(fig, output_dir, f"fig10_gan_components_{scenario}")
                plt.close(fig)


def main() -> None:
    run_no_df(fig10_components, "DL component decomposition")


if __name__ == "__main__":
    main()
