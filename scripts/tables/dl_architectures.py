"""Table 0c: Deep learning architectures summary."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .cli import run_table
from .common import write_tex

_ARCH_TABLE_ROWS: list[tuple[str, str, str, str, str]] = [
    (
        "Autoencoder (AE)",
        r"$512{\times}1{\times}1$ vector",
        r"MSE$_{\mathcal{G}}$",
        "--",
        "Adam",
    ),
    (
        "Variational Autoencoder (VAE)",
        r"$\boldsymbol{\mu},\log\boldsymbol{\sigma}^2 \!\in\! "
        r"\mathbb{R}^{256}$",
        r"MSE$_{\mathcal{G}}$+$\beta$KL",
        "--",
        "Adam",
    ),
    (
        "Generative Adversarial Network (GAN)",
        r"Dilated conv (2,\,4)",
        r"${\ell_1}^{\mathcal{G}}$+BCE$_{\mathrm{adv}}$",
        r"\checkmark",
        r"Adam $(\beta_1{=}0.5,\beta_2{=}0.999)$",
    ),
    (
        "U-Net",
        r"$1024{\times}4{\times}4$ ResBlock",
        r"MSE$_{\mathcal{G}}$",
        r"\checkmark",
        "AdamW",
    ),
    (
        "Vision Transformer (ViT)",
        r"4$\times$ Transformer ($d$=256)",
        r"MSE$_{\mathcal{G}}$",
        "--",
        "AdamW",
    ),
]


def table_dl_architectures(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate the DL architectures table using the LaTeX template."""
    dl = df[df.get("type", "") == "DL"]
    if dl.empty:
        return

    body = [
        " " + " & ".join([model, bottleneck, loss, skip, optimizer]) + r" \\"
        for model, bottleneck, loss, skip, optimizer in _ARCH_TABLE_ROWS
    ]

    lines = [
        r"\begin{table}[t]",
        r" \centering",
        r" \footnotesize",
        r" \caption{Resumo das arquiteturas de \gls{DL}. Entrada:",
        r" $[\mathbf{x};\, \mathbf{z}] \in \mathbb{R}^{5 \times 64 \times 64}$;",  # noqa: E501
        r" Saída: $\hat{\mathbf{y}} \in [0,1]^{4 \times 64 \times 64}$.}",
        r" \label{tab:dl-architectures}",
        r" \resizebox{\columnwidth}{!}{%",
        r" \begin{tabular}{llllc}",
        r" \toprule",
        r" Modelo & \emph{Bottleneck} / \emph{Core} & Função de Perda & \emph{Skip} & Otimizador \\",  # noqa: E501
        r" \midrule",
        *body,
        r" \bottomrule",
        r" \end{tabular}",
        r" }",
        r"\end{table}",
        "",
    ]
    tex = "\n".join(lines)
    write_tex(tex, output_dir / "dl-architectures.tex")


def main() -> None:
    run_table(table_dl_architectures, "DL architectures")


if __name__ == "__main__":
    main()
