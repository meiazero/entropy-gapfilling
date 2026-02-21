import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load style
plt.style.use("images/style.mplstyle")

x = np.random.randn(100)
y = x * 2 + np.random.randn(100) * 0.5
df = pd.DataFrame({"x": x, "y": y})

fig, ax = plt.subplots(figsize=(4.8, 4.4), constrained_layout=True)
palette = sns.color_palette("Set2", 3)

sns.scatterplot(
    data=df,
    x="x",
    y="y",
    ax=ax,
    color=palette[0],
    alpha=0.7,
    s=50,
    edgecolor="white",
    linewidth=1,
    label="Data",
)

# Linear regression and metrics
beta, alpha = np.polyfit(x, y, 1)
y_pred = beta * x + alpha
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1.0 - (ss_res / ss_tot)
n = len(x)
r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - 2)

# Regression line (seaborn)
sns.regplot(
    data=df,
    x="x",
    y="y",
    scatter=False,
    ci=None,
    ax=ax,
    color=palette[1],
    line_kws={"linewidth": 1.5, "label": "Fit"},
)

stats_text = (
    f"y = {alpha:.2f} + {beta:.2f}x\n"
    f"$r^2$ = {r2:.3f}\n"
    f"$r^2_{{adj}}$ = {r2_adj:.3f}\n"
    f"$\\beta$ = {beta:.2f}"
)
ax.text(
    0.02,
    0.98,
    stats_text,
    transform=ax.transAxes,
    va="top",
    ha="left",
    fontsize=10,
    bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend(loc="upper right", frameon=True, framealpha=0.85)
fig.savefig("scatter.png", dpi=300, bbox_inches="tight")
plt.close(fig)
