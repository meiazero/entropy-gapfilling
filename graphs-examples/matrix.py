import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load style
plt.style.use("images/style.mplstyle")

# Generate correlated data
df = pd.DataFrame(
    np.random.randn(100, 4), columns=["Var1", "Var2", "Var3", "Var4"]
)
corr = df.corr()

fig, ax = plt.subplots(figsize=(5.5, 5.0))
im = ax.imshow(corr, aspect="auto")
ax.set_title("Correlation Matrix")
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns)
ax.set_yticklabels(corr.columns)
fig.colorbar(im, ax=ax)
plt.tight_layout()
fig.savefig("matrix.png", dpi=300, bbox_inches="tight")
plt.close(fig)
