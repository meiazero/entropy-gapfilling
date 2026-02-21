import matplotlib.pyplot as plt
import numpy as np

# Load style
plt.style.use("images/style.mplstyle")

# Simulated data similar to the reference image (temperatures in °C)
np.random.seed(48)
years = [2010, 2015, 2020, 2025]
winter = [
    np.random.normal(2, 18, 100),  # ~2010: low mean with negative outliers
    np.random.normal(-5, 10, 100),  # ~2015: lower box
    np.random.normal(-2, 12, 100),  # ~2020: low with outliers
    np.random.normal(0, 15, 100),  # ~2025: wider with low outliers
]
spring_fall = [
    np.random.normal(5, 12, 100),  # ~2010: central box
    np.random.normal(8, 10, 100),  # ~2015: slightly higher
    np.random.normal(20, 8, 100),  # ~2020: higher box
    np.random.normal(10, 15, 100),  # ~2025: wider with high outliers
]
summer = [
    np.random.normal(20, 10, 100),  # ~2010: high box
    np.random.normal(25, 12, 100),  # ~2015: higher
    np.random.normal(15, 15, 100),  # ~2020: wider with low outliers
    np.random.normal(30, 10, 100),  # ~2025: high with outliers
]

# Create figure
fig, ax = plt.subplots(figsize=(5.5, 5))

# Grouped boxplots by year
x = np.arange(len(years)) * 1.2
offset = 0.28
width = 0.22

# Stronger, print-friendly palette for seasonal colors
cmap = plt.get_cmap("Set2")
palette = cmap(np.linspace(0.0, 0.9, 3))

bp_winter = ax.boxplot(
    winter,
    positions=x - offset,
    widths=width,
    patch_artist=True,
    notch=False,
    showmeans=True,
)
bp_spring_fall = ax.boxplot(
    spring_fall,
    positions=x,
    widths=width,
    patch_artist=True,
    notch=False,
    showmeans=True,
)
bp_summer = ax.boxplot(
    summer,
    positions=x + offset,
    widths=width,
    patch_artist=True,
    notch=False,
    showmeans=True,
)

# Apply palette colors to box faces
for patch in bp_winter["boxes"]:
    patch.set_facecolor(palette[0])
for patch in bp_spring_fall["boxes"]:
    patch.set_facecolor(palette[1])
for patch in bp_summer["boxes"]:
    patch.set_facecolor(palette[2])

# Title and labels
# ax.set_title("Seasonal Temperatures in Fake City, Ontario")
ax.set_ylabel("Temperature (°C)")
ax.set_xlabel("Year")
ax.set_xticks(x)
ax.set_xticklabels(years)

# Legend
ax.legend(
    [bp_winter["boxes"][0], bp_spring_fall["boxes"][0], bp_summer["boxes"][0]],
    ["Winter", "Spring/Fall", "Summer"],
    loc="best",
)

plt.tight_layout()
fig.savefig("boxplot_temperaturas.png", dpi=300, bbox_inches="tight")
plt.close(fig)
