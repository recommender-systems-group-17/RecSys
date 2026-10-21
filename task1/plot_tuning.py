"""
Figure for report Section 1.2: validation NDCG@10 over the tuning grids of
the four study models, read from outputs/tuning/<Model>.csv.

Usage (from repo root):
    pip install matplotlib        # once, if not installed
    python task1/plot_tuning.py

Writes:
    outputs/figures/task1_tuning.pdf   (vector, best for LaTeX)
    outputs/figures/task1_tuning.png   (300 dpi, for previews / screenshots)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TUNING = ROOT / "outputs" / "tuning"
FIG = ROOT / "outputs" / "figures"
Y = "valid_ndcg@10"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "lines.markersize": 3.5, "lines.linewidth": 1.3,
})
COLORS = ["#1f4e9c", "#c0392b", "#7f8c8d"]
MARKERS = ["o", "s", "^"]


def load(model):
    return pd.read_csv(TUNING / f"{model}.csv")


def log_ticks(ax, values):
    """Log x-axis that shows exactly the grid values as plain numbers."""
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(values)
    ax.set_xticklabels([f"{v / 1000:g}k" if v >= 1000 else f"{v:g}" for v in values])


def mark_best(ax, x, y):
    ax.plot([x], [y], marker="*", color="gold", markeredgecolor="black",
            markersize=9, zorder=5, linestyle="none")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.05), constrained_layout=True)

    # (a) EASE: regularisation
    ax = axes[0]
    d = load("EASE").sort_values("reg_weight")
    ax.plot(d["reg_weight"], d[Y], marker="o", color=COLORS[0])
    b = d.loc[d[Y].idxmax()]
    mark_best(ax, b["reg_weight"], b[Y])
    log_ticks(ax, sorted(d["reg_weight"].unique()))
    ax.set_title("(a) EASE")
    ax.set_xlabel(r"regularisation $\lambda$")
    ax.set_ylabel("validation NDCG@10")

    # (b) UserKNN: neighbours x shrinkage
    ax = axes[1]
    d = load("UserKNN")
    for j, s in enumerate(sorted(d["shrink"].unique())):
        g = d[d["shrink"] == s].sort_values("k")
        ax.plot(g["k"], g[Y], marker=MARKERS[j], color=COLORS[j], label=f"shrink {s:g}")
    b = d.loc[d[Y].idxmax()]
    mark_best(ax, b["k"], b[Y])
    log_ticks(ax, sorted(d["k"].unique()))
    ax.set_title("(b) UserKNN")
    ax.set_xlabel("neighbours $k$")
    ax.legend(frameon=False, loc="lower center")

    # (c) LightGCN: layers x learning rate
    ax = axes[2]
    d = load("LightGCN")
    for j, lr in enumerate(sorted(d["learning_rate"].unique())):
        g = d[d["learning_rate"] == lr].sort_values("n_layers")
        ax.plot(g["n_layers"], g[Y], marker=MARKERS[j], color=COLORS[j], label=f"lr {lr:g}")
    b = d.loc[d[Y].idxmax()]
    mark_best(ax, b["n_layers"], b[Y])
    ax.set_xticks(sorted(d["n_layers"].unique()))
    ax.set_title("(c) LightGCN")
    ax.set_xlabel("propagation layers")
    ax.legend(frameon=False, loc="lower left")

    # (d) ContentBased: weighting x IDF x decade weight
    ax = axes[3]
    d = load("ContentBased")
    d["idf"] = d["idf"].astype(str).str.lower().isin(["true", "1", "1.0"])
    styles = {("binary", True): (COLORS[0], "o", "-"), ("binary", False): (COLORS[0], "o", "--"),
              ("rating", True): (COLORS[1], "s", "-"), ("rating", False): (COLORS[1], "s", "--")}
    for (w, idf), (c, m, ls) in styles.items():
        g = d[(d["weighting"] == w) & (d["idf"] == idf)].sort_values("year_weight")
        ax.plot(g["year_weight"], g[Y], marker=m, color=c, linestyle=ls,
                label=f"{w}, IDF {'on' if idf else 'off'}")
    b = d.loc[d[Y].idxmax()]
    mark_best(ax, b["year_weight"], b[Y])
    ax.set_xticks(sorted(d["year_weight"].unique()))
    ax.set_ylim(0, d[Y].max() * 1.12)   # room under the lines for the legend
    ax.set_title("(d) ContentBased")
    ax.set_xlabel("release-decade weight")
    ax.legend(frameon=False, loc="lower center", ncol=2, fontsize=5.5,
              columnspacing=0.8, handlelength=1.8)

    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"task1_tuning.{ext}", dpi=300)
    print(f"Saved {FIG.relative_to(ROOT)}/task1_tuning.pdf and .png")


if __name__ == "__main__":
    main()