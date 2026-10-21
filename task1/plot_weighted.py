"""
Figure for report Section 1.3: learned coefficient vs. actual contribution
of each member of the Weighted hybrid.

  (a) coefficient w_m with 95% CI from bootstrap refits
      (outputs/hybrids/Weighted_coefficient_stability.csv)
  (b) change in NDCG@10 when the member is removed, with 95% CI
      (outputs/hybrids/Weighted_ablation.csv)

Usage (from repo root):
    python task1/plot_weighted.py

Writes outputs/figures/task1_weighted.pdf and .png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HYB = ROOT / "outputs" / "hybrids"
FIG = ROOT / "outputs" / "figures"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})
BLUE, GREY, RED = "#1f4e9c", "#9aa5ad", "#c0392b"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    coef = pd.read_csv(HYB / "Weighted_coefficient_stability.csv")
    abl = pd.read_csv(HYB / "Weighted_ablation.csv")
    abl = abl[abl["removed"] != "(none)"].set_index("removed")

    order = coef.sort_values("coefficient")["model"].tolist()   # largest at the top
    coef = coef.set_index("model").loc[order]
    abl = abl.loc[order]
    y = range(len(order))

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 1.75), sharey=True,
                               constrained_layout=True)

    # (a) coefficients
    err = [coef["coefficient"] - coef["ci_low"], coef["ci_high"] - coef["coefficient"]]
    a.barh(y, coef["coefficient"], xerr=err, color=BLUE, alpha=0.85, height=0.55,
           error_kw={"elinewidth": 1, "capsize": 2.5})
    for i, (w, s, hi) in enumerate(zip(coef["coefficient"], coef["share_mean"], coef["ci_high"])):
        a.text(hi + 0.08, i, f"{w:.2f}  ({s:.0%})", va="center", fontsize=7)
    a.set_yticks(list(y))
    a.set_yticklabels(order)
    a.set_xlim(0, coef["ci_high"].max() * 1.35)
    a.set_xlabel("learned coefficient $w_m$ (95% CI, 200 bootstrap refits)")
    a.set_title("(a) Weight in the hybrid")

    # (b) drop-one contribution
    sig = (abl["delta_ci_high"] < 0) | (abl["delta_ci_low"] > 0)
    colors = [RED if s else GREY for s in sig]
    err = [abl["delta"] - abl["delta_ci_low"], abl["delta_ci_high"] - abl["delta"]]
    b.barh(y, abl["delta"], xerr=err, color=colors, height=0.55,
           error_kw={"elinewidth": 1, "capsize": 2.5})
    b.axvline(0, color="black", linewidth=0.8)
    lim = max(abl["delta_ci_high"].abs().max(), abl["delta_ci_low"].abs().max()) * 1.25
    b.set_xlim(-lim, lim)
    b.set_xlabel(r"$\Delta$NDCG@10 when removed (95% CI)")
    b.set_title("(b) Effect of removing the member")
    b.tick_params(axis="y", length=0)

    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"task1_weighted.{ext}", dpi=300)
    print(f"Saved {FIG.relative_to(ROOT)}/task1_weighted.pdf and .png")


if __name__ == "__main__":
    main()