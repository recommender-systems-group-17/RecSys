"""Report figures for Task 2, drawn at print size (6.7 in = full text width).

Reads only the CSVs written by evaluate.py, groups.py and hybrid_analysis.py,
so run those first. Writes outputs/task2/report_*.png.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

T2 = Path(__file__).resolve().parents[1] / "outputs" / "task2"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# one colour per model, the same in every figure
COLOR = {"Pop": "#8a8984", "EASE": "#2a78d6", "UserKNN": "#1baf7a",
         "LightGCN": "#eb6834", "ContentBased": "#6250d6", "Hybrid": "#e34948"}
MEMBERS = ["EASE", "UserKNN", "LightGCN", "ContentBased"]
SHORT = {"Hybrid-Weighted": "Weighted", "Hybrid-Mixed": "Mixed",
         "Hybrid-FeatureAugmentation": "Feat. augm.", "Hybrid-Cascade": "Cascade",
         "Hybrid-MetaLevel": "Meta-level", "Hybrid-FeatureCombination": "Feat. comb.",
         "ContentBased": "Content"}


def color(model):
    return COLOR["Hybrid"] if model.startswith("Hybrid") else COLOR[model]


def tradeoff():
    m = pd.read_csv(T2 / "metrics_k10.csv").set_index("model")
    rows = ["Pop", "EASE", "UserKNN", "LightGCN", "ContentBased",
            "Hybrid-Weighted", "Hybrid-FeatureAugmentation", "Hybrid-Mixed",
            "Hybrid-Cascade", "Hybrid-MetaLevel", "Hybrid-FeatureCombination"]
    panels = [("NDCG", "NDCG@10", "{:.3f}"), ("Novelty", "Novelty", "{:.2f}"),
              ("Coverage", "Coverage", "{:.0%}"), ("ILD", "ILD", "{:.2f}"),
              ("MiscalKL", "MiscalKL\n(lower = better)", "{:.2f}")]
    y = list(range(len(rows)))[::-1]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.7, 3.4), sharey=True)
    for ax, (col, title, fmt) in zip(axes, panels):
        vals = m.loc[rows, col].values
        lo, hi = min(0, vals.min()), vals.max()
        ax.hlines(y, lo, vals, color="#e6e5e0", lw=1.2, zorder=1)
        ax.scatter(vals, y, s=30, color=[color(r) for r in rows],
                   edgecolor="white", linewidth=0.8, zorder=3)
        for yi, v in zip(y, vals):
            ax.text(v + 0.04 * (hi - lo), yi, fmt.format(v), va="center", fontsize=6.5,
                    color="#52514e")
        ax.set_xlim(lo, hi + 0.42 * (hi - lo))
        ax.axhline(5.5, color="#b5b4ae", lw=0.6)   # individuals above, hybrids below
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=7)
        ax.locator_params(axis="x", nbins=3)
    axes[0].set_yticks(y, [SHORT.get(r, r) for r in rows])
    fig.tight_layout(w_pad=0.6)
    fig.savefig(T2 / "report_tradeoff.png")
    plt.close(fig)


def hybrid():
    c = pd.read_csv(T2 / "hybrid_member_contribution.csv", index_col=0).loc[MEMBERS]
    a = pd.read_csv(T2 / "hybrid_ablation.csv").set_index("removed")
    g = pd.read_csv(T2 / "hybrid_coefficients_by_activity.csv", index_col=0).loc[MEMBERS]
    names = [SHORT.get(x, x) for x in MEMBERS]
    cols = [COLOR[x] for x in MEMBERS]
    y = range(len(MEMBERS))[::-1]

    fig, axes = plt.subplots(2, 2, figsize=(6.7, 5.0))
    panels = [(c.coefficient, "(a) Learned coefficient (Task 1.3)", "{:.2f}"),
              (c.share_of_hybrid_hits_also_in_member_top10 * 100,
               "(b) Hybrid hits also in member's top-10 (%)", "{:.0f}%"),
              (a.loc[MEMBERS, "delta_vs_full"], "(c) Change in test NDCG@10 if removed", "{:+.4f}")]
    for ax, (vals, title, fmt) in zip(axes.flat, panels):
        ax.barh(list(y), vals.values, color=cols, height=0.6, edgecolor="white")
        ax.set_yticks(list(y), names)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color="#52514e", lw=0.7)
        span = vals.abs().max()
        for yi, v in zip(y, vals.values):
            ax.text(v + (0.03 * span if v >= 0 else -0.03 * span), yi, fmt.format(v),
                    va="center", ha="left" if v >= 0 else "right", fontsize=7.5)
        lo, hi = min(0, vals.min()), max(0, vals.max())
        ax.set_xlim(lo - 0.6 * span * (lo < 0), hi + 0.3 * span)

    ax = axes[1, 1]
    blues = ["#cfe0f5", "#9cc0ec", "#5f97de", "#2a78d6", "#174f96"]
    h = 0.16
    for j, (grp, b) in enumerate(zip(g.columns, blues)):
        ax.barh([yi + (2 - j) * h for yi in y], g[grp].values, height=h, color=b,
                label=grp, edgecolor="white", linewidth=0.4)
    ax.set_yticks(list(y), names)
    ax.axvline(0, color="#52514e", lw=0.7)
    ax.grid(axis="y", visible=False)
    ax.set_title("(d) Weight share, fitted per activity group", loc="left")
    ax.legend(title="activity group (train interactions)", fontsize=7, title_fontsize=7,
              loc="upper center", bbox_to_anchor=(0.45, -0.08), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(T2 / "report_hybrid.png")
    plt.close(fig)


def user_groups():
    t = pd.read_csv(T2 / "groups_user_k10.csv")
    models = ["Pop", "EASE", "UserKNN", "LightGCN", "ContentBased", "Hybrid-Weighted"]
    markers = dict(zip(models, ["o", "o", "^", "s", "D", "v"]))
    act = sorted(t[t.grouping == "activity"].group.unique(), key=lambda s: int(s[1]))
    main = ["Niche", "Diverse", "Blockbuster"]
    panels = [("activity", act, "NDCG", "(a) NDCG@10 by activity"),
              ("activity", act, "PopLift", "(b) PopLift by activity (lower = closer to taste)"),
              ("mainstream", main, "NDCG", "(c) NDCG@10 by mainstreamness"),
              ("mainstream", main, "MiscalKL", "(d) Miscalibration by mainstreamness")]
    fig, axes = plt.subplots(2, 2, figsize=(6.7, 5.4))
    for ax, (grouping, order, metric, title) in zip(axes.flat, panels):
        sub = t[t.grouping == grouping]
        for mdl in models:
            s = sub[sub.model == mdl].set_index("group").loc[order, metric]
            ax.plot(range(len(order)), s.values, marker=markers[mdl], ms=4.5, lw=1.6,
                    ls="--" if mdl.startswith("Hybrid") else "-", color=color(mdl),
                    label=SHORT.get(mdl, mdl))
        labels = [o.replace(" (", "\n(") for o in order]
        ax.set_xticks(range(len(order)), labels)
        ax.set_title(title, loc="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(T2 / "report_user_groups.png")
    plt.close(fig)


def item_groups():
    m = pd.read_csv(T2 / "metrics_k10.csv").set_index("model")
    models = ["Pop", "EASE", "UserKNN", "LightGCN", "ContentBased",
              "Hybrid-Weighted", "Hybrid-Mixed", "Hybrid-FeatureAugmentation"]
    groups = [("Head", "#2a78d6"), ("Mid", "#eb6834"), ("Tail", "#1baf7a")]
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.9))
    for ax, (prefix, title) in zip(axes, [("Recall", "(a) Recall@10 per item group"),
                                          ("Exposure", "(b) Share of top-10 slots per item group")]):
        w = 0.27
        for j, (grp, col) in enumerate(groups):
            ax.bar([i + (j - 1) * w for i in range(len(models))],
                   m.loc[models, f"{prefix}_{grp}"].values, width=w, color=col,
                   label=grp, edgecolor="white", linewidth=0.4)
        ax.set_xticks(range(len(models)), [SHORT.get(x, x) for x in models],
                      rotation=40, ha="right")
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(T2 / "report_item_groups.png")
    plt.close(fig)


def groups_combined():
    """User and item groups in one 2x3 figure, to fit one report page."""
    t = pd.read_csv(T2 / "groups_user_k10.csv")
    m = pd.read_csv(T2 / "metrics_k10.csv").set_index("model")
    models = ["Pop", "EASE", "UserKNN", "LightGCN", "ContentBased", "Hybrid-Weighted"]
    markers = dict(zip(models, ["o", "o", "^", "s", "D", "v"]))
    act = sorted(t[t.grouping == "activity"].group.unique(), key=lambda s: int(s[1]))
    main = ["Niche", "Diverse", "Blockbuster"]
    fig, axes = plt.subplots(2, 3, figsize=(6.7, 4.0))
    lines = [(axes[0, 0], "activity", act, "NDCG", "(a) NDCG@10 by activity"),
             (axes[0, 1], "activity", act, "PopLift", "(b) PopLift by activity"),
             (axes[1, 0], "mainstream", main, "NDCG", "(d) NDCG@10 by mainstream"),
             (axes[1, 1], "mainstream", main, "MiscalKL", "(e) MiscalKL by mainstream")]
    for ax, grouping, order, metric, title in lines:
        sub = t[t.grouping == grouping]
        for mdl in models:
            s = sub[sub.model == mdl].set_index("group").loc[order, metric]
            ax.plot(range(len(order)), s.values, marker=markers[mdl], ms=3.5, lw=1.4,
                    ls="--" if mdl.startswith("Hybrid") else "-", color=color(mdl),
                    label=SHORT.get(mdl, mdl))
        ax.set_xticks(range(len(order)),
                      [o.split(" ")[0] if grouping == "activity" else o.replace("Blockbuster", "Blockb.")
                       for o in order])
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
    items = ["Pop", "EASE", "UserKNN", "LightGCN", "ContentBased", "Hybrid-Weighted"]
    groups = [("Head", "#2a78d6"), ("Mid", "#eb6834"), ("Tail", "#1baf7a")]
    for ax, prefix, title in [(axes[0, 2], "Recall", "(c) Recall@10 per item group"),
                              (axes[1, 2], "Exposure", "(f) Share of slots per item group")]:
        w = 0.27
        for j, (grp, col) in enumerate(groups):
            ax.bar([i + (j - 1) * w for i in range(len(items))],
                   m.loc[items, f"{prefix}_{grp}"].values, width=w, color=col,
                   label=grp, edgecolor="white", linewidth=0.3)
        ax.set_xticks(range(len(items)), [SHORT.get(x, x) for x in items],
                      rotation=45, ha="right")
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.grid(axis="x", visible=False)
        ax.tick_params(labelsize=7)
    h, l = axes[0, 0].get_legend_handles_labels()
    hb, lb = axes[0, 2].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.36, -0.03),
               fontsize=7, handlelength=2.2, columnspacing=1.0)
    fig.legend(hb, lb, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.88, -0.03),
               fontsize=7, handlelength=1.2, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=0.8, w_pad=0.8)
    fig.savefig(T2 / "report_groups.png")
    plt.close(fig)


if __name__ == "__main__":
    tradeoff()
    hybrid()
    user_groups()
    item_groups()
    groups_combined()
    print("wrote", *sorted(p.name for p in T2.glob("report_*.png")))
