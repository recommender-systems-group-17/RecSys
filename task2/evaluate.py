"""
Task 2.1 + 2.2 - evaluate every model's TEST top-k lists with our own metrics,
compare with the Random / Pop baselines, and test significance.

    python task2/evaluate.py            # k = 10 and 20
    python task2/evaluate.py --k 10

Outputs (outputs/task2/):
    metrics_k<k>.csv           one row per model, all metrics
    per_user_k<k>.csv          per-user metrics (input for groups.py)
    significance_k10.csv       Wilcoxon signed-rank on per-user NDCG@10
    fig_tradeoff.png           accuracy vs beyond-accuracy
"""
import argparse

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data import ALL_MODELS, BASELINES, HYBRIDS, T2, Context, load_recs, user_groups  # noqa: E402
from metrics import (ACCURACY, BEYOND_PER_USER, item_group_recall,  # noqa: E402
                     system_metrics, user_metrics)


def evaluate(ctx, groups, k):
    rows, per_user = [], []
    for m in ALL_MODELS:
        recs = load_recs(m, "test")
        pu = user_metrics(recs, ctx, k)
        df = pd.DataFrame(pu).T
        row = {"model": m, "type": "hybrid" if m in HYBRIDS else
               "baseline" if m in BASELINES else "individual"}
        row.update(df[ACCURACY + BEYOND_PER_USER].mean().to_dict())
        row.update(system_metrics(recs, pu, ctx, groups["active"].to_dict(), k))
        row.update({f"Recall_{g}": v for g, v in item_group_recall(recs, ctx, k).items()})
        rows.append(row)
        per_user.append(df.assign(model=m).rename_axis("user_id").reset_index())
        print(f"  {m:28s} NDCG@{k}={row['NDCG']:.4f}  Cov={row['Coverage']:.3f}  "
              f"Nov={row['Novelty']:.2f}  KL={row['MiscalKL']:.3f}")
    return pd.DataFrame(rows), pd.concat(per_user, ignore_index=True)


def holm(p):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, (len(p) - r) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def significance(per_user, reference):
    """Paired Wilcoxon test of every model against `reference` on per-user NDCG."""
    wide = per_user.pivot(index="user_id", columns="model", values="NDCG")
    rows = []
    for m in wide.columns.drop(reference):
        stat, p = wilcoxon(wide[m], wide[reference], zero_method="zsplit")
        rows.append({"model": m, "reference": reference,
                     "mean_diff": wide[m].mean() - wide[reference].mean(),
                     "wins": int((wide[m] > wide[reference]).sum()),
                     "losses": int((wide[m] < wide[reference]).sum()), "p": p})
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p"].values)
    df["significant"] = df["p_holm"] < 0.05
    return df.sort_values("mean_diff", ascending=False)


def plot_tradeoff(table, k):
    panels = [("Novelty", "Novelty (higher = less popular items)"),
              ("Coverage", "Catalog coverage"),
              ("ILD", "Intra-list diversity (genres)"),
              ("MiscalKL", "Miscalibration KL (lower = better)")]
    colors = {"baseline": "#888888", "individual": "#1f77b4", "hybrid": "#d62728"}
    # Random is far off-scale on every axis (it is in the CSV), so it is left out
    table = table[table["model"] != "Random"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (col, label) in zip(axes, panels):
        for _, r in table.iterrows():
            ax.scatter(r[col], r["NDCG"], c=colors[r["type"]], s=30)
            ax.annotate(r["model"].replace("Hybrid-", "H-"), (r[col], r["NDCG"]),
                        fontsize=6.5, xytext=(3, 2), textcoords="offset points")
        ax.set_xlabel(label)
        ax.set_ylabel(f"NDCG@{k}")
        ax.grid(alpha=0.3)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=t) for t, c in colors.items()]
    axes[0].legend(handles=handles, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(T2 / f"fig_tradeoff_k{k}.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[10, 20])
    args = ap.parse_args()

    T2.mkdir(parents=True, exist_ok=True)
    ctx = Context()
    groups = user_groups(ctx)
    groups.to_csv(T2 / "user_groups.csv")

    for k in args.k:
        print(f"== k = {k}")
        table, per_user = evaluate(ctx, groups, k)
        table.round(4).to_csv(T2 / f"metrics_k{k}.csv", index=False)
        per_user.to_csv(T2 / f"per_user_k{k}.csv", index=False)
        plot_tradeoff(table, k)
        if k == 10:
            best_ind = table[table["type"] == "individual"].sort_values("NDCG").iloc[-1]["model"]
            sig = pd.concat([significance(per_user, best_ind), significance(per_user, "Pop")])
            sig.round(5).to_csv(T2 / "significance_k10.csv", index=False)
            print(sig[sig["reference"] == best_ind].round(4).to_string(index=False))
    print(f"written to {T2}")


if __name__ == "__main__":
    main()
