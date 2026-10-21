"""
Task 3.4 - who gains and who pays when the rerankers are applied?

    python task3/groups.py                 # figure for EASE
    python task3/groups.py --model Hybrid-Weighted

Uses the per-user test metrics written by sweep.py at lambda = 0 and at the
lambda chosen on validation. User groups are the Task 2 groups (activity,
Active/Inactive, Niche/Diverse/Blockbuster, gender, age); item groups are the
Head/Mid/Tail popularity groups.

Outputs (outputs/task3/):
    groups_user_change.csv   per (model, reranker, user group): metrics before/after
    groups_item_change.csv   per (model, reranker): exposure and recall per item group
    itemfair_order_effect.csv  ItemFairness NDCG change by the user's position in
                             the processing order (quintiles), with 95% CIs
    fig_groups_<model>.png
"""
import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import T3, candidates, contexts, paired_ci, user_groups  # noqa: E402
from rerankers import RERANKERS, visit_order  # noqa: E402

DIMENSIONS = ["active", "mainstream", "age_group", "gender"]
METRICS = ["NDCG", "Recall", "ILD", "MiscalKL", "APLT", "PopMiscal"]
ITEM_COLS = [f"{m}_{g}" for m in ("Exposure", "Recall") for g in ("Head", "Mid", "Tail")]


def user_change(pu, groups):
    pu = pu.join(groups[DIMENSIONS], on="user_id")
    base = pu[pu["reranker"] == "Original"].set_index(["model", "user_id"])
    rows = []
    for (model, name, lam), g in pu[pu["reranker"] != "Original"].groupby(["model", "reranker", "lambda"]):
        b = base.loc[model].loc[g["user_id"]]
        for dim in DIMENSIONS:
            for grp, idx in g.groupby(dim).groups.items():
                after = g.loc[idx]
                before = b.loc[after["user_id"]]
                row = {"model": model, "reranker": name, "lambda": lam, "dimension": dim,
                       "group": grp, "n_users": len(after)}
                for m in METRICS:
                    row[f"{m}_before"] = before[m].mean()
                    row[f"{m}_after"] = after[m].mean()
                row["NDCG_rel_change_%"] = 100 * (row["NDCG_after"] / row["NDCG_before"] - 1)
                rows.append(row)
    return pd.DataFrame(rows)


def item_change(sweep, selected):
    t = sweep[sweep["split"] == "test"]
    base = t[t["reranker"] == "Original"].set_index("model")[ITEM_COLS]
    rows = []
    for _, s in selected.iterrows():
        a = t[(t["model"] == s["model"]) & (t["reranker"] == s["reranker"]) &
              np.isclose(t["lambda"], s["lambda"])].iloc[0]
        row = {"model": s["model"], "reranker": s["reranker"], "lambda": s["lambda"]}
        for c in ITEM_COLS:
            row[f"{c}_before"] = base.loc[s["model"], c]
            row[f"{c}_after"] = a[c]
        rows.append(row)
    return pd.DataFrame(rows)


def order_effect(pu):
    """ItemFairness processes users in a fixed random order, so later users face
    more accumulated exposure penalties. Is their accuracy cost larger?"""
    rows = []
    for model, g in pu[pu["reranker"] == "ItemFairness"].groupby("model"):
        order = visit_order(candidates(model, "test"))
        pos = pd.Series(np.arange(len(order)) / len(order), index=order)
        base = pu[(pu["model"] == model) & (pu["reranker"] == "Original")].set_index("user_id")
        after = g.set_index("user_id")
        users = after.index.intersection(base.index)
        quint = pd.qcut(pos.loc[users], 5, labels=[f"Q{i}" for i in range(1, 6)])
        for q, us in users.groupby(quint.values).items():
            m, lo, hi = paired_ci(after.loc[us, "NDCG"].values, base.loc[us, "NDCG"].values)
            rows.append({"model": model, "lambda": after["lambda"].iloc[0],
                         "visit_quintile": q, "n_users": len(us),
                         "dNDCG": m, "dNDCG_ci_low": lo, "dNDCG_ci_high": hi})
    return pd.DataFrame(rows)


def plot(uc, ic, model):
    u = uc[uc["model"] == model]
    order = [(d, g) for d in DIMENSIONS for g in sorted(u.loc[u["dimension"] == d, "group"].unique())]
    labels = [g for _, g in order]

    def matrix(value):
        return np.array([[u[(u["reranker"] == r) & (u["dimension"] == d) & (u["group"] == g)][value].iloc[0]
                          for r in RERANKERS] for d, g in order])

    nd = matrix("NDCG_rel_change_%")
    pm = matrix("PopMiscal_after") - matrix("PopMiscal_before")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    for ax, m, title, cmap in [(axes[0], nd, "NDCG@10 change (%)", "RdBu"),
                               (axes[1], pm, "PopMiscal change (abs., lower = better)", "RdBu_r")]:
        lim = np.abs(m).max()
        ax.imshow(m, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
        for (i, j), v in np.ndenumerate(m):
            ax.text(j, i, f"{v:.1f}" if "NDCG" in title else f"{v:.3f}", ha="center", va="center",
                    fontsize=7)
        ax.set_xticks(range(len(RERANKERS)), RERANKERS, rotation=20, fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_title(title, fontsize=10)
    axes[1].set_yticklabels([])

    i = ic[ic["model"] == model].set_index("reranker").loc[RERANKERS]
    groups = ["Head", "Mid", "Tail"]
    bars = [("Original", [i[f"Exposure_{g}_before"].iloc[0] for g in groups])] + \
        [(r, [i.loc[r, f"Exposure_{g}_after"] for g in groups]) for r in RERANKERS]
    x = np.arange(len(groups))
    wdt = 0.8 / len(bars)
    for n, (r, v) in enumerate(bars):
        axes[2].bar(x + n * wdt, v, wdt, label=r)
    axes[2].set_xticks(x + 0.4 - wdt / 2, groups)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("share of top-10 slots (log)")
    axes[2].set_title("Item exposure per popularity group", fontsize=10)
    axes[2].legend(fontsize=8)
    lams = ", ".join(f"{r}={i.loc[r, 'lambda']:g}" for r in RERANKERS)
    fig.suptitle(f"{model}: effect of each reranker at its chosen lambda ({lams})", fontsize=11)
    fig.tight_layout()
    fig.savefig(T3 / f"fig_groups_{model}.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EASE")
    args = ap.parse_args()

    groups = user_groups(contexts()["test"])
    pu = pd.read_csv(T3 / "per_user_selected.csv", dtype={"user_id": str})
    sweep = pd.read_csv(T3 / "sweep.csv")
    selected = pd.read_csv(T3 / "selected.csv")

    uc = user_change(pu, groups)
    ic = item_change(sweep, selected)
    uc.round(4).to_csv(T3 / "groups_user_change.csv", index=False)
    oe = order_effect(pu)
    oe.round(5).to_csv(T3 / "itemfair_order_effect.csv", index=False)
    print(f"== ItemFairness: NDCG change by position in the processing order\n"
          f"{oe.round(4).to_string(index=False)}")
    ic.round(4).to_csv(T3 / "groups_item_change.csv", index=False)
    plot(uc, ic, args.model)

    show = uc[uc["model"] == args.model].pivot_table(index=["dimension", "group"], columns="reranker",
                                                     values="NDCG_rel_change_%")
    print(f"== {args.model}: NDCG change (%) per user group\n{show.round(1).to_string()}")
    print(f"written to {T3}")


if __name__ == "__main__":
    main()
