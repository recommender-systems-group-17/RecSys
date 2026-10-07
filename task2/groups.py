"""
Task 2.5 + 2.6 - accuracy and beyond-accuracy per user group and item group.
Run after evaluate.py (reads its per-user metrics).

    python task2/groups.py

Outputs (outputs/task2/):
    groups_user_k10.csv          mean metrics per (grouping, group, model)
    groups_best_model_k10.csv    best model per user group, and EASE's gap to it
    groups_mainstream_by_activity_k10.csv  mainstream effect with activity held fixed
    oracle_k10.csv               how much a perfect per-user / per-group switch could gain
    fig_user_groups.png          NDCG / PopLift / MiscalKL per activity and mainstream group
    fig_item_groups.png          recall and exposure per item popularity group
"""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data import INDIVIDUAL, T2  # noqa: E402

K = 10
GROUPINGS = ["activity", "active", "mainstream", "gender", "age_group"]
METRICS = ["NDCG", "Recall", "Novelty", "ILD", "MiscalKL", "PopLift", "APLT"]
# models shown in figures (all models are in the CSVs)
SHOWN = ["Pop", "ItemKNN", "UserKNN", "BPR", "SLIMElastic", "EASE",
         "LightGCN", "NeuMF", "Hybrid-Weighted", "Hybrid-Mixed"]
MEMBERS = [m for m in INDIVIDUAL if m != "Random"]


def per_group(per_user, groups):
    df = per_user.merge(groups, left_on="user_id", right_index=True)
    out = []
    for g in GROUPINGS:
        t = df.groupby([g, "model"])[METRICS].mean().reset_index().rename(columns={g: "group"})
        t.insert(0, "grouping", g)
        t["n_users"] = t["group"].map(df.drop_duplicates("user_id")[g].value_counts())
        out.append(t)
    return pd.concat(out, ignore_index=True)


def best_per_group(table):
    rows = []
    for (g, grp), t in table.groupby(["grouping", "group"]):
        t = t.set_index("model")
        ind = t.loc[MEMBERS, "NDCG"]
        rows.append({"grouping": g, "group": grp, "n_users": int(t["n_users"].iloc[0]),
                     "best_individual": ind.idxmax(), "best_individual_NDCG": ind.max(),
                     "EASE_NDCG": t.loc["EASE", "NDCG"],
                     "best_overall": t["NDCG"].idxmax(), "best_overall_NDCG": t["NDCG"].max()})
    return pd.DataFrame(rows)


def mainstream_within_activity(per_user, groups):
    """Mainstreaminess is confounded with activity (niche users have far longer
    profiles), so compare mainstream groups inside each activity group."""
    df = per_user[per_user["model"].isin(["EASE", "Hybrid-Weighted", "Pop", "NeuMF"])]
    df = df.merge(groups, left_on="user_id", right_index=True)
    return df.pivot_table(index=["model", "activity"], columns="mainstream",
                          values="NDCG", aggfunc="mean")[["Niche", "Diverse", "Blockbuster"]]


def oracle(per_user, groups):
    """Upper bounds for switching between the individual members:
    per-user oracle (pick the best member for every user, using test labels)
    vs per-activity-group oracle vs the single best member."""
    wide = per_user[per_user["model"].isin(MEMBERS)].pivot(
        index="user_id", columns="model", values="NDCG")
    act = groups.loc[wide.index, "activity"]
    group_best = wide.groupby(act).mean().idxmax(axis=1)
    per_group_choice = np.array([wide.loc[u, group_best[act[u]]] for u in wide.index])
    best_single = wide.mean().idxmax()
    picked = wide.idxmax(axis=1)[wide.max(axis=1) > 0]
    return pd.DataFrame([
        {"strategy": f"best single member ({best_single})", "NDCG": wide[best_single].mean()},
        {"strategy": "oracle per activity group", "NDCG": per_group_choice.mean()},
        {"strategy": "oracle per user", "NDCG": wide.max(axis=1).mean()},
    ]), picked.value_counts(normalize=True).rename("share_of_users_where_best")


def plot_user_groups(table):
    specs = [("activity", "NDCG"), ("activity", "PopLift"),
             ("mainstream", "NDCG"), ("mainstream", "MiscalKL")]
    order = {"mainstream": ["Niche", "Diverse", "Blockbuster"]}
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.3))
    for ax, (g, metric) in zip(axes, specs):
        t = table[(table["grouping"] == g) & table["model"].isin(SHOWN)]
        piv = t.pivot(index="group", columns="model", values=metric)
        piv = piv.loc[order.get(g, sorted(piv.index))][SHOWN]
        for m in SHOWN:
            style = "--" if m.startswith("Hybrid") else "-"
            ax.plot(range(len(piv)), piv[m], style, marker="o", ms=3, label=m)
        ax.set_xticks(range(len(piv)))
        ax.set_xticklabels(piv.index, rotation=20, fontsize=8)
        ax.set_title(f"{metric}@{K} by {g}", fontsize=10)
        ax.grid(alpha=0.3)
    axes[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(T2 / "fig_user_groups.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_item_groups(metrics):
    m = metrics.set_index("model").loc[SHOWN]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    x, w = np.arange(len(m)), 0.27
    for j, g in enumerate(["Head", "Mid", "Tail"]):
        axes[0].bar(x + (j - 1) * w, m[f"Recall_{g}"], w, label=g)
        axes[1].bar(x + (j - 1) * w, m[f"Exposure_{g}"], w, label=g)
    axes[0].set_title(f"Recall@{K} of relevant items, per item popularity group", fontsize=10)
    axes[1].set_title(f"Share of top-{K} slots per item popularity group", fontsize=10)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("Hybrid-", "H-") for s in m.index], rotation=35, fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(T2 / "fig_item_groups.png", dpi=200)
    plt.close(fig)


def main():
    per_user = pd.read_csv(T2 / f"per_user_k{K}.csv", dtype={"user_id": str})
    groups = pd.read_csv(T2 / "user_groups.csv", dtype={"user_id": str}).set_index("user_id")
    metrics = pd.read_csv(T2 / f"metrics_k{K}.csv")

    table = per_group(per_user, groups)
    table.round(4).to_csv(T2 / f"groups_user_k{K}.csv", index=False)
    best = best_per_group(table)
    best.round(4).to_csv(T2 / f"groups_best_model_k{K}.csv", index=False)
    print(best.round(4).to_string(index=False))

    mw = mainstream_within_activity(per_user, groups)
    mw.round(4).to_csv(T2 / f"groups_mainstream_by_activity_k{K}.csv")
    print(mw.loc["EASE"].round(4).to_string())

    orc, share = oracle(per_user, groups)
    orc.round(4).to_csv(T2 / f"oracle_k{K}.csv", index=False)
    share.round(4).to_csv(T2 / f"oracle_best_member_share_k{K}.csv")
    print(orc.round(4).to_string(index=False))
    print(share.round(3).to_string())

    plot_user_groups(table)
    plot_item_groups(metrics)
    print(f"written to {T2}")


if __name__ == "__main__":
    main()
