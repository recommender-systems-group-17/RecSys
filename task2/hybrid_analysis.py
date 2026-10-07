"""
Task 2.3 + 2.4 - how does each member contribute to the weighted hybrid?

    python task2/hybrid_analysis.py

1. Coefficients learned in Task 1.3 next to the members' score correlations.
2. Hit attribution: for every correct item in Hybrid-Weighted's test top-10,
   which members also ranked it in their own top-10, and which hits are found
   by exactly one member.
3. Weighted-hybrid replica built from the exported top-100 lists (the full
   score matrices are not shared). Same recipe as Task 1.3: per-user min-max
   normalised member scores, logistic regression with the same C, fitted on
   validation. Used for
     - per-member contribution to the final score of recommended items,
     - leave-one-member-out ablation (test NDCG@10 drop),
     - coefficients fitted separately per user activity group (reported as
       share of total |weight|, since C is fixed and groups are smaller).
   The replica's test NDCG is printed next to the original's as a check.

Outputs: outputs/task2/hybrid_*.csv, fig_hybrid_analysis.png
"""
import matplotlib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data import OUT, T2, Context, load_recs, load_recs_with_scores, truth_sets  # noqa: E402
from metrics import ndcg  # noqa: E402

K = 10
DEPTH_FIT = 50  # candidate depth used in Task 1.3 when fitting the regression
PARAMS = yaml.safe_load((OUT / "hybrids" / "hybrid_params.yaml").read_text())["Weighted"]
MEMBERS = list(PARAMS["weights"])
C = float(PARAMS["C"])


# ------------------------------------------------------------- hit attribution

def hit_attribution(ctx):
    hyb = load_recs("Hybrid-Weighted", "test", K)
    tops = {m: {u: set(r) for u, r in load_recs(m, "test", K).items()} for m in MEMBERS}
    found = {m: 0 for m in MEMBERS}
    unique = {m: 0 for m in MEMBERS}
    overlap = {m: [] for m in MEMBERS}
    n_hits, n_none = 0, 0
    for u in ctx.users:
        h = set(hyb[u])
        for m in MEMBERS:
            overlap[m].append(len(h & tops[m][u]) / K)
        for i in h & ctx.test[u]:
            n_hits += 1
            who = [m for m in MEMBERS if i in tops[m][u]]
            for m in who:
                found[m] += 1
            if len(who) == 1:
                unique[who[0]] += 1
            n_none += not who
    df = pd.DataFrame({
        "coefficient": pd.Series(PARAMS["weights"]),
        "share_of_hybrid_hits_also_in_member_top10": pd.Series(found) / n_hits,
        "hits_only_this_member_had": pd.Series(unique),
        "list_overlap_with_hybrid@10": pd.Series({m: np.mean(v) for m, v in overlap.items()}),
    })
    print(f"hybrid test hits: {n_hits}, found by no member's top-{K}: {n_none}")
    return df.sort_values("coefficient", ascending=False)


# --------------------------------------------------------------------- replica

def features(part):
    """Long -> wide table (user, item) x member of normalised scores.
    Normalisation: per user and member, min-max over that member's top-100
    list; items outside the list get 0."""
    frames = []
    for m in MEMBERS:
        rows = []
        for u, (items, scores) in load_recs_with_scores(m, part).items():
            s = np.asarray(scores, dtype=float)
            n = (s - s.min()) / max(s.max() - s.min(), 1e-12)
            rows.append(pd.DataFrame({"user_id": u, "item_id": items, "rank": np.arange(len(items)),
                                      m: n}))
        frames.append(pd.concat(rows).set_index(["user_id", "item_id"]))
    wide = pd.concat([f[[m]] for f, m in zip(frames, MEMBERS)], axis=1).fillna(0.0)
    best_rank = pd.concat([f["rank"] for f in frames], axis=1).min(axis=1)
    return wide, best_rank


def label(wide, truth):
    u = wide.index.get_level_values(0)
    i = wide.index.get_level_values(1)
    return np.fromiter((it in truth.get(us, ()) for us, it in zip(u, i)), bool, len(wide))


def fit(Xv, yv, members, C=C):
    lr = LogisticRegression(C=C, max_iter=5000).fit(Xv[members].values, yv)
    return dict(zip(members, lr.coef_[0]))


def rank_ndcg(Xt, w, truth, users=None):
    score = sum(w[m] * Xt[m] for m in w)
    res = []
    for u, s in score.groupby(level=0):
        if users is not None and u not in users:
            continue
        top = list(s.nlargest(K).index.get_level_values(1))
        res.append(ndcg(top, truth[u], K))
    return float(np.mean(res))


def replica_analysis(ctx, groups):
    valid_truth, test_truth = truth_sets("valid"), ctx.test
    Xv, rv = features("valid")
    Xv = Xv[rv < DEPTH_FIT]              # same candidate set as Task 1.3
    yv = label(Xv, valid_truth)
    Xt, _ = features("test")

    w = fit(Xv, yv, MEMBERS)
    full = rank_ndcg(Xt, w, test_truth)
    print(f"replica test NDCG@{K} = {full:.4f}   (original Hybrid-Weighted: see metrics_k10.csv)")

    # contribution of each member to the final score of the recommended items
    score = sum(w[m] * Xt[m] for m in MEMBERS)
    top = score.groupby(level=0, group_keys=False).nlargest(K).index
    parts = pd.DataFrame({m: w[m] * Xt.loc[top, m] for m in MEMBERS})
    is_hit = label(parts, test_truth)
    share = parts.div(parts.abs().sum(axis=1), axis=0)
    contrib = pd.DataFrame({"replica_coefficient": pd.Series(w),
                            "score_share_all_recs": share.mean(),
                            "score_share_hits": share[is_hit].mean()})

    # leave one member out
    abl = []
    for m in MEMBERS:
        rest = [x for x in MEMBERS if x != m]
        abl.append({"removed": m, "test_NDCG": rank_ndcg(Xt, fit(Xv, yv, rest), test_truth)})
    abl = pd.DataFrame(abl)
    abl["delta_vs_full"] = abl["test_NDCG"] - full
    abl = pd.concat([pd.DataFrame([{"removed": "(none)", "test_NDCG": full, "delta_vs_full": 0.0}]),
                     abl.sort_values("delta_vs_full")], ignore_index=True)

    # coefficients per activity group (fit on that group's validation users only)
    act = groups["activity"]
    per_group, group_ndcg = {}, []
    user_v = Xv.index.get_level_values(0)
    for g in sorted(act.unique()):
        us = set(act[act == g].index)
        mask = np.isin(user_v, list(us))
        per_group[g] = fit(Xv[mask], yv[mask], MEMBERS)
        group_ndcg.append({"group": g, "global_weights": rank_ndcg(Xt, w, test_truth, us),
                           "group_weights": rank_ndcg(Xt, per_group[g], test_truth, us)})
    # C is fixed, so absolute coefficients shrink with fewer training rows;
    # compare each member's share of the total |weight| instead.
    per_group = pd.DataFrame(per_group)
    per_group = per_group.abs().div(per_group.abs().sum()) * np.sign(per_group)
    group_ndcg = pd.DataFrame(group_ndcg)
    return contrib, abl, per_group, group_ndcg


def plot(attr, contrib, abl, per_group):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.3))
    order = attr.index
    axes[0].barh(order, attr["coefficient"], color="#1f77b4")
    axes[0].set_title("Task 1.3 coefficients (C=%g)" % C, fontsize=10)
    axes[1].barh(order, attr["share_of_hybrid_hits_also_in_member_top10"], color="#2ca02c")
    axes[1].set_title(f"Share of hybrid hits also in member's top-{K}", fontsize=10)
    a = abl[abl["removed"] != "(none)"].set_index("removed").loc[order]
    axes[2].barh(order, a["delta_vs_full"], color="#d62728")
    axes[2].set_title(f"Leave-one-out: change in test NDCG@{K}", fontsize=10)
    per_group.loc[order].plot.barh(ax=axes[3], width=0.8)
    axes[3].set_title("Weight share, fitted per activity group", fontsize=10)
    axes[3].legend(fontsize=7)
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(alpha=0.3, axis="x")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.tight_layout()
    fig.savefig(T2 / "fig_hybrid_analysis.png", dpi=200)
    plt.close(fig)


def main():
    T2.mkdir(parents=True, exist_ok=True)
    ctx = Context()
    groups = pd.read_csv(T2 / "user_groups.csv", dtype={"user_id": str}).set_index("user_id")

    corr = pd.read_csv(OUT / "hybrids" / "Weighted_score_correlation.csv", index_col=0)
    attr = hit_attribution(ctx)
    attr["mean_corr_with_other_members"] = (corr.sum() - 1) / (len(corr) - 1)
    contrib, abl, per_group, group_ndcg = replica_analysis(ctx, groups)
    attr = attr.join(contrib)

    attr.round(4).to_csv(T2 / "hybrid_member_contribution.csv")
    abl.round(4).to_csv(T2 / "hybrid_ablation.csv", index=False)
    per_group.round(4).to_csv(T2 / "hybrid_coefficients_by_activity.csv")
    group_ndcg.round(4).to_csv(T2 / "hybrid_groupwise_weights_ndcg.csv", index=False)
    for name, df in [("member contribution", attr), ("ablation", abl),
                     ("coefficients per activity group", per_group),
                     ("global vs group-specific weights", group_ndcg)]:
        print(f"\n== {name}\n{df.round(4).to_string()}")
    plot(attr, contrib, abl, per_group)
    print(f"written to {T2}")


if __name__ == "__main__":
    main()
