"""
Task 1 evidence for the report: uncertainty of model choices and
complementarity of the hybrid members.

Works only from the exported files (outputs/splits/*.tsv and
outputs/recommendations/*/*.json), independent of RecBole.

  A. Uncertainty of the individual-model choices (VALIDATION split, because
     all choices were made on validation):
       individual_ndcg_ci.csv     mean NDCG@10 + 95% bootstrap CI per model
                                  (valid and test)
       selection_pairs.csv        paired comparisons behind each selection
                                  decision: mean difference, 95% paired
                                  bootstrap CI, Wilcoxon and paired t-test
                                  p-values, Holm-adjusted

  B. Do the hybrids beat their best member? (TEST split; the validation
     lists of the learned hybrids are in-sample)
       hybrid_vs_members.csv      every hybrid vs. EASE and vs. its best member

  C. Complementarity of the hybrid members (VALIDATION split):
       member_overlap.csv         Jaccard overlap of the top-10 lists
       member_hits.csv            hits per member, hits no other member finds
                                  ("unique hits"), and the oracle upper bound

Usage (from repo root):
    python task1/analysis.py
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
AOUT = OUT / "analysis"
K = 10
N_BOOT = 2000
SEED = 2020

MEMBERS = ["EASE", "UserKNN", "LightGCN", "ContentBased"]
BASELINES = ["Pop", "Random"]
HYBRIDS = ["Weighted", "Switching", "Mixed", "Cascade",
           "FeatureCombination", "FeatureAugmentation", "MetaLevel"]

# Each selection decision in Task 1.1/1.2 and the comparison that supports it.
SELECTION_PAIRS = [
    ("EASE", "SLIMElastic", "linear item-item: EASE kept"),
    ("UserKNN", "ItemKNN", "neighbourhood: UserKNN kept"),
    ("LightGCN", "NGCF", "graph: LightGCN kept"),
    ("LightGCN", "BPR", "latent factor covered by LightGCN"),
    ("ContentBased", "Pop", "content-based vs popularity baseline"),
    ("EASE", "LightGCN", "best vs 2nd-best study model"),
    ("EASE", "UserKNN", "best vs 3rd-best study model"),
    ("LightGCN", "UserKNN", "2nd vs 3rd study model"),
]


# ------------------------------------------------------------------ loading

def load_truth(split):
    df = pd.read_csv(OUT / "splits" / f"ml-100k.{split}.tsv", sep="\t", dtype=str)
    return df.groupby(df.columns[0])[df.columns[1]].apply(set).to_dict()


def load_recs(model, split):
    path = OUT / "recommendations" / split / f"ml-100k_{model}_top100.json"
    if not path.exists():
        return None
    with open(path) as f:
        return {u: r["items"][:K] for u, r in json.load(f)["recommendations"].items()}


DISC = 1.0 / np.log2(np.arange(2, K + 2))


def per_user_ndcg(recs, truth, users):
    out = np.zeros(len(users))
    for j, u in enumerate(users):
        rel = truth[u]
        items = recs.get(u, [])
        dcg = sum(DISC[r] for r, i in enumerate(items) if i in rel)
        out[j] = dcg / DISC[:min(len(rel), K)].sum()
    return out


# ------------------------------------------------------------------ statistics

def boot_ci(x, idx):
    m = x[idx].mean(1)
    return np.quantile(m, 0.025), np.quantile(m, 0.975)


def compare(a, b, idx):
    d = a - b
    lo, hi = boot_ci(d, idx)
    if not np.any(d != 0):  # identical lists for every user (e.g. Switching == EASE)
        p_w = 1.0
    else:
        try:
            p_w = stats.wilcoxon(a, b, zero_method="wilcox").pvalue
        except ValueError:
            p_w = 1.0
    p_t = stats.ttest_rel(a, b).pvalue if np.any(d != 0) else 1.0
    return {"mean_a": a.mean(), "mean_b": b.mean(), "diff": d.mean(),
            "diff_ci_low": lo, "diff_ci_high": hi,
            "p_wilcoxon": p_w, "p_ttest": p_t,
            "users_a_better": int((d > 0).sum()), "users_b_better": int((d < 0).sum())}


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


# ------------------------------------------------------------------ main

def main():
    AOUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    truth = {s: load_truth(s) for s in ("valid", "test")}
    users = {s: sorted(truth[s]) for s in truth}
    idx = {s: rng.integers(0, len(users[s]), (N_BOOT, len(users[s]))) for s in users}

    def ndcg(model, split):
        recs = load_recs(model, split)
        return None if recs is None else per_user_ndcg(recs, truth[split], users[split])

    # ---- A. uncertainty of individual-model choices
    individual = sorted({m for pair in SELECTION_PAIRS for m in pair[:2]} | set(MEMBERS + BASELINES))
    scores = {s: {m: ndcg(m, s) for m in individual} for s in ("valid", "test")}
    rows = []
    for m in individual:
        if scores["valid"][m] is None:
            continue
        row = {"model": m, "in_study": m in MEMBERS + BASELINES}
        for s in ("valid", "test"):
            lo, hi = boot_ci(scores[s][m], idx[s])
            row.update({f"{s}_ndcg@10": scores[s][m].mean(), f"{s}_ci_low": lo, f"{s}_ci_high": hi})
        rows.append(row)
    ci = pd.DataFrame(rows).sort_values("valid_ndcg@10", ascending=False)
    ci.round(4).to_csv(AOUT / "individual_ndcg_ci.csv", index=False)

    rows = []
    for a, b, why in SELECTION_PAIRS:
        if scores["valid"].get(a) is None or scores["valid"].get(b) is None:
            continue
        rows.append({"a": a, "b": b, "decision": why, "split": "valid",
                     **compare(scores["valid"][a], scores["valid"][b], idx["valid"])})
    sel = pd.DataFrame(rows)
    if not sel.empty:
        sel["p_wilcoxon_holm"] = holm(sel["p_wilcoxon"])
        sel["significant_0.05"] = sel["p_wilcoxon_holm"] < 0.05
    sel.round(4).to_csv(AOUT / "selection_pairs.csv", index=False)

    # ---- B. hybrids vs members (test)
    member_test = {m: ndcg(m, "test") for m in MEMBERS}
    best_member = max(MEMBERS, key=lambda m: member_test[m].mean())
    rows = []
    for h in HYBRIDS:
        hs = ndcg(f"Hybrid-{h}", "test")
        if hs is None:
            continue
        rows.append({"hybrid": h, "vs": best_member,
                     **compare(hs, member_test[best_member], idx["test"])})
    hyb = pd.DataFrame(rows)
    if not hyb.empty:
        hyb["p_wilcoxon_holm"] = holm(hyb["p_wilcoxon"])
        hyb["significant_0.05"] = hyb["p_wilcoxon_holm"] < 0.05
    hyb.round(4).to_csv(AOUT / "hybrid_vs_members.csv", index=False)

    # ---- C. complementarity of members (valid)
    recs = {m: load_recs(m, "valid") for m in MEMBERS}
    U = users["valid"]
    overlap = pd.DataFrame(index=MEMBERS, columns=MEMBERS, dtype=float)
    for a in MEMBERS:
        for b in MEMBERS:
            overlap.loc[a, b] = np.mean([len(set(recs[a][u]) & set(recs[b][u])) /
                                         max(len(set(recs[a][u]) | set(recs[b][u])), 1) for u in U])
    overlap.round(3).to_csv(AOUT / "member_overlap.csv")

    hits = {m: {u: set(recs[m][u]) & truth["valid"][u] for u in U} for m in MEMBERS}
    rows = []
    for m in MEMBERS:
        others = [x for x in MEMBERS if x != m]
        total = sum(len(hits[m][u]) for u in U)
        unique = sum(len(hits[m][u] - set().union(*(hits[o][u] for o in others))) for u in U)
        rows.append({"model": m, "hits": total, "unique_hits": unique,
                     "unique_share_of_own_hits": unique / max(total, 1),
                     "users_with_hit": sum(bool(hits[m][u]) for u in U) / len(U)})
    union_hits = sum(len(set().union(*(hits[m][u] for m in MEMBERS))) for u in U)
    rows.append({"model": "union of all members (oracle)", "hits": union_hits,
                 "unique_hits": np.nan, "unique_share_of_own_hits": np.nan,
                 "users_with_hit": sum(any(hits[m][u] for m in MEMBERS) for u in U) / len(U)})
    pd.DataFrame(rows).round(4).to_csv(AOUT / "member_hits.csv", index=False)

    # ---- print summary
    pd.set_option("display.width", 200)
    print("\n=== A. NDCG@10 with 95% bootstrap CI ===")
    print(ci.round(4).to_string(index=False))
    print("\n=== A. Selection decisions (validation, paired) ===")
    print(sel[["a", "b", "diff", "diff_ci_low", "diff_ci_high", "p_wilcoxon_holm",
               "significant_0.05"]].round(4).to_string(index=False))
    print(f"\n=== B. Hybrids vs best member ({best_member}) on test ===")
    print(hyb[["hybrid", "diff", "diff_ci_low", "diff_ci_high", "p_wilcoxon_holm",
               "significant_0.05"]].round(4).to_string(index=False))
    print("\n=== C. Top-10 overlap (Jaccard) between members, validation ===")
    print(overlap.round(3).to_string())
    print("\n=== C. Hits and unique hits, validation ===")
    print(pd.read_csv(AOUT / "member_hits.csv").to_string(index=False))
    print(f"\nAll tables written to {AOUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
