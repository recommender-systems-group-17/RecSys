"""
Task 2.1 - our own implementation of every metric (accuracy and beyond-accuracy).

Relevance is binary: an item is relevant for a user if it is in the user's test
set. Recommendation lists already exclude train and validation items.

Two kinds of metrics:
  * per-user metrics (`user_metrics`): averaged over users, and also used for
    user-group analysis and significance tests;
  * list-set metrics (`system_metrics`): need all users' lists at once
    (coverage, Gini, exposure shares, user-fairness gaps).
"""
from itertools import combinations

import numpy as np

ALPHA_CALIBRATION = 0.01  # Steck (2018) smoothing of q, so KL stays finite


# ------------------------------------------------------------------ accuracy

def precision(recs, rel, k):
    return len(set(recs[:k]) & rel) / k


def recall(recs, rel, k):
    return len(set(recs[:k]) & rel) / len(rel)


def ndcg(recs, rel, k):
    dcg = sum(1.0 / np.log2(r + 2) for r, i in enumerate(recs[:k]) if i in rel)
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(rel), k)))
    return dcg / idcg


def mrr(recs, rel, k):
    for r, i in enumerate(recs[:k]):
        if i in rel:
            return 1.0 / (r + 1)
    return 0.0


def hit_rate(recs, rel, k):
    return float(any(i in rel for i in recs[:k]))


def average_precision(recs, rel, k):
    hits, total = 0, 0.0
    for r, i in enumerate(recs[:k]):
        if i in rel:
            hits += 1
            total += hits / (r + 1)
    return total / min(len(rel), k)


# ------------------------------------------------------------ beyond accuracy

def arp(recs, ctx, k):
    """Average Recommendation Popularity: mean #train interactions of the list."""
    return float(np.mean([ctx.popularity(i) for i in recs[:k]]))


def aplt(recs, ctx, k):
    """Average Percentage of Long Tail: share of the list outside the Head group."""
    return float(np.mean([ctx.item_groups.get(i, "Tail") != "Head" for i in recs[:k]]))


def novelty(recs, ctx, k):
    """Mean self-information -log2(p(i)), p(i) = share of users who interacted with i."""
    p = [max(ctx.popularity(i), 1) / ctx.n_users for i in recs[:k]]
    return float(np.mean(-np.log2(p)))


def ild(recs, ctx, k):
    """Intra-list diversity: mean pairwise Jaccard distance between genre sets."""
    gs = [set(ctx.genres.get(i, ["unknown"])) for i in recs[:k]]
    d = [1 - len(a & b) / len(a | b) for a, b in combinations(gs, 2)]
    return float(np.mean(d))


def miscalibration(recs, profile, ctx, k, alpha=ALPHA_CALIBRATION):
    """KL(p || q~) between the genre distribution of the user's train profile (p)
    and of the list (q), with q~ = (1-alpha) q + alpha p  (Steck, 2018). 0 = calibrated."""
    p = ctx.genre_dist(profile)
    q = (1 - alpha) * ctx.genre_dist(recs[:k]) + alpha * p
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


def popularity_lift(recs, profile, ctx, k):
    """(ARP of the list - mean popularity of the profile) / mean popularity of the
    profile. > 0 means the model pushes this user towards more popular items."""
    prof = np.mean([ctx.popularity(i) for i in profile])
    return (arp(recs, ctx, k) - prof) / prof


# --------------------------------------------------------------- aggregation

ACCURACY = ["Precision", "Recall", "NDCG", "MRR", "HR", "MAP"]
BEYOND_PER_USER = ["ARP", "APLT", "Novelty", "ILD", "MiscalKL", "PopLift"]


def user_metrics(recs, ctx, k=10):
    """One row per user with all per-user metrics @k."""
    rows = {}
    for u in ctx.users:
        r, rel, prof = recs.get(u, []), ctx.test[u], ctx.profile[u]
        rows[u] = {
            "Precision": precision(r, rel, k), "Recall": recall(r, rel, k),
            "NDCG": ndcg(r, rel, k), "MRR": mrr(r, rel, k),
            "HR": hit_rate(r, rel, k), "MAP": average_precision(r, rel, k),
            "ARP": arp(r, ctx, k), "APLT": aplt(r, ctx, k),
            "Novelty": novelty(r, ctx, k), "ILD": ild(r, ctx, k),
            "MiscalKL": miscalibration(r, prof, ctx, k),
            "PopLift": popularity_lift(r, prof, ctx, k),
        }
    return rows


def gini(x):
    """Gini index of a non-negative vector (0 = equal, 1 = all mass on one entry)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float(np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / (n * x.sum()))


def exposure(recs, ctx, k):
    """Number of times each catalog item appears in a top-k list."""
    counts = dict.fromkeys(ctx.catalog, 0)
    for u in ctx.users:
        for i in recs.get(u, [])[:k]:
            if i in counts:
                counts[i] += 1
    return counts


def system_metrics(recs, per_user, ctx, user_groups, k=10):
    """Metrics over the full set of lists.

    user_groups: {user: "Active"/"Inactive"} used for the user-fairness gap.
    """
    exp = exposure(recs, ctx, k)
    total = sum(exp.values())
    out = {
        "Coverage": sum(v > 0 for v in exp.values()) / len(exp),
        "Gini": gini(list(exp.values())),
    }
    # item-side: share of recommendation slots per popularity group
    for g in ("Head", "Mid", "Tail"):
        out[f"Exposure_{g}"] = sum(v for i, v in exp.items() if ctx.item_groups[i] == g) / total
    # user-side: NDCG gap between active and inactive users (Li et al., 2021)
    nd = {g: np.mean([per_user[u]["NDCG"] for u in ctx.users if user_groups[u] == g])
          for g in ("Active", "Inactive")}
    out["UserGap_NDCG"] = nd["Active"] - nd["Inactive"]
    return out


def item_group_recall(recs, ctx, k=10):
    """For each item popularity group: share of the relevant test items of that
    group that were recommended (summed over users), i.e. how well a model
    retrieves Head / Mid / Tail items when they are relevant."""
    hit, rel = {"Head": 0, "Mid": 0, "Tail": 0}, {"Head": 0, "Mid": 0, "Tail": 0}
    for u in ctx.users:
        top = set(recs.get(u, [])[:k])
        for i in ctx.test[u]:
            g = ctx.item_groups.get(i, "Tail")
            rel[g] += 1
            hit[g] += i in top
    return {g: hit[g] / rel[g] for g in hit}
