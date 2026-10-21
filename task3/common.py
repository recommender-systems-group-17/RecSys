"""
Shared helpers for Task 3: paths, loading candidate lists, and the evaluation
of a set of reranked lists with the Task 2 metrics (plus the two extra
user-side metrics the popularity-calibration reranker targets).

Task 3 reuses task2/data.py and task2/metrics.py unchanged, so every number is
comparable with Task 2.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "task2"))

from data import OUT, Context, load_recs_with_scores, truth_sets, user_groups  # noqa: E402
from metrics import (ACCURACY, BEYOND_PER_USER, item_group_recall,  # noqa: E402,F401
                     system_metrics, user_metrics)

T3 = OUT / "task3"
K = 10
# finer near 1: objective gains are small next to relevance differences, so
# the item- and user-fairness rerankers only act strongly for lambda > 0.9
LAMBDAS = [round(x, 2) for x in np.linspace(0, 0.9, 10)] + [0.95, 0.98, 1.0]
POP_GROUPS = ("Head", "Mid", "Tail")

# Base models the rerankers are applied to (3.2), all from the Task 1 study set:
# the most accurate individual model (EASE), a graph model (LightGCN), the
# content-based model, which ignores popularity entirely, and the weighted hybrid.
MODELS = ["EASE", "LightGCN", "ContentBased", "Hybrid-Weighted"]


def candidates(model, part):
    """{user: (items, normalised scores)} from the exported top-100 list.
    Scores are min-max normalised per user to [0, 1] so lambda means the same
    for every model."""
    out = {}
    for u, (items, scores) in load_recs_with_scores(model, part).items():
        s = np.asarray(scores, dtype=float)
        out[u] = (items, (s - s.min()) / max(s.max() - s.min(), 1e-12))
    return out


def contexts():
    """Context for test (Task 2's) and a copy whose ground truth is the
    validation split, used to choose lambda."""
    test = Context()
    valid = Context.__new__(Context)
    valid.__dict__.update(test.__dict__)
    valid.test = truth_sets("valid")
    return {"test": test, "valid": valid}


# ---------------------------------------------------- extra user-side metrics

def pop_distribution(items, ctx):
    """Share of the items in the Head / Mid / Tail popularity groups."""
    g = [ctx.item_groups.get(i, "Tail") for i in items]
    return np.array([g.count(x) for x in POP_GROUPS], dtype=float) / max(len(g), 1)


def jensen_shannon(p, q):
    m = (p + q) / 2
    kl = lambda a, b: float(np.sum(a[a > 0] * np.log2(a[a > 0] / b[a > 0])))  # noqa: E731
    return (kl(p, m) + kl(q, m)) / 2


def pop_miscalibration(recs, profile, ctx, k=K):
    """JS divergence between the popularity-group distribution of the user's
    train profile and of the list (Abdollahpouri et al., 2021). 0 = the list
    is as mainstream as the user."""
    return jensen_shannon(pop_distribution(profile, ctx), pop_distribution(recs[:k], ctx))


def evaluate(recs, ctx, groups, k=K):
    """All Task 2 metrics for one set of lists, plus PopMiscal and the user-side
    gaps. Returns (summary row, per-user DataFrame)."""
    pu = user_metrics(recs, ctx, k)
    df = pd.DataFrame(pu).T
    df["PopMiscal"] = [pop_miscalibration(recs.get(u, []), ctx.profile[u], ctx, k) for u in df.index]
    row = df[ACCURACY + BEYOND_PER_USER + ["PopMiscal"]].mean().to_dict()
    row.update(system_metrics(recs, pu, ctx, groups["active"].to_dict(), k))
    row.update({f"Recall_{g}": v for g, v in item_group_recall(recs, ctx, k).items()})
    ms = groups.loc[df.index, "mainstream"]
    row["UserGap_PopMiscal"] = df.loc[ms == "Niche", "PopMiscal"].mean() - \
        df.loc[ms == "Blockbuster", "PopMiscal"].mean()
    return row, df.rename_axis("user_id")


N_BOOT, SEED = 2000, 2020


def paired_ci(after, before, n_boot=N_BOOT, seed=SEED):
    """Mean of (after - before) over users with a 95% paired bootstrap CI."""
    d = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


__all__ = ["ROOT", "OUT", "T3", "K", "LAMBDAS", "MODELS", "candidates", "contexts",
           "evaluate", "user_groups", "pop_distribution", "jensen_shannon", "paired_ci"]
