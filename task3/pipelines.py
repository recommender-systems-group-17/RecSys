"""
Task 3.3 - where should the reranker go in a hybrid pipeline?

    python task3/pipelines.py

Both pipelines use the same weighted hybrid (Task 1.3 recipe, members and C
from outputs/hybrids/hybrid_params.yaml):

  A  "rerank members, then combine": every member's top-100 list is reranked
     into a new order first, and the hybrid combines the reranked lists.
  B  "combine, then rerank": the hybrid combines the original member lists,
     and the reranker is applied to the hybrid's top-100.

To combine reranked lists, members must be represented by their RANK (a
reranker changes the order, not the scores). So the hybrid here uses a
positional feature per member, 1 - rank/100 (0 if the item is not in the
member's top-100), with logistic-regression weights fitted on the validation
lists exactly as in Task 1.3. Members (EASE, UserKNN, LightGCN, ContentBased)
come from hybrid_params.yaml. At lambda = 0 both pipelines are this same
hybrid; its test NDCG is printed next to Hybrid-Weighted's as a check.

Since lambda acts on different inputs in A and B, the pipelines are compared
by their trade-off curves (NDCG vs. targeted metric), not at equal lambda.

Outputs (outputs/task3/):
    pipelines.csv          test metrics per (reranker, pipeline, lambda)
    pipeline_weights.csv   coefficients of the positional hybrid
    fig_pipelines.png
"""
import time

import matplotlib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import OUT, T3, K, candidates, contexts, evaluate, user_groups  # noqa: E402
from data import truth_sets  # noqa: E402
from rerankers import RERANKERS, ItemTable, rerank  # noqa: E402
from sweep import TARGET  # noqa: E402

DEPTH = 100
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95]
PARAMS = yaml.safe_load((OUT / "hybrids" / "hybrid_params.yaml").read_text())["Weighted"]
MEMBERS = list(PARAMS["weights"])
C = float(PARAMS["C"])


def positional(lists):
    """{member: {user: [items]}} -> {user: DataFrame items x members}."""
    out = {}
    users = next(iter(lists.values())).keys()
    for u in users:
        cols = {m: pd.Series(1 - np.arange(len(lists[m][u])) / DEPTH, index=lists[m][u])
                for m in MEMBERS}
        out[u] = pd.DataFrame(cols).fillna(0.0)
    return out


def fit_weights(feats, truth):
    X = pd.concat(feats.values())
    y = np.concatenate([np.isin(f.index, list(truth[u])) for u, f in feats.items()])
    lr = LogisticRegression(C=C, max_iter=5000).fit(X[MEMBERS].values, y)
    return pd.Series(lr.coef_[0], index=MEMBERS)


def combine(feats, w, depth=DEPTH):
    """Hybrid top-`depth` with min-max normalised scores, as reranker input."""
    out = {}
    for u, f in feats.items():
        s = (f[MEMBERS] @ w).sort_values(ascending=False)[:depth]
        v = s.values
        out[u] = (list(s.index), (v - v.min()) / max(v.max() - v.min(), 1e-12))
    return out


def main():
    T3.mkdir(parents=True, exist_ok=True)
    ctxs = contexts()
    ctx = ctxs["test"]
    groups = user_groups(ctx)
    tab = ItemTable(ctx)

    valid_lists = {m: {u: it for u, (it, _) in candidates(m, "valid").items()} for m in MEMBERS}
    w = fit_weights(positional(valid_lists), truth_sets("valid"))
    w.rename("coefficient").round(4).to_csv(T3 / "pipeline_weights.csv")

    member_cands = {m: candidates(m, "test") for m in MEMBERS}
    base_feats = positional({m: {u: it for u, (it, _) in c.items()} for m, c in member_cands.items()})
    hybrid = combine(base_feats, w)

    rows = []
    row, _ = evaluate({u: it[:K] for u, (it, _) in hybrid.items()}, ctx, groups)
    print(f"positional hybrid test NDCG@{K} = {row['NDCG']:.4f}")
    for name in RERANKERS:
        rows.append({"reranker": name, "pipeline": "A", "lambda": 0.0, **row})
        rows.append({"reranker": name, "pipeline": "B", "lambda": 0.0, **row})
    for name in RERANKERS:
        for lam in LAMBDAS[1:]:
            t = time.time()
            # A: rerank every member's full top-100 into a new order, then combine
            lists = {m: rerank(c, name, lam, tab, k=DEPTH) for m, c in member_cands.items()}
            a = combine(positional(lists), w, K)
            ra, _ = evaluate({u: it for u, (it, _) in a.items()}, ctx, groups)
            # B: combine the original lists, then rerank the hybrid's top-100
            rb, _ = evaluate(rerank(hybrid, name, lam, tab), ctx, groups)
            rows.append({"reranker": name, "pipeline": "A", "lambda": lam, **ra})
            rows.append({"reranker": name, "pipeline": "B", "lambda": lam, **rb})
            col = TARGET[name][0]
            print(f"  {name:13s} lambda={lam:<5} A: NDCG={ra['NDCG']:.4f} {col}={ra[col]:.4f} | "
                  f"B: NDCG={rb['NDCG']:.4f} {col}={rb[col]:.4f}   ({time.time() - t:.0f}s)")

    df = pd.DataFrame(rows)
    df.round(5).to_csv(T3 / "pipelines.csv", index=False)
    plot(df)
    print(f"written to {T3}")


def plot(df):
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    style = {"A": ("#ff7f0e", "A: rerank members, then combine"),
             "B": ("#1f77b4", "B: combine, then rerank")}
    for ax, name in zip(axes, RERANKERS):
        col, higher = TARGET[name]
        for p, (c, label) in style.items():
            g = df[(df["reranker"] == name) & (df["pipeline"] == p)].sort_values("lambda")
            ax.plot(g[col], g["NDCG"], "-o", ms=4, color=c, label=label)
            for _, r in g.iterrows():
                ax.annotate(f"{r['lambda']:g}", (r[col], r["NDCG"]), fontsize=6.5,
                            xytext=(3, 3), textcoords="offset points", color=c)
        ax.set_title(f"{name} ({'higher' if higher else 'lower'} {col} = better)", fontsize=10)
        ax.set_xlabel(f"{col}@{K}")
        ax.set_ylabel(f"NDCG@{K}")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Weighted hybrid: reranking before vs. after combining (labels = lambda)", fontsize=11)
    fig.tight_layout()
    fig.savefig(T3 / "fig_pipelines.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
