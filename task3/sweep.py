"""
Task 3.1 + 3.2 - apply every reranker to every base model for a grid of
lambdas, choose lambda on validation, and report the accuracy / beyond-accuracy
trade-off on test.

    python task3/sweep.py
    python task3/sweep.py --tolerance 0.05

Lambda selection: for each (model, reranker) the LARGEST lambda whose
validation NDCG@10 is at most `tolerance` (default 5%) below that model's
validation NDCG@10 without reranking. That is, as much of the beyond-accuracy
objective as we can get for a fixed, explicit accuracy budget. Hybrid-Weighted
was fitted on validation, so its validation NDCG is optimistic; the relative
drop that drives the selection is much less affected.

Outputs (outputs/task3/):
    sweep.csv               every (model, reranker, lambda, split) with all metrics
    selected.csv            chosen lambda per (model, reranker) and its test metrics
    selected_change.csv     % change of every metric vs. no reranking (test)
    per_user_selected.csv   per-user test metrics at lambda = 0 and the chosen lambda
    fig_tradeoff_curves.png NDCG@10 vs targeted metric along lambda (test)
"""
import argparse
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import K, LAMBDAS, MODELS, T3, candidates, contexts, evaluate, user_groups  # noqa: E402
from rerankers import RERANKERS, ItemTable, rerank  # noqa: E402

# metric each reranker is meant to improve, and whether higher is better
TARGET = {"Diversity": ("ILD", True), "Calibration": ("MiscalKL", False),
          "ItemFairness": ("Gini", False), "UserFairness": ("UserGap_PopMiscal", False)}
REPORT = ["NDCG", "Recall", "ILD", "MiscalKL", "Novelty", "APLT", "Coverage", "Gini",
          "Exposure_Tail", "PopMiscal", "UserGap_PopMiscal", "UserGap_NDCG"]


def sweep(ctxs, groups, tab):
    rows, per_user = [], []
    for model in MODELS:
        cands = {part: candidates(model, part) for part in ctxs}
        for name in RERANKERS:
            t = time.time()
            for lam in LAMBDAS:
                for part, ctx in ctxs.items():
                    if lam == 0 and name != RERANKERS[0]:
                        continue       # lambda = 0 is the same for every reranker
                    recs = rerank(cands[part], name, lam, tab)
                    row, pu = evaluate(recs, ctx, groups)
                    rows.append({"model": model, "reranker": name if lam else "Original",
                                 "lambda": lam, "split": part, **row})
                    if part == "test":
                        per_user.append(pu.assign(model=model, reranker=name if lam else "Original",
                                                  **{"lambda": lam}).reset_index())
            print(f"  {model:16s} {name:13s} {time.time() - t:5.1f}s")
    return pd.DataFrame(rows), pd.concat(per_user, ignore_index=True)


def with_baseline(df):
    """Copy the lambda = 0 rows to every reranker so each curve starts there."""
    base = df[df["reranker"] == "Original"]
    return pd.concat([df[df["reranker"] != "Original"]] +
                     [base.assign(reranker=r) for r in RERANKERS], ignore_index=True)


def select(df, tolerance):
    v = with_baseline(df[df["split"] == "valid"])
    t = with_baseline(df[df["split"] == "test"]).set_index(["model", "reranker", "lambda"])
    rows = []
    for (model, name), g in v.groupby(["model", "reranker"]):
        base = g.loc[g["lambda"] == 0, "NDCG"].iloc[0]
        ok = g[g["NDCG"] >= (1 - tolerance) * base]
        lam = ok["lambda"].max()
        rows.append({"model": model, "reranker": name, "lambda": lam,
                     "valid_NDCG_drop": 1 - ok.loc[ok["lambda"] == lam, "NDCG"].iloc[0] / base,
                     **t.loc[(model, name, lam), REPORT].to_dict()})
    sel = pd.DataFrame(rows)
    base = t.xs(0.0, level="lambda").groupby("model")[REPORT].first()
    change = sel.set_index(["model", "reranker"])[REPORT]
    change = (change - base.reindex(change.index.get_level_values(0)).values) / \
        base.reindex(change.index.get_level_values(0)).abs().values * 100
    return sel, change.replace([np.inf, -np.inf], np.nan)   # undefined when the base is 0


def plot(df, sel):
    t = with_baseline(df[df["split"] == "test"])
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    colors = dict(zip(MODELS, ["#1f77b4", "#2ca02c", "#9467bd", "#d62728"]))
    for ax, name in zip(axes, RERANKERS):
        col, higher = TARGET[name]
        for model in MODELS:
            g = t[(t["model"] == model) & (t["reranker"] == name)].sort_values("lambda")
            ax.plot(g[col], g["NDCG"], "-o", ms=3, color=colors[model], label=model)
            s = sel[(sel["model"] == model) & (sel["reranker"] == name)].iloc[0]
            ax.scatter(s[col], s["NDCG"], marker="*", s=180, color=colors[model],
                       edgecolor="black", zorder=5)
        ax.set_title(f"{name}  (target: {col}, {'higher' if higher else 'lower'} = better)",
                     fontsize=10)
        ax.set_xlabel(f"{col}@{K}")
        ax.set_ylabel(f"NDCG@{K}")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Test trade-off along lambda = 0 ... 1 (star = lambda chosen on validation)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(T3 / "fig_tradeoff_curves.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance", type=float, default=0.05,
                    help="max relative validation NDCG loss when choosing lambda")
    args = ap.parse_args()

    T3.mkdir(parents=True, exist_ok=True)
    ctxs = contexts()
    groups = user_groups(ctxs["test"])
    tab = ItemTable(ctxs["test"])

    df, per_user = sweep(ctxs, groups, tab)
    df.round(5).to_csv(T3 / "sweep.csv", index=False)
    sel, change = select(df, args.tolerance)
    sel.round(4).to_csv(T3 / "selected.csv", index=False)
    change.round(2).to_csv(T3 / "selected_change.csv")

    keep = sel[["model", "reranker", "lambda"]].assign(chosen=True)
    pu = per_user.merge(keep, on=["model", "reranker", "lambda"], how="left")
    pu = pu[(pu["reranker"] == "Original") | pu["chosen"].fillna(False).astype(bool)]
    pu.drop(columns="chosen").to_csv(T3 / "per_user_selected.csv", index=False)

    plot(df, sel)
    print("\n== chosen lambda and test metrics\n", sel.round(4).to_string(index=False))
    print("\n== % change vs no reranking (test)\n", change.round(1).to_string())
    print(f"written to {T3}")


if __name__ == "__main__":
    main()
