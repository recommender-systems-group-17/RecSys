"""
Task 1.1 + 1.2: tune every individual model on validation NDCG@10, retrain
with the best settings, and export everything the rest of the group needs.

Usage (from repo root):
    python task1/individual.py                    # tune + final, all models
    python task1/individual.py --models BPR EASE  # only some models
    python task1/individual.py --stage final      # reuse outputs/best_params.yaml
    python task1/individual.py --quick            # smoke test: 1 config, 2 epochs
    python task1/individual.py --models ContentBased   # our own content-based model
    python task1/individual.py --models LightGCN --resume   # only run configs not yet in
                                                            # outputs/tuning/LightGCN.csv

Writes:
    outputs/tuning/<Model>.csv            every config tried + validation scores
    outputs/best_params.yaml              chosen hyperparameters per model
    outputs/splits/ml-100k.*.tsv          shared train/valid/test split
    outputs/recommendations/{valid,test}/ml-100k_<Model>_top100.json
    outputs/individual_summary.csv        RecBole's own valid/test metrics (sanity check)
    cache/scores/{valid,test}/<Model>.npy full score matrices (local, used by hybrids.py)
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (MODELS, OUT, check_split, clear_cli_args, export_split,  # noqa: E402
                    export_topk, load_yaml, save_cache, save_yaml, score_matrix, split_users, train)

from content_based import train_content  # noqa: E402

GRIDS = Path(__file__).resolve().parent / "grids.yaml"
BEST = OUT / "best_params.yaml"


def run(name, params):
    return train_content(params) if name == "ContentBased" else train(name, params)


def combos(spec, quick):
    grid = spec.get("grid", {})
    keys = list(grid)
    values = [grid[k][:1] if quick else grid[k] for k in keys]
    fixed = dict(spec.get("fixed", {}))
    if quick and "epochs" in fixed:
        fixed.update(epochs=2, eval_step=1)
    for vals in itertools.product(*values):
        yield {**fixed, **dict(zip(keys, vals))}


def _same(a, b):
    try:
        return bool(np.isclose(float(a), float(b)))
    except (TypeError, ValueError):
        return str(a) == str(b)


def _previous(name, params, old):
    """Row of an earlier tuning run with exactly these params, or None."""
    if old is None:
        return None
    for _, row in old.iterrows():
        if all(k in row and _same(row[k], v) for k, v in params.items()):
            return row.to_dict()
    return None


def tune(models, quick, resume=False):
    grids = load_yaml(GRIDS)
    best = load_yaml(BEST) if BEST.exists() else {}
    (OUT / "tuning").mkdir(parents=True, exist_ok=True)
    for name in models:
        path = OUT / "tuning" / f"{name}.csv"
        old = pd.read_csv(path) if resume and path.exists() else None
        tried, rows = list(combos(grids.get(name, {}), quick)), []
        for params in tried:
            prev = _previous(name, params, old)
            if prev is not None:
                rows.append({**prev, **params})
                print(f"[tune] {name} {params} -> reused valid ndcg@10={prev['valid_ndcg@10']:.4f}",
                      flush=True)
                continue
            t0 = time.time()
            v = run(name, params)["valid_result"]
            rows.append({**params, **{f"valid_{m}": x for m, x in v.items()},
                         "seconds": round(time.time() - t0, 1)})
            print(f"[tune] {name} {params} -> valid ndcg@10={v['ndcg@10']:.4f}", flush=True)
        df = pd.DataFrame(rows)
        best[name] = tried[int(df["valid_ndcg@10"].idxmax())]
        df.sort_values("valid_ndcg@10", ascending=False).to_csv(
            OUT / "tuning" / f"{name}.csv", index=False)
        save_yaml(best, BEST)


def final(models, k):
    best = load_yaml(BEST) if BEST.exists() else {}
    summary_path = OUT / "individual_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    split_done = (OUT / "splits" / "ml-100k.test.tsv").exists()
    for name in models:
        params = best.get(name, {})
        print(f"[final] training {name} with {params}", flush=True)
        res = run(name, params)
        if split_done:
            check_split(res)
        else:
            export_split(res)
            split_done = True
        for split in ("valid", "test"):
            if "scores" in res:  # ContentBased computes its scores directly
                scores, users = res["scores"][split]
            else:
                users = split_users(res[split])
                scores = score_matrix(res["model"], res[split], users, res["config"]["device"])
            save_cache(name, split, scores, users)
            export_topk(name, scores, users, res["dataset"], split, k, res["config"])
        row = {"model": name, "params": str(params),
               **{f"recbole_valid_{m}": x for m, x in res["valid_result"].items()},
               **{f"recbole_test_{m}": x for m, x in res["test_result"].items()}}
        if not summary.empty:
            summary = summary[summary["model"] != name]
        summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)
        summary.to_csv(summary_path, index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["tune", "final", "all"], default="all")
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--quick", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="reuse configs already in outputs/tuning/<Model>.csv, only run new ones")
    p.add_argument("--export_k", type=int, default=100, help="length of exported lists")
    a = p.parse_args()
    clear_cli_args()
    if a.stage in ("tune", "all"):
        tune(a.models, a.quick, a.resume)
    if a.stage in ("final", "all"):
        final(a.models, a.export_k)


if __name__ == "__main__":
    main()