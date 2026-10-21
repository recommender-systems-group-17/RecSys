"""
Content-based recommender (not available in RecBole, so implemented here).

Item representation: one-hot genres (19) + one-hot release decade, from
dataset/ml-100k/ml-100k.item. The decade block is scaled by `year_weight`.
Optional IDF weighting down-weights very common features (e.g. Drama).

User profile: weighted sum of the feature vectors of the items the user
interacted with in TRAIN only.
  weighting = "binary": every train item counts once
  weighting = "rating": weight = rating - user's mean rating, so disliked
                        items push the profile away from their features

Score(user, item) = cosine(profile, item features).

Ties: with only ~30 features, many movies share the exact same feature
vector, so nearly every user's top-10 contains tied scores. Left alone, ties
are broken by RecBole's internal item id, which follows the order of first
appearance in the data and so correlates with popularity. To keep the model
purely content-based, ties are broken randomly (seeded) instead.

Output has the same shape and masking as the RecBole models, so it plugs
into the same export, cache and hybrid code:
  valid scores mask train items; test scores mask train + valid items;
  the padding item (internal id 0) is always masked.
"""
import numpy as np
import pandas as pd

from common import (ROOT, build_config, ground_truth, ndcg_at_k, split_users,
                    train_matrix)
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed


def item_features(dataset, year_weight=1.0, idf=False):
    i_tok = dataset.field2token_id[dataset.iid_field]
    items = pd.read_csv(ROOT / "dataset/ml-100k/ml-100k.item", sep="\t", dtype=str)
    items.columns = [c.split(":")[0] for c in items.columns]

    genres = sorted({g for gs in items["class"].dropna() for g in gs.split()})
    years = pd.to_numeric(items["release_year"], errors="coerce")
    decades = sorted({int(y) // 10 * 10 for y in years.dropna()})

    F = np.zeros((dataset.item_num, len(genres) + len(decades)), dtype=np.float64)
    for (_, row), year in zip(items.iterrows(), years):
        i = i_tok.get(row["item_id"])
        if i is None:
            continue
        for g in str(row["class"]).split():
            if g in genres:
                F[i, genres.index(g)] = 1.0
        if not np.isnan(year):
            F[i, len(genres) + decades.index(int(year) // 10 * 10)] = year_weight

    if idf:
        df = (F > 0).sum(0)
        F *= np.log((F.shape[0] + 1) / (df + 1)) + 1.0
    return F


def train_content(params=None):
    """Same return structure as common.train(), plus precomputed scores."""
    p = {"weighting": "binary", "idf": False, "year_weight": 1.0, **(params or {})}
    config = build_config("Pop")  # only used for data settings (seed, split)
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    tr, va, te = data_preparation(config, dataset)
    n_u, n_i = dataset.user_num, dataset.item_num

    X = train_matrix(tr, n_u, n_i)
    if p["weighting"] == "rating":
        R = train_matrix(tr, n_u, n_i, dataset.config["RATING_FIELD"])
        mean = np.divide(R.sum(1), X.sum(1), out=np.zeros(n_u, np.float32), where=X.sum(1) > 0)
        W = (R - mean[:, None]) * X
    else:
        W = X

    F = item_features(dataset, p["year_weight"], p["idf"])
    P = W.astype(np.float64) @ F
    P /= np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    Fn = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)
    full = P @ Fn.T
    # seeded random tie-breaking, far smaller than any real score difference
    full += np.random.default_rng(config["seed"]).uniform(0, 1e-7, full.shape)

    Xv = train_matrix(va, n_u, n_i)
    scores, results = {}, {}
    for split, loader, seen in (("valid", va, X), ("test", te, X + Xv)):
        users = split_users(loader)
        S = full[users].copy()  # float64: float32 would undo the tie-breaking
        S[seen[users] > 0] = -np.inf
        S[:, 0] = -np.inf
        scores[split] = (S, users)
        results[split] = {"ndcg@10": float(ndcg_at_k(S, ground_truth(loader, users)).mean())}

    return {"config": config, "dataset": dataset, "train": tr, "valid": va, "test": te,
            "scores": scores, "valid_result": results["valid"], "test_result": results["test"]}