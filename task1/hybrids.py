"""
Task 1.3-1.5: hybrid recommenders (Burke's 7 types), built on top of the
individual models exported by task1/individual.py.

  Weighted             scores of all members, normalised per user, combined with
                       weights learned by logistic regression            (1.3)
  Switching            users bucketed by a user characteristic (activity,
                       mainstreamness or genre diversity); a bucket switches
                       away from the overall best member only if another
                       member is significantly better there (95% paired
                       bootstrap CI > 0)                                 (1.4)
  Mixed                reciprocal rank fusion of several members' lists  (1.4)
  Cascade              member A picks the top-N candidates, member B
                       re-orders them                                    (1.4)
  FeatureCombination   ONE learner (gradient boosting) on collaborative +
                       content/demographic features; uses no member outputs (1.4)
  FeatureAugmentation  member A's top-n predictions are added to the training
                       matrix as pseudo-interactions, then EASE is fit on it (1.4)
  MetaLevel            the item-item model learned by EASE is used as the user
                       representation inside a user-based kNN            (1.4)

Members (default, one model per family; change with --members):
  EASE (linear item-item), UserKNN (neighbourhood), LightGCN (graph /
  latent factor), ContentBased (content-based).
Pop and Random are baselines only. FeatureAugmentation and MetaLevel always
use EASE internally, whatever the members are.

Tuning (1.5)
  Hybrids with learned parameters (Weighted, Switching, FeatureCombination):
  validation users are split 50/50 into FIT and SELECT halves. Parameters are
  fitted on FIT, hyperparameters chosen on SELECT, then refitted on all
  validation users and applied to test.
  Hybrids without learned parameters (the rest): hyperparameters are chosen on
  all validation users. Test is never used for any choice.

Extra evidence written for the report (Weighted hybrid):
  Weighted_coefficient_stability.csv  bootstrap CIs of the coefficients
  Weighted_ablation.csv               drop-one-member effect on SELECT users
Switching_groups_all.csv: per-group NDCG of every member, every criterion.

All models only ever see the TRAIN split as input, so valid and test scores
are produced the same way as RecBole's individual models.

Usage (from repo root):
    python task1/hybrids.py                 # all hybrids, default members
    python task1/hybrids.py --only Cascade MetaLevel
    python task1/hybrids.py --members UserKNN EASE BPR   # other member set
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (MODELS, OUT, ROOT, build_config, clear_cli_args,  # noqa: E402
                    export_topk, ground_truth, load_cache, load_yaml, ndcg_at_k,
                    save_yaml, train_matrix)
from recbole.data import create_dataset, data_preparation  # noqa: E402
from recbole.utils import init_seed  # noqa: E402

HOUT = OUT / "hybrids"
K_EXPORT = 100
SEED = 2020
DEFAULT_MEMBERS = ["EASE", "UserKNN", "LightGCN", "ContentBased"]
N_BOOT = 200            # bootstrap refits for coefficient stability
SWITCH_CRITERIA = ["activity", "mainstreamness", "genre_diversity"]
HYBRIDS = ["Weighted", "Switching", "Mixed", "Cascade",
           "FeatureCombination", "FeatureAugmentation", "MetaLevel"]


# ============================ shared utilities ===============================

def normalize(S):
    """Per-user min-max to [0, 1] over non-masked items; masked items -> 0."""
    fin = np.isfinite(S)
    lo = np.where(fin, S, np.inf).min(1, keepdims=True)
    hi = np.where(fin, S, -np.inf).max(1, keepdims=True)
    N = (S - lo) / np.maximum(hi - lo, 1e-12)
    N[~fin] = 0.0
    return N.astype(np.float32)


def masked(S, mask):
    S = S.astype(np.float32, copy=True)
    S[mask] = -np.inf
    return S


def mean_ndcg(S, truth, rows=None):
    if rows is not None:
        S, truth = S[rows], [t for t, r in zip(truth, rows) if r]
    return float(ndcg_at_k(S, truth).mean())


def ease_weights(X, reg):
    """Closed-form EASE item-item matrix (Steck, 2019)."""
    G = X.T.astype(np.float64) @ X
    G[np.diag_indices_from(G)] += reg
    P = np.linalg.inv(G)
    B = -P / np.diag(P)
    B[np.diag_indices_from(B)] = 0.0
    return B.astype(np.float32)


def topn_mask(S, n):
    idx = np.argpartition(-S, n, axis=1)[:, :n]
    M = np.zeros(S.shape, dtype=bool)
    np.put_along_axis(M, idx, True, axis=1)
    return M


# ================================ hybrids ====================================

class Hybrids:
    def __init__(self, members):
        self.members = members
        self.config, self.dataset, loaders = self._load_split()
        ds = self.dataset
        self.n_users, self.n_items = ds.user_num, ds.item_num

        # full train matrices, indexed by internal id
        self.X = train_matrix(loaders["train"], self.n_users, self.n_items)
        self.R = train_matrix(loaders["train"], self.n_users, self.n_items, ds.config["RATING_FIELD"])

        self.G = self._genre_matrix()
        crit_all = self._user_characteristics()

        self.S = {}
        for split in ("valid", "test"):
            scores = {m: load_cache(m, split)[0] for m in members}
            users = load_cache(members[0], split)[1]
            mask = ~np.isfinite(scores[members[0]])
            self.S[split] = {
                "users": users, "mask": mask, "scores": scores,
                "norm": {m: normalize(s) for m, s in scores.items()},
                "truth": ground_truth(loaders[split], users),
                "counts": self.X[users].sum(1),
                "crit": {c: v[users] for c, v in crit_all.items()},
            }
        V = self.S["valid"]
        rng = np.random.default_rng(SEED)
        self.fit = rng.random(len(V["users"])) < 0.5
        self.sel = ~self.fit
        self.per_model_v = {m: ndcg_at_k(V["scores"][m], V["truth"]) for m in members}
        self.rank = sorted(members, key=lambda m: -self.per_model_v[m].mean())
        best = load_yaml(OUT / "best_params.yaml") if (OUT / "best_params.yaml").exists() else {}
        self.ease_reg = float(best.get("EASE", {}).get("reg_weight", 250.0))
        print("members by validation NDCG@10:",
              ", ".join(f"{m}={self.per_model_v[m].mean():.4f}" for m in self.rank))

    def _genre_matrix(self):
        ds = self.dataset
        i_tok = ds.field2token_id[ds.iid_field]
        items = pd.read_csv(ROOT / "dataset/ml-100k/ml-100k.item", sep="\t", dtype=str)
        items.columns = [c.split(":")[0] for c in items.columns]
        genres = sorted({g for gs in items["class"].dropna() for g in gs.split()})
        G = np.zeros((self.n_items, len(genres)), dtype=np.float32)
        for item_id, gs in zip(items["item_id"], items["class"]):
            i = i_tok.get(item_id)
            if i is None:
                continue
            for g in str(gs).split():
                if g in genres:
                    G[i, genres.index(g)] = 1.0
        return G

    def _user_characteristics(self):
        """Per-user values (all internal ids) used as switching criteria; TRAIN only.
          activity        number of train interactions
          mainstreamness  mean log-popularity of the user's train items
          genre_diversity entropy of the genre distribution of the user's train items"""
        X = self.X
        n_u = np.maximum(X.sum(1), 1.0)
        log_pop = np.log1p(X.sum(0))
        prof = X @ self.G
        p = prof / np.maximum(prof.sum(1, keepdims=True), 1e-12)
        ent = -(p * np.log(np.where(p > 0, p, 1.0))).sum(1)
        return {"activity": X.sum(1), "mainstreamness": (X @ log_pop) / n_u,
                "genre_diversity": ent}

    @staticmethod
    def _load_split():
        config = build_config("Pop")
        init_seed(config["seed"], config["reproducibility"])
        dataset = create_dataset(config)
        tr, va, te = data_preparation(config, dataset)
        return config, dataset, {"train": tr, "valid": va, "test": te}

    def save_tuning(self, name, rows, key):
        df = pd.DataFrame(rows).sort_values(key, ascending=False)
        df.to_csv(HOUT / f"{name}_tuning.csv", index=False)
        return df

    # ---------------------------------------------------------------- 1.3
    def _fit_lr(self, rows, C, depth=50, members=None):
        """Logistic regression on candidate (user, item) pairs = union of each
        member's top-`depth` items. Label = item is in the validation set.
        rows: boolean mask over validation users, or an array of row indices
        (may contain repeats, used for bootstrapping)."""
        members = members or self.members
        V, Xs, ys = self.S["valid"], [], []
        rows = np.asarray(rows)
        idx = np.where(rows)[0] if rows.dtype == bool else rows
        tops = {m: np.argsort(-V["norm"][m][idx], axis=1)[:, :depth] for m in members}
        for r, u in enumerate(idx):
            cand = np.unique(np.concatenate([tops[m][r] for m in members]))
            Xs.append(np.stack([V["norm"][m][u, cand] for m in members], axis=1))
            ys.append(np.isin(cand, list(V["truth"][u])).astype(int))
        lr = LogisticRegression(C=C, max_iter=5000).fit(np.vstack(Xs), np.concatenate(ys))
        return dict(zip(members, lr.coef_[0].astype(float)))

    def _weighted(self, split, w):
        D = self.S[split]
        return masked(sum(x * D["norm"][m] for m, x in w.items()), D["mask"])

    def _coefficient_stability(self, C, w_full):
        """Refit on bootstrap samples of validation users -> how stable is each weight?"""
        rng = np.random.default_rng(SEED)
        n = len(self.S["valid"]["users"])
        boot = []
        for b in range(N_BOOT):
            boot.append(self._fit_lr(rng.integers(0, n, n), C))
            if (b + 1) % 50 == 0:
                print(f"  bootstrap {b + 1}/{N_BOOT}", flush=True)
        B = pd.DataFrame(boot)[self.members]
        share = B.abs().div(B.abs().sum(1), axis=0)
        largest = B.idxmax(axis=1)
        out = pd.DataFrame({
            "model": self.members,
            "coefficient": [w_full[m] for m in self.members],
            "boot_mean": B.mean().values,
            "ci_low": B.quantile(0.025).values,
            "ci_high": B.quantile(0.975).values,
            "share_mean": share.mean().values,
            "share_ci_low": share.quantile(0.025).values,
            "share_ci_high": share.quantile(0.975).values,
            "frac_positive": (B > 0).mean().values,
            "frac_largest_weight": [float((largest == m).mean()) for m in self.members],
        })
        out.sort_values("coefficient", ascending=False).round(4).to_csv(
            HOUT / "Weighted_coefficient_stability.csv", index=False)

    def _ablation(self, C):
        """Drop one member at a time: fit on FIT users, evaluate on SELECT users.
        Paired bootstrap CI of the per-user NDCG difference vs. all members."""
        V = self.S["valid"]
        truth_sel = [t for t, r in zip(V["truth"], self.sel) if r]

        def per_user(members):
            w = self._fit_lr(self.fit, C, members=members)
            return ndcg_at_k(self._weighted("valid", w)[self.sel], truth_sel)

        base = per_user(self.members)
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(base), (2000, len(base)))
        rows = [{"removed": "(none)", "select_ndcg@10": base.mean(), "delta": 0.0,
                 "delta_ci_low": 0.0, "delta_ci_high": 0.0}]
        for m in self.members:
            d = per_user([x for x in self.members if x != m]) - base
            boots = d[idx].mean(1)
            rows.append({"removed": m, "select_ndcg@10": base.mean() + d.mean(), "delta": d.mean(),
                         "delta_ci_low": np.quantile(boots, 0.025),
                         "delta_ci_high": np.quantile(boots, 0.975)})
            print(f"  without {m}: delta NDCG@10={d.mean():+.4f}")
        pd.DataFrame(rows).round(4).to_csv(HOUT / "Weighted_ablation.csv", index=False)

    def weighted(self):
        V, rows = self.S["valid"], []
        for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
            w = self._fit_lr(self.fit, C)
            s = mean_ndcg(self._weighted("valid", w), V["truth"], self.sel)
            rows.append({"C": C, "select_ndcg@10": s, **{f"w_{m}": x for m, x in w.items()}})
            print(f"  C={C}: select NDCG@10={s:.4f}")
        df = self.save_tuning("Weighted", rows, "select_ndcg@10")
        C = float(df.iloc[0]["C"])
        w = self._fit_lr(np.ones_like(self.fit), C)

        coef = pd.DataFrame({"model": list(w), "coefficient": list(w.values())})
        coef["abs_share"] = coef["coefficient"].abs() / coef["coefficient"].abs().sum()
        coef.sort_values("coefficient", ascending=False).to_csv(
            HOUT / "Weighted_coefficients.csv", index=False)
        # correlated inputs make single coefficients hard to read -> save correlations
        keep = ~V["mask"]
        sample = np.stack([V["norm"][m][keep] for m in self.members], axis=1)
        pd.DataFrame(np.corrcoef(sample.T), index=self.members,
                     columns=self.members).round(3).to_csv(HOUT / "Weighted_score_correlation.csv")
        print("  coefficient stability (bootstrap)...", flush=True)
        self._coefficient_stability(C, w)
        print("  drop-one-member ablation...", flush=True)
        self._ablation(C)
        return {"C": C, "weights": w}, self._weighted("valid", w), self._weighted("test", w)

    # ---------------------------------------------------------------- switching
    @staticmethod
    def _groups(values, edges):
        return np.digitize(values, edges[1:-1], right=True)

    def _switch(self, split, crit, edges, choice):
        D = self.S[split]
        g = self._groups(D["crit"][crit], edges)
        out = D["scores"][self.rank[0]].copy()  # fallback for empty groups
        for k, m in choice.items():
            out[g == k] = D["scores"][m][g == k]
        return out

    def _choose(self, rows):
        """Member for a group of validation users (boolean mask `rows`).
        Default = the best member overall. Switch to another member only if it
        is better on this group with a 95% paired bootstrap CI above zero;
        otherwise near-ties (noise) would decide the assignment.
        Returns (member, lower CI bound of its gain over the default)."""
        default = self.rank[0]
        best = max(self.members, key=lambda m: self.per_model_v[m][rows].mean())
        if best == default:
            return default, 0.0
        d = self.per_model_v[best][rows] - self.per_model_v[default][rows]
        rng = np.random.default_rng(SEED)
        lo = float(np.quantile(d[rng.integers(0, len(d), (2000, len(d)))].mean(1), 0.025))
        return (best if lo > 0 else default), lo

    def _group_table(self, crit, n):
        V = self.S["valid"]
        vals = V["crit"][crit]
        edges = np.quantile(vals, np.linspace(0, 1, n + 1))
        g = self._groups(vals, edges)
        table = pd.DataFrame({m: [self.per_model_v[m][g == k].mean() if (g == k).any() else np.nan
                                  for k in range(n)] for m in self.members})
        table.insert(0, "range", [f"{edges[k]:.2f}-{edges[k + 1]:.2f}" for k in range(n)])
        table.insert(1, "n_users", [int((g == k).sum()) for k in range(n)])
        table["best"] = table[self.members].idxmax(axis=1)
        table["margin_over_2nd"] = table[self.members].apply(
            lambda r: r.nlargest(2).iloc[0] - r.nlargest(2).iloc[-1], axis=1)
        guarded = [self._choose(g == k) if (g == k).any() else (self.rank[0], 0.0)
                   for k in range(n)]
        table["gain_ci_low_vs_default"] = [lo for _, lo in guarded]
        table["chosen"] = [m for m, _ in guarded]
        return edges, g, table

    def switching(self):
        V, rows, groups_all = self.S["valid"], [], []
        for crit in SWITCH_CRITERIA:
            vals = V["crit"][crit]
            for n in [2, 3, 4, 5]:
                edges = np.quantile(vals, np.linspace(0, 1, n + 1))
                g = self._groups(vals, edges)
                choice = {k: self._choose(self.fit & (g == k))[0]
                          for k in range(n) if (self.fit & (g == k)).any()}
                s = mean_ndcg(self._switch("valid", crit, edges, choice), V["truth"], self.sel)
                rows.append({"criterion": crit, "n_groups": n, "select_ndcg@10": s,
                             "assignment": str(choice),
                             "switches": len(set(choice.values())) > 1})
                print(f"  {crit}, groups={n}: select NDCG@10={s:.4f} {choice}")
                _, _, t = self._group_table(crit, n)
                t.insert(0, "group", range(n))
                t.insert(0, "n_groups", n)
                t.insert(0, "criterion", crit)
                groups_all.append(t)
        pd.concat(groups_all).round(4).to_csv(HOUT / "Switching_groups_all.csv", index=False)
        best = self.save_tuning("Switching", rows, "select_ndcg@10").iloc[0]
        crit, n = best["criterion"], int(best["n_groups"])

        # refit the assignment on all validation users (same guard)
        edges, g, table = self._group_table(crit, n)
        choice = {k: str(table.loc[k, "chosen"]) for k in range(n)}
        table.round(4).to_csv(HOUT / "Switching_assignment.csv", index_label="group")
        if len(set(choice.values())) == 1:
            print(f"  note: no member beats {choice[0]} significantly in any group, so "
                  f"Switching equals {choice[0]}. See Switching_groups_all.csv.")
        params = {"criterion": crit, "n_groups": n, "edges": [float(e) for e in edges],
                  "assignment": choice}
        return (params, self._switch("valid", crit, edges, choice),
                self._switch("test", crit, edges, choice))

    # ---------------------------------------------------------------- mixed
    def _rrf(self, split, models, c):
        D = self.S[split]
        out = np.zeros(D["mask"].shape, dtype=np.float32)
        for m in models:
            ranks = np.argsort(np.argsort(-D["scores"][m], axis=1), axis=1) + 1
            out += 1.0 / (c + ranks)
        return masked(out, D["mask"])

    def mixed(self):
        V, rows = self.S["valid"], []
        for n_models in range(2, len(self.rank) + 1):
            for c in [1, 5, 10, 30, 60, 100]:
                ms = self.rank[:n_models]
                s = mean_ndcg(self._rrf("valid", ms, c), V["truth"])
                rows.append({"models": "+".join(ms), "n_models": n_models, "c": c, "valid_ndcg@10": s})
        best = self.save_tuning("Mixed", rows, "valid_ndcg@10").iloc[0]
        ms, c = best["models"].split("+"), int(best["c"])
        print(f"  best: {ms}, c={c}, valid NDCG@10={best['valid_ndcg@10']:.4f}")
        return {"models": ms, "c": c}, self._rrf("valid", ms, c), self._rrf("test", ms, c)

    # ---------------------------------------------------------------- cascade
    def _cascade(self, split, a, b, n):
        D = self.S[split]
        cand = topn_mask(D["scores"][a], n)
        # candidates of A first (ordered by B), everything else after (ordered by A)
        out = np.where(cand, 2.0 + D["norm"][b], D["norm"][a])
        return masked(out, D["mask"])

    def cascade(self):
        V, rows, top = self.S["valid"], [], self.rank[:4]
        for a in top:
            for b in top:
                if a == b:
                    continue
                for n in [20, 50, 100]:
                    s = mean_ndcg(self._cascade("valid", a, b, n), V["truth"])
                    rows.append({"stage1": a, "stage2": b, "N": n, "valid_ndcg@10": s})
        best = self.save_tuning("Cascade", rows, "valid_ndcg@10").iloc[0]
        a, b, n = best["stage1"], best["stage2"], int(best["N"])
        print(f"  best: {a} -> top-{n} -> {b}, valid NDCG@10={best['valid_ndcg@10']:.4f}")
        return ({"stage1": a, "stage2": b, "N": n},
                self._cascade("valid", a, b, n), self._cascade("test", a, b, n))

    # ---------------------------------------------------------------- feature combination
    def _side_features(self):
        """Per (user, item) feature matrices, computed from TRAIN only."""
        ds, X, R = self.dataset, self.X, self.R
        u_tok = ds.field2token_id[ds.uid_field]
        i_tok = ds.field2token_id[ds.iid_field]

        items = pd.read_csv(ROOT / "dataset/ml-100k/ml-100k.item", sep="\t", dtype=str)
        items.columns = [c.split(":")[0] for c in items.columns]
        genres = sorted({g for gs in items["class"].dropna() for g in gs.split()})
        G = np.zeros((self.n_items, len(genres)), dtype=np.float32)
        year = np.zeros(self.n_items, dtype=np.float32)
        for _, r in items.iterrows():
            i = i_tok.get(r["item_id"])
            if i is None:
                continue
            for g in str(r["class"]).split():
                if g in genres:
                    G[i, genres.index(g)] = 1.0
            year[i] = float(r["release_year"]) if str(r["release_year"]).isdigit() else np.nan
        year = np.nan_to_num(year, nan=np.nanmean(year[year > 0]))

        users = pd.read_csv(ROOT / "dataset/ml-100k/ml-100k.user", sep="\t", dtype=str)
        users.columns = [c.split(":")[0] for c in users.columns]
        demo = {}
        for col in ("gender", "occupation", "age"):
            vals = users[col] if col != "age" else (users["age"].astype(int) // 10).astype(str)
            lab = np.zeros(self.n_users, dtype=int)
            cats = sorted(vals.unique())
            for t, v in zip(users["user_id"], vals):
                if t in u_tok:
                    lab[u_tok[t]] = cats.index(v)
            demo[col] = (lab, len(cats))

        n_u = X.sum(1, keepdims=True)
        n_i = X.sum(0, keepdims=True)
        item_avg = np.divide(R.sum(0), n_i[0], out=np.zeros(self.n_items, np.float32), where=n_i[0] > 0)
        user_avg = np.divide(R.sum(1), n_u[:, 0], out=np.zeros(self.n_users, np.float32), where=n_u[:, 0] > 0)

        # content: cosine between the user's genre profile and the item's genres
        prof = X @ G
        prof /= np.maximum(np.linalg.norm(prof, axis=1, keepdims=True), 1e-9)
        Gn = G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-9)
        genre_aff = prof @ Gn.T

        # collaborative: mean item-item cosine similarity to the user's history
        Xn = X / np.maximum(np.sqrt(n_i), 1e-9)
        sim = Xn.T @ Xn
        np.fill_diagonal(sim, 0.0)
        cf_sim = (X @ sim) / np.maximum(n_u, 1.0)

        feats = {
            "log_item_pop": np.broadcast_to(np.log1p(n_i), X.shape),
            "item_avg_rating": np.broadcast_to(item_avg[None, :], X.shape),
            "release_year": np.broadcast_to(year[None, :], X.shape),
            "log_user_activity": np.broadcast_to(np.log1p(n_u), X.shape),
            "user_avg_rating": np.broadcast_to(user_avg[:, None], X.shape),
            "genre_affinity": genre_aff,
            "cf_item_similarity": cf_sim,
        }
        # demographic: share of the user's demographic group that interacted with the item
        for col, (lab, k) in demo.items():
            onehot = np.eye(k, dtype=np.float32)[lab]
            grp_items = onehot.T @ X
            grp_size = np.maximum(onehot.sum(0)[:, None], 1.0)
            feats[f"{col}_group_pop"] = (grp_items / grp_size)[lab]
        return feats

    def _fc_matrix(self, users, items=None):
        cols = [self.F[f][users] if items is None else self.F[f][users, items] for f in self.fnames]
        return np.stack(cols, axis=-1)

    def _fc_fit(self, rows, params, neg_per_pos=10):
        V = self.S["valid"]
        rng = np.random.default_rng(SEED)
        us, its, ys = [], [], []
        for r in np.where(rows)[0]:
            u, pos = V["users"][r], list(V["truth"][r])
            if not pos:
                continue
            pool = np.where(~V["mask"][r])[0]
            neg = rng.choice(pool, size=min(len(pool), neg_per_pos * len(pos)), replace=False)
            neg = neg[~np.isin(neg, pos)]
            us += [u] * (len(pos) + len(neg)); its += pos + neg.tolist()
            ys += [1] * len(pos) + [0] * len(neg)
        Xf = self._fc_matrix(np.array(us), np.array(its))
        return HistGradientBoostingClassifier(random_state=SEED, **params).fit(Xf, np.array(ys))

    def _fc_score(self, split, clf):
        D = self.S[split]
        out = np.zeros(D["mask"].shape, dtype=np.float32)
        for i in range(0, len(D["users"]), 100):
            u = D["users"][i:i + 100]
            Xf = self._fc_matrix(u).reshape(-1, len(self.fnames))
            out[i:i + 100] = clf.predict_proba(Xf)[:, 1].reshape(len(u), -1)
        return masked(out, D["mask"])

    def feature_combination(self):
        self.F = self._side_features()
        self.fnames = list(self.F)
        V, rows = self.S["valid"], []
        grid = [{"max_iter": it, "learning_rate": lr, "max_leaf_nodes": leaves}
                # extended after the first run: the optimum (300, 0.05, 15) was at the grid edge
                for it in [300, 600] for lr in [0.02, 0.05, 0.1] for leaves in [7, 15, 31]]
        for params in grid:
            clf = self._fc_fit(self.fit, params)
            s = mean_ndcg(self._fc_score("valid", clf), V["truth"], self.sel)
            rows.append({**params, "select_ndcg@10": s})
            print(f"  {params}: select NDCG@10={s:.4f}")
        best = self.save_tuning("FeatureCombination", rows, "select_ndcg@10").iloc[0]
        params = {k: (int(best[k]) if k != "learning_rate" else float(best[k]))
                  for k in ("max_iter", "learning_rate", "max_leaf_nodes")}
        clf = self._fc_fit(np.ones_like(self.fit), params)
        return ({"features": self.fnames, **params},
                self._fc_score("valid", clf), self._fc_score("test", clf))

    # ---------------------------------------------------------------- feature augmentation
    def _augmented_ease(self, source, n, alpha):
        """Add `source`'s top-n predictions (made from train only) as
        pseudo-interactions with weight alpha, then fit EASE on the result."""
        V = self.S["valid"]
        Xa = self.X.copy()
        pseudo = topn_mask(V["scores"][source], n).astype(np.float32) * alpha
        Xa[V["users"]] = np.maximum(Xa[V["users"]], pseudo)
        return Xa @ ease_weights(Xa, self.ease_reg)

    def feature_augmentation(self):
        V, rows = self.S["valid"], []
        for source in self.rank[:3]:
            for n in [5, 10, 20]:
                for alpha in [0.1, 0.25, 0.5, 1.0]:  # 0.1 added: 0.25 was at the grid edge
                    full = self._augmented_ease(source, n, alpha)
                    s = mean_ndcg(masked(full[V["users"]], V["mask"]), V["truth"])
                    rows.append({"source": source, "n_pseudo": n, "alpha": alpha, "valid_ndcg@10": s})
        base = mean_ndcg(masked((self.X @ ease_weights(self.X, self.ease_reg))[V["users"]], V["mask"]),
                         V["truth"])
        rows.append({"source": "none (plain EASE)", "n_pseudo": 0, "alpha": 0.0, "valid_ndcg@10": base})
        best = self.save_tuning("FeatureAugmentation", rows, "valid_ndcg@10").iloc[0]
        if best["n_pseudo"] == 0:
            print("  note: no augmentation setting beat plain EASE on validation")
            best = pd.DataFrame(rows[:-1]).sort_values("valid_ndcg@10", ascending=False).iloc[0]
        p = {"source": best["source"], "n_pseudo": int(best["n_pseudo"]),
             "alpha": float(best["alpha"]), "ease_reg_weight": self.ease_reg}
        print(f"  best: {p}, valid NDCG@10={best['valid_ndcg@10']:.4f} (plain EASE: {base:.4f})")
        full = self._augmented_ease(p["source"], p["n_pseudo"], p["alpha"])
        return (p, masked(full[V["users"]], V["mask"]),
                masked(full[self.S["test"]["users"]], self.S["test"]["mask"]))

    # ---------------------------------------------------------------- meta-level
    def meta_level(self):
        """EASE's learned item-item model maps each user's history to a dense
        profile; a user-based kNN finds neighbours in that profile space."""
        V, rows = self.S["valid"], []
        P = self.X @ ease_weights(self.X, self.ease_reg)
        P /= np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
        sim = P @ P.T
        np.fill_diagonal(sim, 0.0)
        order = np.argsort(-sim, axis=1)

        def scores(k):
            Sk = np.zeros_like(sim)
            idx = order[:, :k]
            np.put_along_axis(Sk, idx, np.take_along_axis(sim, idx, axis=1), axis=1)
            return Sk @ self.X

        for k in [20, 50, 100, 200]:
            full = scores(k)
            s = mean_ndcg(masked(full[V["users"]], V["mask"]), V["truth"])
            rows.append({"k": k, "valid_ndcg@10": s})
            print(f"  k={k}: valid NDCG@10={s:.4f}")
        k = int(self.save_tuning("MetaLevel", rows, "valid_ndcg@10").iloc[0]["k"])
        full = scores(k)
        return ({"k": k, "ease_reg_weight": self.ease_reg},
                masked(full[V["users"]], V["mask"]),
                masked(full[self.S["test"]["users"]], self.S["test"]["mask"]))


# ================================== main =====================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="+", choices=HYBRIDS, default=HYBRIDS)
    p.add_argument("--members", nargs="+", choices=[m for m in MODELS if m != "Random"],
                   default=DEFAULT_MEMBERS)
    a = p.parse_args()
    clear_cli_args()
    HOUT.mkdir(parents=True, exist_ok=True)

    cached = [m for m in MODELS if (ROOT / "cache/scores/test" / f"{m}.npy").exists()]
    missing = [m for m in a.members if m not in cached]
    if missing:
        sys.exit(f"No cached scores for {missing}. Run: python task1/individual.py "
                 f"--stage final --models {' '.join(missing)}")
    if "EASE" not in a.members:
        print("note: FeatureAugmentation and MetaLevel still use EASE internally")
    print("hybrid members:", ", ".join(a.members))
    h = Hybrids(a.members)

    methods = {"Weighted": h.weighted, "Switching": h.switching, "Mixed": h.mixed,
               "Cascade": h.cascade, "FeatureCombination": h.feature_combination,
               "FeatureAugmentation": h.feature_augmentation, "MetaLevel": h.meta_level}
    params_path = HOUT / "hybrid_params.yaml"
    all_params = load_yaml(params_path) if params_path.exists() else {}
    summary_path = OUT / "task1_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()

    def add_row(name, kind, sv, st):
        nonlocal summary
        row = {"model": name, "type": kind,
               "valid_ndcg@10": mean_ndcg(sv, h.S["valid"]["truth"]),
               "test_ndcg@10": mean_ndcg(st, h.S["test"]["truth"])}
        if not summary.empty:
            summary = summary[summary["model"] != name]
        summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)

    for m in cached:  # individual models, scored with the same code
        add_row(m, "individual", load_cache(m, "valid")[0], load_cache(m, "test")[0])

    for name in a.only:
        print(f"[{name}]", flush=True)
        params, sv, st = methods[name]()
        all_params[name] = {"members": list(a.members), **params}
        save_yaml(all_params, params_path)
        full = f"Hybrid-{name}"
        export_topk(full, sv, h.S["valid"]["users"], h.dataset, "valid", K_EXPORT, h.config)
        export_topk(full, st, h.S["test"]["users"], h.dataset, "test", K_EXPORT, h.config)
        add_row(full, "hybrid", sv, st)
        summary.round(4).to_csv(summary_path, index=False)

    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()