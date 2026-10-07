"""
Loads everything Task 2 needs from the shared Task 1 outputs: the split, the
exported top-100 lists, item genres and user demographics. No RecBole needed.

All ids are kept as the original MovieLens tokens (strings).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
T2 = OUT / "task2"
DATA = ROOT / "dataset" / "ml-100k"
DATASET = "ml-100k"

INDIVIDUAL = ["Random", "Pop", "ItemKNN", "UserKNN", "BPR", "FISM",
              "SLIMElastic", "EASE", "LightGCN", "NGCF", "NeuMF"]
HYBRIDS = ["Hybrid-Weighted", "Hybrid-Switching", "Hybrid-Mixed", "Hybrid-Cascade",
           "Hybrid-FeatureCombination", "Hybrid-FeatureAugmentation", "Hybrid-MetaLevel"]
ALL_MODELS = INDIVIDUAL + HYBRIDS
BASELINES = ["Random", "Pop"]


def load_split(part):
    """DataFrame (user_id, item_id) of one split, ids as strings."""
    return pd.read_csv(OUT / "splits" / f"{DATASET}.{part}.tsv", sep="\t", dtype=str)


def truth_sets(part):
    """{user: set(items)} for the given split."""
    return load_split(part).groupby("user_id")["item_id"].apply(set).to_dict()


def load_recs(model, part="test", k=None):
    """{user: [items]} from the exported top-100 list, optionally cut to k."""
    path = OUT / "recommendations" / part / f"{DATASET}_{model}_top100.json"
    recs = json.loads(path.read_text())["recommendations"]
    return {u: r["items"][:k] if k else r["items"] for u, r in recs.items()}


def load_recs_with_scores(model, part="test"):
    path = OUT / "recommendations" / part / f"{DATASET}_{model}_top100.json"
    recs = json.loads(path.read_text())["recommendations"]
    return {u: (r["items"], r["scores"]) for u, r in recs.items()}


def item_genres():
    """{item: [genres]} from ml-100k.item."""
    df = pd.read_csv(DATA / f"{DATASET}.item", sep="\t", dtype=str)
    df.columns = [c.split(":")[0] for c in df.columns]
    return {i: g.split() for i, g in zip(df["item_id"], df["class"].fillna("unknown"))}


def users():
    """User table (user_id, age, gender, occupation) with ids as strings."""
    df = pd.read_csv(DATA / f"{DATASET}.user", sep="\t", dtype=str)
    df.columns = [c.split(":")[0] for c in df.columns]
    df["age"] = df["age"].astype(int)
    return df.set_index("user_id")


class Context:
    """Shared, model-independent information every metric needs.

    Popularity, profiles and genre distributions are computed from the TRAIN
    split only, i.e. from what the models could see.
    """

    def __init__(self):
        self.train = load_split("train")
        self.test = truth_sets("test")
        self.profile = self.train.groupby("user_id")["item_id"].apply(list).to_dict()
        self.users = sorted(self.test, key=int)
        self.n_users = self.train["user_id"].nunique()

        self.pop = self.train["item_id"].value_counts()          # #train interactions
        self.catalog = list(self.pop.index)                      # items seen in train
        self.genres = item_genres()
        self.genre_list = sorted({g for gs in self.genres.values() for g in gs})
        self._gidx = {g: j for j, g in enumerate(self.genre_list)}
        self.item_groups = popularity_groups(self.pop)

    def popularity(self, item):
        return int(self.pop.get(item, 0))

    def genre_vector(self, item):
        """Item genre distribution: 1/|genres| on each of its genres (Steck, 2018)."""
        v = np.zeros(len(self.genre_list))
        gs = self.genres.get(item, ["unknown"])
        for g in gs:
            v[self._gidx[g]] += 1.0 / len(gs)
        return v

    def genre_dist(self, items):
        """Average genre distribution of a list of items."""
        if not items:
            return np.zeros(len(self.genre_list))
        return np.mean([self.genre_vector(i) for i in items], axis=0)


def popularity_groups(pop, head=0.2, mid=0.3):
    """Split items by train popularity: top `head` share of items = Head,
    next `mid` share = Mid, the rest = Tail. Returns {item: group}."""
    ranked = list(pop.sort_values(ascending=False).index)
    n_head, n_mid = int(round(head * len(ranked))), int(round(mid * len(ranked)))
    return {i: ("Head" if r < n_head else "Mid" if r < n_head + n_mid else "Tail")
            for r, i in enumerate(ranked)}


def user_groups(ctx):
    """One row per user with every grouping used in Task 2.5:

    activity    5 quantile groups of #train interactions (same idea as the
                Switching hybrid's groups)
    active      top 20% users by #train interactions = Active, rest = Inactive
    mainstream  share of Head items in the train profile: bottom 20% users =
                Niche, top 20% = Blockbuster, rest = Diverse (Abdollahpouri et al., 2019)
    gender, age_group  from ml-100k.user
    """
    n = pd.Series({u: len(ctx.profile[u]) for u in ctx.users})
    head_share = pd.Series({u: np.mean([ctx.item_groups[i] == "Head" for i in ctx.profile[u]])
                            for u in ctx.users})
    q = pd.qcut(n, 5, labels=False, duplicates="drop")
    edges = n.groupby(q).agg(["min", "max"])
    labels = {g: f"Q{g + 1} ({lo}-{hi})" for g, (lo, hi) in edges.iterrows()}
    lo20, hi80 = head_share.quantile(0.2), head_share.quantile(0.8)
    demo = users()

    df = pd.DataFrame({
        "n_train": n,
        "activity": q.map(labels),
        "active": np.where(n >= n.quantile(0.8), "Active", "Inactive"),
        "head_share": head_share,
        "mainstream": np.select([head_share <= lo20, head_share >= hi80],
                                ["Niche", "Blockbuster"], "Diverse"),
        "gender": demo.loc[n.index, "gender"].values,
        "age_group": pd.cut(demo.loc[n.index, "age"].values, [0, 24, 34, 49, 120],
                            labels=["<25", "25-34", "35-49", "50+"]).astype(str),
    })
    df.index.name = "user_id"
    return df
