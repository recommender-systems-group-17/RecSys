"""
Shared helpers for Task 1: building configs, training a RecBole model,
computing full score matrices, and exporting results in the course format.

All paths are relative to the repository root.
"""
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # RecBole configs use data_path: dataset/ (relative)
sys.path.insert(0, str(ROOT))

from recbole.config import Config  # noqa: E402
from recbole.data import create_dataset, data_preparation  # noqa: E402
from recbole.utils import get_model, get_trainer, init_seed  # noqa: E402
from recbole.utils.case_study import full_sort_scores  # noqa: E402

DATASET = "ml-100k"
OUT = ROOT / "outputs"
CACHE = ROOT / "cache"

# Our model name -> (RecBole class name, config file).
# UserKNN is RecBole's ItemKNN with knn_method: 'user' (see its config).
MODELS = {
    "Random":      ("Random",      "recbole/config/Random/ml-100k.yaml"),
    "Pop":         ("Pop",         "recbole/config/Pop/ml-100k.yaml"),
    "ItemKNN":     ("ItemKNN",     "recbole/config/ItemKNN/ml-100k.yaml"),
    "UserKNN":     ("ItemKNN",     "recbole/config/UserKNN/ml-100k.yaml"),
    "BPR":         ("BPR",         "recbole/config/BPR/ml-100k.yaml"),
    "FISM":        ("FISM",        "recbole/config/FISM/ml-100k.yaml"),
    "SLIMElastic": ("SLIMElastic", "recbole/config/SLIMElastic/ml-100k.yaml"),
    "EASE":        ("EASE",        "recbole/config/EASE/ml-100k.yaml"),
    "LightGCN":    ("LightGCN",    "recbole/config/LightGCN/ml-100k.yaml"),
    "NGCF":        ("NGCF",        "recbole/config/NGCF/ml-100k.yaml"),
    "NeuMF":       ("NeuMF",       "recbole/config/NeuMF/ml-100k.yaml"),
}

# Applied to every run. seed and eval_args are NOT touched, so all models
# share the exact same train/valid/test split.
COMMON_OVERRIDES = {
    "valid_metric": "NDCG@10",
    "show_progress": False,
    "checkpoint_dir": str(CACHE / "checkpoints"),
}


def clear_cli_args():
    """RecBole's Config also parses sys.argv (e.g. --k=5 would silently change
    ItemKNN's k). Call this after argparse so only our config is used."""
    del sys.argv[1:]


def build_config(name, params=None):
    recbole_name, cfg_file = MODELS[name]
    overrides = dict(COMMON_OVERRIDES)
    overrides.update(params or {})
    return Config(model=recbole_name, dataset=DATASET,
                  config_file_list=[cfg_file], config_dict=overrides)


def train(name, params=None, keep_checkpoint=False):
    """Train one model. Returns dict with model, data loaders and valid/test results."""
    config = build_config(name, params)
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    _, valid_result = trainer.fit(train_data, valid_data, saved=True, show_progress=False)
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=False)
    if not keep_checkpoint and os.path.exists(trainer.saved_model_file):
        os.remove(trainer.saved_model_file)
    return {
        "config": config, "model": trainer.model, "dataset": dataset,
        "train": train_data, "valid": valid_data, "test": test_data,
        "valid_result": dict(valid_result), "test_result": dict(test_result),
    }


def split_users(loader):
    """Internal ids of users that have at least one interaction in this split."""
    uid_field = loader.dataset.uid_field
    return np.array(sorted(set(loader.dataset.inter_feat[uid_field].numpy().tolist())))


@torch.no_grad()
def score_matrix(model, loader, users, device, batch=128):
    """Scores for every item, history items and padding set to -inf.
    valid loader masks train items; test loader masks train+valid items."""
    parts = []
    for i in range(0, len(users), batch):
        s = full_sort_scores(users[i:i + batch].tolist(), model, loader, device)
        parts.append(s.float().cpu().numpy())
    return np.vstack(parts)


def _split_df(res, part):
    ds = res["dataset"]
    feat = res[part].dataset.inter_feat
    return pd.DataFrame({
        ds.uid_field: ds.id2token(ds.uid_field, feat[ds.uid_field].numpy()),
        ds.iid_field: ds.id2token(ds.iid_field, feat[ds.iid_field].numpy()),
    })


def export_split(res, out_dir=OUT / "splits"):
    out_dir.mkdir(parents=True, exist_ok=True)
    for part in ("train", "valid", "test"):
        _split_df(res, part).to_csv(out_dir / f"{DATASET}.{part}.tsv", sep="\t", index=False)


def check_split(res, out_dir=OUT / "splits"):
    """Stop if this machine produces a different split than the committed one
    (can happen with different library versions)."""
    for part in ("train", "valid", "test"):
        saved = pd.read_csv(out_dir / f"{DATASET}.{part}.tsv", sep="\t", dtype=str)
        now = _split_df(res, part).astype(str)
        key = lambda d: set(zip(d.iloc[:, 0], d.iloc[:, 1]))  # noqa: E731
        if key(saved) != key(now):
            raise RuntimeError(
                f"The {part} split on this machine differs from outputs/splits/. "
                "Results would not be comparable. Use the committed outputs instead "
                "of retraining, or align library versions with the person who made them.")


def export_topk(name, scores, users, dataset, split, k, config, out_root=OUT / "recommendations"):
    """Write top-k lists in the same JSON format as save_recommendations.py,
    using original user/item ids."""
    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = np.argsort(-scores, axis=1)[:, :k]
    top_scores = np.take_along_axis(scores, idx, axis=1)
    user_tok = dataset.id2token(dataset.uid_field, users)
    item_tok = dataset.id2token(dataset.iid_field, idx)
    recs = {
        str(u): {"items": item_tok[r].tolist(),
                 "scores": [round(float(x), 6) for x in top_scores[r]]}
        for r, u in enumerate(user_tok)
    }
    payload = {"dataset": DATASET, "model": name, "split": split, "k": k,
               "seed": config["seed"], "eval_args": config["eval_args"],
               "recommendations": recs}
    with open(out_dir / f"{DATASET}_{name}_top{k}.json", "w") as f:
        json.dump(payload, f, separators=(",", ":"))


def save_cache(name, split, scores, users):
    d = CACHE / "scores" / split
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{name}.npy", scores.astype(np.float32))
    np.save(d / "users.npy", users)


def load_cache(name, split):
    d = CACHE / "scores" / split
    return np.load(d / f"{name}.npy"), np.load(d / "users.npy")


def ground_truth(loader, users):
    """List of sets of relevant internal item ids, aligned with `users`."""
    feat = loader.dataset.inter_feat
    uid, iid = loader.dataset.uid_field, loader.dataset.iid_field
    df = pd.DataFrame({"u": feat[uid].numpy(), "i": feat[iid].numpy()})
    g = df.groupby("u")["i"].apply(set).to_dict()
    return [g.get(u, set()) for u in users]


def ndcg_at_k(scores, truth, k=10):
    """Per-user NDCG@k (binary relevance). Used only for tuning; the full
    metric suite is implemented separately in Task 2."""
    top = np.argsort(-scores, axis=1)[:, :k]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    out = np.zeros(len(truth))
    for r, rel in enumerate(truth):
        if not rel:
            continue
        hits = np.array([i in rel for i in top[r]], dtype=float)
        idcg = disc[:min(len(rel), k)].sum()
        out[r] = (hits * disc).sum() / idcg
    return out


def train_matrix(loader, n_users, n_items, field=None):
    """Dense user x item matrix (internal ids) of the given split.
    field=None -> binary interactions, otherwise that field's values."""
    feat = loader.dataset.inter_feat
    u = feat[loader.dataset.uid_field].numpy()
    i = feat[loader.dataset.iid_field].numpy()
    X = np.zeros((n_users, n_items), dtype=np.float32)
    X[u, i] = 1.0 if field is None else feat[field].numpy()
    return X


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _plain(x):
    """numpy scalars/arrays -> plain python so YAML can store them."""
    if isinstance(x, dict):
        return {_plain(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def save_yaml(obj, path):
    obj = _plain(obj)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)
