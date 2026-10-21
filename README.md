# DSAIT4335 Recommender Systems - Final Project (Group 17)

This repo is a copy of the course RecBole repo ([masoudmansoury/RecBole_DSAIT4335](https://github.com/masoudmansoury/RecBole_DSAIT4335)) plus our group's work. All experiments run on **MovieLens 100K** (already included in `dataset/ml-100k/`).

The original RecBole README is in `README_RecBole.md`. Details on Task 1 (model choice, hybrids, results, design choices) are in `task1/README.md`.

---

## 1. Setup

**Requirement:** conda (Anaconda or Miniconda).

```bash
git clone https://github.com/recommender-systems-group-17/RecSys.git
cd RecSys
bash setup.sh
conda activate recsys
```

`setup.sh` creates a conda environment called `recsys` (Python 3.10), installs RecBole from this repo, and prints `RecBole <version> installed OK` at the end.

After that, run `conda activate recsys` every time you open a new terminal.

**Windows:** run `setup.sh` from Git Bash, or run these commands one by one:

```bash
conda create -n recsys python=3.10 -y
conda activate recsys
pip install -e .
pip install "numpy<2" "ray[tune]"
```

### Troubleshooting

| Message | Meaning |
|---|---|
| `np.float_ was removed in the NumPy 2.0 release` | NumPy 2 is installed. Run `pip install "numpy<2"`. |
| `Can't import ray.tune` | Run `pip install "ray[tune]"`. |
| `command line args [...] will not be used in RecBole` | Harmless. The config file is still applied. |
| `Could not save embeddings: BPR.forward() ...` | Harmless message from the course trainer. |
| `FutureWarning` / `UserWarning` from pandas / torch | Harmless deprecation notices. |

---

## 2. Rules for consistent results

1. **Don't retrain models.** All models are trained and exported once, and everyone works from the files in `outputs/`.
2. **Use the split in `outputs/splits/`.** This is the exact split every model was trained and tested on. It is identical to the output of the course's fixed `save_split.py` (verified after the Oct 2026 fix).
3. **Only use the models in the study** (section 4). Files of the other models are still in `outputs/` but are not part of the analysis.
4. **Report numbers from the `test` lists only** (`outputs/recommendations/test/`). The validation lists of the learned hybrids are in-sample.
5. **Don't use the metrics RecBole prints in the report.** We implement our own (Task 2.1).
6. **Pull before you start working:** `git pull`.
7. **Work on your own branch** and merge into `main` through a pull request. `main` should always run.

---

## 3. Repo layout

```
RecSys/
├── setup.sh                    one-time environment setup
├── run_task1.sh                reproduces all of Task 1
├── run_task2.sh / run_task3.sh reproduce Task 2 / Task 3 (no retraining)
├── task1/                      Task 1 code (see task1/README.md)
│   ├── individual.py           1.1–1.2: tune + train individual models
│   ├── content_based.py        our content-based model (not in RecBole)
│   ├── hybrids.py              1.3–1.5: 7 hybrid models
│   ├── common.py               shared helpers
│   └── grids.yaml              hyperparameter grids
├── task2/                      Task 2 code (see task2/README.md)
├── task3/                      Task 3 code (see task3/README.md)
├── dataset/ml-100k/            MovieLens 100K (.inter, .item, .user)
├── recbole/config/<Model>/     RecBole config per model
├── outputs/                    SHARED results, committed to git
│   ├── splits/                 ml-100k.{train,valid,test}.tsv
│   ├── recommendations/
│   │   ├── test/               ml-100k_<Model>_top100.json  ← use these
│   │   └── valid/              ml-100k_<Model>_top100.json
│   ├── tuning/                 every hyperparameter setting tried, per model
│   ├── best_params.yaml        chosen hyperparameters per model
│   ├── hybrids/                hybrid tuning, settings, coefficients
│   └── task1_summary.csv       valid/test NDCG@10 of every model
├── cache/                      local score matrices, NOT committed
└── saved/                      local checkpoints, NOT committed
```

---

## 4. Models in the study

| Role | Models | Files in `outputs/recommendations/test/` |
|---|---|---|
| Individual models | EASE, UserKNN, LightGCN, ContentBased | `ml-100k_<Model>_top100.json` |
| Baselines | Pop, Random | `ml-100k_Pop_top100.json`, `ml-100k_Random_top100.json` |
| Hybrids (7) | Weighted, Switching, Mixed, Cascade, FeatureCombination, FeatureAugmentation, MetaLevel | `ml-100k_Hybrid-<Name>_top100.json` |

- **One individual model per category:** linear item-item (EASE), neighbourhood CF (UserKNN), graph / latent factor (LightGCN), content-based (ContentBased).
- **The hybrids are built from these 4 models.** Pop and Random are baselines only.
- **Why these four:** we first tuned 12 models and kept one per category. Within each category the alternatives were statistically indistinguishable on validation. See `task1/README.md`.

**Not in the study** (files kept for reference only): ItemKNN, BPR, FISM, SLIMElastic, NGCF, NeuMF.

### Test NDCG@10 (quick Task 1 metric; final numbers come from Task 2)

| Model | NDCG@10 | | Hybrid | NDCG@10 | vs EASE |
|---|---|---|---|---|---|
| EASE | 0.3302 | | Weighted | 0.3347 | +0.0045 (n.s.) |
| LightGCN | 0.3020 | | FeatureAugmentation | 0.3346 | +0.0044 (n.s.) |
| UserKNN | 0.2877 | | Mixed | 0.3316 | +0.0014 (n.s.) |
| Pop | 0.0985 | | Switching | 0.3302 | identical to EASE |
| ContentBased | 0.0322 | | Cascade | 0.3301 | −0.0001 (n.s.) |
| Random | 0.0076 | | MetaLevel | 0.2998 | −0.0304 (sig.) |
| | | | FeatureCombination | 0.2542 | −0.0760 (sig.) |

n.s. = not significant, sig. = significant (Wilcoxon, Holm-corrected). Confidence intervals and the evidence behind the model choices are in `outputs/analysis/`.

---

## 5. Output file formats

### Splits: `outputs/splits/ml-100k.{train,valid,test}.tsv`

Tab-separated, original MovieLens IDs:

```
user_id	item_id
196	242
```

### Recommendations: `outputs/recommendations/{test,valid}/ml-100k_<Model>_top100.json`

```json
{
  "dataset": "ml-100k",
  "model": "EASE",
  "split": "test",
  "k": 100,
  "seed": 2020,
  "eval_args": {...},
  "recommendations": {
    "1": {"items": ["50", "181", ...], "scores": [0.58, 0.51, ...]},
    ...
  }
}
```

- One entry per user in that split, with items ranked best first (top 100).
- Test lists exclude items the user had in train or valid. Validation lists exclude train items.
- IDs are strings in the original MovieLens format, the same as in the split files.
- Scores are on each model's own scale, so they can't be compared across models.

### Loading in Python

```python
import json
import pandas as pd

MODELS = ["EASE", "UserKNN", "LightGCN", "ContentBased", "Pop", "Random"]
HYBRIDS = ["Weighted", "Switching", "Mixed", "Cascade",
           "FeatureCombination", "FeatureAugmentation", "MetaLevel"]

test = pd.read_csv("outputs/splits/ml-100k.test.tsv", sep="\t", dtype=str)
truth = test.groupby("user_id")["item_id"].apply(set).to_dict()

def load(name):
    with open(f"outputs/recommendations/test/ml-100k_{name}_top100.json") as f:
        return json.load(f)["recommendations"]

recs = {m: load(m) for m in MODELS} | {f"Hybrid-{h}": load(f"Hybrid-{h}") for h in HYBRIDS}
top10 = {u: r["items"][:10] for u, r in recs["EASE"].items()}
```

Item metadata (genres, release year) for beyond-accuracy metrics is in `dataset/ml-100k/ml-100k.item`, and user metadata (age, gender, occupation) is in `dataset/ml-100k/ml-100k.user`.

---

## 6. Reproducing Task 1 (only if needed)

```bash
bash run_task1.sh --quick   # smoke test, a few minutes
bash run_task1.sh           # full run, several hours on a laptop
```

This tunes all 12 models (the 11 RecBole models plus ContentBased), then builds the hybrids from the 4 study models. To rebuild only the hybrids:

```bash
python task1/hybrids.py
```

If your machine produces a different split than the one in `outputs/splits/`, the script stops instead of overwriting results.

---

## 7. Pulling course updates

The course repo is set up as the `upstream` remote. To bring in updates:

```bash
git pull --no-rebase upstream main
git push
```

Coordinate in the group chat before doing this, so only one person does it.

---

## 8. Status

- [x] Environment setup (`setup.sh`)
- [x] Shared split (`outputs/splits/`), verified against the fixed `save_split.py`
- [x] Task 1.1–1.2: individual models trained, tuned and exported; final study set chosen (EASE, UserKNN, LightGCN, ContentBased)
- [x] Task 1.3–1.5: 7 hybrid models built, tuned and exported
- [x] Task 2: evaluation metrics and analysis (`task2/README.md`, `bash run_task2.sh`)
- [x] Task 3: rerankers, pipelines and group analysis (`task3/README.md`, `bash run_task3.sh`)
- [ ] Report
