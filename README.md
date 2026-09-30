# DSAIT4335 Recommender Systems - Final Project (Group 17)

This repo is a copy of the course RecBole repo ([masoudmansoury/RecBole_DSAIT4335](https://github.com/masoudmansoury/RecBole_DSAIT4335)) plus our group's work. All experiments run on **MovieLens 100K** (already included in `dataset/ml-100k/`).

The original RecBole README is in `README_RecBole.md`.

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
| `FutureWarning` from pandas / torch | Harmless deprecation notices. |

---

## 2. Rules for consistent results

1. **Do not change `seed` or `eval_args`** in any config file. They define the train/valid/test split, and every model must use the same split.
2. **Use the shared split in `outputs/splits/`.** Don't regenerate it. Different library versions can produce a slightly different split even with the same seed.
3. **Evaluate from the files in `outputs/`.** Models are trained on one machine and exported there, so everyone evaluates exactly the same recommendation lists. Don't use the metrics RecBole prints in the report; we implement our own (Task 2.1).
4. **Pull before you start working:** `git pull`.
5. **Work on your own branch** and merge into `main` through a pull request. `main` should always run.

---

## 3. Repo layout

```
RecSys/
├── setup.sh                   # one-time environment setup
├── dataset/ml-100k/           # MovieLens 100K (.inter, .item, .user)
├── recbole/config/<Model>/    # one config per model: ml-100k.yaml
├── run_recbole.py             # train a model
├── save_split.py              # export train/valid/test split
├── save_recommendations.py    # export top-k recommendations of a trained model
├── score_from_saved.py        # example evaluation (Recall/Precision/F1) from exported files
├── outputs/                   # SHARED results, committed to git
│   ├── splits/                #   ml-100k.{train,valid,test}.tsv
│   └── recommendations/       #   ml-100k_<Model>_top<k>.json
└── saved/                     # local checkpoints (.pth), NOT committed
```

---

## 4. Output file formats

### Splits: `outputs/splits/ml-100k.{train,valid,test}.tsv`

Tab-separated, original MovieLens IDs:

```
user_id	item_id
196	242
```

### Recommendations: `outputs/recommendations/ml-100k_<Model>_top<k>.json`

```json
{
  "dataset": "ml-100k",
  "model": "Pop",
  "k": 100,
  "seed": 2020,
  "eval_args": {...},
  "recommendations": {
    "1": {"items": ["50", "181", ...], "scores": [0.58, 0.51, ...]},
    ...
  }
}
```

- One entry per user in the test set, with items ranked best first.
- Items the user already interacted with in train/valid are excluded.
- IDs are strings in the original MovieLens format, the same as in the split files.

### Loading in Python

```python
import json
import pandas as pd

test = pd.read_csv("outputs/splits/ml-100k.test.tsv", sep="\t", dtype=str)
truth = test.groupby("user_id")["item_id"].apply(set).to_dict()

with open("outputs/recommendations/ml-100k_Pop_top100.json") as f:
    recs = json.load(f)["recommendations"]

top10 = {u: r["items"][:10] for u, r in recs.items()}
```

Item metadata (genres, release date) for beyond-accuracy metrics is in `dataset/ml-100k/ml-100k.item`, and user metadata (age, gender, occupation) is in `dataset/ml-100k/ml-100k.user`.

---

## 5. Training and exporting a model (reference)

Only needed if you are producing the shared outputs.

```bash
# train
python run_recbole.py --model BPR --dataset ml-100k \
  --config_files recbole/config/BPR/ml-100k.yaml

# export recommendations (use the checkpoint name printed by `ls saved/`)
python save_recommendations.py \
  --model_file saved/ml-100k-BPR-<timestamp>.pth \
  --k 100 --output_dir outputs/recommendations

# example evaluation
python score_from_saved.py \
  --test_tsv outputs/splits/ml-100k.test.tsv \
  --rec_json outputs/recommendations/ml-100k_BPR_top100.json --k 10
```

UserKNN is run as ItemKNN with the UserKNN config:

```bash
python run_recbole.py --model ItemKNN --dataset ml-100k \
  --config_files recbole/config/UserKNN/ml-100k.yaml
```

---

## 6. Pulling course updates

The course repo is set up as the `upstream` remote. To bring in updates:

```bash
git pull upstream main
git push
```

Coordinate in the group chat before doing this, so only one person does it.

---

## 7. Status

- [x] Environment setup (`setup.sh`)
- [x] Shared split exported (`outputs/splits/`)
- [x] Pop baseline exported
- [ ] Individual models trained, tuned and exported
- [ ] Hybrid models
- [ ] Evaluation metrics (Task 2)
- [ ] Rerankers (Task 3)