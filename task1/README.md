# Task 1 – Individual and hybrid recommenders

## How to run

From the repo root, inside the `recsys` conda environment:

```bash
bash run_task1.sh --quick   # ~5-10 min smoke test, checks everything runs
bash run_task1.sh           # full run (tuning takes hours, see below)
```

Or step by step:

```bash
python task1/individual.py --models EASE SLIMElastic   # tune + export some models
python task1/individual.py --stage final               # retrain with saved best params, no tuning
python task1/hybrids.py                                # all 7 hybrids
python task1/hybrids.py --only Cascade MetaLevel       # some hybrids
```

`--quick` writes 2-epoch "best" parameters, so always do a full run afterwards.

**Runtime:** tuning is per model, so it can be split over several runs with `--models`. Fast: Pop, Random, ItemKNN, UserKNN, EASE, SLIMElastic. Slow on a laptop CPU: BPR, FISM, LightGCN, NGCF, NeuMF. Run those separately, e.g. overnight. `hybrids.py` needs every individual model's cached scores, so run it last.

## Models

**Individual (11):** Random, Pop, ItemKNN, UserKNN, BPR, FISM, SLIMElastic, EASE, LightGCN, NGCF, NeuMF.
UserKNN is RecBole's ItemKNN with `knn_method: 'user'`.

**Hybrids (7, Burke's taxonomy).** Members are all individual models except Random.

| Hybrid | What it does | Tuned on validation |
|---|---|---|
| Weighted (1.3) | Per-user min-max normalised member scores, weighted by logistic-regression coefficients | regularisation C |
| Switching | Users split into activity quantiles (#train interactions); each group uses its best member | number of groups |
| Mixed | Reciprocal rank fusion of the best m members | m, fusion constant c |
| Cascade | Member A selects top-N candidates, member B re-orders them | A, B, N |
| FeatureCombination | One gradient-boosting model on collaborative + content + demographic features (item popularity/avg rating/year, user activity/avg rating, genre affinity, item-item CF similarity, demographic-group popularity). Uses no member outputs | trees, learning rate, leaves |
| FeatureAugmentation | Member A's top-n predictions are added to the train matrix as pseudo-interactions (weight α), then EASE is fit on the augmented matrix | A, n, α |
| MetaLevel | EASE's learned item-item matrix turns each user's history into a dense profile; user-kNN finds neighbours in that space | k |

## Design choices (for the report)

- **Hyperparameter selection uses validation only.** Individual models are selected on validation NDCG@10 (`valid_metric: NDCG@10`; the course default is MRR@10). `seed` and `eval_args` are never changed.
- **Hybrids with learned parameters** (Weighted, Switching, FeatureCombination): validation users are split 50/50. Parameters are fitted on one half, hyperparameters are chosen on the other, and the final model is refitted on all validation users. This avoids choosing hyperparameters on the same data the weights were fitted on.
- **Weighted hybrid training data:** for each validation user, candidates are the union of every member's top-50 items. The label is 1 if the item is in the user's validation set. Scores are min-max normalised per user, so the coefficients are on a comparable scale. Members are correlated (see `Weighted_score_correlation.csv`), which matters when interpreting the coefficients in Task 2.3.
- **All models only see the train split as input.** Validation lists exclude train items; test lists exclude train + valid items. This is the same as RecBole.
- **FeatureAugmentation and MetaLevel** use a closed-form EASE implemented in numpy, with EASE's tuned `reg_weight`. Plain numpy EASE reproduces RecBole's EASE validation score, which serves as a sanity check; that row is in `FeatureAugmentation_tuning.csv`.
- **Validation lists of the learned hybrids are in-sample**, since their parameters were fit on validation. Only use the **test** lists for reporting.

## Outputs

```
outputs/
├── splits/                          shared split (train/valid/test TSV)
├── best_params.yaml                 chosen hyperparameters per individual model
├── tuning/<Model>.csv               every config tried, validation metrics
├── individual_summary.csv           RecBole's own metrics (sanity check only)
├── recommendations/
│   ├── valid/ml-100k_<Model>_top100.json
│   └── test/ml-100k_<Model>_top100.json      <- use these for Task 2/3
├── hybrids/
│   ├── hybrid_params.yaml           chosen settings per hybrid
│   ├── <Hybrid>_tuning.csv          every setting tried
│   ├── Weighted_coefficients.csv    learned weights (Task 2.3)
│   ├── Weighted_score_correlation.csv
│   └── Switching_assignment.csv     per-group NDCG of every member + chosen model
└── task1_summary.csv                valid/test NDCG@10 for every model (quick overview;
                                     the full metric suite is Task 2)
cache/scores/                        full score matrices, local only (gitignored)
```

Hybrid files are named `ml-100k_Hybrid-<Name>_top100.json` and use the same JSON format as `save_recommendations.py`.
