# Task 1 – Individual and hybrid recommenders

## Final model set

| Role | Models |
|---|---|
| **Individual models in the study** | EASE, UserKNN, LightGCN, ContentBased |
| **Baselines** | Pop, Random (not hybrid members) |
| **Hybrids** | all 7 below, built from the 4 individual models |

The four individual models are one per category: linear item-item (EASE), neighbourhood CF (UserKNN), graph / latent factor (LightGCN) and content-based (ContentBased). Within each category, the model with the best validation NDCG@10 was picked.

**How the set was chosen.** We first trained and tuned 11 models: Random, Pop, ItemKNN, UserKNN, BPR, FISM, SLIMElastic, EASE, LightGCN, NGCF and NeuMF. Many were highly correlated (`outputs/hybrids/Weighted_score_correlation.csv` from the 10-member run):

- BPR, LightGCN and NGCF: 0.93–0.96 → kept LightGCN
- EASE and SLIMElastic: 0.85 → kept EASE
- ItemKNN is weaker than UserKNN → kept UserKNN
- FISM correlates 0.93 with Pop (it behaves like a popularity model) → dropped
- NeuMF: dropped in favour of a content-based model, which brings a completely different signal (item features instead of interactions)

The recommendation lists of the dropped models are still in `outputs/recommendations/`, but they are **not part of the study**. Use only the models in the table above.

UserKNN is RecBole's ItemKNN with `knn_method: 'user'`. ContentBased is not in RecBole and is implemented in `task1/content_based.py`.

## Results (test NDCG@10, our own quick metric)

| Model | Test NDCG@10 | | Hybrid | Test NDCG@10 |
|---|---|---|---|---|
| EASE | 0.3302 | | FeatureAugmentation | 0.3315 |
| LightGCN | 0.3028 | | Weighted | 0.3310 |
| UserKNN | 0.2877 | | Switching | 0.3302 |
| ContentBased | 0.0322 | | Cascade | 0.3301 |
| Pop | 0.0985 | | Mixed | 0.3283 |
| Random | 0.0076 | | MetaLevel | 0.2998 |
| | | | FeatureCombination | 0.2542 |

Weighted hybrid coefficients: LightGCN 3.37 (47%), EASE 1.58 (22%), UserKNN 1.58 (22%), ContentBased 0.69 (10%).

**In short:** with one model per category, the best hybrids match EASE within 0.001 NDCG@10. EASE won every user-activity group, so Switching is identical to EASE. In an earlier run with all 10 members, the hybrids reached 0.3349 (+0.005 over EASE), mostly thanks to SLIMElastic, which is in the same family as EASE.

These numbers come from the quick NDCG in `hybrids.py`. Final numbers for the report should come from the Task 2 metrics code.

## How to run

From the repo root, inside the `recsys` conda environment:

```bash
# individual models (tune + export)
python task1/individual.py --models Random Pop EASE UserKNN ContentBased   # fast
python task1/individual.py --models LightGCN                               # slow on CPU

# hybrids (default members: EASE UserKNN LightGCN ContentBased)
python task1/hybrids.py
```

Other options:

```bash
python task1/individual.py --stage final --models EASE   # retrain with saved best params, no tuning
python task1/hybrids.py --only Cascade MetaLevel         # only some hybrids
python task1/hybrids.py --members EASE UserKNN BPR       # a different member set
bash run_task1.sh --quick                                # smoke test of everything
```

- `--quick` writes 2-epoch "best" parameters, so always do a full run afterwards.
- `hybrids.py` needs the cached scores (`cache/scores/`) of every member, so run it last. If a member is missing, it stops and prints the command to fix it.

## The content-based model

- **Item features:** one-hot genres (19), optionally plus release decade.
- **User profile:** sum of the features of the user's train items.
- **Score:** cosine similarity between profile and item.
- **Tuned on validation:** item weighting (binary vs rating minus the user's mean), IDF weighting of features, release-decade weight.
- **Best setting:** binary weighting, IDF on, no release decade (valid NDCG@10 = 0.037).
  - IDF roughly doubled the score, because without it common genres such as Drama dominate every profile.
  - Release decade didn't help.
- **Ties:** many movies share the same features, so scores tie. Ties are broken randomly (seeded), because RecBole's item order would otherwise break them in favour of popular items.

## Hybrids (Burke's taxonomy)

| Hybrid | What it does | Tuned on validation | Chosen |
|---|---|---|---|
| Weighted (1.3) | Per-user min-max normalised member scores, weighted by logistic-regression coefficients | regularisation C | C = 0.1; weights LightGCN 3.37, EASE 1.58, UserKNN 1.58, ContentBased 0.69 |
| Switching | Users split into activity quantiles (#train interactions); each group uses its best member | number of groups | 2 groups (16–53 and 53–591 train interactions), EASE for both |
| Mixed | Reciprocal rank fusion of the best m members | m, fusion constant c | all 4 members (EASE, LightGCN, UserKNN, ContentBased), c = 10 |
| Cascade | Member A selects top-N candidates, member B re-orders them | A, B, N | LightGCN top-50 → re-ordered by EASE |
| FeatureCombination | One gradient-boosting model on collaborative + content + demographic features (item popularity/avg rating/year, user activity/avg rating, genre affinity, item-item CF similarity, demographic-group popularity). Uses no member outputs | trees, learning rate, leaves | 300 trees, lr 0.05, 15 leaves |
| FeatureAugmentation | Member A's top-n predictions are added to the train matrix as pseudo-interactions (weight α), then EASE is fit on the augmented matrix | A, n, α | LightGCN top-10, α = 0.25 |
| MetaLevel | EASE's learned item-item matrix turns each user's history into a dense profile; user-kNN finds neighbours in that space | k | k = 50 |

FeatureCombination, FeatureAugmentation and MetaLevel are our interpretations of these Burke categories. FeatureCombination and MetaLevel don't depend on the member set, so their results are the same as in the 10-member run.

## Design choices (for the report)

- **Hyperparameter selection uses validation only.** Individual models are selected on validation NDCG@10 (`valid_metric: NDCG@10`; the course default is MRR@10). `seed` and `eval_args` are never changed.
- **The split** in `outputs/splits/` is exported with the same seeded procedure used in training, and is identical to the output of the course's fixed `save_split.py`.
- **Hybrids with learned parameters** (Weighted, Switching, FeatureCombination): validation users are split 50/50. Parameters are fitted on one half, hyperparameters are chosen on the other, and the final model is refitted on all validation users. This avoids choosing hyperparameters on the same data the weights were fitted on.
- **Weighted hybrid training data:** for each validation user, candidates are the union of every member's top-50 items. The label is 1 if the item is in the user's validation set. Scores are min-max normalised per user, so the coefficients are on a comparable scale.
- **Interpreting the coefficients:** a coefficient reflects a model's *unique* contribution given the other members, not its accuracy alone. LightGCN gets the largest weight because it is the least correlated with EASE and UserKNN, which overlap with each other (0.85).
- **All models only see the train split as input.** Validation lists exclude train items; test lists exclude train + valid items. This is the same as RecBole.
- **FeatureAugmentation and MetaLevel** use a closed-form EASE implemented in numpy, with EASE's tuned `reg_weight`. Plain numpy EASE reproduces RecBole's EASE validation score, which serves as a sanity check; that row is in `FeatureAugmentation_tuning.csv`.
- **Validation lists of the learned hybrids are in-sample**, since their parameters were fit on validation. Only use the **test** lists for reporting.

## Limitations

- Some tuned values sit at the edge of their grid: LightGCN (3 layers) and Mixed's fusion constant (c = 10, the smallest tried). The Weighted hybrid's C (0.1) is inside its grid.
- Switching chose EASE for both groups, so it never switches and is identical to EASE.
- ContentBased uses only genres and release decade, so it is weak on accuracy (0.032). It is useful for its different signal, not on its own.
- With these 4 members, the hybrids add no meaningful accuracy over EASE.

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
│   ├── hybrid_params.yaml           chosen settings + members per hybrid
│   ├── <Hybrid>_tuning.csv          every setting tried
│   ├── Weighted_coefficients.csv    learned weights (Task 2.3)
│   ├── Weighted_score_correlation.csv
│   └── Switching_assignment.csv     per-group NDCG of every member + chosen model
└── task1_summary.csv                valid/test NDCG@10 for every model (quick overview;
                                     the full metric suite is Task 2)
cache/scores/                        full score matrices, local only (gitignored)
```

Hybrid files are named `ml-100k_Hybrid-<Name>_top100.json`. ContentBased is `ml-100k_ContentBased_top100.json`. All use the same JSON format as `save_recommendations.py`.

**Files for Tasks 2 and 3** (all in `outputs/recommendations/test/`):
`EASE`, `UserKNN`, `LightGCN`, `ContentBased`, `Pop`, `Random`, and the 7 `Hybrid-*` files.