# Task 2 – Evaluation of effectiveness

## How to run

Task 2 only reads the shared Task 1 outputs (`outputs/splits`, `outputs/recommendations/test`, `outputs/hybrids`) and `dataset/ml-100k`. No models are retrained and RecBole is not needed. Packages: numpy, pandas, scipy, scikit-learn, matplotlib, pyyaml.

```bash
bash run_task2.sh              # everything, ~1 min
# or step by step (order matters):
python task2/evaluate.py         # 2.1 + 2.2  all metrics, significance, trade-off plot
python task2/groups.py           # 2.5 + 2.6  user/item group analysis, oracle
python task2/hybrid_analysis.py  # 2.3 + 2.4  weighted-hybrid coefficient analysis
```

## Files

| File | Content |
|---|---|
| `data.py` | Loading the split, rec lists, genres, demographics; item and user groups |
| `metrics.py` | Our own implementation of every metric (2.1) |
| `evaluate.py` | The study models (Random, Pop, EASE, UserKNN, LightGCN, ContentBased and the 7 hybrids) at k = 10 and 20; NDCG with 95% bootstrap CIs; Wilcoxon tests and paired bootstrap CIs vs the best individual model and vs Pop |
| `groups.py` | Metrics per user group and per item popularity group, best model per group (with a paired bootstrap CI of its lead over EASE), oracle switching |
| `hybrid_analysis.py` | Coefficients, hit attribution, leave-one-out ablation, per-group coefficients |

Outputs go to `outputs/task2/`. The `.csv` files have all numbers and the `fig_*.png` files are the report figures.

## Metrics (all @k, k = 10 unless stated)

Relevance is binary: the item is in the user's **test** set. Popularity, profiles and genre distributions are computed from the **train** split only.

| Metric | Definition | Measures |
|---|---|---|
| Precision, Recall, NDCG, MRR, HR, MAP | Standard binary-relevance definitions; IDCG and MAP use min(\|rel\|, k) | accuracy |
| ARP | Mean #train interactions of recommended items | popularity bias |
| APLT | Share of recommended items outside the Head group | popularity bias |
| PopLift | (ARP of list − mean popularity of user's profile) / mean popularity of profile; >0 = pushed to more popular items than the user's taste | popularity bias (user side) |
| Novelty | Mean self-information −log2(#users who interacted / #users) | novelty |
| ILD | Mean pairwise Jaccard distance between the genre sets of listed items | diversity |
| MiscalKL | KL(p‖q̃) between genre distribution of the train profile p and of the list q, q̃ = 0.99q + 0.01p (Steck, 2018) | calibration |
| Coverage | Share of train items recommended to at least one user | diversity (aggregate) |
| Gini | Gini index of item exposure counts over all train items | item-side fairness / concentration |
| Exposure_Head/Mid/Tail | Share of all top-k slots going to each item popularity group | item-side fairness |
| Recall_Head/Mid/Tail | Of the relevant test items in a group, share that was recommended | accuracy per item group |
| UserGap_NDCG | NDCG(Active users) − NDCG(Inactive users) | user-side fairness |

## Groups

- **Item popularity groups** (by #train interactions): Head = top 20% of items, Mid = next 30%, Tail = remaining 50%.
- **Activity**: 5 quantile groups of #train interactions per user.
- **Active / Inactive**: the top 20% of users by #train interactions are Active (Li et al., 2021, use 5%; 20% gives a more stable group with 943 users).
- **Mainstream**: the share of Head items in the user's profile. The bottom 20% of users are Niche, the top 20% Blockbuster, the rest Diverse (Abdollahpouri et al., 2019). Niche users have much longer profiles, so `groups_mainstream_by_activity_k10.csv` holds activity fixed.
- **Gender, age group**: from `ml-100k.user`.

## Design choices (for the report)

- **Models:** only the Task 1 study set: EASE, UserKNN, LightGCN and ContentBased (one per family), Pop and Random as baselines, and the 7 hybrids built from the four members. The other RecBole models from Task 1 are not evaluated.
- **Significance:** paired Wilcoxon signed-rank on per-user NDCG@10, with Holm correction across models, plus a 95% paired bootstrap CI of the mean difference. We compare against the best individual model (EASE) and against Pop. Hybrid-Switching is identical to EASE (Task 1.5), so its p-value is set to 1.
- **Per-group winners:** `groups_best_model_k10.csv` reports whether the best member of a group beats EASE significantly (95% paired bootstrap CI above zero). Without this, a near-tie in a small group looks like a finding.
- **Oracle switching** (`oracle_k10.csv`) uses test labels. It is an upper bound for analysis, not a model: how much a perfect per-user or per-group choice of member could gain.
- **Weighted-hybrid replica.** The full score matrices are local to whoever ran Task 1, so `hybrid_analysis.py` rebuilds the weighted hybrid from the exported top-100 lists. It uses the same recipe, the same members and the same C (from `hybrid_params.yaml`), normalising min-max over each top-100 list, and is fitted on validation. The script prints its test NDCG@10 next to the original's (0.3347) as a check. Coefficients from Task 1.3 are reported as they are.
- **Relation to Task 1:** Task 1 already reports the coefficients with bootstrap CIs and a drop-one ablation on validation (`outputs/hybrids/Weighted_coefficient_stability.csv`, `Weighted_ablation.csv`). Task 2.3 adds the ablation on test, hit attribution and group-specific weights. Use the same wording for the main finding (a large coefficient does not mean a large contribution).
- **Per-group coefficients** are shown as the share of total |weight|. C is fixed, so absolute coefficients shrink when a group has fewer training rows.
