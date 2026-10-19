# Task 3 – Rerankers and societal aspects

## How to run

Task 3 only reads the shared Task 1 outputs (`outputs/splits`, `outputs/recommendations/{valid,test}`, `outputs/hybrids/hybrid_params.yaml`) and `dataset/ml-100k`. It reuses `task2/data.py` and `task2/metrics.py` unchanged, so every number is comparable with Task 2. No models are retrained and RecBole is not needed.

```bash
bash run_task3.sh                 # everything, ~6 min
# or step by step (order matters):
python task3/sweep.py             # 3.1 + 3.2  lambda sweep, lambda selection, trade-off curves
python task3/pipelines.py         # 3.3        rerank before vs. after the weighted hybrid
python task3/groups.py            # 3.4        effects per user / item group (--model to change the figure)
```

## Files

| File | Content |
|---|---|
| `common.py` | Paths, candidate loading, lambda grid, base models, evaluation (Task 2 metrics + `PopMiscal`, `UserGap_PopMiscal`) |
| `rerankers.py` | The four rerankers (3.1) |
| `sweep.py` | Every reranker × base model × lambda on validation and test; lambda selection |
| `pipelines.py` | Weighted hybrid with rerankers on the members (A) or on the hybrid output (B) |
| `groups.py` | Before/after metrics per user group and per item popularity group |

Outputs go to `outputs/task3/`: `.csv` files with all numbers and `fig_*.png` for the report.

## Rerankers (3.1)

All rerankers are greedy post-processors. Each one takes a model's exported **top-100** list with scores min-max normalised per user to [0, 1] (s_i), and builds a new top-10 by repeatedly adding the item that maximises `(1 − λ)·s_i + λ·gain_i`. At λ = 0 you get the original list.

| Reranker | Objective | gain_i | Target metric |
|---|---|---|---|
| Diversity | Genre variety within a list | MMR (Carbonell & Goldstein, 1998): −max Jaccard genre similarity to the items already chosen | ILD |
| Calibration | Genre mix of the list matches the user's profile | Steck (2018): −KL(p ‖ q̃) of the list after adding i, with the same α = 0.01 as the metric | MiscalKL |
| ItemFairness | Spread exposure over more items | Amortised exposure (in the spirit of Biega et al., 2018): users are visited in a fixed random order (seed 2020), and the gain is −(share of the users visited so far who were already shown i) | Gini, Coverage, Exposure_Mid/Tail |
| UserFairness | Every user gets a list as mainstream as their own taste | Calibrated Popularity (Abdollahpouri et al., 2021): −JS between the Head/Mid/Tail shares of the profile and of the list | UserGap_PopMiscal (Niche − Blockbuster) |

Why these choices:

- **ItemFairness is system-level.** The penalty depends on what earlier users were shown, so it targets aggregate exposure (Gini). A per-user method such as xQuAD only guarantees a few tail items in each list.
- **UserFairness targets the users who are treated worst.** Task 2 showed that popularity bias hurts Niche users most: their lists are far more mainstream than their profiles. CP reduces that mismatch for each user, which closes the Niche–Blockbuster gap.
- **Why not equalise NDCG between Active and Inactive users** (Li et al., 2021)? A post-processor cannot raise true accuracy for inactive users, so it could only close that gap by lowering it for active users. `UserGap_NDCG` is still reported for every run.
- **Two new metrics** are added on top of the Task 2 metrics: `PopMiscal` (per user, Abdollahpouri et al., 2021) and `UserGap_PopMiscal`. Both are defined in `common.py`.

## Lambda selection (3.2)

- **Grid:** λ ∈ {0, 0.1, …, 0.9, 0.95, 0.98, 1}. It is finer near 1 because the item- and user-fairness gains are small next to differences in relevance.
- **Rule:** for each (model, reranker), choose the **largest λ whose validation NDCG@10 is at most 5% below the model without reranking** (`--tolerance`). This means getting as much of the beyond-accuracy objective as possible within an explicit accuracy budget.
- **Base models:**
  - EASE: the most accurate individual model
  - LightGCN: a graph model in between
  - NeuMF: the least popularity-biased model
  - Hybrid-Weighted
- **Caveat:** Hybrid-Weighted was fitted on validation, so its validation NDCG is optimistic. Selection depends on the relative drop, which is less affected.

## Pipelines (3.3)

- **A (rerank, then combine):** each of the 10 members' top-100 lists is reranked into a new order, then the hybrid combines the reranked lists.
- **B (combine, then rerank):** the reranker is applied to the hybrid's top-100.
- **Positional features:** a reranker changes the order but not the scores, so for both pipelines the hybrid represents each member by its positional score `1 − rank/100`. The logistic-regression weights are fitted on validation with the Task 1.3 recipe and C (`pipeline_weights.csv`). This positional hybrid's test NDCG@10 is 0.3349, the same as Hybrid-Weighted.
- **How to compare:** λ means different things in A and B, so compare the trade-off curves rather than equal λ values.
- **ItemFairness in A** counts exposure over the members' full top-100 lists.

## Main observations (for the report)

- **Candidate pool limits:** every base model puts almost all top-100 candidates in the Head group (EASE: 99% of top-10 slots are Head, 0 Tail). Rerankers can only redistribute within the top-100, so tail exposure stays below 2% even at the chosen λ.
- **Cost of the targeted gain:** within the 5% budget, MiscalKL drops 19–34%, ILD rises 6–9%, Gini falls 1–6%, and UserGap_PopMiscal falls 72–95%. The relative cost differs: for EASE, ItemFairness costs only −0.5% NDCG for a 27% coverage gain (`selected_change.csv`).
- **Side effects:** Diversity and Calibration reinforce each other (both change the genre mix), but neither changes popularity. UserFairness raises coverage by 9–51% and novelty, so the user-side and item-side objectives partly overlap.
- **Base model matters:** NeuMF and LightGCN are already less popularity-biased, so the same λ gains them less on popularity and fairness. NeuMF's validation NDCG also drops faster, so it gets λ = 0.1 for three rerankers.
- **Group effects (EASE):**
  - Calibration costs Active users and users aged 35–49 the most NDCG.
  - ItemFairness slightly *improves* NDCG for Inactive, Diverse, 50+ and female users, but costs Niche and Blockbuster users.
  - UserFairness lowers PopMiscal most for Niche users (−0.19, against −0.04 for Blockbuster users), as intended.
- **Pipelines:**
  - B (rerank last) reaches much further along every objective: MiscalKL 0.11 against 0.35, and UserGap 0.008 against 0.07.
  - A keeps more NDCG at moderate objective levels. A likely explanation: combining the members partly undoes each member's reranking, because the hybrid pulls the items the members agree on (popular, similar ones) back to the top.
  - For ItemFairness, A collapses at high λ: every member pushes away the same over-exposed items, and the combined list loses relevance.
