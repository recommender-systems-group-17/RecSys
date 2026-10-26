"""
Task 3.1 - four post-processing rerankers.

Every reranker takes a model's top-100 candidate list per user with relevance
scores min-max normalised to [0, 1] (s_i) and greedily builds a new top-k list
S. At each step it adds the candidate that maximises

    (1 - lambda) * s_i  +  lambda * objective gain of adding i to S

lambda = 0 returns the original top-k; lambda = 1 ignores relevance entirely
(within the top-100 candidates).

  Diversity    MMR (Carbonell & Goldstein, 1998). Gain = -max Jaccard genre
               similarity between i and the items already in S. Targets ILD.
  Calibration  Steck (2018). Gain = -KL(p || q~(S + i)), p = genre distribution
               of the user's train profile, q~ smoothed genre distribution of
               the list. Targets MiscalKL.
  Item fairness  Amortised exposure reranker (in the spirit of Biega et al.,
               2018). Users are processed one after another in a fixed random
               order; gain = -share of the users processed so far who were
               already shown item i. Items that have had a lot of exposure are
               pushed down for later users. Targets Gini, Coverage, Exposure_Tail.
  User fairness  Calibrated popularity, CP (Abdollahpouri et al., 2021).
               Gain = -JS(P_u || Q(S + i)) between the Head/Mid/Tail share of
               the user's profile and of the list, so every user gets a list as
               mainstream as their own taste. Popularity bias hurts niche users
               most (Task 2), so this targets the gap between user groups
               (UserGap_PopMiscal) rather than one global quantity.
"""
import numpy as np

from common import K, POP_GROUPS  # noqa: F401
from metrics import ALPHA_CALIBRATION

RERANKERS = ["Diversity", "Calibration", "ItemFairness", "UserFairness"]
SEED = 2020


class ItemTable:
    """Per-item feature rows (genre distribution, genre set, popularity group)
    built once, so each greedy step is a vectorised operation."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.idx, rows_g, rows_b, rows_h = {}, [], [], []
        items = set(ctx.catalog) | set(ctx.genres)
        for i in sorted(items):
            self.idx[i] = len(rows_g)
            rows_g.append(ctx.genre_vector(i))
            rows_b.append(rows_g[-1] > 0)
            rows_h.append([ctx.item_groups.get(i, "Tail") == g for g in POP_GROUPS])
        self.G = np.array(rows_g)                       # genre distribution per item
        self.B = np.array(rows_b, dtype=float)          # genre indicator per item
        self.H = np.array(rows_h, dtype=float)          # popularity group one-hot
        self._p_genre, self._p_pop = {}, {}

    def rows(self, items):
        return np.array([self.idx[i] for i in items])

    def profile_genre(self, u):
        if u not in self._p_genre:
            self._p_genre[u] = self.G[self.rows(self.ctx.profile[u])].mean(axis=0)
        return self._p_genre[u]

    def profile_pop(self, u):
        if u not in self._p_pop:
            self._p_pop[u] = self.H[self.rows(self.ctx.profile[u])].mean(axis=0)
        return self._p_pop[u]


def _greedy(s, lam, gain, k):
    """Generic greedy selection. gain(chosen, mask) -> objective value of
    adding each candidate to the chosen positions (array over candidates)."""
    chosen, free = [], np.ones(len(s), bool)
    for _ in range(min(k, len(s))):
        val = (1 - lam) * s + lam * gain(chosen)
        val[~free] = -np.inf
        j = int(np.argmax(val))
        chosen.append(j)
        free[j] = False
    return chosen


def mmr(items, s, u, lam, tab, k=K):
    B = tab.B[tab.rows(items)]
    inter = B @ B.T
    sizes = B.sum(axis=1)
    sim = inter / (sizes[:, None] + sizes[None, :] - inter)   # Jaccard of genre sets

    def gain(chosen):
        return -sim[:, chosen].max(axis=1) if chosen else np.zeros(len(s))
    return [items[j] for j in _greedy(s, lam, gain, k)]


def _kl(p, Q):
    """KL(p || q) for one p and a matrix of candidate q's (rows)."""
    m = p > 0
    return np.sum(p[m] * np.log(p[m] / Q[:, m]), axis=1)


def calibration(items, s, u, lam, tab, k=K, alpha=ALPHA_CALIBRATION):
    G = tab.G[tab.rows(items)]
    p = tab.profile_genre(u)

    def gain(chosen):
        q = (G[chosen].sum(axis=0) + G) / (len(chosen) + 1)     # q(S + i) for every i
        return -_kl(p, (1 - alpha) * q + alpha * p)
    return [items[j] for j in _greedy(s, lam, gain, k)]


def _js(p, Q):
    M = (p + Q) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p > 0, p * np.log2(p / M), 0).sum(axis=1)
        b = np.where(Q > 0, Q * np.log2(Q / M), 0).sum(axis=1)
    return (a + b) / 2


def calibrated_popularity(items, s, u, lam, tab, k=K):
    H = tab.H[tab.rows(items)]
    p = tab.profile_pop(u)

    def gain(chosen):
        return -_js(p, (H[chosen].sum(axis=0) + H) / (len(chosen) + 1))
    return [items[j] for j in _greedy(s, lam, gain, k)]


def rerank(cands, name, lam, tab, k=K):
    """Rerank every user's candidates. cands: {user: (items, scores)}.
    Returns {user: [top-k items]}."""
    if lam == 0:
        return {u: items[:k] for u, (items, _) in cands.items()}
    if name == "ItemFairness":
        return amortised_exposure(cands, lam, k)
    f = {"Diversity": mmr, "Calibration": calibration, "UserFairness": calibrated_popularity}[name]
    return {u: f(items, s, u, lam, tab, k) for u, (items, s) in cands.items()}


def visit_order(users, seed=SEED):
    """Order in which amortised_exposure processes users (fixed random order).
    Users later in the order face more accumulated exposure penalties, so
    groups.py checks whether their accuracy cost is larger."""
    users = sorted(users, key=str)
    np.random.default_rng(seed).shuffle(users)
    return users


def amortised_exposure(cands, lam, k=K, seed=SEED):
    """Item-side fairness. Users are visited in a fixed random order; an item's
    penalty is the share of previously visited users whose list contained it.
    The penalty does not depend on the rest of the list, so within one user
    this is a plain re-sort."""
    users = visit_order(cands, seed)
    shown, out = {}, {}
    for n, u in enumerate(users):
        items, s = cands[u]
        pen = np.array([shown.get(i, 0) for i in items]) / max(n, 1)
        top = np.argsort(-((1 - lam) * s - lam * pen), kind="stable")[:k]
        out[u] = [items[j] for j in top]
        for i in out[u][:K]:
            shown[i] = shown.get(i, 0) + 1
    return out
