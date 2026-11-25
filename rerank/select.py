"""Evidence selection: which reviews go under which candidate (Appendix C).

Every function returns an allocation: one list per candidate of indices into that candidate's pool.
`pools[c]` is an (n_c, d) array of unit-norm review embeddings and `q` the unit-norm request
embedding, so a dot product is a cosine. No rule sees the target.

Rules = an objective x an allocation:
  objectives   relevance, contrastive, mmr, dpp
  allocations  one_each (one passage per candidate) or shared (B passages over the pooled reviews)
Controls (random, from an rng seeded per instance):
  presence     one random passage per candidate
  count        B random passages from the union of all pools
  resampled    a rule's allocation, every slot refilled at random from the same candidate
  swapped      a rule's blocks re-filed among the candidates it covered (see swap_owners)
"""
import numpy as np

LAMBDA = BETA = RHO = 0.5


def _contrastive_scores(pools, q):
    """cos(e, q) - lambda * max over other candidates j of cos(e, c_j), c_j the unit mean of pool j:
    a passage every candidate could have supplied says nothing about which is the target."""
    cents = np.stack([p.mean(0) / np.linalg.norm(p.mean(0)) for p in pools])
    out = []
    for c, p in enumerate(pools):
        comp = p @ cents.T
        comp[:, c] = -np.inf
        out.append(p @ q - LAMBDA * comp.max(1))
    return out


def _scores(pools, q, objective):
    return _contrastive_scores(pools, q) if objective == "contrastive" else [p @ q for p in pools]


def _empty(pools):
    return [[] for _ in pools]


def _top_shared(scores, budget):
    """The `budget` best passages over all pools; ties go to the earlier candidate and passage."""
    flat = sorted(((-s, c, k) for c, sc in enumerate(scores) for k, s in enumerate(sc)))[:budget]
    alloc = [[] for _ in scores]
    for _, c, k in flat:
        alloc[c].append(k)
    return alloc


def _greedy(pools, q, budget, objective, one_each):
    """MMR or the determinantal rule, chosen greedily over the pooled passages. With one_each, a
    candidate leaves the race once it holds a passage (the partition constraint).

    For the determinantal rule, adding passage i to the chosen set A multiplies det L_A by the Schur
    complement L_ii - L_iA L_AA^-1 L_Ai, so the greedy step maximises that (in logs)."""
    E = np.concatenate(pools)
    owner = np.concatenate([[c] * len(p) for c, p in enumerate(pools)])
    local = np.concatenate([np.arange(len(p)) for p in pools])
    rel = E @ q
    S = E @ E.T
    g = np.exp(BETA * rel / 2)
    L = g[:, None] * (S + 1e-6 * np.eye(len(E))) * g[None, :]
    chosen, open_ = [], np.ones(len(E), bool)
    for _ in range(budget):
        if not open_.any():
            break
        if objective == "mmr":
            val = BETA * rel - RHO * (S[:, chosen].max(1) if chosen else 0.0)
        else:
            gain = np.diag(L).copy()
            if chosen:
                LA = L[:, chosen]
                gain -= np.einsum("ij,ij->i", LA @ np.linalg.inv(L[np.ix_(chosen, chosen)]), LA)
            val = np.log(np.maximum(gain, 1e-300))
        best = int(np.argmax(np.where(open_, val, -np.inf)))
        chosen.append(best)
        open_[best] = False
        if one_each:
            open_[owner == owner[best]] = False
    alloc = _empty(pools)
    for i in chosen:
        alloc[owner[i]].append(int(local[i]))
    return alloc


def rule(objective, allocation, pools, q, budget):
    if objective in ("mmr", "dpp"):
        return _greedy(pools, q, budget if allocation == "shared" else len(pools), objective, allocation == "one_each")
    scores = _scores(pools, q, objective)
    if allocation == "one_each":
        return [[int(np.argmax(s))] for s in scores]
    return _top_shared(scores, budget)



def designed(kind, pools, q, budget, prior):
    """Relevance picks the passages inside each pool; only the split of the budget comes from the
    reader's scores `prior` (log-probabilities over these candidates).
      leader   fill the reader's top candidate, then the next, in score order
      softmax  in proportion to softmax(prior)
      inverse  in proportion to 1 - softmax(prior): starves the candidate the reader prefers"""
    order_in_pool = [list(np.argsort(-(p @ q), kind="stable")) for p in pools]
    cap = np.array([len(p) for p in pools])
    if kind == "leader":
        counts, left = np.zeros(len(pools), int), budget
        for c in np.argsort(-np.asarray(prior), kind="stable"):
            counts[c] = min(left, cap[c]); left -= counts[c]
    else:
        w = np.exp(np.asarray(prior) - np.max(prior)); w /= w.sum()
        counts = _largest_remainder(w if kind == "softmax" else 1 - w, budget, cap)
    return [[int(k) for k in order_in_pool[c][:n]] for c, n in enumerate(counts)]


def _largest_remainder(weights, budget, cap):
    """Integer counts summing to budget (or to the total capacity), proportional to weights."""
    w = np.asarray(weights, float) / np.sum(weights)
    counts = np.zeros(len(w), int)
    while counts.sum() < min(budget, cap.sum()):
        open_ = counts < cap
        share = w * open_ / (w * open_).sum() * (min(budget, cap.sum()) - counts.sum())
        add = np.minimum(np.floor(share).astype(int), cap - counts)
        if add.sum() == 0:
            frac = np.where(open_, share - np.floor(share), -1)
            add = np.zeros(len(w), int); add[int(np.argmax(frac))] = 1
        counts += add
    return counts



def presence(pools, rng):
    return [[rng.randrange(len(p))] for p in pools]


def count(pools, budget, rng):
    pairs = [(c, k) for c, p in enumerate(pools) for k in range(len(p))]
    alloc = _empty(pools)
    for c, k in rng.sample(pairs, min(budget, len(pairs))):
        alloc[c].append(k)
    return alloc


def resampled(alloc, pools, rng):
    """Keep how many passages each candidate got; draw which ones at random from its own pool."""
    return [sorted(rng.sample(range(len(p)), len(a))) for a, p in zip(alloc, pools)]


def swap_owners(alloc, rng):
    """For the swapped control: owner[c] = the candidate whose whole block candidate c now shows.
    A random derangement over the covered candidates, so no block stays under its owner; the
    prompt holds exactly the same text."""
    covered = [c for c, a in enumerate(alloc) if a]
    owner = list(range(len(alloc)))
    if len(covered) < 2:
        return owner
    while True:
        perm = covered[:]
        rng.shuffle(perm)
        if all(a != b for a, b in zip(covered, perm)):
            break
    for c, src in zip(covered, perm):
        owner[c] = src
    return owner
