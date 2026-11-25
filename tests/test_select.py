"""Properties of evidence-selection rules and matched controls."""
import random

import numpy as np
import pytest

from rerank import select


def _pools(seed=0, sizes=(5, 1, 7, 3, 20), d=16):
    rng = np.random.default_rng(seed)
    unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
    return [unit(rng.normal(size=(n, d))) for n in sizes], unit(rng.normal(size=d))


def _count(alloc):
    return sum(len(a) for a in alloc)


@pytest.mark.parametrize("objective", ["relevance", "contrastive", "mmr", "dpp"])
def test_one_each_gives_every_candidate_exactly_one(objective):
    pools, q = _pools()
    alloc = select.rule(objective, "one_each", pools, q, len(pools))
    assert [len(a) for a in alloc] == [1] * len(pools)
    assert all(0 <= a[0] < len(p) for a, p in zip(alloc, pools))


@pytest.mark.parametrize("objective", ["relevance", "contrastive", "mmr", "dpp"])
@pytest.mark.parametrize("budget", [3, 10, 20])
def test_shared_spends_exactly_the_budget_without_repeats(objective, budget):
    pools, q = _pools()
    alloc = select.rule(objective, "shared", pools, q, budget)
    assert _count(alloc) == budget
    assert all(len(set(a)) == len(a) and all(0 <= k < len(p) for k in a) for a, p in zip(alloc, pools))


@pytest.mark.parametrize("objective", ["relevance", "contrastive", "mmr", "dpp"])
def test_rules_are_deterministic(objective):
    pools, q = _pools()
    assert select.rule(objective, "shared", pools, q, 7) == select.rule(objective, "shared", pools, q, 7)


def test_relevance_shared_takes_the_top_cosines():
    pools, q = _pools()
    alloc = select.rule("relevance", "shared", pools, q, 5)
    picked = sorted(float(pools[c][k] @ q) for c, a in enumerate(alloc) for k in a)
    every = sorted((float(x) for p in pools for x in p @ q), reverse=True)[:5]
    assert np.allclose(sorted(every), picked)


def test_dpp_greedy_matches_brute_force_log_det():
    """The Schur-complement step must pick what recomputing log det for every extension picks."""
    pools, q = _pools(seed=3, sizes=(4, 4, 4))
    alloc = select.rule("dpp", "shared", pools, q, 4)
    E = np.concatenate(pools); g = np.exp(0.5 * (E @ q) / 2)
    L = g[:, None] * (E @ E.T + 1e-6 * np.eye(len(E))) * g[None, :]
    chosen = []
    for _ in range(4):
        vals = [np.linalg.slogdet(L[np.ix_(chosen + [i], chosen + [i])])[1] if i not in chosen else -np.inf
                for i in range(len(E))]
        chosen.append(int(np.argmax(vals)))
    offsets = np.cumsum([0] + [len(p) for p in pools])
    expect = [[i - offsets[c] for i in chosen if offsets[c] <= i < offsets[c + 1]] for c in range(3)]
    assert [sorted(a) for a in alloc] == [sorted(e) for e in expect]


def test_mmr_with_no_redundancy_weight_is_relevance():
    pools, q = _pools()
    old = select.RHO
    try:
        select.RHO = 0.0
        assert [sorted(a) for a in select.rule("mmr", "shared", pools, q, 6)] == \
               [sorted(a) for a in select.rule("relevance", "shared", pools, q, 6)]
    finally:
        select.RHO = old


def test_presence_is_one_random_passage_each():
    pools, _ = _pools()
    alloc = select.presence(pools, random.Random(0))
    assert [len(a) for a in alloc] == [1] * len(pools)
    assert alloc == select.presence(pools, random.Random(0))


def test_count_matched_conserves_the_budget():
    pools, _ = _pools()
    for B in (3, 10, 20):
        assert _count(select.count(pools, B, random.Random(B))) == B


def test_resampled_keeps_the_allocation_exactly():
    pools, q = _pools()
    alloc = select.rule("contrastive", "shared", pools, q, 10)
    res = select.resampled(alloc, pools, random.Random(1))
    assert [len(a) for a in res] == [len(a) for a in alloc]


def test_swap_moves_every_covered_block_once_and_never_to_its_owner():
    """Swapping preserves every covered block and changes its owner."""
    alloc = [[0, 1, 2], [], [4], [1, 3], [], [0]]
    for seed in range(50):
        owners = select.swap_owners(alloc, random.Random(seed))
        covered = [c for c, a in enumerate(alloc) if a]
        assert sorted(owners[c] for c in covered) == covered
        assert all(owners[c] != c for c in covered)
        assert all(owners[c] == c for c in range(len(alloc)) if not alloc[c])


@pytest.mark.parametrize("kind", ["leader", "softmax", "inverse"])
def test_designed_allocations_spend_the_budget_within_capacity(kind):
    pools, q = _pools(sizes=(2, 1, 5))
    prior = [-0.1, -3.0, -1.0]
    alloc = select.designed(kind, pools, q, 6, prior)
    assert _count(alloc) == 6 and all(len(a) <= len(p) for a, p in zip(alloc, pools))


def test_leader_fills_the_favourite_first_and_inverse_starves_it():
    pools, q = _pools(sizes=(20, 20, 20))
    prior = [-2.0, -0.01, -5.0]
    assert [len(a) for a in select.designed("leader", pools, q, 3, prior)] == [0, 3, 0]
    assert len(select.designed("inverse", pools, q, 3, prior)[1]) == 0
