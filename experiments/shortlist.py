"""Evidence selection on a fixed top-K shortlist from an initial reasoning read.

Targets outside the shortlist count as misses. Verified pools compare selection rules
and designed allocations; filled pools separate allocation from passage content.
Random controls use three draws. The omission condition removes empty evidence blocks.
"""
import random

from rerank import build, data, embed, engine, select
from experiments.fulllist import SEEDS, seeded

KS = (3, 10)
EMPTY = "placeholder"
OBJECTIVES = ("relevance", "contrastive", "mmr", "dpp")


def first_stage(model):
    return {r["info"]["instance"] + "|" + r["info"]["setting"]: r["scores"]
            for r in engine.load(model, "fulllist") if r["info"]["arm"] == "none" and r["info"]["think"]}


def shortlists(model, corpus, pool, limit=None):
    """Every (view, setting, K, shortlist, first-stage order) with the shortlist scorable under
    `pool`: the denominator of the shortlist tables. A target outside the shortlist stays in the
    denominator as a miss; a shortlisted candidate with an empty pool drops the instance."""
    prior = first_stage(model)
    for v in data.views(corpus, pool):
        if limit is not None and v.idx >= limit:
            continue
        for setting in build.SETTINGS:
            scores = prior.get(f"{v.key}|{setting}")
            assert scores is not None, f"no first stage for {v.key}; finish fulllist first"
            order = sorted(range(20), key=lambda i: (-scores[i], i))
            for K in KS:
                if all(v.items[c]["reviews"] for c in order[:K]):
                    yield v, setting, K, order[:K], order, scores


def specs(model, limit=None, part="main"):
    out = []
    for corpus in ("amazon", "reddit"):
        for pool in ("verified", "filled"):
            if part == "omit" and pool == "filled":
                continue
            E = embed.embeddings(corpus, pool)
            for v, setting, K, short, order, scores in shortlists(model, corpus, pool, limit):
                if v.target not in short:
                    continue
                shown = short[:]
                random.Random(f"{v.key}|{K}|shuffle").shuffle(shown)
                pools, q = embed.pools_of(v, shown, E)
                rng = lambda name, seed=0: random.Random(f"{v.key}|{K}|{name}" + (f"|{seed}" if seed else ""))
                arms = {seeded("presence", seed): select.presence(pools, rng("presence", seed)) for seed in SEEDS}
                if pool == "verified":
                    arms["none"] = None
                    for obj in OBJECTIVES:
                        for alloc in ("one_each", "shared"):
                            arms[f"{obj}-{alloc}"] = select.rule(obj, alloc, pools, q, K)
                    for kind in ("leader", "softmax", "inverse"):
                        arms[kind] = select.designed(kind, pools, q, K, [scores[c] for c in shown])
                else:
                    for obj in ("relevance", "contrastive"):
                        arms[f"{obj}-shared"] = a = select.rule(obj, "shared", pools, q, K)
                        for seed in SEEDS:
                            arms[seeded(f"{obj}-resampled", seed)] = select.resampled(a, pools, rng(f"resample-{obj}", seed))
                info = {"instance": v.key, "setting": setting, "K": K, "pool": pool,
                        "target": shown.index(v.target), "tiebreak": [order.index(c) for c in shown]}
                labels = ["%02d" % p for p in range(K)]
                for arm, alloc in arms.items():
                    if part == "omit":
                        if alloc is None or all(alloc):
                            continue
                        arm, empty = arm + "-omit", "omit"
                    else:
                        empty = EMPTY
                    blocks = build.review_blocks(v, shown, labels, alloc, None, empty) if alloc is not None else None
                    out.append({"key": f"{v.key}|{setting}|K{K}|{pool}|{arm}", "user": build.user(v, shown, setting, blocks),
                                "think": False, "n": K, "info": {**info, "arm": arm}})
    return out
