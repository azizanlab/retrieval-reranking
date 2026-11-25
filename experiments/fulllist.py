"""Full-list ranking configurations over twenty candidates.

The grid compares retrieval and reasoning with filled review pools; selection uses
verified pools. The omission ablation removes empty-candidate evidence blocks.
Random controls use three seeded draws, averaged within instance for analysis.
"""
import random

from rerank import build, data, embed, select

BUDGETS = (3, 5, 10, 20, 40)
EMPTY = "placeholder"


def _spec(view, setting, arm, user, think=False):
    return {"key": f"{view.key}|{setting}|{arm}|{int(think)}", "user": user, "think": think, "n": 20,
            "info": {"instance": view.key, "setting": setting, "arm": arm, "think": think, "target": view.target}}


SEEDS = (0, 1, 2)


def seeded(arm, seed):
    """The name of draw `seed` of a random arm; draw 0 keeps the plain name."""
    return arm if seed == 0 else f"{arm}-s{seed}"


def _evidence(view, alloc, owners=None, empty=EMPTY):
    return build.review_blocks(view, list(range(20)), ["%02d" % p for p in range(20)], alloc, owners, empty)


def specs(limit=None, part="all"):
    parts = {"grid", "selection"} if part == "all" else {part}
    out = []
    for corpus in ("amazon", "reddit"):
        if "grid" in parts:

            for v in data.views(corpus):
                if limit is None or v.idx < limit:
                    for setting in build.SETTINGS:
                        u = build.user(v, range(20), setting)
                        out += [_spec(v, setting, "none", u), _spec(v, setting, "none", u, think=True)]
        for pool in ("filled", "verified"):
            if pool == "verified" and not parts & {"selection", "omit"} or pool == "filled" and not parts & {"grid", "omit"}:
                continue
            views = [v for v in data.views(corpus, pool) if all(it["reviews"] for it in v.items)]
            views = [v for v in views if limit is None or v.idx < limit]
            E = embed.embeddings(corpus, pool)
            for v in views:
                pools, q = embed.pools_of(v, range(20), E)
                rng = lambda name, seed=0: random.Random(f"{v.key}|{name}" + (f"|{seed}" if seed else ""))
                arms, owners, think = {}, {}, {"titles": [], "meta": []}
                if pool == "filled":
                    for B in BUDGETS:
                        arms[f"contrastive-{B}"] = select.rule("contrastive", "shared", pools, q, B)
                        for seed in SEEDS:
                            arms[seeded(f"count-{B}", seed)] = select.count(pools, B, rng(f"count-{B}", seed))
                    arms["refiled-10"] = arms["contrastive-10"]
                    owners = {"refiled-10": select.swap_owners(arms["contrastive-10"], rng("swap-10"))}
                    think = {"titles": [f"contrastive-{B}" for B in BUDGETS], "meta": ["contrastive-10"]}
                else:
                    arms = {seeded("v-presence", seed): select.presence(pools, rng("presence", seed)) for seed in SEEDS}
                    arms["v-relevance-each"] = select.rule("relevance", "one_each", pools, q, 20)
                    for obj in ("relevance", "contrastive", "mmr", "dpp"):
                        arms[f"v-{obj}-shared20"] = select.rule(obj, "shared", pools, q, 20)
                for setting in build.SETTINGS:
                    for arm, alloc in arms.items():
                        if "omit" in parts and all(alloc):
                            continue
                        if "omit" in parts:
                            u = build.user(v, range(20), setting, _evidence(v, alloc, owners.get(arm), "omit"))
                            out.append(_spec(v, setting, arm + "-omit", u))
                            continue
                        u = build.user(v, range(20), setting, _evidence(v, alloc, owners.get(arm)))
                        out.append(_spec(v, setting, arm, u))
                        if arm in think[setting]:
                            out.append(_spec(v, setting, arm, u, think=True))
    return out
