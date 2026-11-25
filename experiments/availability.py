"""Attach controlled evidence blocks to fixed sets of three candidates.

The paper reports review blocks with titles-only candidate lists, averaging three
seeded presentations. Metadata, bottom-three sets, and rank-order presentations
are supported as optional diagnostic conditions.
"""
import random

from rerank import build, data, prompts
from experiments.shortlist import first_stage

SHOWN, CONTENTS, MODES = ("top", "rand", "bot"), ("real", "swap", "filler", "template"), ("reviews", "metadata")


def _item(view, c, src, mode):
    """Candidate c's heading with candidate src's content."""
    label, title = "%02d" % c, view.titles[c]
    if mode == "reviews":
        return prompts.format_reviews(label, title, view.items[src]["reviews"], 1)
    meta = dict(view.items[src]["meta"]); meta["title"] = title
    return prompts.format_metadata(label, meta)


def blocks(view, shown, content, mode):
    out = []
    for j, c in enumerate(shown):
        f = _item(view, c, shown[(j + 1) % len(shown)] if content == "swap" else c, mode)
        out.append(prompts.filler_evidence(f) if content == "filler" else
                   prompts.template_evidence(f) if content == "template" else f)
    return out


def shown_sets(view, scores):
    """Select favored, bottom-ranked, and seeded random sets from the initial read."""
    order = sorted(range(20), key=lambda i: (-scores[i], i))
    return {"top": order[:3], "bot": order[-3:], "rand": random.Random(f"{view.key}|rand3").sample(range(20), 3)}


ORDERS = (0, 1, 2)


def specs(model, limit=None, part="random"):
    """Build paired conditions in three seeded block orders, or one rank-order presentation."""
    prior = first_stage(model)
    out = []
    for corpus in ("amazon", "reddit"):
        for v in data.views(corpus):
            scores = prior.get(f"{v.key}|titles")
            if limit is not None and v.idx >= limit:
                continue
            assert scores is not None, f"no first stage for {v.key}; finish fulllist first"
            for name, ranked in shown_sets(v, scores).items():
                orders = {"rank": ranked} if part == "rank" else {}
                for o in (ORDERS if part == "random" else ()):
                    shown = ranked[:]
                    random.Random(f"{v.key}|{name}|order{o}").shuffle(shown)
                    orders[o] = shown
                for o, shown in orders.items():
                    suffix = "" if o == "rank" else f"|o{o}"
                    for mode in MODES:
                        for content in CONTENTS:
                            user = build.user(v, range(20), "titles", None) + prompts.HEADER_AVAILABILITY + \
                                prompts.retrieved_block(blocks(v, shown, content, mode), mode)
                            out.append({"key": f"{v.key}|{name}|{mode}|{content}{suffix}", "user": user, "think": False, "n": 20,
                                        "info": {"instance": v.key, "shown_set": name, "shown": shown, "mode": mode,
                                                 "content": content, "order": o, "target": v.target}})
    return out
