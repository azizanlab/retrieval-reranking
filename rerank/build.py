"""Assemble the user prompt of one read from a view, the candidates it shows and their evidence.

A read shows `cands` (view positions) under labels 00, 01, ... in that order. Evidence is an
allocation (one list of review indices per shown candidate) plus, for the swapped control, `owners`:
owners[p] is the candidate whose reviews are printed under position p.
"""
from rerank import prompts

SETTINGS = ("titles", "meta")


def candidate_block(view, cands, labels, setting):
    if setting == "titles":
        return None
    out = []
    for lab, c in zip(labels, cands):
        meta = dict(view.items[c]["meta"]); meta["title"] = view.titles[c]
        out.append(prompts.format_metadata(lab, meta))
    return "\n".join(out)


def review_blocks(view, cands, labels, alloc, owners=None, empty="placeholder"):
    """One block per shown candidate holding its allocated reviews. A candidate allocated nothing
    prints the "(no reviews available)" line under empty="placeholder", or no block under "omit"."""
    owners = owners or list(range(len(cands)))
    out = []
    for p, (lab, c) in enumerate(zip(labels, cands)):
        picks = alloc[owners[p]]
        if not picks and empty == "omit":
            continue
        revs = [view.items[cands[owners[p]]]["reviews"][k] for k in picks]
        out.append(prompts.format_reviews(lab, view.titles[c], revs, len(revs)))
    return out


def user(view, cands, setting, blocks=None, header=prompts.HEADER_SELECTION, mode="reviews"):
    labels = ["%02d" % p for p in range(len(cands))]
    text = prompts.user_prompt(view.history, view.query, [(l, view.titles[c]) for l, c in zip(labels, cands)],
                               candidate_block(view, cands, labels, setting))
    if blocks:
        text += header + prompts.retrieved_block(blocks, mode)
    return text
