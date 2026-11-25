"""Benchmark instances, seeded candidate order, and candidate-specific review pools.

Published pools contain up to five verified Amazon reviews; verified and filled pools
contain up to twenty, with filled pools supplementing missing verified reviews.
Reddit uses candidate-specific comment pools for all three settings.
"""
import gzip, json, os, random
from dataclasses import dataclass

DATA = os.environ.get("RERANK_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
AMAZON = ("Toys_and_Games", "Grocery_and_Gourmet_Food", "Office_Products")
PER_CATEGORY = 250
SIDECAR = {"verified": "deeprev20", "filled": "deeprev20u"}


@dataclass
class View:
    key: str
    idx: int
    query: str
    history: list
    titles: list
    items: list
    target: int


def datasets(corpus):
    return [f"hq_{c}_r200pm" for c in AMAZON] if corpus == "amazon" else ["hq_reddit750_r200pm"]


def _open(path):
    if os.path.exists(path):
        return open(path, encoding="utf-8")
    return gzip.open(path + ".gz", "rt", encoding="utf-8")


def load(dataset):
    with _open(os.path.join(DATA, dataset + ".jsonl")) as f:
        rows = [json.loads(l) for l in f]
    return rows[:PER_CATEGORY] if dataset != "hq_reddit750_r200pm" else rows


def make_view(dataset, idx, inst):
    """Shuffle the target and nineteen negatives with a seed fixed by instance index."""
    rng = random.Random(idx)
    target = next(c for c in inst["candidates"] if c["item_id"] == inst["target_item_id"])
    others = [c for c in inst["candidates"] if c["item_id"] != inst["target_item_id"]]
    chosen = [target] + rng.sample(others, 19)
    rng.shuffle(chosen)
    items = []
    for c in chosen:
        meta = dict(c.get("meta") or {})
        meta.setdefault("title", c.get("title", ""))
        meta.setdefault("description", c.get("meta_text", ""))
        items.append({"item_id": c["item_id"], "meta": meta, "reviews": c.get("reviews", [])})
    return View(key=f"{dataset}/{idx}", idx=idx, query=inst.get("q_full") or "",
                history=[h["title"] for h in inst["history"] if h["title"]],
                titles=[c["title"] or "(no title)" for c in chosen], items=items,
                target=next(i for i, c in enumerate(chosen) if c["item_id"] == inst["target_item_id"]))


def _sidecar(dataset, kind):
    path = os.path.join(DATA, f"{SIDECAR[kind]}_{dataset[len('hq_'):]}.jsonl")
    with _open(path) as f:
        return {r["item_id"]: r["reviews"] for r in map(json.loads, f)}


def _is_source(review, inst):
    """A review the request was written from cannot be evidence: same reviewer, or text whose
    token set overlaps the source review at a Jaccard of 0.6 or more."""
    if review.get("user_id") and review["user_id"] == inst.get("user_id"):
        return True
    a = set(" ".join((review.get("text") or "").lower().split()).split())
    b = set(" ".join((inst.get("src_review") or "").lower().split()).split())
    return bool(a and b) and len(a & b) / len(a | b) >= 0.6


def views(corpus, pool="published"):
    """Every evaluated instance of a corpus as a View, with its candidates' reviews from `pool`."""
    out = []
    for ds in datasets(corpus):
        side = _sidecar(ds, pool) if corpus == "amazon" and pool != "published" else None
        for idx, inst in enumerate(load(ds)):
            v = make_view(ds, idx, inst)
            if side is not None:
                for it in v.items:
                    if it["item_id"] in side:
                        it["reviews"] = [r for r in side[it["item_id"]] if not _is_source(r, inst)][:20]
            out.append(v)
    return out
