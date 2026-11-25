"""all-MiniLM-L6-v2 embeddings of requests and reviews, unit-norm, cached per corpus and pool."""
import os

import numpy as np

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
CACHE = os.path.join(os.environ.get("RERANK_RESULTS_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")), "emb")


def embeddings(corpus, pool):
    """{text: vector} for every request and review of a corpus under a pool, encoded once and cached.
    Always the whole corpus, so a run on a few instances never leaves a partial cache behind."""
    from rerank import data
    path = os.path.join(CACHE, f"{corpus}-{pool}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return dict(zip(z["texts"], z["vecs"]))
    views = data.views(corpus, pool)
    texts = sorted({v.query for v in views} | {r.get("text", "") for v in views for it in v.items for r in it["reviews"]})
    from sentence_transformers import SentenceTransformer
    vecs = SentenceTransformer(ENCODER).encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, texts=np.array(texts, dtype=object), vecs=vecs)
    return dict(zip(texts, vecs))


def pools_of(view, cands, E):
    """Review-embedding pools of the candidates `cands` of a view, and the request vector."""
    return [np.stack([E[r.get("text", "")] for r in view.items[c]["reviews"]]) for c in cands], E[view.query]
