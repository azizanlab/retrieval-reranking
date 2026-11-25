"""Metrics and the paired statistics every table uses."""
import numpy as np


def rank_of(scores, target):
    """1-based rank of the target; ties broken by presentation position, as a sort would."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return order.index(target) + 1


def metrics(rank):
    """HR@1, HR@10, MRR and NDCG@10 for one instance, from the target's rank."""
    return {"hr1": float(rank == 1), "hr10": float(rank <= 10), "mrr": 1.0 / rank,
            "ndcg10": 1.0 / np.log2(rank + 1) if rank <= 10 else 0.0}


def bootstrap(d, n_boot=20000, seed=0):
    """Mean of paired differences d with a 95% percentile interval over instances, and a two-sided
    bootstrap p-value (the share of resampled means on the other side of zero, doubled)."""
    d = np.asarray(d, float)
    means = np.random.default_rng(seed).choice(d, (n_boot, len(d))).mean(1)
    p = min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean()))
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5), p


def holm(pvalues):
    """Holm-adjusted p-values, in the input order."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (len(p) - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def shown_mass_logodds(scores, shown, clip=1e-3):
    """log(m / (1 - m)) with m the probability the softmax over the twenty puts on the shown set."""
    s = np.asarray(scores, float)
    p = np.exp(s - s.max()); p /= p.sum()
    m = np.clip(p[list(shown)].sum(), clip, 1 - clip)
    return float(np.log(m / (1 - m)))
