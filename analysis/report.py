"""Aggregate ranking metrics and paired contrasts from scored reads.

Random-control draws and availability presentations are averaged within instance
before bootstrapping. Outputs record cohort sizes and the number of available draws.
"""
import argparse, collections, glob, json, os

import numpy as np

from rerank import data, engine, stats
from experiments import availability, shortlist

CORPORA, SETTINGS = ("amazon", "reddit"), ("titles", "meta")
N_BOOT_AVAIL = 5000
DRAWS = ("", "-s1", "-s2")
METRICS = ("hr1", "hr10", "mrr", "ndcg10")


def _random(arm):
    return arm.startswith(("count-", "v-presence", "presence")) or "-resampled" in arm


def draws(arm):
    """The names of every draw of an arm: three for a random control, else the arm itself."""
    if arm.endswith("-omit"):
        return [d + "-omit" for d in draws(arm[:-len("-omit")])]
    return [arm + d for d in DRAWS] if _random(arm) else [arm]


def _paired(a, b, n_boot=20000):
    m, lo, hi, p = stats.bootstrap(np.asarray(a, float) - np.asarray(b, float), n_boot)
    return {"delta": m, "lo": lo, "hi": hi, "p": p}


def _level(a, n_boot=20000):
    m, lo, hi, _ = stats.bootstrap(np.asarray(a, float), n_boot)
    return {"mean": m, "lo": lo, "hi": hi}


def _holm(contrasts):
    for c, ph in zip(contrasts, stats.holm([c["p"] for c in contrasts])):
        c["holm"] = ph
    return contrasts


class Arms:
    """Per-instance values of an arm, averaged over its draws. `value(row)` maps a read to a number;
    an arm is available when its first draw is there for every instance, and then every draw present
    for every instance is used."""

    def __init__(self, lookup, keys):
        self.lookup, self.keys = lookup, keys

    def n_draws(self, arm, *rest):
        names = draws(arm)
        full = [d for d in names if all(self.lookup((k, d) + rest) is not None for k in self.keys)]
        return len(full) if full and full[0] == names[0] else 0

    def values(self, arm, value, *rest):
        names = draws(arm)[:self.n_draws(arm, *rest)]
        if not names:
            return None
        return [float(np.mean([value(self.lookup((k, d) + rest)) for d in names])) for k in self.keys]



def fulllist(model, limit):
    rows = engine.load(model, "fulllist")
    idx = {(r["info"]["instance"], r["info"]["setting"], r["info"]["arm"], r["info"]["think"]): r for r in rows}
    out = {k: [] for k in ("table1", "table7", "table8", "budgets", "table10", "omit")}
    for corpus in CORPORA:
        for pool in ("filled", "verified"):
            keys = [v.key for v in data.views(corpus, pool) if all(it["reviews"] for it in v.items)
                    and (limit is None or v.idx < limit)]
            for setting in SETTINGS:
                A = Arms(lambda t: idx.get((t[0], setting, t[1], t[2] if len(t) > 2 else False)), keys)
                metric = lambda m: (lambda r: stats.metrics(stats.rank_of(r["scores"], r["info"]["target"]))[m])
                hr1 = lambda arm, think=False: A.values(arm, metric("hr1"), think)
                cell = {"corpus": corpus, "setting": setting, "n": len(keys)}
                if pool == "filled":
                    configs = [("no retrieval, no reasoning", "none", False), ("arbitrary retrieval, no reasoning", "count-10", False),
                               ("retrieval, no reasoning", "contrastive-10", False), ("reasoning, no retrieval", "none", True),
                               ("retrieval and reasoning", "contrastive-10", True)]
                    for name, arm, think in configs:
                        if not A.n_draws(arm, think):
                            continue
                        m = {k: float(np.mean(A.values(arm, metric(k), think))) for k in METRICS}
                        lv = _level(hr1(arm, think))
                        out["table1"].append({**cell, "config": name, **m, "hr1_lo": lv["lo"], "hr1_hi": lv["hi"],
                                              "draws": A.n_draws(arm, think),
                                              "prompt": float(np.mean(A.values(arm, lambda r: r["prompt_tokens"], think))),
                                              "generated": float(np.mean(A.values(arm, lambda r: r["trace_tokens"], think))),
                                              "truncated": float(np.mean(A.values(arm, lambda r: r["trace_tokens"] >= engine.THINK_BUDGET, think)))})
                    t8 = [("retrieval, no reasoning - neither", ("contrastive-10", False), ("none", False)),
                          ("against arbitrary retrieval", ("contrastive-10", False), ("count-10", False)),
                          ("retrieval and reasoning - reasoning only", ("contrastive-10", True), ("none", True)),
                          ("reasoning, no retrieval - neither", ("none", True), ("none", False)),
                          ("retrieval and reasoning - retrieval only", ("contrastive-10", True), ("contrastive-10", False)),
                          ("retrieval only - reasoning only", ("contrastive-10", False), ("none", True))]
                    for name, a, b in t8:
                        if hr1(*a) and hr1(*b):
                            out["table8"].append({**cell, "contrast": name, **_paired(hr1(*a), hr1(*b))})
                    four = ("none", "count-10", "contrastive-10", "refiled-10")
                    if all(hr1(a) for a in four):
                        t7 = [("retrieved - arbitrary", "contrastive-10", "count-10"), ("arbitrary - none", "count-10", "none"),
                              ("retrieved - none", "contrastive-10", "none"), ("re-filed - none", "refiled-10", "none"),
                              ("retrieved - re-filed", "contrastive-10", "refiled-10")]
                        out["table7"] += _holm([{**cell, "contrast": name, **_paired(hr1(a), hr1(b))} for name, a, b in t7])
                        out["table7"].append({**cell, "contrast": "level", **{a: float(np.mean(hr1(a))) for a in four}})
                    fam = []
                    for B in (3, 5, 10, 20, 40):
                        for arm, think in ((f"contrastive-{B}", False), (f"contrastive-{B}", True), (f"count-{B}", False)):
                            if hr1(arm, think):
                                lv = _level(hr1(arm, think))
                                out["budgets"].append({**cell, "B": B, "arm": arm.split("-")[0], "think": think, "hr1": lv["mean"],
                                                       "lo": lv["lo"], "hi": lv["hi"], "draws": A.n_draws(arm, think),
                                                       "tokens": float(np.mean(A.values(arm, lambda r: r["prompt_tokens"] + r["trace_tokens"], think)))})
                        if hr1(f"count-{B}") and hr1("none"):
                            fam.append({**cell, "B": B, "contrast": "arbitrary - none", **_paired(hr1(f"count-{B}"), hr1("none"))})
                    out["budgets"] += _holm(fam)
                    omit_arms = [f"{a}-{B}" for B in (3, 5, 10, 20, 40) for a in ("contrastive", "count")] + ["refiled-10"]
                else:
                    arms = ["none", "v-relevance-each", "v-relevance-shared20", "v-contrastive-shared20", "v-dpp-shared20", "v-mmr-shared20"]
                    base = hr1("v-presence")
                    if base and all(hr1(a) for a in arms):
                        fam = [{**cell, "arm": a, "hr1": float(np.mean(hr1(a))), **_paired(hr1(a), base)} for a in arms]
                        out["table10"] += _holm(fam)
                        out["table10"].append({**cell, "arm": "v-presence", "hr1": float(np.mean(base)), "draws": A.n_draws("v-presence")})
                    omit_arms = [f"v-{o}-shared20" for o in ("relevance", "contrastive", "mmr", "dpp")]

                for arm in omit_arms:
                    a, b = hr1(arm), hr1(arm + "-omit")
                    if a and b:
                        out["omit"].append({**cell, "where": "full list", "arm": arm, "with_line": float(np.mean(a)),
                                            "omitted": float(np.mean(b)), **_paired(b, a)})
                for B in ((3, 5, 10) if pool == "filled" else ()):
                    a, b = hr1(f"count-{B}-omit"), hr1("none")
                    if a and b:
                        out["omit"].append({**cell, "where": "full list", "arm": f"arbitrary-{B} - none, line omitted", **_paired(a, b)})
    return out



def _short_hr1(row):
    """Top-1 on the shortlist, ties broken by the first-stage order."""
    s, tb = row["scores"], row["info"]["tiebreak"]
    return float(min(range(len(s)), key=lambda i: (-s[i], tb[i])) == row["info"]["target"])


def shortlist_tables(model, limit):
    rows = engine.load(model, "shortlist")
    idx = {(r["info"]["instance"], r["info"]["setting"], r["info"]["K"], r["info"]["pool"], r["info"]["arm"]): r for r in rows}
    out = {"table5": [], "table4": [], "table3": [], "omit_short": []}
    for corpus in CORPORA:
        for pool in ("verified", "filled"):
            groups = collections.defaultdict(list)
            for v, setting, K, short, *_ in shortlist.shortlists(model, corpus, pool, limit):
                groups[(setting, K)].append((v, v.target in short))
            for (setting, K), members in groups.items():
                cell = {"corpus": corpus, "setting": setting, "K": K, "n": len(members),
                        "recall": float(np.mean([hit for _, hit in members]))}
                read = [v.key for v, hit in members if hit]
                A = Arms(lambda t: idx.get((t[0], setting, K, pool, t[1])), read)
                def hr1(arm):
                    """Outside the shortlist is a miss for every arm; inside, the read's top-1."""
                    vals = A.values(arm, _short_hr1)
                    if vals is None:
                        return None
                    it = iter(vals)
                    return [next(it) if hit else 0.0 for _, hit in members]
                base = hr1("presence")
                if base is None:
                    continue
                cell["presence_draws"] = A.n_draws("presence")
                if pool == "verified":
                    rules = [f"{o}-{a}" for a in ("one_each", "shared") for o in shortlist.OBJECTIVES]
                    if all(hr1(a) for a in rules + ["none"]):
                        fam = [{**cell, "arm": a, "hr1": float(np.mean(hr1(a))), **_paired(hr1(a), base)} for a in rules]
                        out["table5"] += _holm(fam)
                        out["table5"].append({**cell, "arm": "presence - none", **_paired(base, hr1("none"))})
                    designed = ["relevance-one_each", "relevance-shared", "leader", "softmax", "inverse"]
                    if all(hr1(a) for a in designed):
                        out["table4"] += _holm([{**cell, "arm": a, **_paired(hr1(a), base)} for a in designed])
                    for arm in [f"{o}-shared" for o in shortlist.OBJECTIVES] + ["leader", "softmax", "inverse"]:
                        a, b = hr1(arm), hr1(arm + "-omit")
                        if a and b:
                            out["omit_short"].append({**cell, "arm": arm, "with_line": float(np.mean(a)) - float(np.mean(base)),
                                                      "omitted": float(np.mean(b)) - float(np.mean(base)), **_paired(b, a)})
                else:
                    for obj in ("relevance", "contrastive"):
                        rule, res = hr1(f"{obj}-shared"), hr1(f"{obj}-resampled")
                        if rule and res:
                            extra = {"resample_draws": A.n_draws(f"{obj}-resampled")}
                            out["table3"].append({**cell, **extra, "rule": obj, "term": "total", **_paired(rule, base)})
                            out["table3"].append({**cell, **extra, "rule": obj, "term": "allocation", **_paired(res, base)})
                            out["table3"].append({**cell, **extra, "rule": obj, "term": "content", **_paired(rule, res)})
    fams = collections.defaultdict(list)
    for r in out["table3"]:
        if r["term"] != "total":
            fams[(r["corpus"], r["setting"], r["K"])].append(r)
    for fam in fams.values():
        _holm(fam)
    return out



def _top1(r):
    return float(int(np.argmax(r["scores"])) in r["info"]["shown"])


def availability_tables(model, limit):
    rows = engine.load(model, "availability")
    none = {r["info"]["instance"]: r for r in engine.load(model, "fulllist")
            if r["info"]["arm"] == "none" and r["info"]["setting"] == "titles" and not r["info"]["think"]}
    idx = {(r["info"]["instance"], r["info"]["shown_set"], r["info"]["mode"], r["info"]["content"], r["info"].get("order", "rank")): r
           for r in rows}
    labelorder = {}
    for f in glob.glob(os.path.join(engine.RESULTS, model, "availability_labelorder", "scores.*.jsonl")):
        for r in map(json.loads, open(f)):
            labelorder[(r["info"]["instance"], r["info"]["shown_set"], r["info"]["mode"], r["info"]["content"])] = r
    out = {"rates": [], "rates_rank": [], "table6": [], "order_effect": []}
    for corpus in CORPORA:
        keys = [v.key for v in data.views(corpus) if limit is None or v.idx < limit]
        if any(k not in none for k in keys):
            continue
        for name in availability.SHOWN:
            for mode in availability.MODES:
                cell = {"corpus": corpus, "shown_set": name, "mode": mode, "n": len(keys)}
                get = lambda k, c, o: idx.get((k, name, mode, c, o))

                if all(get(k, c, "rank") for k in keys for c in availability.CONTENTS):
                    shown = [get(k, "real", "rank")["info"]["shown"] for k in keys]
                    out["rates_rank"].append({**cell, "content": "none",
                                              "rate": float(np.mean([_top1({**none[k], "info": {"shown": s}}) for k, s in zip(keys, shown)]))})
                    for c in availability.CONTENTS:
                        out["rates_rank"].append({**cell, "content": c, "rate": float(np.mean([_top1(get(k, c, "rank")) for k in keys]))})

                if not all(get(k, c, o) for k in keys for c in availability.CONTENTS for o in availability.ORDERS):
                    continue
                shown = [get(k, "real", 0)["info"]["shown"] for k in keys]
                base = [none[k] for k in keys]
                lo0 = [stats.shown_mass_logodds(b["scores"], s) for b, s in zip(base, shown)]
                pre = [_top1({**b, "info": {"shown": s}}) for b, s in zip(base, shown)]
                out["rates"].append({**cell, "content": "none", **_level(pre, N_BOOT_AVAIL), "rate": float(np.mean(pre)),
                                     "mass": float(np.mean([1 / (1 + np.exp(-x)) for x in lo0]))})
                lodds, hits = {}, {}
                for c in availability.CONTENTS:
                    hits[c] = np.array([[_top1(get(k, c, o)) for o in availability.ORDERS] for k in keys])
                    per = hits[c].mean(1)
                    out["rates"].append({**cell, "content": c, **_level(per, N_BOOT_AVAIL), "rate": float(per.mean())})
                    lodds[c] = [np.mean([stats.shown_mass_logodds(get(k, c, o)["scores"], get(k, c, o)["info"]["shown"])
                                         for o in availability.ORDERS]) for k in keys]
                    out["table6"].append({**cell, "content": c, **_paired(lodds[c], lo0, N_BOOT_AVAIL)})
                out["table6"].append({**cell, "content": "real - swap", **_paired(lodds["real"], lodds["swap"], N_BOOT_AVAIL)})

                for c in availability.CONTENTS:
                    h = hits[c]
                    row = {**cell, "content": c, **{f"order{o}": float(h[:, i].mean()) for i, o in enumerate(availability.ORDERS)},
                           "order_spread": float(h.mean(0).max() - h.mean(0).min()),
                           "flips_across_orders": float(np.mean(h.min(1) != h.max(1))),
                           "flips_own_vs_template": float(np.mean(hits["real"][:, 0] != hits["template"][:, 0]))}
                    if all(get(k, c, "rank") for k in keys):
                        row["rank_order"] = float(np.mean([_top1(get(k, c, "rank")) for k in keys]))
                    if all((k, name, mode, c) in labelorder for k in keys):
                        row["label_order"] = float(np.mean([_top1(labelorder[(k, name, mode, c)]) for k in keys]))
                    out["order_effect"].append(row)
    return out



def _fmt(x):
    return f"{x:+.3f}" if isinstance(x, float) else str(x)


def _md(title, rows, cols):
    if not rows:
        return f"## {title}\n\n(no results yet)\n"
    lines = [f"## {title}\n", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _ci(r):
    if "delta" in r:
        r = dict(r); r["delta"] = f"{r['delta']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]" + ("*" if r.get("holm", 1) < 0.05 else "")
    return r


def figures(model, t):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = engine.out_dir(model, "")
    styles = {"no retrieval, no reasoning": ("o", "0.5"), "arbitrary retrieval, no reasoning": ("^", "C2"),
              "retrieval, no reasoning": ("s", "C0"), "reasoning, no retrieval": ("D", "C3"), "retrieval and reasoning": ("*", "C1")}


    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, corpus in zip(axes, CORPORA):
        rows = [r for r in t["table1"] if r["corpus"] == corpus and r["setting"] == "titles"]
        for r in rows:
            m, c = styles[r["config"]]
            ax.errorbar(r["prompt"] + r["generated"], r["hr1"], yerr=[[r["hr1"] - r["hr1_lo"]], [r["hr1_hi"] - r["hr1"]]],
                        fmt=m, color=c, ms=7, capsize=3, label=r["config"])
        if rows:
            ax.axhline(next(r["hr1"] for r in rows if r["config"] == "no retrieval, no reasoning"), ls="--", color="0.6", lw=1)
            ax.set(xscale="log", xlabel="tokens per query (prompt + generated, log)", title=f"{corpus}, titles only (n={rows[0]['n']})")
    axes[0].set_ylabel("HR@1 (95% interval)"); axes[1].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Ranking quality against cost ({model})", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(d, "fig1_cost.png"), dpi=150); plt.close(fig)


    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, corpus in zip(axes, CORPORA):
        pts = [r for r in t["budgets"] if r["corpus"] == corpus and r["setting"] == "titles" and "hr1" in r]
        for arm, think, label, style in (("contrastive", False, "retrieval, no reasoning", "o-"),
                                         ("contrastive", True, "retrieval and reasoning", "s-"),
                                         ("count", False, "arbitrary retrieval, no reasoning", "^--")):
            sel = sorted((r for r in pts if r["arm"] == arm and r["think"] == think), key=lambda r: r["B"])
            if sel:
                ax.errorbar([r["B"] for r in sel], [r["hr1"] for r in sel], capsize=3, fmt=style, label=label,
                            yerr=[[r["hr1"] - r["lo"] for r in sel], [r["hi"] - r["hr1"] for r in sel]])
        for r in t["table1"]:
            if r["corpus"] == corpus and r["setting"] == "titles" and r["config"] in ("no retrieval, no reasoning", "reasoning, no retrieval"):
                ax.axhline(r["hr1"], ls=":" if "no reasoning" in r["config"] else "--", color="grey", label=r["config"])
        n = next((r["n"] for r in t["table1"] if r["corpus"] == corpus), "?")
        ax.set(xscale="log", xticks=[3, 5, 10, 20, 40], xticklabels=["3", "5", "10", "20", "40"],
               xlabel="reviews in the prompt, in total", title=f"{corpus}, titles only (n={n})")
    axes[0].set_ylabel("HR@1 (95% interval)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(f"Ranking quality as the review budget grows ({model})", fontsize=10)
    fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(os.path.join(d, "fig3_budgets.png"), dpi=150); plt.close(fig)


    fig, axes = plt.subplots(2, 2, figsize=(9, 5.5))
    for ax, (corpus, mode) in zip(axes.T.ravel(), [(c, m) for c in CORPORA for m in availability.MODES]):
        rates = {r["content"]: r for r in t["rates"] if r["corpus"] == corpus and r["mode"] == mode and r["shown_set"] == "top"}
        if not rates:
            continue
        cs = list(availability.CONTENTS)
        ax.barh(range(len(cs)), [100 * rates[c]["rate"] for c in cs], color=["C0", "C1", "0.6", "0.8"],
                xerr=[[100 * (rates[c]["rate"] - rates[c]["lo"]) for c in cs], [100 * (rates[c]["hi"] - rates[c]["rate"]) for c in cs]], capsize=3)
        ax.axvline(100 * rates["none"]["rate"], ls="--", color="k", label="without evidence")
        ax.set(yticks=range(len(cs)), yticklabels=["own text", "swapped text", "matched filler", "template only"], xlim=(0, 100),
               title=f"{corpus}, {mode} blocks", xlabel="queries with a shown candidate ranked first (%)")
        ax.invert_yaxis()
    axes[0, 0].legend(fontsize=7)
    n = next((r["n"] for r in t["rates"]), "?")
    fig.suptitle(f"Attaching text promotes the reader's top three ({model}, n={n} per corpus, mean of 3 block orders)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(d, "fig2_availability.png"), dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    t = {**fulllist(a.model, a.limit), **shortlist_tables(a.model, a.limit), **availability_tables(a.model, a.limit)}
    d = engine.out_dir(a.model, "")
    with open(os.path.join(d, "tables.json"), "w") as f:
        json.dump(t, f, indent=1)
    cells = ["corpus", "setting"]
    md = [f"# Results: {a.model}\n", "Generated by analysis/report.py from results/; every number is in tables.json. "
          "Contrasts are paired over instances with a 95% bootstrap interval; * survives Holm in its family. "
          "Random controls are the per-instance mean of their draws (column `draws`).\n",
          _md("Table 1: retrieval and reasoning", t["table1"], cells + ["n", "config", "draws", "hr1", "hr10", "mrr", "ndcg10", "prompt", "generated", "truncated"]),
          _md("Table 8: paired contrasts at budget 10", [_ci(r) for r in t["table8"]], cells + ["n", "contrast", "delta"]),
          _md("Table 7: attribution on the full list", [_ci(r) for r in t["table7"]], cells + ["n", "contrast", "delta", "none", "count-10", "contrastive-10", "refiled-10"]),
          _md("Figure 3: budgets", [_ci(r) for r in t["budgets"]], cells + ["B", "arm", "think", "draws", "hr1", "lo", "hi", "contrast", "delta"]),
          _md("Table 10: selection on the full list (vs presence)", [_ci(r) for r in t["table10"]], cells + ["n", "arm", "draws", "hr1", "delta"]),
          _md("Table 5: eight rules on the shortlist (vs presence)", [_ci(r) for r in t["table5"]], cells + ["K", "n", "recall", "presence_draws", "arm", "hr1", "delta"]),
          _md("Table 4: designed allocations (vs presence)", [_ci(r) for r in t["table4"]], cells + ["K", "n", "presence_draws", "arm", "delta"]),
          _md("Table 3: allocation and content (filled pools)", [_ci(r) for r in t["table3"]], cells + ["K", "n", "presence_draws", "resample_draws", "rule", "term", "delta"]),
          _md("Figures 2, 4: shown set ranked first (mean of 3 block orders)", t["rates"], ["corpus", "shown_set", "mode", "n", "content", "rate", "lo", "hi", "mass"]),
          _md("Table 6: shift in the shown set's log-odds (mean of 3 block orders)", [_ci(r) for r in t["table6"]], ["corpus", "shown_set", "mode", "n", "content", "delta"]),
          _md("Block order: how much it moves the availability rates", t["order_effect"],
              ["corpus", "shown_set", "mode", "content", "order0", "order1", "order2", "order_spread", "rank_order", "label_order",
               "flips_across_orders", "flips_own_vs_template"]),
          _md("Without the empty-candidate line: full list (delta = omitted - with line)", [_ci(r) for r in t["omit"]],
              cells + ["n", "arm", "with_line", "omitted", "delta"]),
          _md("Without the empty-candidate line: shortlist gain over presence (delta = omitted - with line)", [_ci(r) for r in t["omit_short"]],
              cells + ["K", "n", "arm", "with_line", "omitted", "delta"])]
    with open(os.path.join(d, "REPORT.md"), "w") as f:
        f.write("\n".join(md))
    figures(a.model, t)
    print(f"wrote {d}/REPORT.md, tables.json, figures")


if __name__ == "__main__":
    main()
