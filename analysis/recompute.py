"""Recompute publication summaries from scored reads, without model inference."""
import argparse
import functools
import json
import math
from pathlib import Path

import numpy as np

from analysis import report
from experiments import shortlist
from rerank import data, engine, stats

MODELS = ("qwen3-8b", "gpt-oss-20b")
DIAGNOSTIC_KEYS = {"rates_rank", "order_effect", "omit_short"}


def checked_load(model, experiment):
    records = {}
    pattern = str(Path(engine.RESULTS) / model / experiment / "scores.*.jsonl")
    for record in engine._read(pattern):
        assert all(math.isfinite(s) for s in record["scores"]), record["key"]
        assert 0 <= record["info"]["target"] < len(record["scores"]), record["key"]
        if record["key"] in records:
            assert records[record["key"]] == record, f"Conflicting duplicate read: {record['key']}"
        records[record["key"]] = record
    if not records:
        raise FileNotFoundError(f"No scored reads found for {model}/{experiment}")
    return list(records.values())


def compare(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                yield path + "/" + k
            else:
                yield from compare(a[k], b[k], path + "/" + k)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            yield from compare(x, y, path + f"/{i}")
    elif isinstance(a, (float, int)) and isinstance(b, (float, int)):
        if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12):
            yield f"{path}: {a} vs {b}"
    elif a != b:
        yield f"{path}: {a} vs {b}"


def extras(model):
    idx = {(r["info"]["instance"], r["info"]["setting"], r["info"]["arm"], r["info"]["think"]): r
           for r in engine.load(model, "fulllist")}
    out = {"selected10_vs_random40": [], "budget_seed_means": [], "availability_empty": []}
    for corpus in report.CORPORA:
        keys = [v.key for v in data.views(corpus, "filled") if all(it["reviews"] for it in v.items)]
        for setting in report.SETTINGS:
            def hits(arm):
                return np.array([stats.rank_of(idx[k, setting, arm, False]["scores"],
                                             idx[k, setting, arm, False]["info"]["target"]) == 1 for k in keys], float)
            sel, rand = hits("contrastive-10"), np.mean([hits(a) for a in report.draws("count-40")], axis=0)
            out["selected10_vs_random40"].append(dict(corpus=corpus, setting=setting, n=len(keys), **report._paired(sel, rand)))
            for B in (3, 5, 10, 20, 40):
                means = [float(hits(a).mean()) for a in report.draws(f"count-{B}")]
                out["budget_seed_means"].append(dict(corpus=corpus, setting=setting, B=B, means=means))
                if model == "qwen3-8b" and B in (3, 5, 10):


                    for arm in report.draws(f"count-{B}-omit"):
                        assert len(hits(arm)) == len(keys)
        prior = shortlist.first_stage(model)
        n_empty = sum(any(not v.items[c]["reviews"] for c in sorted(range(20), key=lambda c: (-prior[v.key + "|titles"][c], c))[:3])
                      for v in data.views(corpus))
        out["availability_empty"].append(dict(corpus=corpus, n=750, any_empty_top3=n_empty))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--check", action="store_true", help="Compare with saved summaries without changing them")
    args = parser.parse_args()
    engine.load = functools.lru_cache(maxsize=3)(checked_load)
    data.views = functools.lru_cache(maxsize=None)(data.views)
    shortlist.first_stage = functools.lru_cache(maxsize=1)(shortlist.first_stage)
    failed = False
    for model in ([args.model] if args.model else MODELS):
        print(f"Recomputing {model}", flush=True)
        result = {}
        for fn in (report.fulllist, report.shortlist_tables, report.availability_tables):
            result.update(fn(model, None))
            print(f"  {fn.__name__} complete", flush=True)
        for name in ("table1", "budgets", "table10", "table5", "table4", "table3"):
            for row in result[name]:
                for key in ("draws", "presence_draws", "resample_draws"):
                    if key in row:
                        assert row[key] in (1, 3), (name, key, row)
                if row.get("arm", "").startswith("count") or row.get("config", "").startswith("arbitrary"):
                    assert row["draws"] == 3, row
                assert row.get("presence_draws", 3) == 3 and row.get("resample_draws", 3) == 3, row
        result = {key: [row for row in rows if row.get("mode", "reviews") == "reviews"]
                  for key, rows in result.items() if key not in DIAGNOSTIC_KEYS}
        result.update(extras(model))
        path = Path(engine.RESULTS) / model / "tables.json"
        if args.check:
            differences = list(compare(result, json.loads(path.read_text())))
            print(f"{model}: {len(differences)} differences from saved summaries", flush=True)
            for difference in differences[:20]:
                print("  " + difference)
            failed |= bool(differences)
        else:
            path.write_text(json.dumps(result, indent=2) + "\n")
            print(f"Wrote {path}", flush=True)
        engine.load.cache_clear()
        shortlist.first_stage.cache_clear()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
