"""The task, the prompts and the statistics."""
import json, os, random

import numpy as np
import pytest

from rerank import build, data, prompts, stats

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def test_corpus_sizes_and_one_target_per_view():
    for corpus in ("amazon", "reddit"):
        vs = data.views(corpus)
        assert len(vs) == 750
        assert all(len(v.titles) == 20 and v.items[v.target]["item_id"] != "" for v in vs)


def test_deep_pools_extend_the_published_reviews():
    pub = {v.key: v for v in data.views("amazon")}
    for v in data.views("amazon", "verified")[:200]:
        for a, b in zip(v.items, pub[v.key].items):
            if a["reviews"] and b["reviews"]:
                assert [r["text"] for r in a["reviews"][:len(b["reviews"])]] == [r["text"] for r in b["reviews"]]


def test_eligible_counts_match_the_paper_denominators():
    """746 (filled) and 604 (verified) instances have a review for all twenty candidates."""
    full = lambda pool: sum(all(it["reviews"] for it in v.items) for v in data.views("amazon", pool))
    assert (full("filled"), full("verified")) == (746, 604)


def test_source_review_is_excluded():
    inst = {"user_id": "u", "src_review": "Great sleeves, they lay flat in the binder"}
    assert data._is_source({"user_id": "u", "text": "anything"}, inst)
    assert data._is_source({"text": "Great sleeves, they lay flat in the binder"}, inst)
    assert not data._is_source({"text": "Terrible dice"}, inst)


@pytest.mark.model_assets
def test_answers_have_equal_length_under_both_tokenizers():
    from transformers import AutoTokenizer
    for hf in ("Qwen/Qwen3-8B", "openai/gpt-oss-20b"):
        tok = AutoTokenizer.from_pretrained(hf)
        assert len({len(tok.encode(prompts.answer(i), add_special_tokens=False)) for i in range(20)}) == 1


def test_filler_matches_the_word_count_and_template_has_none():
    f = prompts.format_reviews("03", "Some title", [{"rating": 4, "text": "one two three four five six"}], 1)
    words = lambda s: sum(len(l.split()) for l in s.splitlines()[1:])
    assert words(prompts.filler_evidence(f)) == words(f)
    assert prompts.template_evidence(f).splitlines()[1].strip() == "- " + prompts.NO_INFO


@pytest.mark.model_assets
def test_prompt_tokens_match_benchmark_fixture():
    """Check prompt lengths against the fixed benchmark regression fixture."""
    from transformers import AutoTokenizer
    from rerank import embed, engine, select
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    with open(os.path.join(FIXTURES, "prompt_tokens.json")) as f:
        ref = {tuple(k.split("|")): v for k, v in json.load(f).items()}
    names = {"amazon": "Amazon-20", "reddit": "Reddit"}
    settings = {"titles": "titles-only", "meta": "titles + metadata"}
    for corpus in ("amazon", "reddit"):
        views = [v for v in data.views(corpus, "filled") if all(it["reviews"] for it in v.items)]
        E = embed.embeddings(corpus, "filled")
        for setting in ("titles", "meta"):
            n_none, n_con = [], []
            for v in views:
                pools, q = embed.pools_of(v, range(20), E)
                lab = ["%02d" % i for i in range(20)]
                blocks = build.review_blocks(v, list(range(20)), lab, select.rule("contrastive", "shared", pools, q, 10))
                n_none.append(len(engine.prompt_ids(tok, "qwen3-8b", build.user(v, range(20), setting))))
                n_con.append(len(engine.prompt_ids(tok, "qwen3-8b", build.user(v, range(20), setting, blocks))))
            b, s = names[corpus], settings[setting]
            assert round(np.mean(n_none)) == ref[(b, s, "none")]
            assert round(np.mean(n_con)) == ref[(b, s, "global_contrast")]


def test_bootstrap_and_holm():
    d = np.r_[np.ones(60), np.zeros(40)]
    mean, lo, hi, p = stats.bootstrap(d, 2000)
    assert mean == 0.6 and lo > 0.5 and hi <= 0.7 and p < 0.001
    assert np.allclose(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_rank_breaks_ties_by_position():
    assert stats.rank_of([0.0, 0.0, 1.0], 0) == 2 and stats.rank_of([0.0, 0.0, 1.0], 1) == 3


@pytest.mark.model_assets
def test_gpt_oss_prompt_does_not_depend_on_the_day():
    from transformers import AutoTokenizer
    from rerank import engine
    tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    text = tok.decode(engine.prompt_ids(tok, "gpt-oss-20b", "x"))
    assert f"Current date: {engine.GPT_OSS_DATE}" in text and "Reasoning: medium" in text


def test_random_arms_average_their_three_draws_and_nothing_else():
    from analysis.report import draws
    assert draws("count-3") == ["count-3", "count-3-s1", "count-3-s2"]
    assert draws("count-3-omit") == ["count-3-omit", "count-3-s1-omit", "count-3-s2-omit"]
    assert draws("v-presence") == ["v-presence", "v-presence-s1", "v-presence-s2"]
    assert draws("relevance-resampled") == ["relevance-resampled", "relevance-resampled-s1", "relevance-resampled-s2"]
    assert draws("contrastive-10") == ["contrastive-10"] and draws("v-contrastive-shared20") == ["v-contrastive-shared20"]
