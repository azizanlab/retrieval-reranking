"""Generate reasoning traces and score candidate labels for deterministic experiment specifications.

Outputs are resumable JSONL records, keyed by instance and experimental condition.
Qwen traces use vLLM; gpt-oss traces and both label readouts use transformers.
"""
import argparse, glob, gzip, importlib, inspect, json, os, zlib

from rerank import prompts

MODELS = {"qwen3-8b": ("Qwen/Qwen3-8B", "qwen"), "gpt-oss-20b": ("openai/gpt-oss-20b", "gpt-oss")}
THINK_BUDGET = 4096
GPT_OSS_DATE = "2026-11-18"
PREFILL_TOKENS, PREFILL_CHUNK = 6000, 1024
RESULTS = os.environ.get("RERANK_RESULTS_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))


def out_dir(model, exp):
    d = os.path.join(RESULTS, model, exp)
    os.makedirs(d, exist_ok=True)
    return d


def _read(pattern):
    out = []
    paths = glob.glob(pattern) + glob.glob(pattern + ".gz")
    for f in sorted(paths, key=lambda p: (p.removesuffix(".gz"), not p.endswith(".gz"))):
        opener = gzip.open if f.endswith(".gz") else open
        with opener(f, "rt", encoding="utf-8") as fh:
            out += [json.loads(l) for l in fh if l.strip()]
    return out


def load(model, exp):
    """Every scored read of an experiment, one row per key (a restarted shard can repeat a key)."""
    return list({r["key"]: r for r in _read(os.path.join(RESULTS, model, exp, "scores.*.jsonl"))}.values())


def specs(model, exp, shard="0/1", limit=None):
    """`exp` is a module of experiments/, optionally with a part: "fulllist:grid"."""
    name, _, part = exp.partition(":")
    fn = importlib.import_module(f"experiments.{name}").specs
    kw = {"part": part} if part else {}
    all_ = fn(model, limit, **kw) if "model" in inspect.signature(fn).parameters else fn(limit, **kw)
    assert len({s["key"] for s in all_}) == len(all_), "duplicate read keys"


    i, n = map(int, shard.split("/"))
    return [s for s in all_ if zlib.crc32(s["key"].encode()) % n == i]


def prompt_ids(tok, model, user):
    msgs = [{"role": "system", "content": prompts.SYSTEM}, {"role": "user", "content": user}]


    kw = {"reasoning_effort": "medium", "strftime_now": lambda fmt: GPT_OSS_DATE} if MODELS[model][1] == "gpt-oss" else {}
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, **kw)


def encode(tok, text):
    return tok.encode(text, add_special_tokens=False)


def think(model, exp, shard, limit):
    """Generate the reasoning traces this shard's reasoning reads need."""
    from transformers import AutoTokenizer
    d = out_dir(model, exp.partition(":")[0])
    done = {t["key"] for t in _read(os.path.join(d, "traces.*.jsonl"))}
    todo = [s for s in specs(model, exp, shard, limit) if s["think"] and s["key"] not in done]
    if not todo:
        return
    hf_id, fam = MODELS[model]
    path = os.path.join(d, f"traces.{shard.replace('/', 'of')}.jsonl")
    if fam == "gpt-oss":
        return _think_transformers(model, todo, path)
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(hf_id)
    stop = [tok.convert_tokens_to_ids("</think>"), tok.eos_token_id]
    llm = LLM(hf_id, dtype="bfloat16", enable_prefix_caching=True, max_model_len=32768, gpu_memory_utilization=0.85)
    params = SamplingParams(temperature=0.0, max_tokens=THINK_BUDGET, stop_token_ids=stop)
    for start in range(0, len(todo), 256):
        batch = todo[start:start + 256]
        ids = [prompt_ids(tok, model, s["user"]) + encode(tok, prompts.THINK[fam]["open"]) for s in batch]
        res = llm.generate([{"prompt_token_ids": x} for x in ids], params, use_tqdm=False)
        with open(path, "a") as f:
            for s, r in zip(batch, res):
                trace = list(r.outputs[0].token_ids)
                if trace and trace[-1] in stop:
                    trace = trace[:-1]
                f.write(json.dumps({"key": s["key"], "trace": trace}) + "\n")


def _think_transformers(model, todo, path):
    """Generate greedy gpt-oss analysis traces, batching prompts by length and chunking prefill."""
    torch = __import__("torch")
    sc = Scorer(model)
    tok, fam = sc.tok, prompts.THINK[sc.fam]
    stop = tok.convert_tokens_to_ids("<|end|>")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = {s["key"]: prompt_ids(tok, model, s["user"]) + encode(tok, fam["open"]) for s in todo}
    order = sorted(ids, key=lambda k: len(ids[k]))
    i = 0
    while i < len(order):




        bs = max(1, PREFILL_TOKENS // min(len(ids[order[i]]), PREFILL_CHUNK))
        keys = order[i:i + bs]; i += bs
        L = max(len(ids[k]) for k in keys)
        x = torch.tensor([[pad] * (L - len(ids[k])) + ids[k] for k in keys], device="cuda")
        m = torch.tensor([[0] * (L - len(ids[k])) + [1] * len(ids[k]) for k in keys], device="cuda")
        with torch.no_grad():
            out = sc.model.generate(input_ids=x, attention_mask=m, max_new_tokens=THINK_BUDGET, do_sample=False,
                                    eos_token_id=[stop, tok.eos_token_id], pad_token_id=pad, prefill_chunk_size=PREFILL_CHUNK)
        with open(path, "a") as f:
            for k, row in zip(keys, out[:, L:].tolist()):
                trace = []
                for t in row:
                    if t in (stop, tok.eos_token_id):
                        break
                    trace.append(t)
                f.write(json.dumps({"key": k, "trace": trace}) + "\n")


class Scorer:
    def __init__(self, model):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        hf_id, self.fam = MODELS[model]
        self.model_name, self.torch = model, torch
        self.tok = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.bfloat16, device_map="cuda").eval()
        self.answers = [encode(self.tok, prompts.answer(i)) for i in range(20)]

        assert len({len(a) for a in self.answers}) == 1

    def ids(self, spec, trace):
        fam = prompts.THINK[self.fam]
        ids = prompt_ids(self.tok, self.model_name, spec["user"])
        if spec["think"]:
            return ids + encode(self.tok, fam["open"]) + trace + encode(self.tok, fam["close"])
        return ids + encode(self.tok, fam["empty"])

    def _logprobs(self, hidden):


        W = self.model.lm_head.weight
        return (hidden.float() @ W.float().T).log_softmax(-1)

    def scores(self, ids, n):
        """Score all candidate answers using a shared, chunked prompt cache and float32 log probabilities."""
        torch = self.torch
        with torch.no_grad():
            x, out = torch.tensor([ids], device="cuda"), None
            for a in range(0, len(ids), PREFILL_CHUNK):
                out = self.model.model(x[:, a:a + PREFILL_CHUNK], past_key_values=out and out.past_key_values, use_cache=True)
            first = self._logprobs(out.last_hidden_state[0, -1])
            cache = out.past_key_values
            cache.batch_repeat_interleave(n)
            A = torch.tensor(self.answers[:n], device="cuda")
            rest = self._logprobs(self.model.model(A[:, :-1], past_key_values=cache, use_cache=False).last_hidden_state)
            total = first[A[:, 0]] + rest.gather(-1, A[:, 1:, None]).squeeze(-1).sum(1)
        return total.tolist()


def score(model, exp, shard, limit):
    d = out_dir(model, exp.partition(":")[0])
    done = {r["key"] for r in _read(os.path.join(d, "scores.*.jsonl"))}
    todo = [s for s in specs(model, exp, shard, limit) if s["key"] not in done]
    if not todo:
        return
    traces = {t["key"]: t["trace"] for t in _read(os.path.join(d, "traces.*.jsonl"))} if any(s["think"] for s in todo) else {}
    missing = [s["key"] for s in todo if s["think"] and s["key"] not in traces]
    assert not missing, f"{len(missing)} reasoning reads have no trace; run the think step first"
    sc = Scorer(model)
    with open(os.path.join(d, f"scores.{shard.replace('/', 'of')}.jsonl"), "a") as f:
        for s in todo:
            trace = traces.get(s["key"], [])
            prompt = prompt_ids(sc.tok, model, s["user"])
            f.write(json.dumps({"key": s["key"], "info": s["info"], "scores": sc.scores(sc.ids(s, trace), s["n"]),
                                "prompt_tokens": len(prompt), "trace_tokens": len(trace)}) + "\n")
            f.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["think", "score"])
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--exp", required=True, help="fulllist[:grid|selection|omit], shortlist[:main|omit], availability[:random|rank]")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=None, help="only instances with index below this limit")
    a = ap.parse_args()
    (think if a.step == "think" else score)(a.model, a.exp, a.shard, a.limit)
