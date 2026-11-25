# Data and result formats

Amazon-20 is derived from Amazon Reviews 2023 in Toys and Games, Grocery and Gourmet Food, and Office Products. Reddit is derived from the Reddit-Movie recommendation corpus. These are the curated instances and review pools used by the accompanying paper, rather than the complete source datasets.

Requests were paraphrased with Llama-3.1-8B. Candidate lists use similarity-based negative mining and popularity matching. Source reviews or comments are excluded from the evidence pool. The paper's task appendix gives the construction procedure and audits.

## Benchmark files

- `hq_*_r200pm.jsonl.gz`: instance records containing the request, user history, target, candidate metadata, and published review pools. The loader retains the first 250 instances of each Amazon category and all 750 Reddit instances.
- `deeprev20_*.jsonl.gz`: Amazon pools of up to twenty verified reviews per item.
- `deeprev20u_*.jsonl.gz`: Amazon pools supplemented with unverified reviews where verified reviews are unavailable.

Full-list retrieval/reasoning comparisons use 746 eligible Amazon instances and 750 Reddit instances. Full-list selection with verified pools uses 604 Amazon instances. Availability retains all 750 instances in each corpus; a candidate without a review receives its placeholder. Shortlist cohort sizes depend on the model, prompt setting, and shortlist size and are recorded in the summaries.

## Scored reads

Each `scores.*.jsonl.gz` record contains:

| Field | Meaning |
|---|---|
| `key` | Instance and condition identifier, used to resume and deduplicate reads |
| `info` | Condition metadata, target position, and any shortlist or shown-set fields |
| `scores` | Log probabilities of the allowed candidate-label answers |
| `prompt_tokens` | Prompt length under the model's tokenizer |
| `trace_tokens` | Length of the generated reasoning trace; zero for direct reads |

These files store scores and token counts, not model weights or reasoning traces. Results are grouped by model and experiment. Records for optional diagnostic conditions may be present; the publication builder selects the paper's comparisons.

## Summary files

`results/<model>/tables.json` contains ranking metrics, paired contrasts, confidence intervals, Holm-adjusted probabilities where applicable, cohort sizes, and draw counts. Availability entries in these summaries use review blocks. `analysis.recompute` rebuilds these summaries from the scored reads.

`MANIFEST.json` lists record counts and SHA-256 hashes of the decompressed data and score files. Compression preserves their contents exactly.
