# Retrieval and reasoning in LLM reranking

Code and experiment artifacts for **Understanding Retrieval and Reasoning in LLM Reranking under Availability Bias**.

The experiments compare Qwen3-8B and gpt-oss-20b on Amazon-20 and Reddit recommendation tasks. They separate the contributions of reasoning, review selection, evidence allocation, and evidence assignment using paired controls.

## Install

Use Python 3.10 or later from a checkout of this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

The default installation supports analysis, figure generation, and the local tests. It does not download language models.

## Reproduce the paper artifacts

The repository includes compressed benchmark inputs, scored reads for both models, and result summaries. The loaders read the compressed files directly.

```bash
python -m analysis.build_paper
python -m analysis.recompute --check
python -m pytest
```

The first command writes ten LaTeX result tables and four PDF/PNG figures to `paper/`. The second recomputes the reported statistics from the scored reads and compares them with the saved summaries; it uses CPU only and takes longer than figure generation. Omit `--check` to write recomputed summaries. Use `--model qwen3-8b` or `--model gpt-oss-20b` to process one model.

The LaTeX tables use `booktabs` and the following interval macro:

```latex
\newcommand{\ci}[2]{{\scriptsize$[#1,#2]$}}
```

## Run model inference

Inference requires a CUDA GPU, the model weights, and the MiniLM review encoder. Install the corresponding dependencies:

```bash
python -m pip install -e ".[inference]"
python -m pip install -e ".[qwen]"
```

The `qwen` extra supplies vLLM for Qwen reasoning traces. gpt-oss traces and both models' label readouts use transformers.

```bash
bash run.sh qwen3-8b
bash run.sh gpt-oss-20b
GPUS="0 1" bash run.sh qwen3-8b
bash run.sh qwen3-8b 5
```

`run.sh` uses one worker per selected GPU on a single Linux node and waits for each stage to finish. The optional instance limit is applied within each dataset file. It writes fresh evaluations to `runs/`, leaving the supplied scored reads in `results/` available for analysis. Set `RERANK_RESULTS_DIR` to choose another output directory. Finished reads are skipped when a run resumes.

For individual stages or externally managed distributed jobs:

```bash
RERANK_RESULTS_DIR=runs python -m rerank.engine think --model qwen3-8b --exp fulllist:grid --shard 0/1
RERANK_RESULTS_DIR=runs python -m rerank.engine score --model qwen3-8b --exp fulllist:grid --shard 0/1
```

The first-stage full-list scores must exist before shortlist or availability reads. Set `RERANK_DATA_DIR` to use a different benchmark directory. The optional tokenizer and encoder checks run with `python -m pytest --run-model-checks` after installing the inference dependencies; they may download pretrained assets.

## Repository layout

| Path | Contents |
|---|---|
| `rerank/` | Data loading, prompts, selection rules, model scoring, and statistics |
| `experiments/` | Full-list, shortlist, and availability configurations |
| `analysis/` | Result aggregation, verification, and publication figures and tables |
| `tests/` | Checks for task construction, selection constraints, controls, and statistics |
| `data/` | Compressed curated instances and Amazon review-pool sidecars |
| `results/<model>/` | Compressed scored reads and `tables.json` summaries |
| `paper/` | Generated figures and LaTeX tables |
| `MANIFEST.json` | Record counts and SHA-256 hashes of the uncompressed artifacts |

## Experimental conventions

- Each instance has twenty candidates and one target. Candidate order is seeded and shared across paired conditions.
- Greedy reasoning is capped at 4,096 tokens. gpt-oss uses medium reasoning effort and a fixed chat-template date of 2026-11-18.
- Ranking uses forced label log probabilities, chunked prefill, and a float32 output projection.
- Selection rules are deterministic given their inputs. Count-matched, presence-matched, and content-resampled controls average three seeded draws within each instance. Full-list block re-filing uses one derangement.
- Availability averages three block presentations. The paper figures and summaries use review blocks; the experiment module also supports diagnostic metadata and candidate-set conditions.
- Confidence intervals resample instances after draw averaging: 20,000 bootstrap resamples for ranking contrasts and 5,000 for availability. Holm correction uses the comparison families described in the paper. Determinism does not remove instance-sampling uncertainty.
- The Qwen placeholder ablation omits empty evidence blocks. It is included in Qwen's default inference stages.

The supplied scores reproduce the two-model comparisons and Qwen placeholder ablation. The paper's supplementary Qwen model-size, latency, reasoning-cap, and candidate-order studies use separate evaluation runs and are outside these inference runners. Model weights, generated reasoning traces, embedding caches, and cluster logs are not included. Fresh generations can vary with hardware and library versions.

See [DATA.md](DATA.md) for benchmark provenance and record formats.
