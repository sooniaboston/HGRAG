# HGRAG — Association by Design

**Association by Design: A Brain-Inspired Hybrid Retrieval Architecture for
Multi-Hop Retrieval-Augmented Generation**

Zhao Shukuo — Department of Precision Instruments, Tsinghua University, Beijing, China
Corresponding author: sofiazhao@foxmail.com

This repository is the reference implementation and released data for the HGRAG
manuscript. It contains everything needed to (i) run the three-route retriever,
(ii) recompute every significance test reported in the paper from released
per-question result vectors, and (iii) audit the artifacts behind the published
numbers.

---

## What HGRAG is

Flat RAG scores each passage in isolation through a single similarity signal.
Multi-hop questions need an **evidence set** whose members may not resemble the
question at all. HGRAG treats this as a failure of *association* rather than of
ranking, and fixes it with three routes that fail in different ways and are
therefore integrated rather than concatenated:

| Route | Mechanism | Cognitive principle |
|---|---|---|
| **Graph** | activation spreads over an entity–relation graph (≤ 2 hops, decay ρ = 0.6, firing threshold, IDF-style specificity discounting) | bounded spread, convergent accumulation, discounting of generic concepts |
| **Dense** | exact inner-product search over frozen `paraphrase-multilingual-MiniLM-L12-v2` embeddings | semantic proximity |
| **Lexical** | BM25 over the shared Elasticsearch 8.11.3 index | surface-form matching |

A **scale-fair fusion operator** normalizes each channel on its own score
distribution before combining them (rank normalization for dense and graph,
max normalization for BM25), so no route can dominate the aggregate by
construction.

---

## Headline results

Frozen, distractor-controlled HotpotQA benchmark: 1,172 passages, 111 questions,
seeded 44/67 development/test split (seed `20240917`). All systems share one
index and one embedding model.

| System | Recall@1 | Recall@10 | Mean latency |
|---|---|---|---|
| **HGRAG** | 41.79% | **96.27%** | 21.8 ms (`HGRAG-Fast`) |
| HippoRAG-style PPR control | 32.09% | 88.81% | 21.5 ms |
| BM25 + dense RRF | 39.55% | 88.06% | — |
| RAGFlow (production hybrid) | 43.28% | 87.31% | 12.6 ms |
| BM25-only | 44.03% | 85.07% | — |
| Dense-only | 34.33% | 76.12% | — |

HGRAG vs. RAGFlow at Recall@10: **+8.96 pp**, 95% CI [+4.48, +14.18], `p = 0.0010`
(paired bootstrap, 10,000 resamples). Every ablation is an exact McNemar test.

The latency figure above belongs to **HGRAG-Fast**, the released low-latency
implementation of the retrieval paths. It returns a *bit-identical* top-10 list on
every test question as the reference implementation, which costs 55.8 ms.

---

## Repository layout

```
hgrag/                      core implementation
  activation_graphrag.py      entity–relation extraction, canonicalisation,
                              activation network, three-route retriever
  hgrag4_experiments.py       corpus loading, retriever construction, statistics
  hgrag4_fusion.py            scale-fair fusion operator + design variants
  hgrag4_rrf.py               BM25 + dense reciprocal-rank-fusion baseline
  hgrag4_hipporag.py          HippoRAG-style personalised-PageRank control
  hgrag4_fastimpl.py          released low-latency implementation of the paths
  hgrag4_splits.py            seeded dev/test partition
  hgrag4_significance_sweeps.py  paired bootstrap + exact McNemar machinery
  hgrag4_latency_bench*.py    controlled latency benchmark
  hgrag4_build_corpus.py      corpus construction from the public release
  hgrag4_build_graph.py       graph construction (requires an LLM endpoint)
  hgrag4_export_sample.py     exports the redistributable sample
  hgrag4_verify_sample.py     *self-check*: the sample runs and recomputes
  hgrag4_verify_paper.py      *self-check*: every number in the .tex has a source
  hgrag4_proxy.py             OpenAI-compatible deployment proxy
sample/hgrag4_sample/        redistributable benchmark sample (see below)
```

---

## Quick start

```bash
pip install -r requirements.txt
```

### 1. Run the retriever on the released sample

```bash
cd hgrag
python hgrag4_run.py --kg-dir ../sample/hgrag4_sample/kg --out-dir ./out --stages 1
```

### 2. Verify the sample and recompute the paper's statistics

```bash
python hgrag4_verify_sample.py --sample ../sample/hgrag4_sample
```

This performs four checks: SHA-256 integrity of every shipped file; graph
connectivity (the activation network must have edges, otherwise spreading
activation cannot be reproduced); a live retrieval run; and an **independent
recomputation** of the paper's headline paired-bootstrap test from the shipped
per-question vectors, compared against the released numbers digit by digit.

Expected output (tail):

```
[4/4] 用样本自带的逐题结果向量独立复算论文的显著性检验 ...
      复算 HGRAG vs RAGFlow @ Recall@10: Δ = +8.96 pp, 95% CI [+4.48, +14.18], p = 0.0010
      released:            Δ = +8.96 pp, 95% CI [+4.48, +14.18], p = 0.0010
      [OK] 独立复算与 released 结果 *逐位* 一致
```

### 3. Audit every number in the manuscript

```bash
python hgrag4_verify_paper.py \
    --results ../sample/hgrag4_sample/results/summary.json \
    --tex path/to/hgrag_neurocomputing.tex
```

This scans the manuscript for every reported percentage, percentage-point delta
and latency figure and checks each one against the released results, failing
loudly on any number that has no experimental source.

---

## The released sample

`sample/hgrag4_sample/` contains the artifacts behind the paper:

| Path | Contents |
|---|---|
| `kg/passages.json` | 200-passage corpus slice (`title`, `text`) |
| `kg/questions.json` | benchmark questions with their original human annotations |
| `kg/splits.json` | the seeded 44/67 dev/test partition (seed `20240917`) |
| `kg/entities.json` | extracted entities |
| `kg/relations.json` | relation triples |
| `kg/activation_network.json` | materialized activation network **with edge weights** |
| `kg/raw_extractions.jsonl` | raw LLM extraction records, so the graph can be rebuilt without any API call |
| `results/*.json` | per-question result vectors and summary statistics for every experiment |
| `MANIFEST.json` | file list with SHA-256 digests |

**The sample is self-contained for verification.** Even though it contains only a
200-passage / 19-question slice of the corpus, it ships the *complete* split file
and the *complete* per-question result vectors for all 67 test questions, so a
reader who downloads only this directory can recompute every `p`-value and
confidence interval reported in the paper.

---

## Reproducing the full pipeline

The full corpus (1,172 passages / 111 questions) is the **HotpotQA distractor
split**, taken from its official public release. The entity–relation graph was
produced with a hosted LLM endpoint and is shipped as a *frozen artifact*; it is
therefore not bit-reproducible across different extraction models, but every
retrieval number in the paper is computed against the frozen graph shipped here.

```bash
# 1. corpus from the official HotpotQA release
python hgrag4_prepare_corpus.py --out-dir ./kg

# 2. graph extraction (needs an LLM endpoint; the raw records are resumable)
python hgrag4_build_graph.py --kg-dir ./kg --out-dir ./kg \
    --deepseek-api-key "$DEEPSEEK_API_KEY"

# 3. evaluation
python hgrag4_run.py --kg-dir ./kg --out-dir ./out
```

---

## Provenance and license

The corpus and questions are the HotpotQA *distractor* split, taken from its
official public release. Human annotations (`gold_titles`, `answer`, `type`,
`level`) are used verbatim; **no question was generated by an LLM and no synthetic
distractor was introduced**. The entity/relation extraction was produced by a
hosted LLM endpoint; the raw records are included so that the graph can be
reconstructed without any further API call.

HotpotQA passages and questions are **CC BY-SA 4.0** (Wikipedia text), taken
from the official public release. The derived artifacts and the code in this
repository are released under the same license as the paper. See `LICENSE`.

## Citation

```bibtex
@article{zhao2026association,
  title   = {Association by Design: A Brain-Inspired Hybrid Retrieval
             Architecture for Multi-Hop Retrieval-Augmented Generation},
  author  = {Zhao, Shukuo},
  journal = {Neurocomputing},
  year    = {2026}
}
```

## Declarations

See the manuscript for competing interests, funding, CRediT author
contributions, and the data-availability statement.
