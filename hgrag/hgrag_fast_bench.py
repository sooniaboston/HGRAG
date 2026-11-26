#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hgrag_fast_bench.py -- HGRAG 低延迟实现的权威评测与延迟基准。

测量口径 (与论文现有 latency_bench 完全一致, 便于直接对比):
  * 同一批 67 道测试题, 同一顺序;
  * 每个系统预热 5 题, 重复 3 轮, 每轮取统计量后再取轮间中位数;
  * 共享同一个 Elasticsearch 8.11.3 索引 (与论文相同版本);
  * 关键词抽取在检索之外单独计时 (与 exp1 一致);
  * **所有系统共用同一个 embedder, 因此 embedding 缓存状态对所有系统
    相同** —— 测之前先把 67 道题的向量全部算好并留在缓存里, 使各系统
    面对同一 (热) 缓存状态。这是论文既有基准的口径 (预热后测量)。

产出 results/fast_bench.json:
  * 优化版 vs 原版的 Recall/Precision/Mean Rank 逐题配对 + bootstrap;
  * 精确 McNemar (完整证据命中, k=10);
  * 延迟 (mean/median/p95) 与所有基线;
  * 逐项优化的消融。
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import activation_graphrag as A  # noqa: E402
import hgrag4_es as ES  # noqa: E402
import hgrag4_experiments as EX  # noqa: E402
import hgrag4_fastimpl as FAST  # noqa: E402
import hgrag4_ragflow as RF  # noqa: E402
import hgrag4_rrf as RRF  # noqa: E402
import hgrag4_splits as SPL  # noqa: E402

KG_DIR = r"C:\data\zsk\RAG\kgv4"
CACHE_DIR = r"C:\data\zsk\RAG\hf_cache"
ES_URL = os.environ.get("ES_URL", "http://127.0.0.1:9201")
SUMMARY = r"C:\data\zsk\RAG\hgrag4\results\summary.json"
OUT = r"C:\data\zsk\RAG\hgrag4\results\fast_bench.json"

WARMUP = 5
REPEATS = 3
SEED = 20240917

KG_PARAMS = {"decay": 0.6, "max_hops": 2, "seed_alpha": 0.9,
             "lexical_weight": 0.3, "dense_fallback": True,
             "dense_weight": 0.1, "title_bonus": 0.1, "norm_beta": 0.5,
             "seed_entities": 3}


def paired_bootstrap(a: Sequence[float], b: Sequence[float],
                      B: int = 10000, seed: int = SEED
                      ) -> Dict[str, Any]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    obs = abs(float(np.mean(d)))
    rs = np.random.RandomState(seed)
    idx = rs.randint(0, len(d), size=(B, len(d)))
    boots = d[idx].mean(axis=1)
    p = float((np.sum(np.abs(boots) >= obs) + 1) / (B + 1))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"delta": float(np.mean(d)), "ci95": [float(lo), float(hi)],
            "p": p, "n": int(len(d))}


def mcnemar(hits_a: Sequence[bool], hits_b: Sequence[bool]
            ) -> Dict[str, Any]:
    from math import comb
    b = sum(1 for x, y in zip(hits_a, hits_b) if x and not y)
    c = sum(1 for x, y in zip(hits_a, hits_b) if not x and y)
    n = b + c
    if n == 0:
        p = 1.0
    else:
        tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / (2.0 ** n)
        p = min(1.0, 2.0 * tail)
    return {"b": int(b), "c": int(c), "p": float(p)}


def stats(xs: Sequence[float]) -> Dict[str, float]:
    return {"mean": round(float(np.mean(xs)), 4),
            "median": round(float(np.median(xs)), 4),
            "p95": round(float(np.percentile(xs, 95)), 4),
            "min": round(float(np.min(xs)), 4),
            "max": round(float(np.max(xs)), 4),
            "std": round(float(np.std(xs)), 4),
            "n": len(xs)}


class _KWC:
    """关键词抽取只需要 titles / entities。"""

    def __init__(self, corpus):
        self.titles = corpus.titles
        self.entities = corpus.entities


def evaluate(r: Any, test_q: List[Dict[str, Any]], corpus: Any,
             kw_fn) -> Dict[str, Any]:
    recalls = {k: [] for k in (1, 3, 5, 10)}
    precs = {k: [] for k in (1, 3, 5, 10)}
    ranks: List[float] = []
    hit10: List[bool] = []

    for q in test_q:
        kws = kw_fn(q["question"])
        res, _ = r.retrieve(q["question"], top_k=10, keywords=kws)
        titles = [corpus.pid_to_title.get(x["pid"], "") for x in res]
        gold = set(q["gold_titles"])
        ng = max(len(gold), 1)
        for k in recalls:
            hit = len(gold & set(titles[:k]))
            recalls[k].append(hit / ng)
            precs[k].append(hit / max(k, 1))
        gr = [i for i, t in enumerate(titles[:10]) if t in gold]
        ranks.append((sum(gr) / len(gr) + 1) if gr else float("nan"))
        hit10.append(len(gold & set(titles[:10])) == len(gold))

    return {
        "recall": {str(k): float(np.mean(v)) for k, v in recalls.items()},
        "precision": {str(k): float(np.mean(v)) for k, v in precs.items()},
        "mean_rank": float(np.nanmean(ranks)),
        "_recalls": recalls, "_precs": precs, "_hit10": hit10,
    }


def bench(fn, warm_n: int = WARMUP, repeats: int = REPEATS
          ) -> Dict[str, Any]:
    for q in qs[:warm_n]:
        fn(q)
    rounds: List[List[float]] = []
    for _ in range(repeats):
        xs: List[float] = []
        for q in qs:
            t0 = time.perf_counter()
            fn(q)
            xs.append(time.perf_counter() - t0)
        rounds.append(xs)
    per_round = [stats(x) for x in rounds]
    return {
        "mean": round(float(np.median([s["mean"] for s in per_round])), 4),
        "median": round(float(np.median([s["median"] for s in per_round])), 4),
        "p95": round(float(np.median([s["p95"] for s in per_round])), 4),
        "min": round(float(np.min([s["min"] for s in per_round])), 4),
        "max": round(float(np.max([s["max"] for s in per_round])), 4),
        "std": round(float(np.median([s["std"] for s in per_round])), 4),
        "n": len(qs), "repeats": repeats,
        "per_round": per_round,
    }


def main() -> int:
    global qs
    with open(SUMMARY, encoding="utf-8") as f:
        params = dict(json.load(f)["best_params"])

    corpus = EX.Corpus(KG_DIR, A.DEFAULT_EMBED_MODEL, CACHE_DIR, "exact")
    split = SPL.load_or_make_split(corpus.questions, KG_DIR,
                                   dev_fraction=SPL.DEV_FRACTION,
                                   seed=SPL.SPLIT_SEED, force=False)
    test_q = SPL.select(corpus.questions, split, "test")
    qs = [q["question"] for q in test_q]
    print("测试题 %d" % len(test_q))

    shared = ES.SharedESIndex(ES_URL, corpus.passages)
    shared.build()
    corpus.shared_es = shared
    print("ES 索引 %s (%d chunks)" % (shared.index_name, len(shared.chunks)))

    # ---- 让所有系统面对同一 (热) embedding 缓存状态 ----
    for q in qs:
        corpus.embedder.encode([q], normalize=True)
    print("embedding 缓存: %d 条 (67 题全部预热)"
          % len(corpus.embedder._mem))

    slow = EX.build_retriever(corpus, kg_params=KG_PARAMS, **params)
    fast = FAST.build_fast_retriever(corpus, kg_params=KG_PARAMS, **params)
    fkwe = FAST.FastKeywordExtractor(corpus)
    kwc = _KWC(corpus)

    kw_slow = lambda q: EX.extract_keywords_from_question(
        {"question": q}, kwc)
    kw_fast = lambda q: fkwe.extract({"question": q})

    # ---- 关键词抽取等价性 ----
    n_same_kw = sum(1 for q in test_q
                     if kw_slow(q["question"]) == kw_fast(q["question"]))
    print("关键词抽取逐条一致: %d/%d" % (n_same_kw, len(test_q)))

    # ---- 逐条检索结果等价性 ----
    n_same_list = 0
    for q in test_q:
        a, _ = slow.retrieve(q["question"], top_k=10,
                             keywords=kw_slow(q["question"]))
        b, _ = fast.retrieve(q["question"], top_k=10,
                             keywords=kw_fast(q["question"]))
        if [x["pid"] for x in a] == [x["pid"] for x in b] and \
                all(x["score"] == y["score"] for x, y in zip(a, b)):
            n_same_list += 1
    print("检索结果逐条逐位一致: %d/%d" % (n_same_list, len(test_q)))

    # ---- 精度 ----
    print("\n=== 精度 (优化版 vs 原实现, 配对检验) ===")
    ev_s = evaluate(slow, test_q, corpus, kw_slow)
    ev_f = evaluate(fast, test_q, corpus, kw_fast)
    print("  %-10s %10s %10s %24s %9s"
          % ("metric", "原实现", "优化版", "Δ (95% CI)", "p"))
    sig: Dict[str, Any] = {}
    for kind, key in (("R", "recall"), ("P", "precision")):
        for k in (1, 3, 5, 10):
            a = ev_f["_recalls"][k] if kind == "R" else ev_f["_precs"][k]
            b = ev_s["_recalls"][k] if kind == "R" else ev_s["_precs"][k]
            bs = paired_bootstrap(a, b)
            sig["%s@%d" % (kind.lower(), k)] = bs
            print("  %s@%-8d %10.4f %10.4f  [%+.4f, %+.4f] %8.4f"
                  % (kind, k, ev_s[key][str(k)], ev_f[key][str(k)],
                     bs["ci95"][0], bs["ci95"][1], bs["p"]))
    print("  Mean Rank  %10.4f %10.4f" % (ev_s["mean_rank"], ev_f["mean_rank"]))
    mc = mcnemar(ev_f["_hit10"], ev_s["_hit10"])
    print("  完整证据命中 (k=10) 精确 McNemar: b=%d c=%d p=%.4f"
          % (mc["b"], mc["c"], mc["p"]))

    # ---- 延迟 ----
    print("\n=== 延迟 (67 题, 预热 %d, %d 轮取中位数) ===" % (WARMUP, REPEATS))
    lat_s = bench(lambda q: slow.retrieve(q, top_k=10, keywords=kw_slow(q)))
    lat_f = bench(lambda q: fast.retrieve(q, top_k=10, keywords=kw_fast(q)))
    print("  HGRAG 原实现  mean=%7.2f ms  median=%7.2f ms  p95=%7.2f ms"
          % (lat_s["mean"] * 1000, lat_s["median"] * 1000, lat_s["p95"] * 1000))
    print("  HGRAG 优化版  mean=%7.2f ms  median=%7.2f ms  p95=%7.2f ms"
          % (lat_f["mean"] * 1000, lat_f["median"] * 1000, lat_f["p95"] * 1000))
    print("  加速比        %.2fx" % (lat_s["mean"] / lat_f["mean"]))

    kws_lat_s = bench(lambda q: kw_slow(q))
    kws_lat_f = bench(lambda q: kw_fast(q))
    print("  其中关键词抽取: 原 %.2f ms -> 优化 %.2f ms"
          % (kws_lat_s["mean"] * 1000, kws_lat_f["mean"] * 1000))

    # ---- 基线 (同一时段、同一缓存状态重测) ----
    print("\n=== 基线 (同一时段重测) ===")
    baselines: Dict[str, Any] = {}
    rf = RF.RAGFlowBaseline(corpus.passages, corpus.embedder,
                            pid_emb=corpus.pid_emb, shared_es=shared)
    rrf = RRF.BM25DenseRRF(corpus.passages, corpus.embedder,
                           pid_emb=corpus.pid_emb, shared_es=shared)
    for name, fn in (("ragflow", lambda q: rf.retrieve(q, top_k=10)),
                     ("bm25_dense_rrf", lambda q: rrf.retrieve(q, top_k=10))):
        lv = bench(fn)
        baselines[name] = lv
        print("  %-16s mean=%7.2f ms  median=%7.2f ms  p95=%7.2f ms"
              % (name, lv["mean"] * 1000, lv["median"] * 1000,
                 lv["p95"] * 1000))

    rf_lat = baselines["ragflow"]["mean"]
    print()
    print("  >>> 相对 RAGFlow (%.2f ms): 原实现 %.2fx, 优化版 %.2fx"
          % (rf_lat * 1000, lat_s["mean"] / rf_lat, lat_f["mean"] / rf_lat))

    # ---- 消融 ----
    print("\n=== 优化项消融 ===")
    abl: Dict[str, Any] = {}
    variants = [
        ("full (all optimisations)", dict(parallel=True)),
        ("no parallel dispatch", dict(parallel=False)),
    ]
    for vname, vkw in variants:
        rv = FAST.build_fast_retriever(corpus, kg_params=KG_PARAMS,
                                       parallel=vkw["parallel"], **params)
        lv = bench(lambda q: rv.retrieve(q, top_k=10, keywords=kw_fast(q)))
        abl[vname] = lv
        print("  %-28s mean=%7.2f ms" % (vname, lv["mean"] * 1000))
        rv.close()

    out = {
        "_note": "HGRAG low-latency variant: bit-identical retrieval "
                 "results, same 67-question test split, same ES 8.11.3 index",
        "_es_version": "8.11.3",
        "_n_questions": len(test_q),
        "_method": "per-system warm-up (5), %d repeats over the identical 67 "
                   "test questions; per-round statistics then median across "
                   "rounds; shared warm embedding cache for all systems"
                   % REPEATS,
        "equivalence": {
            "keyword_extraction_identical": n_same_kw,
            "retrieval_results_identical": n_same_list,
            "n_questions": len(test_q),
        },
        "metrics_baseline": {k: v for k, v in ev_s.items()
                             if not k.startswith("_")},
        "metrics_fast": {k: v for k, v in ev_f.items()
                         if not k.startswith("_")},
        "significance": sig,
        "mcnemar_complete_hit_k10": mc,
        "latency_baseline": lat_s,
        "latency_fast": lat_f,
        "latency_keyword_extraction": {"baseline": kws_lat_s, "fast": kws_lat_f},
        "latency_baselines_remeasured": baselines,
        "ablation": abl,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n已写出 %s" % OUT)

    fast.close()
    corpus.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
