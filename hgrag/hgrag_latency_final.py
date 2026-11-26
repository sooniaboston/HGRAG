#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hgrag_latency_final.py -- 论文延迟结果的最终受控基准 (v2)。

设计要点 (相对 v1 的改进)
------------------------
1. **漂移鲁棒的交错测量**。机器不是恒定的空闲环境: 同��份代码在不同
   时段测得的绝对延迟可以差 2-3 倍。因此本脚本不再"按系统顺序各测一遍",
   而是 **按轮交错**: 每轮内依次测量所有系统, 重复 R 轮; 对每个系统取
   轮间中位数。这样任何一段全局性的负载尖峰 (例如后台进程抢占) 会同时
   影响所有系统, 而不会只惩罚排在前面测的系统。
2. **每个系统的测量顺序在轮间轮换** (round-robin 起始点), 消除
   "总是第一个测的系统吃亏/占便宜" 的位置偏差。
3. **共享 embedding 缓存与共享 ES 索引**, 所有系统面对完全一致的
   I/O 与缓存状态。
4. 报告 mean / median / p95, 以及 *每轮* 原始统计量, 供审计。

输出的两组口径
--------------
* "检索器口径" (retriever-only): 计时范围 = 完整检索操作 (query 编码 +
  三路执行 + 融合 + top-k 组装), **不含** 关键词抽取。与论文既有的
  latency_bench 完全一致, 因此可直接与论文现有表格对比。
* "部署形态" (deployed): 额外计入关键词抽取 (系统在线上真正要做的事),
  与 exp1 精度评测的口径一致。

输出: results/latency_final.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

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
OUT = r"C:\data\zsk\RAG\hgrag4\results\latency_final.json"

WARMUP = 5
REPEATS = 7
KG_PARAMS = {"decay": 0.6, "max_hops": 2, "seed_alpha": 0.9,
             "lexical_weight": 0.3, "dense_fallback": True,
             "dense_weight": 0.1, "title_bonus": 0.1, "norm_beta": 0.5,
             "seed_entities": 3}


def stats(xs: List[float]) -> Dict[str, float]:
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


def main() -> int:
    with open(SUMMARY, encoding="utf-8") as f:
        summary = json.load(f)
    params = dict(summary["best_params"])

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

    # 所有系统共用同一 embedder -> 先把 67 题向量算好, 缓存状态一致
    for q in qs:
        corpus.embedder.encode([q], normalize=True)
    print("embedding 缓存预热完成")

    # ------------------------------------------------------------------ #
    # 构造系统
    # ------------------------------------------------------------------ #
    systems: List[Tuple[str, Any, str]] = []

    r = EX.build_retriever(corpus, w_v=1.0, w_g=0.0, w_kw=0.0, eta=0.0,
                           use_guarantee=False, guarantee_mode="off")
    systems.append(("milvus_only",
                    lambda q: r.retrieve(q, top_k=10, keywords=None), ""))

    r = EX.build_retriever(corpus, w_v=0.0, w_g=1.0, w_kw=0.0, eta=0.0,
                           use_guarantee=False, guarantee_mode="off")
    systems.append(("kg_only",
                    lambda q: r.retrieve(q, top_k=10, keywords=None), ""))

    r = EX.build_retriever(corpus, w_v=0.0, w_g=0.0, w_kw=0.0, eta=0.0,
                           w_bm25=1.0, lexical="bm25", w_title=0.0,
                           use_guarantee=False, guarantee_mode="off")
    systems.append(("bm25_only",
                    lambda q: r.retrieve(q, top_k=10, keywords=None), ""))

    rf = RF.RAGFlowBaseline(corpus.passages, corpus.embedder,
                            pid_emb=corpus.pid_emb, shared_es=shared)
    systems.append(("ragflow", lambda q: rf.retrieve(q, top_k=10), ""))

    rrf = RRF.BM25DenseRRF(corpus.passages, corpus.embedder,
                           pid_emb=corpus.pid_emb, shared_es=shared)
    systems.append(("bm25_dense_rrf",
                    lambda q: rrf.retrieve(q, top_k=10), ""))

    r = EX.build_retriever(corpus, **params)
    systems.append(("hgrag_full",
                    lambda q: r.retrieve(q, top_k=10, keywords=None),
                    "原实现 (论文当前报告值)"))

    rf2 = FAST.build_fast_retriever(corpus, kg_params=KG_PARAMS, **params)
    systems.append(("hgrag_optimised",
                    lambda q: rf2.retrieve(q, top_k=10, keywords=None),
                    "低延迟实现: 与原实现逐位等价"))

    # ---- 部署形态 (含关键词抽取) ----
    kwc = _KWC(corpus)
    fkwe = FAST.FastKeywordExtractor(corpus)
    systems.append(("hgrag_full_deployed",
                    lambda q: r.retrieve(
                        q, top_k=10,
                        keywords=EX.extract_keywords_from_question(
                            {"question": q}, kwc)),
                    "原实现 + 关键词通道"))
    systems.append(("hgrag_optimised_deployed",
                    lambda q: rf2.retrieve(
                        q, top_k=10, keywords=fkwe.extract({"question": q})),
                    "低延迟实现 + 关键词通道"))

    try:
        from hgrag4_hipporag import (HippoRAGStyleRetriever,
                                     PersonalizedPageRank)
        _ppr = PersonalizedPageRank(corpus.entities, corpus.network["edges"],
                                     alpha=0.5)
        _hp = HippoRAGStyleRetriever(corpus, _ppr, seed_nodes=5, alpha=0.5)
        systems.append(("hipporag_ppr",
                        lambda q: _hp.retrieve(q, top_k=10)[0], ""))
        print("已加入 HippoRAG-style PPR 对照")
    except Exception as exc:  # noqa: BLE001
        print("跳过 HippoRAG 对照: %s" % exc)

    # ------------------------------------------------------------------ #
    # 交错测量
    # ------------------------------------------------------------------ #
    print("\n预热各系统 (%d 题)..." % WARMUP)
    for _name, fn, _note in systems:
        for q in qs[:WARMUP]:
            fn(q)

    n_sys = len(systems)
    rounds: List[List[List[float]]] = [[] for _ in systems]
    for rep in range(REPEATS):
        # 轮间轮换起始系统, 消除位置偏差
        order = [(rep + k) % n_sys for k in range(n_sys)]
        for si in order:
            _name, fn, _note = systems[si]
            xs: List[float] = []
            for q in qs:
                t0 = time.perf_counter()
                fn(q)
                xs.append(time.perf_counter() - t0)
            rounds[si].append(xs)
        print("  第 %d/%d 轮完成" % (rep + 1, REPEATS))

    results: Dict[str, Any] = {}
    for si, (name, _fn, note) in enumerate(systems):
        per_round = [stats(x) for x in rounds[si]]
        agg = {
            "mean": round(float(np.median([s["mean"] for s in per_round])), 4),
            "median": round(float(np.median(
                [s["median"] for s in per_round])), 4),
            "p95": round(float(np.median([s["p95"] for s in per_round])), 4),
            "min": round(float(np.min([s["min"] for s in per_round])), 4),
            "max": round(float(np.max([s["max"] for s in per_round])), 4),
            "std": round(float(np.median([s["std"] for s in per_round])), 4),
            "n": len(qs), "repeats": REPEATS,
            "note": note,
            "per_round": per_round,
        }
        results[name] = agg
        print("  %-26s mean=%7.2f  median=%7.2f  p95=%7.2f ms"
              % (name, agg["mean"] * 1000, agg["median"] * 1000,
                 agg["p95"] * 1000))

    rf_m = results["ragflow"]["mean"]
    hg_m = results["hgrag_full"]["mean"]
    ho_m = results["hgrag_optimised"]["mean"]
    hgd_m = results["hgrag_full_deployed"]["mean"]
    hod_m = results["hgrag_optimised_deployed"]["mean"]
    print()
    print("  [检索器口径, 无关键词通道]")
    print("  HGRAG 原实现 / RAGFlow    = %.2fx" % (hg_m / rf_m))
    print("  HGRAG 优化版 / RAGFlow    = %.2fx" % (ho_m / rf_m))
    print("  HGRAG 优化版 / HGRAG 原版 = %.2fx" % (hg_m / ho_m))
    print()
    print("  [部署形态, 含关键词通道]")
    print("  HGRAG 原实现 / RAGFlow    = %.2fx" % (hgd_m / rf_m))
    print("  HGRAG 优化版 / RAGFlow    = %.2fx" % (hod_m / rf_m))
    print("  HGRAG 优化版 / HGRAG 原版 = %.2fx" % (hgd_m / hod_m))

    out = {
        "_evaluated_on": "test",
        "_n_questions": len(qs),
        "_warmup": WARMUP,
        "_repeats": REPEATS,
        "_method": "interleaved round-robin across systems within each "
                   "round (starting system rotates per round); per-system "
                   "warm-up (%d queries); %d rounds over the identical %d "
                   "test questions; per-round statistics, then median across "
                   "rounds; shared warm embedding cache and shared "
                   "Elasticsearch index for all systems"
                   % (WARMUP, REPEATS, len(qs)),
        "_es_version": "8.11.3",
        "_backend": corpus.milvus_backend,
        "_timing_scope": {
            "retriever_only": "query encoding + route execution + fusion + "
                              "top-k assembly (identical to the paper's "
                              "existing latency_bench scope)",
            "deployed": "retriever_only + automatic keyword extraction "
                        "(identical to the exp1 accuracy-evaluation scope)",
        },
        "results": results,
        "ratios": {
            "hgrag_full_over_ragflow": round(hg_m / rf_m, 4),
            "hgrag_optimised_over_ragflow": round(ho_m / rf_m, 4),
            "hgrag_full_over_hgrag_optimised": round(hg_m / ho_m, 4),
            "deployed_hgrag_full_over_ragflow": round(hgd_m / rf_m, 4),
            "deployed_hgrag_optimised_over_ragflow": round(hod_m / rf_m, 4),
            "deployed_hgrag_full_over_hgrag_optimised": round(hgd_m / hod_m, 4),
        },
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n已写出 %s" % OUT)

    rf2.close()
    corpus.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
