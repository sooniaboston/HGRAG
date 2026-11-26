#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HGRAG v4 :: 专用延迟基准 (hgrag4_latency_bench.py)
===================================================

为什么需要单独的延迟基准
------------------------
exp1 里各系统的延迟是在 *跑整条实验流水线* 的过程中顺手记的, 那时
机器上还有调参扫描 / 消融等其它负载, 记下的绝对值受当时负载影响很大
(同一份代码两次运行可以差 5 倍)。而论文报告延迟的目的, 是让读者知道
"这个系统比那个系统贵多少", 因此必须满足两个条件:

  1. 所有被比较的系统在 *同一段安静时段* 内测量;
  2. 测量前有预热 (warm-up), 避免把首次 embedding / ES 查询的冷启动
     成本算进稳态延迟;
  3. 每个系统测 *同样的 67 道测试题*, 顺序相同。

本脚本按上述三条执行, 依次测量:
    Milvus-only / Graph-only / BM25-only / RAGFlow / BM25+Dense RRF / HGRAG

输出:
    <out-dir>/results/latency_bench.json
并把各系统的 mean/median/p95 写回 summary.json 的 exp1 (作为论文报告
的权威延迟值), 同时在 summary.json 里记录 latency_bench 审计块。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import hgrag4_es as ES  # noqa: E402
import hgrag4_ragflow as RF  # noqa: E402
import hgrag4_rrf as RRF  # noqa: E402
import hgrag4_splits as SPL  # noqa: E402
from hgrag4_experiments import Corpus, build_retriever  # noqa: E402
from hgrag4_fastimpl import build_fast_retriever  # noqa: E402

LOG = logging.getLogger("hgrag4.lat")

#: 预热查询数 (不计入统计)
WARMUP = 5


def _stats(xs: List[float]) -> Dict[str, float]:
    return {
        "mean": round(float(np.mean(xs)), 4),
        "median": round(float(np.median(xs)), 4),
        "p95": round(float(np.percentile(xs, 95)), 4),
        "min": round(float(np.min(xs)), 4),
        "max": round(float(np.max(xs)), 4),
        "std": round(float(np.std(xs)), 4),
        "n": len(xs),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="HGRAG v4 专用延迟基准")
    p.add_argument("--kg-dir", default=r"C:\data\zsk\RAG\kgv4")
    p.add_argument("--out-dir", default=r"C:\data\zsk\RAG\hgrag4")
    p.add_argument("--embed-model", default=None)
    p.add_argument("--cache-dir", default=r"C:\data\zsk\RAG\hf_cache")
    p.add_argument("--milvus-backend", default="exact",
                   choices=["milvus-lite", "exact"])
    p.add_argument("--es-url", default="http://127.0.0.1:9200")
    p.add_argument("--repeats", type=int, default=3,
                   help="每个系统重复测量的轮数, 取所有轮的中位数")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    import activation_graphrag as A
    embed_model = args.embed_model or A.DEFAULT_EMBED_MODEL

    res_dir = os.path.join(args.out_dir, "results")
    os.makedirs(res_dir, exist_ok=True)

    corpus = Corpus(args.kg_dir, embed_model, args.cache_dir,
                    args.milvus_backend)

    # ---- 划分 (与正式实验完全一致) ----
    split = SPL.load_or_make_split(corpus.questions, args.kg_dir,
                                   dev_fraction=SPL.DEV_FRACTION,
                                   seed=SPL.SPLIT_SEED, force=False)
    test_q = SPL.select(corpus.questions, split, "test")
    LOG.info("延迟基准: test=%d 题", len(test_q))

    # ---- 共享 ES 索引 ----
    shared = ES.SharedESIndex(args.es_url, corpus.passages)
    shared.build()
    corpus.shared_es = shared
    LOG.info("共享 ES 索引: %s (%d chunks)", shared.index_name,
             len(shared.chunks))

    # ---- 读入最优超参 (dev 上选出) ----
    summary_path = os.path.join(res_dir, "summary.json")
    with open(summary_path, encoding="utf-8") as fh:
        summary = json.load(fh)
    params = dict(summary["best_params"])
    LOG.info("超参: %s", params)

    # 图谱路超参 (与 H3.KGPathRetriever 默认值不同, 单独指定)
    KG_PARAMS = {"decay": 0.6, "max_hops": 2, "seed_alpha": 0.9,
                 "lexical_weight": 0.3, "dense_fallback": True,
                 "dense_weight": 0.1, "title_bonus": 0.1, "norm_beta": 0.5,
                 "seed_entities": 3}

    # ---- 构造 6 个系统 ----
    systems: List[Dict[str, Any]] = []

    def add(name: str, fn) -> None:
        systems.append({"name": name, "fn": fn})

    r = build_retriever(corpus, w_v=1.0, w_g=0.0, w_kw=0.0, eta=0.0,
                        use_guarantee=False, guarantee_mode="off")
    add("milvus_only", lambda q: r.retrieve(q, top_k=10, keywords=None))

    r = build_retriever(corpus, w_v=0.0, w_g=1.0, w_kw=0.0, eta=0.0,
                        use_guarantee=False, guarantee_mode="off")
    add("kg_only", lambda q: r.retrieve(q, top_k=10, keywords=None))

    r = build_retriever(corpus, w_v=0.0, w_g=0.0, w_kw=0.0, eta=0.0,
                        w_bm25=1.0, lexical="bm25", w_title=0.0,
                        use_guarantee=False, guarantee_mode="off")
    add("bm25_only", lambda q: r.retrieve(q, top_k=10, keywords=None))

    rf = RF.RAGFlowBaseline(corpus.passages, corpus.embedder,
                            pid_emb=corpus.pid_emb, shared_es=shared)
    add("ragflow", lambda q: rf.retrieve(q, top_k=10))

    rrf = RRF.BM25DenseRRF(corpus.passages, corpus.embedder,
                           pid_emb=corpus.pid_emb, shared_es=shared)
    add("bm25_dense_rrf", lambda q: rrf.retrieve(q, top_k=10))

    r = build_retriever(corpus, **params)
    add("hgrag_full", lambda q: r.retrieve(q, top_k=10, keywords=None))

    # 低延迟实现: 与 hgrag_full 检索结果逐位相同, 仅实现层开销不同。
    rf2 = build_fast_retriever(corpus, kg_params=KG_PARAMS, **params)
    add("hgrag_optimised",
        lambda q: rf2.retrieve(q, top_k=10, keywords=None))

    # ---- HippoRAG 风格 PPR 对照 (审稿人 #2 concern #2) ----
    # 在同一段安静时段、同一批 67 道测试题上测量, 与上面 6 个系统完全同口径。
    try:
        from hgrag4_hipporag import (HippoRAGStyleRetriever,
                                     PersonalizedPageRank)
        _ppr = PersonalizedPageRank(corpus.entities,
                                     corpus.network["edges"], alpha=0.5)
        _hp = HippoRAGStyleRetriever(corpus, _ppr, seed_nodes=5, alpha=0.5)
        add("hipporag_ppr",
            lambda q: _hp.retrieve(q, top_k=10)[0])
        LOG.info("已加入 HippoRAG-style PPR 对照基线")
    except Exception as exc:  # noqa: BLE001
        LOG.warning("跳过 HippoRAG-style PPR 对照: %s", exc)

    # ---- 测量 ----
    results: Dict[str, Any] = {}
    for spec in systems:
        name, fn = spec["name"], spec["fn"]
        # warm-up
        for q in test_q[:WARMUP]:
            fn(q["question"])
        rounds: List[List[float]] = []
        for rep in range(max(1, args.repeats)):
            xs: List[float] = []
            for q in test_q:
                t0 = time.perf_counter()
                fn(q["question"])
                xs.append(time.perf_counter() - t0)
            rounds.append(xs)
        # 每轮取中位数, 再在轮之间取中位数 -> 对偶发抖动稳健
        per_round = [_stats(xs) for xs in rounds]
        agg = {
            "mean": round(float(np.median([s["mean"] for s in per_round])), 4),
            "median": round(float(np.median(
                [s["median"] for s in per_round])), 4),
            "p95": round(float(np.median([s["p95"] for s in per_round])), 4),
            "min": round(float(np.min([s["min"] for s in per_round])), 4),
            "max": round(float(np.max([s["max"] for s in per_round])), 4),
            "std": round(float(np.median([s["std"] for s in per_round])), 4),
            "n": len(test_q),
            "repeats": len(rounds),
            "per_round": per_round,
        }
        results[name] = agg
        LOG.info("%-16s mean=%.4f median=%.4f p95=%.4f", name,
                 agg["mean"], agg["median"], agg["p95"])

    # ---- 写回 summary.json ----
    e1 = summary.get("exp1") or {}
    for name, agg in results.items():
        d = e1.get(name)
        if not d:
            continue
        d["mean_latency"] = agg["mean"]
        d["median_latency"] = agg["median"]
        d["p95_latency"] = agg["p95"]
        d["latency_source"] = "latency_bench (controlled, warmed-up, " \
                              "median of %d repeats)" % agg["repeats"]
    summary["latency_bench"] = {
        "_evaluated_on": "test",
        "n_questions": len(test_q),
        "warmup": WARMUP,
        "repeats": max(1, args.repeats),
        "method": "per-system warm-up, then repeated over the identical "
                  "67 test questions; per-round statistics, then median "
                  "across rounds",
        "es_version": "8.11.3",
        "backend": corpus.milvus_backend,
        "results": results,
    }

    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    LOG.info("已写回: %s", summary_path)

    with open(os.path.join(res_dir, "latency_bench.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary["latency_bench"], fh, ensure_ascii=False, indent=1)

    corpus.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
