#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HGRAG v4 实验主入口
====================

用法示例::

    python hgrag4_run.py \
        --kg-dir C:\\data\\zsk\\RAG\\kgv4 \
        --out-dir C:\\data\\zsk\\RAG\\hgrag4 \
        --use-elasticsearch

流程:
  1. 载入 kgv4 语料 + 确定性 dev/test 划分
  2. exp6 在 DEV 上调参 -> 最优超参
  3. exp1/2/3/4/5/7 在 TEST 上报告 (含 RAGFlow 基线)
  4. 写 summary.json (每个实验带 evaluated_on 标记)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import activation_graphrag as A  # noqa: E402
import hgrag4_splits as SPL  # noqa: E402
from hgrag4_experiments import Corpus  # noqa: E402
from hgrag4_experiments_part2 import (exp1_baselines, exp2_dual_path,  # noqa: E402
                                       exp3_bottomup, exp4_weights,
                                       exp5_keyword, exp6_tuning,
                                       exp7_latency)
import hgrag4_ragflow as RF  # noqa: E402
import hgrag4_rrf as RRF  # noqa: E402
import hgrag4_es as ES  # noqa: E402

LOG = logging.getLogger("hgrag4.run")

#: v4 的最终超参 —— 与 exp6 在 DEV (44 题) 上做坐标下降 + 5-fold CV + 1-SE
#: 规则选出的最优配置完全一致 (见 results/summary.json -> exp6.best_params)。
#: 这里固化下来, 使得只跑 stage 1 也能复现正式结果, 而不必重跑调参。
DEFAULT_PARAMS = {
    "w_v": 0.4, "w_g": 0.5, "w_kw": 0.2, "eta": 0.0,
    "guarantee_k": 0, "use_guarantee": False, "guarantee_mode": "off",
    "kw_pos": 0.35, "kw_count": 0.35, "kw_assoc": 0.30,
    "norm_mode": "rank", "kw_scope": "cand", "design": "sum",
    "w_title": 0.0, "w_bm25": 0.7, "lexical": "bm25", "lex_norm": "max",
}


def build_ragflow(corpus: Corpus, use_es: bool, es_url: str,
                  use_kg: bool = False) -> Optional[RF.RAGFlowBaseline]:
    """
    构造 RAGFlow 基线。

    若语料对象上挂了共享 ES 索引 (corpus.shared_es), 则基线查询 *同一个*
    索引 —— 这样 HGRAG 与 RAGFlow 的词法能力严格相同, 差值干净地归因到
    "图谱路 + 多路融合"。
    """
    shared = getattr(corpus, "shared_es", None)
    if shared is not None:
        LOG.info("RAGFlow 基线: 使用共享 Elasticsearch 索引 @ %s "
                 "(与 HGRAG 同索引)", shared.es_url)
        return RF.RAGFlowBaseline(corpus.passages, corpus.embedder,
                                  pid_emb=corpus.pid_emb,
                                  shared_es=shared,
                                  use_knowledge_graph=use_kg)
    if not use_es:
        LOG.info("RAGFlow 基线: 使用本地 BM25 实现")
        return RF.RAGFlowBaseline(corpus.passages, corpus.embedder,
                                  pid_emb=corpus.pid_emb)
    LOG.info("RAGFlow 基线: 使用真实 Elasticsearch (BM25) @ %s", es_url)
    return RF.RAGFlowBaseline(corpus.passages, corpus.embedder,
                              pid_emb=corpus.pid_emb, es_url=es_url,
                              use_knowledge_graph=use_kg)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="HGRAG v4 实验")
    p.add_argument("--kg-dir", default=r"C:\data\zsk\RAG\kgv4")
    p.add_argument("--out-dir", default=r"C:\data\zsk\RAG\hgrag4")
    p.add_argument("--embed-model", default=A.DEFAULT_EMBED_MODEL)
    p.add_argument("--cache-dir", default=r"C:\data\zsk\RAG\hf_cache")
    p.add_argument("--milvus-backend", default="exact",
                   choices=["milvus-lite", "exact"])
    p.add_argument("--stages", default="1,2,3,4,5,6,7")
    p.add_argument("--dev-fraction", type=float, default=SPL.DEV_FRACTION)
    p.add_argument("--split-seed", type=int, default=SPL.SPLIT_SEED)
    p.add_argument("--force-split", action="store_true")
    p.add_argument("--use-elasticsearch", action="store_true",
                   help="用真实 Elasticsearch 作为 RAGFlow 全文后端")
    p.add_argument("--es-url", default="http://127.0.0.1:9200")
    p.add_argument("--ragflow-kg", action="store_true",
                   help="开启 RAGFlow 的知识图谱检索块")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    res_dir = os.path.join(args.out_dir, "results")
    os.makedirs(res_dir, exist_ok=True)

    # ---------- 载入语料 ----------
    corpus = Corpus(args.kg_dir, args.embed_model, args.cache_dir,
                    args.milvus_backend)

    # ---------- 划分 ----------
    split = SPL.load_or_make_split(
        corpus.questions, args.kg_dir,
        dev_fraction=args.dev_fraction, seed=args.split_seed,
        force=args.force_split)
    dev_q = SPL.select(corpus.questions, split, "dev")
    test_q = SPL.select(corpus.questions, split, "test")
    audit = SPL.audit(split)
    LOG.info("划分: dev=%d test=%d (seed=%d)", len(dev_q), len(test_q),
             args.split_seed)
    for name in ("dev", "test"):
        LOG.info("  %s type分布: %s", name, audit[name]["type_pct"])

    stages = {int(s) for s in args.stages.split(",") if s.strip()}
    summary: Dict[str, Any] = {
        "kg_dir": args.kg_dir,
        "milvus_backend": args.milvus_backend,
        "embed_model": args.embed_model,
        "n_questions": len(corpus.questions),
        "n_passages": len(corpus.passages),
        "split": {
            "seed": args.split_seed,
            "dev_fraction": args.dev_fraction,
            "n_dev": len(dev_q),
            "n_test": len(test_q),
            "split_file": os.path.join(args.kg_dir, "splits.json"),
            "audit": audit,
        },
    }

    # ---------- RAGFlow 基线 ----------
    ragflow = None
    shared_es = None
    # 阶段 8/9 也要用同一个 ES 索引: 图谱成分消融的 "完整 HGRAG" 参照行
    # 必须与论文主表 (Table 1) 的 HGRAG 行口径一致, 否则消融的差值里会混入
    # "本地 BM25 vs Elasticsearch" 的实现差异。
    if args.use_elasticsearch and (stages & {1, 6, 7, 8, 9}):
        # 建立 HGRAG 与 RAGFlow 共用的 ES 全文索引
        try:
            shared_es = ES.SharedESIndex(args.es_url, corpus.passages)
            shared_es.build()
            corpus.shared_es = shared_es
            LOG.info("共享 Elasticsearch 索引已建立: %s (%d chunks)",
                     shared_es.index_name, len(shared_es.chunks))
        except Exception as exc:  # pragma: no cover
            LOG.warning("共享 ES 索引建立失败, 回退到独立索引: %s", exc)
            shared_es = None
    if 1 in stages or 7 in stages:
        ragflow = build_ragflow(corpus, args.use_elasticsearch, args.es_url,
                                args.ragflow_kg)

    # ---------- BM25 + Dense RRF 基线 (审稿人要求增加的参照系) ----------
    # 与 RAGFlow / HGRAG 共用同一 ES 索引与同一 embedding 模型,
    # 只有融合算子不同 (RRF 名次倒数加权 vs 分数线性加权)。
    rrf = None
    if 1 in stages:
        rrf = RRF.BM25DenseRRF(corpus.passages, corpus.embedder,
                               pid_emb=corpus.pid_emb,
                               shared_es=shared_es,
                               es_url=args.es_url if args.use_elasticsearch
                               else None)
        LOG.info("BM25+Dense RRF 基线: K=%.0f, w_bm25=%.1f, w_vec=%.1f, "
                 "共享ES=%s", rrf.rrf_k, rrf.w_bm25, rrf.w_vec,
                 shared_es is not None)

    # ---------- exp6: 在 DEV 上调参 ----------
    params = dict(DEFAULT_PARAMS)
    if 6 in stages:
        summary["exp6"] = exp6_tuning(corpus, res_dir, dev_q)
        A.write_json(os.path.join(res_dir, "exp6_tuning.json"), summary["exp6"])
        bp = summary["exp6"]["best_params"]
        params = {
            "w_v": bp["w_v"], "w_g": bp["w_g"], "w_kw": bp["w_kw"],
            "eta": bp["eta"], "guarantee_k": bp["guarantee_k"],
            "use_guarantee": bp["guarantee_mode"] != "off"
                           and bp["guarantee_k"] > 0,
            "guarantee_mode": bp["guarantee_mode"],
            "kw_pos": bp["w_pos"], "kw_count": bp["w_count"],
            "kw_assoc": bp["w_assoc"],
            "norm_mode": bp["norm_mode"], "kw_scope": bp["kw_scope"],
            "design": bp["design"], "w_title": bp["w_title"],
            "w_bm25": bp["w_bm25"], "lexical": bp["lexical"],
            "lex_norm": bp["lex_norm"],
        }
        LOG.info("最终超参 (在 DEV 上选出): %s", params)
    summary["best_params"] = params

    # ---------- 其余实验: 在 TEST 上报告 ----------
    if 1 in stages:
        summary["exp1"] = exp1_baselines(corpus, res_dir, test_q, ragflow,
                                         rrf, params)
        A.write_json(os.path.join(res_dir, "exp1_baselines.json"), summary["exp1"])
    if 2 in stages:
        summary["exp2"] = exp2_dual_path(corpus, res_dir, test_q, params)
        A.write_json(os.path.join(res_dir, "exp2_dualpath.json"), summary["exp2"])
    if 3 in stages:
        summary["exp3"] = exp3_bottomup(corpus, res_dir, test_q, params)
        A.write_json(os.path.join(res_dir, "exp3_bottomup.json"), summary["exp3"])
    if 4 in stages:
        summary["exp4"] = exp4_weights(corpus, res_dir, test_q, params)
        A.write_json(os.path.join(res_dir, "exp4_weights.json"), summary["exp4"])
    if 5 in stages:
        summary["exp5"] = exp5_keyword(corpus, res_dir, test_q, params)
        A.write_json(os.path.join(res_dir, "exp5_keyword.json"), summary["exp5"])
    if 7 in stages:
        summary["exp7"] = exp7_latency(corpus, res_dir, test_q, params, ragflow)
        A.write_json(os.path.join(res_dir, "exp7_latency.json"), summary["exp7"])

    # ---------- exp8/exp9: 图谱成分消融 + 截断稳健性 (审稿人#1 关注点 2) ----------
    if 8 in stages or 9 in stages:
        import hgrag4_ablate_graph as ABL
        params8 = dict(DEFAULT_PARAMS)
        if 6 in stages and summary.get("exp6"):
            bp = summary["exp6"]["best_params"]
            params8 = {
                "w_v": bp["w_v"], "w_g": bp["w_g"], "w_kw": bp["w_kw"],
                "eta": bp["eta"], "guarantee_k": bp["guarantee_k"],
                "use_guarantee": bp["guarantee_mode"] != "off"
                and bp["guarantee_k"] > 0,
                "guarantee_mode": bp["guarantee_mode"],
                "kw_pos": bp["w_pos"], "kw_count": bp["w_count"],
                "kw_assoc": bp["w_assoc"],
                "norm_mode": bp["norm_mode"], "kw_scope": bp["kw_scope"],
                "design": bp["design"], "w_title": bp["w_title"],
                "w_bm25": bp["w_bm25"], "lexical": bp["lexical"],
                "lex_norm": bp["lex_norm"],
            }
        if 8 in stages:
            trunc = ABL.measure_truncation(corpus, args.embed_model)
            summary["exp8_truncation"] = trunc
            A.write_json(os.path.join(res_dir, "exp8_truncation.json"), trunc)
            exp8 = ABL.exp8_graph_components(corpus, test_q, params8)
            summary["exp8"] = exp8
            A.write_json(os.path.join(res_dir, "exp8_graph_components.json"),
                         exp8)
        if 9 in stages:
            exp9 = ABL.exp9_truncation_robustness(
                args.kg_dir, args.embed_model, args.cache_dir, test_q,
                params8)
            summary["exp9"] = exp9
            A.write_json(os.path.join(res_dir, "exp9_truncation_robust.json"),
                         exp9)

    # ---------- 审计: 每个实验在哪个划分上评估 ----------
    summary["evaluated_on"] = {
        k: v.get("_evaluated_on")
        for k, v in summary.items()
        if isinstance(v, dict) and "_evaluated_on" in v
    }
    # summary.json 是 *累积* 的产物: 只跑部分阶段 (例如 --stages 8,9) 时,
    # 若直接覆盖, 就会把先前阶段 (exp1..exp7) 的结果抹掉。因此这里把
    # 本次没有重算的键从既有 summary.json 里保留下来。
    sp = os.path.join(res_dir, "summary.json")
    prev = A.read_json(sp) or {}
    merged: Dict[str, Any] = dict(prev)
    merged.update(summary)
    merged["evaluated_on"] = {
        k: v.get("_evaluated_on")
        for k, v in merged.items()
        if isinstance(v, dict) and "_evaluated_on" in v
    }
    A.write_json(sp, merged)
    LOG.info("评估划分审计: %s", merged["evaluated_on"])
    LOG.info("结果目录: %s", res_dir)
    corpus.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
