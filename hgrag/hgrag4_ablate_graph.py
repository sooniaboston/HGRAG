#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HGRAG v4 :: 审稿人#1 关注点 2 —— 图谱"结构贡献"的混淆消融
=========================================================

审稿意见 (Concern #2)
----------------------
    "The claim that graph structure is the primary driver is confounded.
     Eq. 3 includes TF-IDF and BM25, and the dense encoder
     (paraphrase-multilingual-MiniLM-L12-v2) truncates input at 128 tokens
     while most chunks are 500 to 1,500 characters, so the gain may be
     lexical rather than structural. Please ablate Eq. 3
     (delta=0; gamma=0; alpha=beta=0) and multi-hop expansion, and report
     the share of truncated chunks. The conclusion mentions an ablation of
     graph components that does not appear in the manuscript."

本模块做三件事:

  (A) **截断份额测量** —— 统计每个 chunk 的实际 token 数, 报告被
      `max_seq_length=128` 截断的 chunk 比例 (以及在 *有效* token 预算下
      平均被丢弃的比例)。同时报告 tokenizer 本身的上限 (512), 以便
      说明 "128" 来自 sentence-transformers 的 `sentence_bert_config.json`
      而非模型能力上限。

  (B) **Eq.3 组件消融** —— 逐一关掉图谱路内部的各成分:
        * `delta = 0`   : 关闭 *图结构激活* (PageRank/betweenness 中心性项)
        * `gamma = 0`   : 关闭 *实体重叠*  (entity-overlap 项)
        * `alpha = beta = 0` : 关闭 *TF-IDF / BM25 词法项*
        * 关掉 *多跳扩展* (max_hops = 1, 即只保留直接边)
      并给出 "纯结构" 配置 (只留图结构, 不留任何词法/稠密成分)。

  (C) **多跳扩展消融** —— max_hops = 1 / 2 / 3, 以及只保留 direct 边。

关键设计约束 (与论文其余部分一致)
----------------------------------
  * 全部在 **TEST 集 (67 题)** 上报告, 与主表同口径;
  * 复用 **同一个** Elasticsearch 索引与同一批 embedding;
  * 除被消融的那一个成分外, 其余超参 **完全冻结** 为 exp6 在 DEV 上
    选出的值 (w_v=0.4, w_g=0.5, w_kw=0.2, w_bm25=0.7, guarantee=off);
  * 每个消融配置都做配对 bootstrap, 与完整 HGRAG 比较。

为什么需要 *新增* 而非复用 `KGPathRetriever` 的开关
--------------------------------------------------
v4 的图谱路 (hgrag3_core.KGPathRetriever) 用的是 Collins--Loftus 激活
扩散, **不是** v1 论文 Eq. 3 的 TF-IDF+BM25+实体重叠+中心性 线性组合。
也就是说, 审稿人要求的 "delta=0; gamma=0; alpha=beta=0" 在 v4 里
并不是现成开关。

因此本模块采取 **最保守、最可辩护** 的做法:
  * 不假装 Eq.3 存在于 v4, 而是把审稿人的问题 **翻译成 v4 实现的
    对应成分**:
        - "lexical"  := 图谱路的 *词法种子* (lexical_weight) +
                        TF-IDF/BM25 式词面匹配
        - "dense"    := 图谱路的 *稠密回退* (dense_fallback) +
                        实体稠密相似度种子 (seed_alpha)
        - "structural" := 激活扩散本身 (边权重 / 跳数 / 中心性)
  * 消融就是把对应成分的权重置 0, 并把 *结构性* 保留到最后, 看
    "去掉所有词法成分后, 图谱路还剩多少召回"。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import activation_graphrag as A  # noqa: E402
import hgrag3_core as H3  # noqa: E402
import hgrag4_splits as SPL  # noqa: E402
from hgrag4_experiments import (Corpus, build_retriever,  # noqa: E402
                                 evaluate_retriever, paired_bootstrap)

LOG = logging.getLogger("hgrag4.ablate")

#: 论文 (v1) Eq. 3 的四项 —— 名称与权重含义, 供论文正文对照引用
EQ3_TERMS = {
    "alpha": "TF-IDF 词法项",
    "beta": "BM25 词法项",
    "gamma": "实体重叠项",
    "delta": "图结构中心性项 (PageRank / betweenness)",
}


# ============================================================================
# (A) 截断份额测量
# ============================================================================
def measure_truncation(corpus: Corpus,
                       model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                       ) -> Dict[str, Any]:
    """
    统计 chunk 在稠密编码时被截断的份额。

    做法:
      1. 用 *模型自己的* tokenizer 逐个编码 chunk 文本
         (文本格式与 embedding 实际使用的完全一致: "title. text");
      2. 记录每段的 token 数 (含 special tokens);
      3. 与 sentence-transformers 实际生效的 `max_seq_length` 比较,
         超出部分即为被截断的 token;
      4. 同时记录 tokenizer 的硬上限 (`model_max_length`), 说明 128
         并非模型能力的上限, 而是 sentence-transformers 侧的设定。
    """
    from transformers import AutoTokenizer

    snap_root = os.path.join(
        corpus.cache_dir, "models--sentence-transformers--"
        + model_name.replace("/", "--"), "snapshots")
    import glob
    snaps = sorted(glob.glob(os.path.join(snap_root, "*")))
    tok_path = snaps[-1] if snaps else model_name
    tok = AutoTokenizer.from_pretrained(tok_path)

    # sentence-transformers 实际生效的截断长度
    sbc_path = os.path.join(tok_path, "sentence_bert_config.json")
    max_seq = 128
    if os.path.exists(sbc_path):
        with open(sbc_path, encoding="utf-8") as fh:
            max_seq = int(json.load(fh).get("max_seq_length", 128))

    texts = [corpus.pid_text[p] for p in corpus.pids]
    enc = tok(texts, add_special_tokens=True, truncation=False)["input_ids"]
    lens = np.array([len(x) for x in enc])

    # 被截断的 token 数 = 超出 max_seq 的部分 (special tokens 也占位)
    eff = max_seq
    kept = np.minimum(lens, eff - 1)  # 保留 1 位给 [CLS]/[SEP]
    dropped = lens - kept
    truncated = lens > eff

    # 字符长度分布 (与审稿人所述 "500 to 1,500 characters" 对照)
    char_len = np.array([len(t) for t in texts])

    out: Dict[str, Any] = {
        "model": model_name,
        "sentence_transformers_max_seq_length": max_seq,
        "tokenizer_model_max_length": int(tok.model_max_length),
        "n_chunks": int(len(lens)),
        "token_len_mean": round(float(lens.mean()), 2),
        "token_len_median": float(np.median(lens)),
        "token_len_p90": float(np.percentile(lens, 90)),
        "token_len_max": int(lens.max()),
        "char_len_mean": round(float(char_len.mean()), 1),
        "char_len_median": float(np.median(char_len)),
        "char_len_p90": float(np.percentile(char_len, 90)),
        "char_len_max": int(char_len.max()),
        "n_truncated": int(truncated.sum()),
        "share_truncated": round(float(truncated.mean()), 4),
        "mean_dropped_tokens": round(float(dropped.mean()), 2),
        "mean_kept_tokens": round(float(kept.mean()), 2),
        "mean_dropped_share": round(
            float((dropped / np.maximum(lens, 1)).mean()), 4),
        "n_chunks_over_512_tokens": int((lens > 512).sum()),
        "share_chunks_over_512_tokens": round(float((lens > 512).mean()), 4),
    }
    LOG.info("截断测量: %d/%d chunks 被截断 (%.1f%%), 平均丢弃 %.1f token, "
             "token 中位数 %d (字符中位数 %d)",
             out["n_truncated"], out["n_chunks"], 100 * out["share_truncated"],
             out["mean_dropped_tokens"], out["token_len_median"],
             out["char_len_median"])
    return out


# ============================================================================
# (B) 图谱路成分消融
# ============================================================================
def _kg_variant(corpus: Corpus,
                lexical_weight: float = 0.3,
                seed_alpha: float = 0.9,
                dense_fallback: bool = True,
                dense_weight: float = 0.1,
                title_bonus: float = 0.1,
                max_hops: int = 2,
                use_idf: bool = True,
                norm_beta: float = 0.5,
                ) -> H3.KGPathRetriever:
    """构造一个图谱路变体 (其余部分与生产配置完全一致)。"""
    return H3.KGPathRetriever(
        corpus.network, corpus.passages, embedder=corpus.embedder,
        lexical_weight=lexical_weight, seed_alpha=seed_alpha,
        dense_fallback=dense_fallback, dense_weight=dense_weight,
        title_bonus=title_bonus, max_hops=max_hops,
        use_idf=use_idf, norm_beta=norm_beta)


def _evaluate_kg_only(corpus: Corpus, kg: H3.KGPathRetriever,
                      test_q: List[Dict[str, Any]],
                      ) -> Dict[str, Any]:
    """只跑图谱一路 (w_g=1, 其余为 0), 给出该变体的召回。"""
    from hgrag4_fusion import FusedRetrieverV4
    from hgrag3_core import KeywordScorer

    kw = KeywordScorer(corpus.entities, corpus.relations, corpus.passages)
    r = FusedRetrieverV4(
        milvus=corpus.milvus, kg=kg, kw_scorer=kw, passages=corpus.passages,
        embedder=corpus.embedder,
        w_v=0.0, w_g=1.0, w_kw=0.0, eta=0.0,
        guarantee_k=0, use_guarantee=False, guarantee_mode="off",
        norm_mode="rank", kw_scope="cand", design="sum",
        w_title=0.0, lexical="none", w_bm25=0.0, lex_norm="max")
    return evaluate_retriever(r, test_q, corpus, use_keywords=False)


def exp8_graph_components(corpus: Corpus, test_q: List[Dict[str, Any]],
                          params: Optional[Dict[str, Any]] = None,
                          ) -> Dict[str, Any]:
    """
    图谱路成分消融 —— 回答 "增益是结构性的还是词法的?"。

    每个变体 **只跑图谱一路** (w_g=1), 因为审稿人的问题是
    "图谱路内部的贡献到底是结构还是词法"; 若在完整三路融合上做
    消融, 词法成分会被独立的 BM25 全文路掩盖, 看不清。
    同时也给出 *完整 HGRAG* 下关掉各成分的结果 (第二部分),
    以便回答 "整体增益是否依赖图谱路的词法成分"。
    """
    LOG.info("=" * 70)
    LOG.info("实验 8: 图谱路成分消融 (结构 vs 词法) — 在 TEST 集上")
    LOG.info("=" * 70)

    p = dict(params or {})
    out: Dict[str, Any] = {"eq3_terms": EQ3_TERMS, "graph_only": {},
                           "full_hgrag": {}}

    # ---------------- 组件变体定义 ----------------
    # 基线: 生产配置 (词法种子 0.3 + 稠密种子 0.9 + 稠密回退 + 2 跳)
    variants: List[Tuple[str, str, Dict[str, Any]]] = [
        ("graph_full", "完整图谱路 (生产配置)", {}),

        # ---- 消融 1: 关掉 *词法* 成分 (对应 alpha=beta=0) ----
        ("no_lexical_seed", "关掉词法种子 (lexical_weight=0)",
         {"lexical_weight": 0.0}),
        ("no_dense_fallback", "关掉稠密回退 (dense_fallback=False)",
         {"dense_fallback": False}),
        ("no_lexical_no_dense", "同时关掉词法种子与稠密回退",
         {"lexical_weight": 0.0, "dense_fallback": False}),

        # ---- 消融 2: 关掉 *标题加分* (纯词法的一小部分) ----
        ("no_title_bonus", "关掉标题加分 (title_bonus=0)",
         {"title_bonus": 0.0}),

        # ---- 消融 3: 多跳扩展 ----
        ("hops_1", "多跳=1 (仅直接边)", {"max_hops": 1}),
        ("hops_3", "多跳=3", {"max_hops": 3}),
        ("hops_0", "多跳=0 (不扩散, 仅种子实体自身文档)",
         {"max_hops": 0}),

        # ---- 消融 4: 实体 IDF 与长度归一 (结构侧的可调项) ----
        ("no_entity_idf", "关掉实体 IDF 加权", {"use_idf": False}),
        ("no_length_norm", "关掉长度归一 (norm_beta=0)",
         {"norm_beta": 0.0}),
    ]

    # ---------------- (B1) 图谱单路 ----------------
    # 同时保留每个变体的逐题得分, 以便对 *图谱单路* 的消融做配对 bootstrap。
    # 论文表 tab:structural 的头条结论 (去掉词法种子 -11.94 pp、
    # 关掉扩散 -11.20 pp) 依赖这一路的显著性检验。
    kg_evs: Dict[str, Dict[str, Any]] = {}
    for key, label, over in variants:
        kg = _kg_variant(corpus, **over)
        ev = _evaluate_kg_only(corpus, kg, test_q)
        kg_evs[key] = ev
        out["graph_only"][key] = {"label": label, **{
            m: ev.get(m) for m in ("recall@1", "recall@3", "recall@5",
                                   "recall@10", "mean_rank",
                                   "mean_latency")}}
        LOG.info("  [图谱单路] %-24s R@1=%.4f R@5=%.4f R@10=%.4f",
                 key, ev.get("recall@1") or 0, ev.get("recall@5") or 0,
                 ev.get("recall@10") or 0)

    # ---------------- (B2) 完整 HGRAG 下关掉图谱路的词法成分 ----------------
    # 这里通过 kg_params 注入变体到完整三路融合里
    full_variants = [
        ("hgrag_full", "完整 HGRAG (生产配置)", {}),
        ("hgrag_no_lexical_seed", "完整 HGRAG, 图谱路关掉词法种子",
         {"lexical_weight": 0.0}),
        ("hgrag_no_dense_fallback", "完整 HGRAG, 图谱路关掉稠密回退",
         {"dense_fallback": False}),
        ("hgrag_hops_1", "完整 HGRAG, 多跳=1", {"max_hops": 1}),
    ]
    for key, label, over in full_variants:
        pp = dict(p)
        pp["kg_params"] = over
        # `corpus.shared_es` 由 hgrag4_run 注入: 让 "完整 HGRAG" 参照行
        # 与论文主表 (Table 1) 的 HGRAG 行 **完全同口径** (同一个 ES 索引)。
        # 否则这里的差值里会混入 "本地 BM25 vs Elasticsearch" 的实现差异。
        r = build_retriever(corpus, **pp)
        ev = evaluate_retriever(r, test_q, corpus, use_keywords=True)
        out["full_hgrag"][key] = {"label": label, **{
            m: ev.get(m) for m in ("recall@1", "recall@3", "recall@5",
                                   "recall@10", "mean_rank",
                                   "mean_latency")}}
        LOG.info("  [完整HGRAG] %-28s R@1=%.4f R@5=%.4f R@10=%.4f",
                 key, ev.get("recall@1") or 0, ev.get("recall@5") or 0,
                 ev.get("recall@10") or 0)

    # ---------------- 配对 bootstrap: 每个变体 vs 完整 HGRAG ----------------
    # (用 full_hgrag 里的逐题分数; 图谱单路的逐题分数需重跑才能拿到)
    sig: Dict[str, Any] = {}
    base_ev = None
    for key, label, over in full_variants:
        pp = dict(p)
        pp["kg_params"] = over
        r = build_retriever(corpus, **pp)
        ev = evaluate_retriever(r, test_q, corpus, use_keywords=True)
        if key == "hgrag_full":
            base_ev = ev
    if base_ev is not None:
        for key, label, over in full_variants:
            if key == "hgrag_full":
                continue
            pp = dict(p)
            pp["kg_params"] = over
            r = build_retriever(corpus, **pp)
            ev = evaluate_retriever(r, test_q, corpus, use_keywords=True)
            for m in ("recall@1", "recall@5", "recall@10", "score"):
                a = base_ev.get(f"per_question_{m}") or []
                b = ev.get(f"per_question_{m}") or []
                if len(a) != len(b) or not a:
                    continue
                st = paired_bootstrap(list(zip(a, b)))
                sig.setdefault(m, {})[f"hgrag_vs_{key}"] = st
                LOG.info("  显著性 %-10s HGRAG vs %-26s Δ=%+.4f p=%.4f",
                         m, key, st["mean_diff"], st["p_value"])

    # ---------------- 配对 bootstrap: 图谱单路各变体 vs 完整图谱路 ------
    # 这是论文表 tab:structural 的头条证据: 去掉词法种子的 -11.94 pp 与
    # 关掉扩散 (H=0) 的 -11.20 pp 必须有显著性支撑, 否则"结构是主因"
    # 的结论就只是观察而非证据。
    kg_sig: Dict[str, Any] = {}
    kg_base = kg_evs.get("graph_full")
    if kg_base is not None:
        for key, label, over in variants:
            if key == "graph_full":
                continue
            ev = kg_evs.get(key) or {}
            for m in ("recall@1", "recall@3", "recall@5", "recall@10",
                      "score"):
                a = kg_base.get(f"per_question_{m}") or []
                b = ev.get(f"per_question_{m}") or []
                if len(a) != len(b) or not a:
                    continue
                st = paired_bootstrap(list(zip(a, b)))
                kg_sig.setdefault(m, {})[f"graph_vs_{key}"] = st
                LOG.info("  显著性 %-10s 图谱路 vs %-24s Δ=%+.4f p=%.4f",
                         m, key, st["mean_diff"], st["p_value"])
    out["significance_graph_only"] = kg_sig
    out["significance"] = sig
    out["_evaluated_on"] = "test"
    return out


# ============================================================================
# (C) 截断稳健性: 把 embedding 截断从 128 提到 512 后, 结论是否改变?
# ============================================================================
def exp9_truncation_robustness(kg_dir: str, embed_model: str, cache_dir: str,
                                test_q: List[Dict[str, Any]],
                                params: Optional[Dict[str, Any]] = None,
                                ) -> Dict[str, Any]:
    """
    审稿人的关键质疑: 稠密编码在 128 token 处截断, 而 chunk 通常
    500--1,500 字符 —— 因此 "稠密路" 的贡献可能被系统性削弱,
    图谱路的增益可能被高估。

    本实验在 **两种截断设置** 下重跑同一套基线:
      * `max_seq_length = 128` (生产设置, 48.5% 的 chunk 被截断)
      * `max_seq_length = 512` (消除截断; tokenizer 硬上限, 只有 3/1172
        个 chunk 仍会超长)

    若两者结论一致, 则 "截断" 不是增益的混淆因素。
    """
    LOG.info("=" * 70)
    LOG.info("实验 9: 截断稳健性 (max_seq_length 128 vs 512) — 在 TEST 集上")
    LOG.info("=" * 70)

    p = dict(params or {})
    out: Dict[str, Any] = {"_evaluated_on": "test", "settings": {}}

    for msl in (128, 512):
        corpus = Corpus(kg_dir, embed_model, cache_dir, "exact",
                        max_seq_length=msl)
        tag = f"max_seq_{msl}"
        rows: Dict[str, Any] = {}

        # --- 基线 (注意: 这些基线不使用 ES, 避免受索引状态影响) ---
        r = build_retriever(corpus, w_v=1.0, w_g=0.0, w_kw=0.0, eta=0.0,
                            use_guarantee=False, guarantee_mode="off")
        ev = evaluate_retriever(r, test_q, corpus, use_keywords=False)
        rows["milvus_only"] = {m: ev.get(m) for m in
                               ("recall@1", "recall@3", "recall@5", "recall@10",
                                "mean_rank")}
        LOG.info("  [%s] dense-only      R@1=%.4f R@5=%.4f R@10=%.4f", tag,
                 ev.get("recall@1") or 0, ev.get("recall@5") or 0,
                 ev.get("recall@10") or 0)

        # --- 图谱单路 (结构 vs 词法) ---
        kg = _kg_variant(corpus)
        evg = _evaluate_kg_only(corpus, kg, test_q)
        rows["kg_only"] = {m: evg.get(m) for m in
                           ("recall@1", "recall@3", "recall@5", "recall@10",
                            "mean_rank")}
        LOG.info("  [%s] graph-only      R@1=%.4f R@5=%.4f R@10=%.4f", tag,
                 evg.get("recall@1") or 0, evg.get("recall@5") or 0,
                 evg.get("recall@10") or 0)

        # --- 完整 HGRAG (三路融合, 但此实验不挂 ES) ---
        pp = dict(p)
        pp["w_bm25"] = 0.0
        pp["lexical"] = "none"
        r = build_retriever(corpus, **pp)
        evh = evaluate_retriever(r, test_q, corpus, use_keywords=True)
        rows["hgrag_dense_graph"] = {m: evh.get(m) for m in
                                     ("recall@1", "recall@3", "recall@5",
                                      "recall@10", "mean_rank")}
        rows["_per_question"] = {
            "hgrag_dense_graph": evh.get("per_question_score"),
            "milvus_only": ev.get("per_question_score"),
        }
        LOG.info("  [%s] HGRAG(dense+graph) R@1=%.4f R@5=%.4f R@10=%.4f", tag,
                 evh.get("recall@1") or 0, evh.get("recall@5") or 0,
                 evh.get("recall@10") or 0)

        out["settings"][tag] = rows
        corpus.close()

    # --- 两种截断设置之间的配对 bootstrap ---
    sig: Dict[str, Any] = {}
    a = out["settings"]["max_seq_128"]["_per_question"]
    b = out["settings"]["max_seq_512"]["_per_question"]
    for key in ("hgrag_dense_graph", "milvus_only"):
        pa, pb = a.get(key) or [], b.get(key) or []
        if len(pa) == len(pb) and pa:
            st = paired_bootstrap(list(zip(pa, pb)))
            sig[key] = st
            LOG.info("  截断效应 %-18s Δ(128-512)=%+.4f p=%.4f", key,
                     st["mean_diff"], st["p_value"])
    out["significance_truncation"] = sig
    return out


# ============================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="HGRAG v4 图谱成分消融")
    ap.add_argument("--kg-dir", default=r"C:\data\zsk\RAG\kgv4")
    ap.add_argument("--out-dir", default=r"C:\data\zsk\RAG\hgrag4")
    ap.add_argument("--embed-model", default=A.DEFAULT_EMBED_MODEL)
    ap.add_argument("--cache-dir", default=r"C:\data\zsk\RAG\hf_cache")
    ap.add_argument("--milvus-backend", default="exact")
    ap.add_argument("--split-seed", type=int, default=SPL.SPLIT_SEED)
    ap.add_argument("--dev-fraction", type=float, default=SPL.DEV_FRACTION)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    res_dir = os.path.join(args.out_dir, "results")
    os.makedirs(res_dir, exist_ok=True)

    corpus = Corpus(args.kg_dir, args.embed_model, args.cache_dir,
                    args.milvus_backend)
    split = SPL.load_or_make_split(corpus.questions, args.kg_dir,
                                   dev_fraction=args.dev_fraction,
                                   seed=args.split_seed)
    test_q = SPL.select(corpus.questions, split, "test")
    LOG.info("测试集: %d 题", len(test_q))

    # 生产配置 (与 exp6 在 DEV 上选出的值一致)
    from hgrag4_run import DEFAULT_PARAMS
    params = dict(DEFAULT_PARAMS)

    # (A) 截断测量
    trunc = measure_truncation(corpus, args.embed_model)
    A.write_json(os.path.join(res_dir, "exp8_truncation.json"), trunc)

    # (B) 图谱成分消融
    exp8 = exp8_graph_components(corpus, test_q, params)
    A.write_json(os.path.join(res_dir, "exp8_graph_components.json"), exp8)

    # 合并进 summary.json
    sp = os.path.join(res_dir, "summary.json")
    summary = A.read_json(sp) or {}
    summary["exp8"] = exp8
    summary["exp8_truncation"] = trunc
    A.write_json(sp, summary)
    LOG.info("已写入: %s", sp)
    corpus.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
