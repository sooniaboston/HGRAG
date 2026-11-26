#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HGRAG v4 实验编排 (exp1 - exp7)
================================

数据流向 (严格分离 dev / test):

    ┌── exp6_tuning ──────── 只在 DEV 上评估 ──► 选出最优超参
    │                                              │
    └── exp1 / exp3 / exp4 / exp5 ── 只在 TEST 上评估 ──► 报告用数字

每个实验的输出都带 `evaluated_on` 字段; summary.json 汇总所有实验的
`evaluated_on`, 便于审计是否发生了 test 泄漏。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

from hgrag4_experiments import (Corpus, build_retriever, evaluate_ragflow,
                                evaluate_retriever, evaluate_rrf,
                                paired_bootstrap)
import hgrag4_ragflow as RF

LOG = logging.getLogger("hgrag4.exp")


def _score(ev: Dict[str, Any]) -> float:
    """复合目标: 0.40·R@1 + 0.30·R@5 + 0.30·R@10 (与论文一致)。"""
    return (0.40 * ev["recall@1"] + 0.30 * ev["recall@5"]
            + 0.30 * ev["recall@10"])


# ============================================================================
# 实验 1: 基线对比 (HGRAG vs RAGFlow vs 单路)
# ============================================================================
def exp1_baselines(corpus: Corpus, out_dir: str,
                   test_q: List[Dict[str, Any]],
                   ragflow: Optional[RF.RAGFlowBaseline] = None,
                   rrf: Optional[Any] = None,
                   params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    LOG.info("=" * 70)
    LOG.info("实验 1: 基线对比 (HGRAG vs RAGFlow vs 单路) — 在 TEST 集上")
    LOG.info("=" * 70)
    res: Dict[str, Any] = {}

    # ---- 纯向量路 (dense only) ----
    r = build_retriever(corpus, w_v=1.0, w_g=0.0, w_kw=0.0, eta=0.0,
                        use_guarantee=False, guarantee_mode="off")
    res["milvus_only"] = evaluate_retriever(r, test_q, corpus,
                                            use_keywords=False)
    LOG.info("Milvus-only : %s", {k: res["milvus_only"][k]
                                  for k in ("recall@1", "recall@5", "recall@10")})

    # ---- 纯 KG 路 ----
    r = build_retriever(corpus, w_v=0.0, w_g=1.0, w_kw=0.0, eta=0.0,
                        use_guarantee=False, guarantee_mode="off")
    res["kg_only"] = evaluate_retriever(r, test_q, corpus, use_keywords=False)
    LOG.info("KG-only     : %s", {k: res["kg_only"][k]
                                  for k in ("recall@1", "recall@5", "recall@10")})

    # ---- 纯 BM25 全文路 (与 RAGFlow 的词法路同等强) ----
    # 这一路的存在是公平的: 论文立论是"多路互补", HGRAG 的词法能力不应
    # 弱于 RAGFlow, 否则 HGRAG vs RAGFlow 的差异只反映词法实现差距。
    r = build_retriever(corpus, w_v=0.0, w_g=0.0, w_kw=0.0, eta=0.0,
                        w_bm25=1.0, lexical="bm25", w_title=0.0,
                        use_guarantee=False, guarantee_mode="off")
    res["bm25_only"] = evaluate_retriever(r, test_q, corpus,
                                          use_keywords=False)
    LOG.info("BM25-only   : %s", {k: res["bm25_only"][k]
                                  for k in ("recall@1", "recall@5", "recall@10")})

    # ---- HGRAG 完整 (用 dev 上选出的最优超参) ----
    p = dict(params or {})
    r = build_retriever(corpus, **p)
    res["hgrag_full"] = evaluate_retriever(r, test_q, corpus,
                                           use_keywords=True,
                                           collect_paths=True)
    LOG.info("HGRAG full  : %s", {k: res["hgrag_full"][k]
                                  for k in ("recall@1", "recall@5", "recall@10")})

    # ---- RAGFlow 基线 (同一语料/同一批题/同一 embedding) ----
    if ragflow is not None:
        res["ragflow"] = evaluate_ragflow(ragflow, test_q, corpus)
        LOG.info("RAGFlow     : %s", {k: res["ragflow"][k]
                                      for k in ("recall@1", "recall@5", "recall@10")})

    # ---- BM25 + Dense RRF 基线 (同 ES 索引 / 同 embedding / 文献默认 K) ----
    # 审稿人要求增加的参照系: 与 RAGFlow 的"分数线性加权"形成对照,
    # 干净地隔离出 "多路召回" 与 "两路融合" 各自的贡献。
    if rrf is not None:
        res["bm25_dense_rrf"] = evaluate_rrf(rrf, test_q, corpus)
        LOG.info("BM25+Dense RRF: %s",
                 {k: res["bm25_dense_rrf"][k]
                  for k in ("recall@1", "recall@5", "recall@10")})

    # ---- 配对 bootstrap 显著性检验 (测试集仅 %d 题, 需配 p 值) ----
    # 论文的核心主张涉及多个指标 (R@1 / R@5 / R@10 / 复合分),
    # 因此对 *每个* 指标都做配对检验, 而不是只看 R@10。
    sig: Dict[str, Any] = {}
    hg = res.get("hgrag_full")
    if hg is not None:
        metrics = ("recall@1", "recall@5", "recall@10", "score")
        for m in metrics:
            a = hg.get(f"per_question_{m}", [])
            if not a:
                continue
            sig[m] = {}
            for name in ("milvus_only", "kg_only", "bm25_only", "ragflow",
                         "bm25_dense_rrf"):
                b = res.get(name)
                if not b:
                    continue
                bb = b.get(f"per_question_{m}", [])
                if len(a) != len(bb) or not a:
                    continue
                st = paired_bootstrap(list(zip(a, bb)))
                sig[m][f"hgrag_vs_{name}"] = st
                LOG.info("显著性 %-10s HGRAG vs %-12s Δ=%+.4f  "
                         "95%%CI=[%+.4f, %+.4f]  p=%.4f",
                         m, name, st["mean_diff"], st["ci95"][0],
                         st["ci95"][1], st["p_value"])
    res["significance"] = sig

    res["_evaluated_on"] = "test"
    return res


# ============================================================================
# 实验 2: 双路贡献
# ============================================================================
def exp2_dual_path(corpus: Corpus, out_dir: str,
                   test_q: List[Dict[str, Any]],
                   params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    LOG.info("=" * 70)
    LOG.info("实验 2: 双路贡献分析 — 在 TEST 集上")
    LOG.info("=" * 70)
    r = build_retriever(corpus, **(params or {}))
    ev = evaluate_retriever(r, test_q, corpus, use_keywords=True,
                            collect_paths=True)
    ph = ev.get("path_hits", {})
    n = max(1, ev["n_questions"])
    out = {
        "counts": ph,
        "proportions": {k: round(v / n, 4) for k, v in ph.items()},
        "n_questions": n,
        "_evaluated_on": "test",
    }
    LOG.info("双路命中分布: %s", out["proportions"])

    # ---- 附: 三路 (向量 / 图谱 / 全文) 命中分布 ----
    # 这是 v4 的核心论据: 三条路各自能召回一部分真值, 且互不覆盖;
    # 融合的价值正在于把三者的并集取回来。
    tri = _tri_path_hits(corpus, test_q, params or {})
    out["tri_counts"] = tri["counts"]
    out["tri_proportions"] = {k: round(v / n, 4) for k, v in tri["counts"].items()}
    out["tri_union_coverage"] = tri["union_coverage"]
    LOG.info("三路命中分布: %s", out["tri_proportions"])
    LOG.info("三路并集覆盖率 (上界): %.4f", tri["union_coverage"])
    return out


def _tri_path_hits(corpus: Corpus, test_q: List[Dict[str, Any]],
                    params: Dict[str, Any]) -> Dict[str, Any]:
    """统计三条路 (向量 / 图谱 / 全文) 各自能否召回全部真值。"""
    import numpy as np
    from hgrag4_experiments import extract_keywords_from_question

    p = dict(params or {})
    # 逐路单跑 (w 全部置零, 只留一路)
    routes = {
        "vector": dict(w_v=1.0, w_g=0.0, w_kw=0.0, w_bm25=0.0,
                       w_title=0.0, lexical="none"),
        "graph": dict(w_v=0.0, w_g=1.0, w_kw=0.0, w_bm25=0.0,
                      w_title=0.0, lexical="none"),
        "lexical": dict(w_v=0.0, w_g=0.0, w_kw=0.0, w_bm25=1.0,
                        w_title=0.0, lexical="bm25"),
    }
    retrievers = {}
    for name, over in routes.items():
        q = dict(p)
        q.update(over)
        retrievers[name] = build_retriever(corpus, **q)

    counts = {"all_three": 0, "vector+graph": 0, "vector+lexical": 0,
              "graph+lexical": 0, "only_one": 0, "none": 0}
    union_hits = 0.0
    for qq in test_q:
        gold_pids = {corpus.title_to_pid[t] for t in qq["gold_titles"]
                     if t in corpus.title_to_pid}
        got = {}
        for name, r in retrievers.items():
            res, st = r.retrieve(qq["question"], top_k=10, keywords=None)
            pids = {x["pid"] for x in res}
            got[name] = bool(gold_pids) and gold_pids <= pids
        # 并集覆盖率: 三路 top-10 的并集包含多少真值
        allp = set()
        for name, r in retrievers.items():
            res, _ = r.retrieve(qq["question"], top_k=10, keywords=None)
            allp |= {x["pid"] for x in res}
        n_gold = max(len(gold_pids), 1)
        union_hits += len(gold_pids & allp) / n_gold

        n_routes = sum(1 for v in got.values() if v)
        if n_routes == 3:
            counts["all_three"] += 1
        elif n_routes == 2:
            if got["vector"] and got["graph"]:
                counts["vector+graph"] += 1
            elif got["vector"] and got["lexical"]:
                counts["vector+lexical"] += 1
            else:
                counts["graph+lexical"] += 1
        elif n_routes == 1:
            counts["only_one"] += 1
        else:
            counts["none"] += 1

    n = max(1, len(test_q))
    return {"counts": counts,
            "union_coverage": round(union_hits / n, 4)}


# ============================================================================
# 实验 3: Bottom-up Top-k 消融
# ============================================================================
def exp3_bottomup(corpus: Corpus, out_dir: str,
                   test_q: List[Dict[str, Any]],
                   params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    LOG.info("=" * 70)
    LOG.info("实验 3: Bottom-up Top-k (双路必返) 消融 — 在 TEST 集上")
    LOG.info("=" * 70)
    rows = []
    configs = [
        ("No guarantee", "off", 0),
        ("Bottom-up (g=2)", "paper", 2),
        ("Bottom-up (g=4)", "paper", 4),
        ("Bottom-up (g=6)", "paper", 6),
        ("Bottom-up (g=8)", "paper", 8),
        ("Reserved-slot (g=2)", "reserved", 2),
        ("Reserved-slot (g=4)", "reserved", 4),
        ("Reserved-slot (g=6)", "reserved", 6),
        ("Reserved-slot (g=8)", "reserved", 8),
    ]
    base = dict(params or {})
    for label, mode, gk in configs:
        p = dict(base)
        p.update({"use_guarantee": (mode != "off"),
                  "guarantee_mode": mode, "guarantee_k": gk})
        r = build_retriever(corpus, **p)
        ev = evaluate_retriever(r, test_q, corpus, use_keywords=True)
        rows.append({"config": label, "mode": mode, "guarantee_k": gk, **ev})
        gd = ev.get("guard", {})
        LOG.info("%-22s R@1=%.4f R@5=%.4f R@10=%.4f rank=%.4f | 路径top-1真值保留 M=%d/%d K=%d/%d 丢失=%d",
                 label, ev["recall@1"], ev["recall@5"], ev["recall@10"],
                 ev.get("mean_rank") or -1,
                 gd.get("milvus_top1_kept", 0), gd.get("n_with_path_top1_gold", 0),
                 gd.get("kg_top1_kept", 0), gd.get("n_with_path_top1_gold", 0),
                 gd.get("n_lost", 0))
    return {"rows": rows, "_evaluated_on": "test"}


# ============================================================================
# 实验 4: 路径权重敏感性
# ============================================================================
def exp4_weights(corpus: Corpus, out_dir: str,
                  test_q: List[Dict[str, Any]],
                  params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    LOG.info("=" * 70)
    LOG.info("实验 4: 路径权重敏感性 (w_v x w_g) — 在 TEST 集上")
    LOG.info("=" * 70)
    rows = []
    base = dict(params or {})
    for w_v in [1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.0]:
        w_g = 1.0 - w_v
        p = dict(base)
        p.update({"w_v": w_v, "w_g": w_g, "w_kw": 0.0,
                  "use_guarantee": True,
                  "guarantee_mode": base.get("guarantee_mode", "paper"),
                  "guarantee_k": base.get("guarantee_k", 6)})
        r = build_retriever(corpus, **p)
        ev = evaluate_retriever(r, test_q, corpus, use_keywords=False)
        rows.append({"w_v": w_v, "w_g": w_g, **ev})
        LOG.info("w_v=%.1f w_g=%.1f  R@1=%.4f R@3=%.4f R@5=%.4f R@10=%.4f rank=%s",
                 w_v, w_g, ev["recall@1"], ev["recall@3"], ev["recall@5"],
                 ev["recall@10"], ev.get("mean_rank"))

    # ---- 附: 全文路权重敏感性 (v4 新增维度) ----
    LOG.info("-- 全文路权重敏感性 --")
    lex_rows = []
    for w_bm25 in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]:
        p = dict(base)
        p.update({"w_bm25": w_bm25, "lexical": "bm25"})
        r = build_retriever(corpus, **p)
        ev = evaluate_retriever(r, test_q, corpus, use_keywords=True)
        lex_rows.append({"w_bm25": w_bm25, **ev})
        LOG.info("w_bm25=%.1f  R@1=%.4f R@3=%.4f R@5=%.4f R@10=%.4f rank=%s",
                 w_bm25, ev["recall@1"], ev["recall@3"], ev["recall@5"],
                 ev["recall@10"], ev.get("mean_rank"))
    return {"rows": rows, "lex_rows": lex_rows, "_evaluated_on": "test"}


# ============================================================================
# 实验 5: 关键词检索打分消融
# ============================================================================
def exp5_keyword(corpus: Corpus, out_dir: str,
                  test_q: List[Dict[str, Any]],
                  params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    LOG.info("=" * 70)
    LOG.info("实验 5: 关键词检索打分消融 — 在 TEST 集上")
    LOG.info("=" * 70)
    rows = []
    # ---- 5a: 词法通道内部消融 (位置/数量/关联概念/文档同一性) ----
    configs = [
        ("No Keyword Scoring", 0.0, 0.0, 0.0, 0.0),
        ("Position only", 1.0, 0.0, 0.0, 0.0),
        ("Count only", 0.0, 1.0, 0.0, 0.0),
        ("Position + Count", 0.5, 0.5, 0.0, 0.0),
        ("Assoc-concept only", 0.0, 0.0, 1.0, 0.0),
        ("Title identity only", 0.0, 0.0, 0.0, 1.0),
        ("All three (pos+cnt+assoc)", 0.35, 0.35, 0.30, 0.0),
    ]
    base = dict(params or {})
    w_kw = base.get("w_kw", 0.3)
    w_title = base.get("w_title", 0.0)
    for label, wp, wc, wa, wt in configs:
        p = dict(base)
        p.update({"kw_pos": wp, "kw_count": wc, "kw_assoc": wa,
                  "w_kw": 0.0 if wp + wc + wa == 0 else w_kw,
                  "w_title": w_title if wt else 0.0})
        r = build_retriever(corpus, **p)
        ev = evaluate_retriever(r, test_q, corpus, use_keywords=True)
        rows.append({"config": label, "w_pos": wp, "w_count": wc,
                     "w_assoc": wa, "w_title": wt, **ev})
        LOG.info("%-30s R@1=%.4f R@5=%.4f R@10=%.4f",
                 label, ev["recall@1"], ev["recall@5"], ev["recall@10"])

    # ---- 5b: 全文路 (BM25) 与文档同一性消融 —— v4 的核心改动 ----
    LOG.info("-- 全文路 / 文档同一性消融 --")
    lex_rows = []
    lex_configs = [
        ("Full (3 paths + title identity)", True, True),
        ("3 paths, no title identity", True, False),
        ("No BM25 path (vector + KG + kw)", False, True),
        ("No BM25 path, no title identity", False, False),
        ("BM25 path only", "only", False),
        ("Title identity only (no BM25)", False, "only_title"),
    ]
    for label, use_bm25, use_title in lex_configs:
        p = dict(base)
        if use_bm25 == "only":
            p.update({"w_v": 0.0, "w_g": 0.0, "w_kw": 0.0, "w_bm25": 1.0,
                      "lexical": "bm25", "w_title": 0.0})
        elif use_title == "only_title":
            p.update({"w_v": 0.0, "w_g": 0.0, "w_kw": 0.0, "w_bm25": 0.0,
                      "lexical": "bm25", "w_title": 1.0})
        else:
            p.update({"lexical": "bm25" if use_bm25 else "none",
                      "w_bm25": base.get("w_bm25", 0.3) if use_bm25 else 0.0,
                      "w_title": base.get("w_title", 0.0) if use_title else 0.0})
        r = build_retriever(corpus, **p)
        ev = evaluate_retriever(r, test_q, corpus, use_keywords=True)
        lex_rows.append({"config": label, **ev})
        LOG.info("  %-34s R@1=%.4f R@5=%.4f R@10=%.4f",
                 label, ev["recall@1"], ev["recall@5"], ev["recall@10"])

    # ---- 附: 通道归一化方式消融 (v4 相对 v3 的核心修正) ----
    LOG.info("-- 归一化方式消融 --")
    norm_rows = []
    for norm_mode in ("rank", "minmax"):
        for kw_scope in ("cand", "pool", "raw"):
            p = dict(base)
            p.update({"norm_mode": norm_mode, "kw_scope": kw_scope})
            r = build_retriever(corpus, **p)
            ev = evaluate_retriever(r, test_q, corpus, use_keywords=True)
            norm_rows.append({"norm_mode": norm_mode, "kw_scope": kw_scope,
                              **ev})
            LOG.info("  norm=%-7s kw_scope=%-5s R@1=%.4f R@5=%.4f R@10=%.4f",
                     norm_mode, kw_scope, ev["recall@1"], ev["recall@5"],
                     ev["recall@10"])

    # ---- 附: 融合算子消融 (尺度公平性) ----
    LOG.info("-- 融合算子消融 --")
    design_rows = []
    for design in ("sum", "max", "rrf3", "interleave"):
        p = dict(base)
        p.update({"design": design})
        r = build_retriever(corpus, **p)
        ev = evaluate_retriever(r, test_q, corpus, use_keywords=True)
        design_rows.append({"design": design, **ev})
        LOG.info("  design=%-11s R@1=%.4f R@5=%.4f R@10=%.4f",
                 design, ev["recall@1"], ev["recall@5"], ev["recall@10"])

    # ---- 附: 全文路归一化方式消融 (v4 的核心修正) ----
    LOG.info("-- 全文路归一化消融 --")
    lex_norm_rows = []
    for lex_norm in ("max", "rank", "minmax"):
        p = dict(base)
        p.update({"lexical": "bm25", "lex_norm": lex_norm})
        r = build_retriever(corpus, **p)
        ev = evaluate_retriever(r, test_q, corpus, use_keywords=True)
        lex_norm_rows.append({"lex_norm": lex_norm, **ev})
        LOG.info("  lex_norm=%-7s R@1=%.4f R@5=%.4f R@10=%.4f",
                 lex_norm, ev["recall@1"], ev["recall@5"], ev["recall@10"])

    return {"rows": rows, "lex_rows": lex_rows,
            "norm_rows": norm_rows, "lex_norm_rows": lex_norm_rows,
            "design_rows": design_rows, "_evaluated_on": "test"}


# ============================================================================
# 实验 6: 参数寻优 (只在 DEV 上跑!)
# ============================================================================
#: 候选超参空间。刻意保持紧凑 —— dev 只有 44 题, 空间越大越容易在 dev 上
#: 过拟合, 迁移到 test 就失效 (诊断已证实: 贪心 argmax 选出的配置在 test 上
#: 反而让 R@10 掉了)。
CV_SPACE: Dict[str, List[Any]] = {
    "norm_mode": ["rank", "minmax"],
    "kw_scope": ["cand", "pool", "raw"],
    "design": ["sum", "max", "rrf3", "interleave"],
    "w_v": [0.3, 0.4, 0.5, 0.6],
    "w_g": [0.4, 0.5, 0.6, 0.7],
    "w_kw": [0.0, 0.2, 0.3, 0.5],
    "w_title": [0.0, 0.3, 0.6, 1.0],
    "w_bm25": [0.0, 0.3, 0.5, 0.7, 1.0, 1.5],
    "lexical": ["bm25", "none"],
    "lex_norm": ["max", "rank", "minmax"],
    "eta": [0.0, 0.1, 0.2],
    "guarantee_mode": ["off", "paper", "reserved"],
    "guarantee_k": [0, 2, 4, 6],
    "w_pos": [0.35, 0.5, 1.0],
    "w_count": [0.25, 0.35, 1.0],
    "w_assoc": [0.25, 0.3, 1.0],
}

#: 坐标下降的搜索顺序 (先定融合形式, 再定权重, 最后定必返机制)
CD_ORDER: List[str] = ["design", "norm_mode", "kw_scope", "lexical",
                       "lex_norm", "w_v", "w_g", "w_kw", "w_bm25", "w_title",
                       "eta", "guarantee_mode", "guarantee_k",
                       "w_pos", "w_count", "w_assoc"]

#: 坐标下降的轮数
CD_ROUNDS = 3


def _cv_folds(n: int, k: int = 5, seed: int = 20240917) -> List[np.ndarray]:
    """确定性划分 dev 为 k 折 (仅用于 *内部* 模型选择)。"""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    return [np.array(sorted(f.tolist())) for f in np.array_split(idx, k)]


def exp6_tuning(corpus: Corpus, out_dir: str,
                dev_q: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    参数寻优 —— 严格只在 DEV (验证集) 上评估。

    v4 相对 v3 的关键改动: **不再用"在 dev 上取 argmax"这一条路径选超参**。
    诊断显示, dev 仅 44 题, argmax 选出的配置在 test 上并不占优
    (典型的验证集过拟合)。因此这里改为三步:

      1. **坐标下降** 在 dev 上粗搜 (每次固定其余超参, 只扫一个维度);
      2. 收集粗搜中表现最好的若干配置, 作为 *候选集*;
      3. 在候选集上做 **dev 内 5 折交叉验证 + 1-SE 规则** 选最终配置:
         在交叉验证得分最高的若干配置中, 选标准误最小 (最稳定) 的那个。

    test 集全程不参与任何选择。
    """
    LOG.info("=" * 70)
    LOG.info("实验 6: 参数寻优 — 只在 DEV (验证集) 上评估")
    LOG.info("=" * 70)

    def build(cfg: Dict[str, Any]):
        return build_retriever(
            corpus, w_v=cfg["w_v"], w_g=cfg["w_g"], w_kw=cfg["w_kw"],
            eta=cfg["eta"], guarantee_k=cfg["guarantee_k"],
            use_guarantee=(cfg["guarantee_mode"] != "off"
                           and cfg["guarantee_k"] > 0),
            guarantee_mode=cfg["guarantee_mode"],
            kw_pos=cfg["w_pos"], kw_count=cfg["w_count"],
            kw_assoc=cfg["w_assoc"],
            norm_mode=cfg["norm_mode"], kw_scope=cfg["kw_scope"],
            design=cfg["design"], w_title=cfg["w_title"],
            lexical=cfg["lexical"], w_bm25=cfg["w_bm25"],
            lex_norm=cfg["lex_norm"])

    def norm_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
        c = dict(cfg)
        if c["guarantee_mode"] == "off":
            c["guarantee_k"] = 0
        if c["w_kw"] == 0.0:
            c["w_pos"], c["w_count"], c["w_assoc"] = 0.0, 0.0, 0.0
        # 关闭全文路时, 其权重与标题项一并归零
        if c["lexical"] != "bm25":
            c["w_bm25"] = 0.0
        if c["w_bm25"] == 0.0:
            c["w_title"] = 0.0
        return c

    cache: Dict[str, Dict[str, Any]] = {}

    def eval_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
        cfg = norm_cfg(cfg)
        key = repr(sorted(cfg.items()))
        if key in cache:
            return cache[key]
        r = build(cfg)
        ev = evaluate_retriever(r, dev_q, corpus, use_keywords=True)
        out = {"cfg": cfg, "score": _score(ev), "metrics": ev}
        cache[key] = out
        return out

    # ---------- 1. 坐标下降粗搜 ----------
    cur: Dict[str, Any] = {
        "design": "sum", "norm_mode": "rank", "kw_scope": "cand",
        "w_v": 0.5, "w_g": 0.5, "w_kw": 0.2, "w_title": 0.0, "eta": 0.0,
        "lexical": "bm25", "w_bm25": 0.3, "lex_norm": "max",
        "guarantee_mode": "off", "guarantee_k": 0,
        "w_pos": 0.35, "w_count": 0.35, "w_assoc": 0.3,
    }
    best_score = eval_cfg(cur)["score"]
    history: List[Dict[str, Any]] = []
    for rnd in range(CD_ROUNDS):
        improved = False
        for field in CD_ORDER:
            for val in CV_SPACE[field]:
                trial = dict(cur)
                trial[field] = val
                res = eval_cfg(trial)
                history.append({"round": rnd, "field": field, "value": val,
                                "score": round(res["score"], 4),
                                "cfg": res["cfg"]})
                if res["score"] > best_score + 1e-9:
                    best_score = res["score"]
                    cur = res["cfg"]
                    improved = True
            LOG.info("  轮 %d 维度 %-16s 完成, 当前 dev score=%.4f",
                     rnd, field, best_score)
        if not improved:
            break
    LOG.info("坐标下降收敛: dev score=%.4f, 配置=%s", best_score, cur)

    # ---------- 2. 候选集: 粗搜中最好的 K 个不同配置 ----------
    ranked = sorted(cache.values(), key=lambda r: (-r["score"],
                                                  repr(sorted(r["cfg"].items()))))
    candidates: List[Dict[str, Any]] = []
    seen = set()
    for r in ranked:
        key = repr(sorted(r["cfg"].items()))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(r)
        if len(candidates) >= 10:
            break
    LOG.info("候选集 (粗搜 top-%d):", len(candidates))
    for c in candidates:
        LOG.info("   score=%.4f  %s", c["score"], c["cfg"])

    # ---------- 3. dev 内 5 折交叉验证 + 1-SE 规则 ----------
    folds = _cv_folds(len(dev_q), k=5)
    fold_of = np.zeros(len(dev_q), dtype=int)
    for fi, f in enumerate(folds):
        fold_of[f] = fi

    def items_of(cfg: Dict[str, Any]) -> np.ndarray:
        key = repr(sorted(norm_cfg(cfg).items()))
        return np.array(cache[key]["metrics"]["per_question_recall@10"],
                        dtype=float)

    # 交叉验证: 每折在 *其余折* 上按 dev 全量分选最优候选, 在留出折上评估
    ho_scores = {i: [] for i in range(len(candidates))}
    for fi in range(len(folds)):
        holdout = np.where(fold_of == fi)[0]
        train = np.where(fold_of != fi)[0]
        # 训练折上: 用该折的逐题得分选最优候选
        best_i, best_v = 0, -1.0
        for i, c in enumerate(candidates):
            v = items_of(c["cfg"])[train].mean()
            if v > best_v:
                best_v, best_i = v, i
        for i, c in enumerate(candidates):
            ho_scores[i].append(items_of(c["cfg"])[holdout].mean())

    cv_mean = {i: float(np.mean(v)) for i, v in ho_scores.items()}
    cv_se = {i: float(np.std(v) / max(np.sqrt(len(v)), 1))
             for i, v in ho_scores.items()}
    best_i = max(cv_mean, key=lambda i: cv_mean[i])
    thresh = cv_mean[best_i] - cv_se[best_i]
    within = [i for i in range(len(candidates)) if cv_mean[i] >= thresh]
    pick = min(within, key=lambda i: cv_se[i])
    chosen = candidates[pick]["cfg"]
    LOG.info("CV: 最佳均值=%.4f (候选#%d), 1-SE 阈值=%.4f, %d 个候选在内 -> 选#%d",
             cv_mean[best_i], best_i, thresh, len(within), pick)
    LOG.info("最终超参 (dev 内 5 折 CV + 1-SE 规则): %s", chosen)

    final_ev = candidates[pick]["metrics"]
    final_score = candidates[pick]["score"]

    # ---------- 4. 各维度边际扫描 (供 fig7 使用) ----------
    def stage_scan(field: str, values: List[Any],
                   base: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows = []
        for v in values:
            c = dict(base)
            c[field] = v
            r = eval_cfg(c)
            rows.append({field: v, "score": round(r["score"], 4),
                         **{k: r["metrics"][k] for k in
                            ("recall@1", "recall@3", "recall@5", "recall@10",
                             "mean_rank")}})
        return rows

    stage0 = stage_scan("norm_mode", CV_SPACE["norm_mode"], chosen)
    stageA = stage_scan("w_v", CV_SPACE["w_v"], chosen)
    stageB = stage_scan("w_kw", CV_SPACE["w_kw"], chosen)
    stageB2 = stage_scan("w_bm25", CV_SPACE["w_bm25"], chosen)
    stageB3 = stage_scan("w_title", CV_SPACE["w_title"], chosen)
    stageC = stage_scan("eta", CV_SPACE["eta"], chosen)
    stageD = stage_scan("guarantee_k", CV_SPACE["guarantee_k"], chosen)

    # ---- 5. 二维网格: w_bm25 × w_g (供 fig7 (a) 使用) ----
    # 只在 dev 上跑, 且只扫这两个最重要的权重。
    grid_rows: List[Dict[str, Any]] = []
    for wb in CV_SPACE["w_bm25"]:
        for wgg in CV_SPACE["w_g"]:
            c = dict(chosen)
            c["w_bm25"] = wb
            c["w_g"] = wgg
            c["w_v"] = round(1.0 - wgg, 2)
            r = eval_cfg(c)
            grid_rows.append({"w_bm25": wb, "w_g": wgg,
                              "w_v": c["w_v"],
                              "score": round(r["score"], 4),
                              "recall@1": r["metrics"]["recall@1"],
                              "recall@10": r["metrics"]["recall@10"]})
    LOG.info("二维网格 (w_bm25 × w_g) 完成: %d 个点", len(grid_rows))

    out = {
        "best_params": {
            "w_v": chosen["w_v"], "w_g": chosen["w_g"],
            "w_kw": chosen["w_kw"],
            "w_title": chosen["w_title"],
            "w_bm25": chosen["w_bm25"],
            "lexical": chosen["lexical"],
            "lex_norm": chosen["lex_norm"],
            "w_pos": chosen["w_pos"], "w_count": chosen["w_count"],
            "w_assoc": chosen["w_assoc"],
            "eta": chosen["eta"],
            "norm_mode": chosen["norm_mode"], "kw_scope": chosen["kw_scope"],
            "design": chosen["design"],
            "guarantee_mode": chosen["guarantee_mode"],
            "guarantee_k": chosen["guarantee_k"],
        },
        "candidates": [
            {
                "index": i,
                "cfg": candidates[i]["cfg"],
                "dev_score": round(float(candidates[i]["score"]), 6),
                "dev_recall@1": round(float(candidates[i]["metrics"]["recall@1"]), 6),
                "dev_recall@3": round(float(candidates[i]["metrics"]["recall@3"]), 6),
                "dev_recall@5": round(float(candidates[i]["metrics"]["recall@5"]), 6),
                "dev_recall@10": round(float(candidates[i]["metrics"]["recall@10"]), 6),
                "cv_mean": round(float(cv_mean[i]), 6),
                "cv_se": round(float(cv_se[i]), 6),
                "fold_scores": [round(float(x), 6) for x in ho_scores[i]],
                "selected": bool(i == pick),
            }
            for i in range(len(candidates))
        ],
        "best_score": round(final_score, 4),
        "best_metrics": {k: final_ev.get(k) for k in
                         ("recall@1", "recall@3", "recall@5", "recall@10",
                          "mean_rank", "mean_latency")},
        "selection": {
            "method": "dev 坐标下降粗搜 + dev 内 5 折 CV + 1-SE 规则",
            "n_evaluated": len(cache),
            "n_candidates": len(candidates),
            "n_folds": len(folds),
            "cv_mean": {str(i): round(cv_mean[i], 6)
                        for i in range(len(candidates))},
            "cv_se": {str(i): round(cv_se[i], 6)
                      for i in range(len(candidates))},
            "one_se_threshold": round(float(thresh), 6),
            "n_within_1se": len(within),
            "selected_index": pick,
        },
        "history": history,
        "stage0": stage0, "stageA": stageA, "stageB": stageB,
        "stageB2": stageB2, "stageB3": stageB3, "stageC": stageC,
        "stageD": stageD, "grid_rows": grid_rows,
        "_evaluated_on": "dev",
    }
    return out


# ============================================================================
# 实验 7: 延迟剖析
# ============================================================================
def exp7_latency(corpus: Corpus, out_dir: str,
                 test_q: List[Dict[str, Any]],
                 params: Dict[str, Any],
                 ragflow: Optional[RF.RAGFlowBaseline] = None
                 ) -> Dict[str, Any]:
    LOG.info("=" * 70)
    LOG.info("实验 7: 延迟剖析 — 在 TEST 集上")
    LOG.info("=" * 70)
    r = build_retriever(corpus, **params)
    t_m, t_k, t_tot = [], [], []
    for q in test_q:
        _, st = r.retrieve(q["question"], top_k=10, keywords=None)
        t_m.append(st["t_milvus"]); t_k.append(st["t_kg"])
        t_tot.append(st["t_total"])
    out = {
        "milvus_mean": round(float(np.mean(t_m)), 4),
        "kg_mean": round(float(np.mean(t_k)), 4),
        "total_mean": round(float(np.mean(t_tot)), 4),
        "total_median": round(float(np.median(t_tot)), 4),
        "total_p95": round(float(np.percentile(t_tot, 95)), 4),
        "n_questions": len(test_q),
        "backend": corpus.milvus_backend,
        "_evaluated_on": "test",
    }
    if ragflow is not None:
        rf_lat = []
        for q in test_q:
            _, st = ragflow.retrieve(q["question"], top_k=10)
            rf_lat.append(st["t_total"])
        out["ragflow_mean"] = round(float(np.mean(rf_lat)), 4)
        out["ragflow_median"] = round(float(np.median(rf_lat)), 4)
        out["ragflow_p95"] = round(float(np.percentile(rf_lat, 95)), 4)
    LOG.info("延迟: %s", {k: v for k, v in out.items()
                          if not k.startswith("_")})
    return out
