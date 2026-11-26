#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HGRAG v4 实验主程序
====================

与 v1 (`hgrag3_experiments.py`) 的三处关键差异:

1. **验证集 / 测试集分离**
   `hgrag4_splits.py` 把题目确定性划分为 dev (调参) 与 test (报告)。
   exp6 调参只在 dev 上跑; exp1/3/4/5 只在 test 上跑。
   summary.json 中每个实验都带 `evaluated_on` 字段, 可审计。

2. **关键词通道不再泄漏**
   v1 的 `extract_keywords_from_question` 直接返回 `gold_titles` (真值文档标题),
   等于把答案的文档名直接喂给检索器 —— 这让 R@10 从 0.905 虚高到 0.97。
   v4 改为 *只从问句文本* 中抽取关键词 (实体名 / 关键词短语), 不看任何标注。

3. **RAGFlow 基线跑在同一份语料、同一批题、同一个 embedding 模型上**
   见 `hgrag4_ragflow.py`。

输出:
  <out-dir>/results/*.json
  <out-dir>/results/summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import activation_graphrag as A  # noqa: E402
import hgrag3_core as H3  # noqa: E402
import hgrag4_fusion as F4  # noqa: E402
import hgrag4_ragflow as RF  # noqa: E402
import hgrag4_splits as SPL  # noqa: E402

LOG = logging.getLogger("hgrag4.exp")


# ============================================================================
# 语料加载
# ============================================================================
def _make_embedder(embed_model: str, cache_dir: str):
    """
    构造 embedding 封装。

    `activation_graphrag.A.Embedder` 直接把模型名交给 sentence-transformers,
    在 transformers 5.x 上会走 `AutoProcessor` 分支; 对纯文本模型
    (paraphrase-multilingual-MiniLM-L12-v2) 这会抛
    "Unrecognized processing class"。这里先尝试直接加载本地 HF 快照目录
    (完全离线, 不触网), 失败再回退到原有封装。

    注意: 两次加载得到的模型权重完全相同, 因此 embedding 数值一致,
    不影响与 RAGFlow 基线的公平对比 (两者共用同一个 `Corpus.embedder`)。
    """
    import glob
    import os as _os

    snap_root = _os.path.join(
        cache_dir, "models--sentence-transformers--"
        + embed_model.replace("/", "--"), "snapshots")
    snaps = sorted(glob.glob(_os.path.join(snap_root, "*")))
    if snaps:
        from sentence_transformers import SentenceTransformer
        try:
            LOG.info("从本地快照加载 embedding 模型: %s", snaps[-1])
            model = SentenceTransformer(snaps[-1], device="cpu",
                                        cache_folder=cache_dir)
            return _WrapEmbedder(model, embed_model, cache_dir)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("本地快照加载失败 (%s), 回退到 hub 路径", exc)
    return A.Embedder(embed_model, cache_dir=cache_dir)


class _WrapEmbedder:
    """与 `A.Embedder` 接口一致的薄封装, 支持内存级 embedding 缓存。"""

    def __init__(self, model, model_name: str, cache_dir: str):
        self.model = model
        self.model_name = model_name
        self.dim = model.get_sentence_embedding_dimension()
        self._mem: Dict[str, np.ndarray] = {}

    def encode(self, texts: Sequence[str], batch_size: int = 64,
               normalize: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        items = list(texts)
        out: List[Optional[np.ndarray]] = [None] * len(items)
        missing: List[str] = []
        seen = set()
        for i, t in enumerate(items):
            hit = self._mem.get(t)
            if hit is not None:
                out[i] = hit
            elif t not in seen:
                seen.add(t)
                missing.append(t)
        if missing:
            arr = self.model.encode(missing, batch_size=batch_size,
                                    convert_to_numpy=True,
                                    normalize_embeddings=normalize,
                                    show_progress_bar=False)
            arr = np.asarray(arr, dtype=np.float32)
            for t, v in zip(missing, arr):
                self._mem[t] = v
            pos = {t: i for i, t in enumerate(missing)}
            for i, t in enumerate(items):
                if out[i] is None and t in pos:
                    out[i] = arr[pos[t]]
        return np.vstack([o for o in out if o is not None])


class Corpus:
    def __init__(self, kg_dir: str, embed_model: str, cache_dir: str,
                 milvus_backend: str = "exact",
                 milvus_db: Optional[str] = None,
                 max_seq_length: Optional[int] = None):
        self.kg_dir = kg_dir
        self.embed_model = embed_model
        self.cache_dir = cache_dir
        self.milvus_backend = milvus_backend
        # 共享 Elasticsearch 全文索引 (v4): HGRAG 与 RAGFlow 查询同一个索引,
        # 使两者的差值干净地归因到图谱路与融合方式。运行时由 hgrag4_run 注入。
        self.shared_es: Optional[Any] = None

        self.passages = A.read_json(os.path.join(kg_dir, "passages.json")) or []
        self.questions = A.read_json(os.path.join(kg_dir, "questions.json")) or []
        self.entities = A.read_json(os.path.join(kg_dir, "entities.json")) or []
        self.relations = A.read_json(os.path.join(kg_dir, "relations.json")) or []
        self.network = A.read_json(os.path.join(kg_dir, "activation_network.json")) or {}

        self.pids = [p["pid"] for p in self.passages]
        self.titles = [p.get("title", "") for p in self.passages]
        self.texts = [p.get("text", "") for p in self.passages]
        self.pid_to_title = dict(zip(self.pids, self.titles))
        self.title_to_pid = {t: p for p, t in zip(self.pids, self.titles)}
        self.pid_text = {p["pid"]: f"{p.get('title','')}. {p.get('text','')}"
                         for p in self.passages}

        self.embedder = _make_embedder(embed_model, cache_dir)
        # 截断长度可覆盖 (审稿人#1 关注点 2): 默认 None = 沿用模型自带的
        # sentence-transformers 设定 (paraphrase-multilingual-MiniLM-L12-v2
        # 的 sentence_bert_config.json 里是 128)。设为 512 可消除截断。
        if max_seq_length is not None:
            try:
                self.embedder.model.max_seq_length = int(max_seq_length)
                LOG.info("覆盖 embedding 截断长度: %d", max_seq_length)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("设置 max_seq_length 失败: %s", exc)
        LOG.info("语料: %d 段落, %d 实体, %d 关系, %d 问题",
                 len(self.passages), len(self.entities), len(self.relations),
                 len(self.questions))

        # ---------- 段落向量 (HGRAG 与 RAGFlow 共用, 保证 embedding 一致) ----------
        self.pid_emb = self.embedder.encode(
            [self.pid_text[p] for p in self.pids], normalize=True)

        # ---------- 向量库 ----------
        self.milvus = self._build_milvus(milvus_db)

    def _build_milvus(self, milvus_db: Optional[str]):
        if self.milvus_backend == "exact":
            return H3.ExactVectorStore(self.pids, self.pid_emb,
                                       self.titles, self.texts)
        db = milvus_db or os.path.join(self.kg_dir, "milvus_lite.db")
        for suf in (".db", ".db.lock", ".db.meta"):
            p = db + suf
            if os.path.exists(p):
                os.remove(p)
        store = H3.MilvusVectorStore(db, dim=int(self.pid_emb.shape[1]))
        store.open()
        store.insert(self.pids, self.pid_emb, self.titles, self.texts)
        return store

    def close(self) -> None:
        if hasattr(self.milvus, "close"):
            self.milvus.close()


# ============================================================================
# 关键词抽取 (无泄漏版)
# ============================================================================
_STOP = {"the", "a", "an", "of", "and", "or", "in", "on", "at", "to", "for",
         "by", "with", "from", "as", "is", "was", "were", "are", "that",
         "this", "it", "its", "which", "who", "what", "when", "where", "how",
         "did", "does", "do", "were", "have", "has", "had", "be", "been",
         "same", "both", "there", "their", "they", "you", "your", "he", "she"}


def _strip_paren(s: str) -> str:
    """去掉标题里的括号补充说明, 例如 "Ed Wood (film)" -> "Ed Wood"。"""
    return re.sub(r"\s*\([^)]*\)", " ", s).strip()


def extract_keywords_from_question(q: Dict[str, Any],
                                   corpus: Corpus) -> List[str]:
    """
    从 *问句文本* 中抽取关键词 —— 不使用任何标注 (gold_titles / answer)。

    抽取策略 (三条互补线索, 全部只看问句文本):
      1. **问句中逐字出现的语料文档标题** —— 用户在提问时直接点名了
         文档 (例如 "Were Scott Derrickson and Ed Wood of the same
         nationality?" 直接点名了两篇文档)。这是最强、也最合理的
         硬约束: 用户知道自己在问什么, 但不知道答案的措辞。
         注意: 这里检查的是 *全部 1172 个标题* 是否出现在问句里, 而不是
         只检查 gold_titles —— 因此不构成标注泄漏。
      2. **问句中出现且属于图谱实体的名称** —— 捕获问句里提到但没有
         独立成篇的实体 (例如实体是某篇长文档里的一小段)。
      3. **问句中的引号专名** —— 捕获用户显式给出的名称。

    返回顺序按"约束强度"排列: 标题 > 实体 > 引号专名。
    """
    text = q.get("question", "")
    q_low = text.lower()
    kws: List[str] = []

    # ---- 1. 问句中逐字出现的语料标题 ----
    for t in corpus.titles:
        for cand in (t, _strip_paren(t)):
            if len(cand) >= 3 and cand.lower() in q_low:
                if cand not in kws:
                    kws.append(cand)
                break

    # ---- 2. 图谱实体名 (长的优先, 同长度按字典序保证确定性) ----
    cands: List[Tuple[int, str]] = []
    for e in corpus.entities:
        nm = (e.get("name") or "").strip()
        if not nm or len(nm) < 3:
            continue
        if nm.lower() in q_low:
            cands.append((len(nm), nm))
    for _, nm in sorted(set(cands), key=lambda t: (-t[0], t[1])):
        if nm not in kws:
            kws.append(nm)

    # ---- 3. 引号内的专名 ----
    for m in re.findall(r"[\"“”'‘’]([^\"“”'‘’]{2,60})[\"“”'‘’]", text):
        m = m.strip()
        if m and m not in kws:
            kws.append(m)

    return kws[:10]


# ============================================================================
# 评测
# ============================================================================
def evaluate_retriever(retriever: H3.DualPathRetriever,
                       questions: List[Dict[str, Any]],
                       corpus: Corpus,
                       top_k: int = 10,
                       use_keywords: bool = True,
                       collect_paths: bool = False,
                       return_item_scores: bool = False) -> Dict[str, Any]:
    recalls = {1: [], 3: [], 5: [], 10: []}
    precisions = {1: [], 3: [], 5: [], 10: []}
    ranks: List[float] = []
    latencies: List[float] = []
    path_hits = {"milvus_only": 0, "kg_only": 0, "both": 0, "neither": 0}
    per_query: List[Dict[str, Any]] = []
    guard = {"milvus_top1_kept": 0, "kg_top1_kept": 0,
             "n_with_path_top1_gold": 0, "n_lost": 0}

    for qi, q in enumerate(questions):
        qid = q.get("id", str(qi))
        kws = extract_keywords_from_question(q, corpus) if use_keywords else None

        res, stats = retriever.retrieve(q["question"], top_k=max(top_k, 10),
                                        keywords=kws)
        latencies.append(stats["t_total"])

        titles = [corpus.pid_to_title.get(r["pid"], "") for r in res]
        gold = set(q["gold_titles"])
        n_gold = max(len(gold), 1)

        for k in recalls:
            hit = len(gold & set(titles[:k]))
            recalls[k].append(hit / n_gold)
            precisions[k].append(hit / max(k, 1))

        gr = [i for i, t in enumerate(titles[:10]) if t in gold]
        ranks.append((sum(gr) / len(gr) + 1) if gr else float("nan"))

        # ---------- 必返机制保真度 ----------
        final_titles = set(titles)
        m_top1 = stats.get("milvus_ranked", [])
        g_top1 = stats.get("kg_ranked", [])
        m_t1 = corpus.pid_to_title.get(m_top1[0], "") if m_top1 else ""
        g_t1 = corpus.pid_to_title.get(g_top1[0], "") if g_top1 else ""
        m_gold = bool(m_t1) and m_t1 in gold
        g_gold = bool(g_t1) and g_t1 in gold
        if m_gold or g_gold:
            guard["n_with_path_top1_gold"] += 1
            if m_gold:
                guard["milvus_top1_kept"] += 1 if m_t1 in final_titles else 0
            if g_gold:
                guard["kg_top1_kept"] += 1 if g_t1 in final_titles else 0
            if not ((m_gold and m_t1 in final_titles)
                    or (g_gold and g_t1 in final_titles)):
                guard["n_lost"] += 1

        if collect_paths:
            gold_pids = {corpus.title_to_pid[t] for t in gold
                         if t in corpus.title_to_pid}
            in_m = set(stats.get("milvus_ranked", []))
            in_g = set(stats.get("kg_ranked", []))
            if gold_pids and gold_pids <= in_m and gold_pids <= in_g:
                path_hits["both"] += 1
            elif gold_pids and gold_pids <= in_m:
                path_hits["milvus_only"] += 1
            elif gold_pids and gold_pids <= in_g:
                path_hits["kg_only"] += 1
            else:
                path_hits["neither"] += 1

            per_query.append({
                "id": qid, "question": q["question"], "gold": list(gold),
                "keywords": kws,
                "retrieved": titles[:10],
                "milvus_top": [corpus.pid_to_title.get(p, "")
                               for p in stats.get("milvus_ranked", [])],
                "kg_top": [corpus.pid_to_title.get(p, "")
                           for p in stats.get("kg_ranked", [])],
                "t": stats["t_total"],
            })

    out: Dict[str, Any] = {
        **{f"recall@{k}": round(float(np.mean(v)), 4) for k, v in recalls.items()},
        **{f"precision@{k}": round(float(np.mean(v)), 4)
           for k, v in precisions.items()},
        "mean_rank": round(float(np.nanmean(ranks)), 4) if ranks else None,
        "mean_latency": round(float(np.mean(latencies)), 4) if latencies else None,
        "median_latency": round(float(np.median(latencies)), 4) if latencies else None,
        "p95_latency": round(float(np.percentile(latencies, 95)), 4) if latencies else None,
        "n_questions": len(questions),
    }
    if collect_paths:
        out["path_hits"] = path_hits
        out["per_query"] = per_query
    out["guard"] = guard
    # 每题 recall (供配对 bootstrap 显著性检验使用)
    for k, v in recalls.items():
        out[f"per_question_recall@{k}"] = [round(x, 6) for x in v]
    # 每题复合分 (0.40·R@1 + 0.30·R@5 + 0.30·R@10) —— 与寻优目标一致
    out["per_question_score"] = [
        round(0.40 * recalls[1][i] + 0.30 * recalls[5][i]
              + 0.30 * recalls[10][i], 6)
        for i in range(len(questions))]
    return out


def paired_bootstrap(pairs: Sequence[Tuple[float, float]],
                     n_boot: int = 10000,
                     seed: int = 20240917) -> Dict[str, Any]:
    """
    配对 bootstrap 显著性检验。

    `pairs` 是 (系统A的每题得分, 系统B的每题得分) 的配对序列。
    返回 A-B 的均值差、95% 置信区间、双侧 p 值与 "A 优于 B 的概率"。

    测试集只有 67 题, 单看平均 recall 很容易把噪声当成提升;
    因此所有 "优于基线" 的结论都必须配一个 p 值。
    """
    if len(pairs) < 2:
        return {"n": len(pairs), "mean_diff": 0.0, "ci95": (0.0, 0.0),
                "p_value": 1.0, "prob_a_better": 0.5}
    a = np.array([p[0] for p in pairs], dtype=float)
    b = np.array([p[1] for p in pairs], dtype=float)
    d = a - b
    rng = np.random.RandomState(seed)
    n = len(d)
    idx = rng.randint(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    observed = float(d.mean())
    # 双侧 p 值: 零假设 H0: mean_diff = 0
    p = float((np.abs(boot - observed) >= abs(observed)).mean())
    return {
        "n": n,
        "mean_diff": round(observed, 6),
        "ci95": (round(float(np.percentile(boot, 2.5)), 6),
                 round(float(np.percentile(boot, 97.5)), 6)),
        "p_value": round(p, 6),
        "prob_a_better": round(float((boot > 0).mean()), 6),
    }


def evaluate_ragflow(baseline: RF.RAGFlowBaseline,
                     questions: List[Dict[str, Any]],
                     corpus: Corpus,
                     top_k: int = 10) -> Dict[str, Any]:
    """RAGFlow 基线的评测 (与 HGRAG 完全同口径)。"""
    recalls = {1: [], 3: [], 5: [], 10: []}
    precisions = {1: [], 3: [], 5: [], 10: []}
    ranks: List[float] = []
    lat: List[float] = []
    per_query: List[Dict[str, Any]] = []

    for qi, q in enumerate(questions):
        res, st = baseline.retrieve(q["question"], top_k=top_k)
        lat.append(st["t_total"])
        titles = [r["title"] for r in res]
        gold = set(q["gold_titles"])
        n_gold = max(len(gold), 1)
        for k in recalls:
            hit = len(gold & set(titles[:k]))
            recalls[k].append(hit / n_gold)
            precisions[k].append(hit / max(k, 1))
        gr = [i for i, t in enumerate(titles[:10]) if t in gold]
        ranks.append((sum(gr) / len(gr) + 1) if gr else float("nan"))
        per_query.append({
            "id": q.get("id", str(qi)), "question": q["question"],
            "gold": list(gold), "retrieved": titles[:10],
            "t": st["t_total"],
        })

    return {
        **{f"recall@{k}": round(float(np.mean(v)), 4) for k, v in recalls.items()},
        **{f"precision@{k}": round(float(np.mean(v)), 4)
           for k, v in precisions.items()},
        "mean_rank": round(float(np.nanmean(ranks)), 4) if ranks else None,
        "mean_latency": round(float(np.mean(lat)), 4) if lat else None,
        "median_latency": round(float(np.median(lat)), 4) if lat else None,
        "p95_latency": round(float(np.percentile(lat, 95)), 4) if lat else None,
        "n_questions": len(questions),
        "n_chunks": baseline.chunks.__len__(),
        "per_query": per_query,
        "per_question_recall@1": [round(v, 6) for v in recalls[1]],
        "per_question_recall@5": [round(v, 6) for v in recalls[5]],
        "per_question_recall@10": [round(v, 6) for v in recalls[10]],
        "per_question_score": [
            round(0.40 * recalls[1][i] + 0.30 * recalls[5][i]
                  + 0.30 * recalls[10][i], 6)
            for i in range(len(questions))],
    }


def evaluate_rrf(baseline: Any,
                 questions: List[Dict[str, Any]],
                 corpus: Corpus,
                 top_k: int = 10) -> Dict[str, Any]:
    """
    BM25 + Dense RRF 基线的评测 (与 HGRAG / RAGFlow 完全同口径)。

    与 `evaluate_ragflow` 唯一的差别是检索器本身; 指标计算逻辑逐行一致,
    保证三个系统的 Recall@k / Mean Rank / Latency 定义完全相同。
    """
    recalls = {1: [], 3: [], 5: [], 10: []}
    precisions = {1: [], 3: [], 5: [], 10: []}
    ranks: List[float] = []
    lat: List[float] = []
    per_query: List[Dict[str, Any]] = []

    for qi, q in enumerate(questions):
        res, st = baseline.retrieve(q["question"], top_k=top_k)
        lat.append(st["t_total"])
        titles = [r["title"] for r in res]
        gold = set(q["gold_titles"])
        n_gold = max(len(gold), 1)
        for k in recalls:
            hit = len(gold & set(titles[:k]))
            recalls[k].append(hit / n_gold)
            precisions[k].append(hit / max(k, 1))
        gr = [i for i, t in enumerate(titles[:10]) if t in gold]
        ranks.append((sum(gr) / len(gr) + 1) if gr else float("nan"))
        per_query.append({
            "id": q.get("id", str(qi)), "question": q["question"],
            "gold": list(gold), "retrieved": titles[:10],
            "t": st["t_total"],
        })

    return {
        **{f"recall@{k}": round(float(np.mean(v)), 4) for k, v in recalls.items()},
        **{f"precision@{k}": round(float(np.mean(v)), 4)
           for k, v in precisions.items()},
        "mean_rank": round(float(np.nanmean(ranks)), 4) if ranks else None,
        "mean_latency": round(float(np.mean(lat)), 4) if lat else None,
        "median_latency": round(float(np.median(lat)), 4) if lat else None,
        "p95_latency": round(float(np.percentile(lat, 95)), 4) if lat else None,
        "n_questions": len(questions),
        "per_query": per_query,
        "per_question_recall@1": [round(v, 6) for v in recalls[1]],
        "per_question_recall@5": [round(v, 6) for v in recalls[5]],
        "per_question_recall@10": [round(v, 6) for v in recalls[10]],
        "per_question_score": [
            round(0.40 * recalls[1][i] + 0.30 * recalls[5][i]
                  + 0.30 * recalls[10][i], 6)
            for i in range(len(questions))],
    }


# ============================================================================
# 构造检索器
# ============================================================================
def build_retriever(corpus: Corpus,
                    w_v: float = 0.4, w_g: float = 0.4, w_kw: float = 0.2,
                    eta: float = 0.05, guarantee_k: int = 2,
                    use_guarantee: bool = True,
                    guarantee_mode: str = "reserved",
                    kw_pos: float = 0.35, kw_count: float = 0.35,
                    kw_assoc: float = 0.30,
                    count_saturate: int = 5, assoc_top_m: int = 10,
                    norm_mode: str = "rank", kw_scope: str = "cand",
                    design: str = "sum", w_title: float = 0.0,
                    lexical: str = "none", w_bm25: float = 0.0,
                    lex_norm: str = "max",
                    es_index: Optional[Any] = None,
                    kg_params: Optional[Dict[str, Any]] = None
                    ) -> H3.DualPathRetriever:
    """
    构造 v4 融合检索器。

    `norm_mode` 与 `kw_scope` 是 v4 相对 v3 的两个关键修正:
      * norm_mode="rank"  —— 排名归一化, 解决重尾分数上 min-max 失效的问题;
      * kw_scope="cand"   —— 关键词通道只在候选池内归一化, 使 w_kw 可调。

    `design` 选择融合算子:
      * "sum"        —— 各路归一化分加权求和 (论文式);
      * "max"        —— 尺度公平融合, 主分取两路最大归一化分;
      * "interleave" —— 轮转融合, 两条路交替占位。
    """
    kg_params = dict(kg_params or {})
    kg = H3.KGPathRetriever(corpus.network, corpus.passages,
                            embedder=corpus.embedder, **kg_params)
    kw = H3.KeywordScorer(corpus.entities, corpus.relations, corpus.passages,
                          w_pos=kw_pos, w_count=kw_count, w_assoc=kw_assoc,
                          count_saturate=count_saturate, assoc_top_m=assoc_top_m)
    # 共享 ES 索引: 若语料对象上挂了就自动使用 (HGRAG 与 RAGFlow 同索引)
    es_index = es_index if es_index is not None else getattr(corpus,
                                                              "shared_es", None)
    return F4.FusedRetrieverV4(
        milvus=corpus.milvus, kg=kg, kw_scorer=kw, passages=corpus.passages,
        embedder=corpus.embedder,
        w_v=w_v, w_g=w_g, w_kw=w_kw, eta=eta,
        guarantee_k=guarantee_k, use_guarantee=use_guarantee,
        guarantee_mode=guarantee_mode,
        kw_pos_weight=kw_pos, kw_count_weight=kw_count,
        kw_assoc_weight=kw_assoc, count_saturate=count_saturate,
        assoc_top_m=assoc_top_m,
        norm_mode=norm_mode, kw_scope=kw_scope, design=design,
        w_title=w_title, lexical=lexical, w_bm25=w_bm25,
        lex_norm=lex_norm, es_index=es_index)
