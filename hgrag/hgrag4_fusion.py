#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HGRAG v4 :: 融合算子 (FusedRetrieverV4)
=========================================

本模块是 v4 相对 v1/v3 的核心方法学修正, 只新增不改动旧文件。

为什么需要新算子 —— v4 首轮实验暴露了三个可复现的缺陷:

1. **min-max 归一化在重尾分数上失效**
   KG 激活扩散分数是典型的重尾分布: top-1 与 top-2 差距很大, 而第 5~20 名
   挤在一起。对这种分布做 min-max, 排名第 2~20 的文档几乎全被压到 ≈0,
   于是任何一条向量路的文档都能把它们挤掉 —— 表现为 "融合后的 R@10 反而低于
   KG 单路"。

   修正: 改用 **排名归一化 (rank normalization)** —— 每条路内部按名次线性
   衰减到 [0,1] (第 1 名 = 1.0, 第 N 名 = 0)。它对分数分布的尾部形状不敏感,
   且严格保序, 因此每条路内部的排序信息被完整保留。

2. **关键词通道被融合算子"压死"**
   v3 的关键词分在 *全语料* 上做 min-max。关键词命中文档数远多于两条主路的
   候选数, 归一化后任意一个"命中一般的文档"都能拿到 0.3~0.5, 于是一旦
   w_kw > 0, 关键词通道就压过两条主通道。调参因此把 w_kw 选成 0, 关键词
   通道完全失效 (消融表六行结果一模一样)。

   修正: 关键词分只在 **候选池 (s_v ∪ s_g)** 内做排名归一化 ("cand" 作用域),
   三条通道因此处在可比的尺度上, w_kw 才具备可调性。

3. **必返机制在 gk ≤ 4 时是空操作**
   "reserved" 模式把保证项塞进 top-k 的保留槽位, 但当保证项本身就排在
   融合分前列时, 它与纯按分数排序的结果完全一致 —— 机制不起作用。
   修正: 显式区分两种语义, 并在 dev 上实测:
     "paper"    —— 论文式 Bottom-up: 保证集整体排在前部
     "reserved" —— 保留槽位式: 保证项必须入选, 顺序仍由融合分决定

所有选择只在 dev (验证集) 上做出; test 集全程不参与任何选择。
"""
from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import hgrag3_core as H3
import hgrag4_ragflow as RF

# ============================================================================
# 归一化
# ============================================================================
def rank_normalize(order: Sequence[str],
                   scores: Optional[Dict[str, float]] = None
                   ) -> Dict[str, float]:
    """
    排名归一化: 按名次线性衰减到 [0,1]。

    第 1 名 -> 1.0, 第 N 名 -> 0.0, 中间线性插值。
    严格保序, 且不依赖分数分布的形状 (对重尾鲁棒)。
    """
    n = len(order)
    if n == 0:
        return {}
    if n == 1:
        return {order[0]: 1.0}
    return {pid: 1.0 - i / (n - 1) for i, pid in enumerate(order)}


def minmax_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """经典 min-max (保留用于消融对照)。"""
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


NORMALIZERS = {
    "rank": "排名归一化 (对重尾鲁棒, 保序)",
    "minmax": "min-max 归一化 (v3 做法)",
}


# ============================================================================
# 关键词通道 (v4): 在 v3 的 KeywordScorer 之上增加"文档同一性"信号
# ============================================================================
def _norm_title(s: str) -> str:
    """标题归一化: 去括号补充、小写、压空白。"""
    s = re.sub(r"\s*\([^)]*\)", " ", s or "").strip().lower()
    return re.sub(r"\s+", " ", s)


class KeywordChannelV4:
    """
    v4 关键词通道 = v3 的 KeywordScorer (位置/数量/关联概念) + **文档同一性**。

    为什么需要"文档同一性"这一项:
      HotpotQA 这类多跳问答里, 问句经常直接点名文档 (本语料 44 题 dev 中
      有 70.5% 的问句逐字包含至少一个真值文档标题)。用户当然知道自己在问
      哪篇文档 —— 检索系统应当把"问句里点名的文档"识别出来, 而不是把它
      和"正文里偶然提到该词"的文档混为一谈。

      v3 的位置分/数量分把"标题命中"和"正文命中"同等对待, 因此无法体现
      这一强先验; RAGFlow 恰恰靠 BM25 的 title 字段 boost 拿到了更高的
      R@1 (0.4328 vs HGRAG 0.3731)。加入同一性信号后, HGRAG 的第三条路
      才真正携带了两条主路都没有的信息, 双路互补的立论才成立。

    该信号只看 *问句文本* 与 *语料标题* 的字面匹配, 不使用任何标注。
    """

    def __init__(self, base: H3.KeywordScorer, passages: List[Dict[str, Any]],
                 w_title: float = 0.0):
        self.base = base
        self.w_title = w_title
        self.pid_title = {p["pid"]: p.get("title", "") for p in passages}
        self.pid_title_norm = {pid: _norm_title(t)
                               for pid, t in self.pid_title.items()}
        # 归一化标题 -> pid (同一标题可能对应多个段落)
        self.title_index: Dict[str, List[str]] = defaultdict(list)
        for pid, t in self.pid_title_norm.items():
            if t:
                self.title_index[t].append(pid)

    # ------------------------------------------------------------------ #
    def match_entities(self, keywords: Sequence[str]) -> List[str]:
        return self.base.match_entities(keywords)

    def title_hits(self, keywords: Sequence[str]) -> Dict[str, float]:
        """
        找出"问句点名"的文档。

        三级匹配 (逐级放宽, 分数递减):
          1.0  标题与关键词完全一致
          0.85 标题去掉括号后与关键词完全一致
          0.6  关键词是标题的 *前缀* 且关键词长度 >= 4
                (例如关键词 "Scott Derrickson" 命中标题
                 "Scott Derrickson (director)")
        """
        hits: Dict[str, float] = {}
        for kw in keywords:
            k = _norm_title(kw)
            if not k or len(k) < 3:
                continue
            for pid in self.title_index.get(k, []):
                hits[pid] = max(hits.get(pid, 0.0), 1.0)
            # 前缀匹配 (需足够长, 避免 "The" 之类误命中)
            if len(k) >= 4:
                for t, pids in self.title_index.items():
                    if t.startswith(k + " ") or t.startswith(k + "("):
                        for pid in pids:
                            hits[pid] = max(hits.get(pid, 0.0), 0.6)
        return hits

    # ------------------------------------------------------------------ #
    def score(self, keywords: Sequence[str],
              candidate_pids: Optional[Sequence[str]] = None
              ) -> Dict[str, Dict[str, Any]]:
        """对候选段落批量打分, 返回 v3 的细节 + 新的 s_title。"""
        base = self.base.score(keywords, candidate_pids)
        th = self.title_hits(keywords) if self.w_title > 0 else {}
        out: Dict[str, Dict[str, Any]] = {}
        pids = list(candidate_pids) if candidate_pids is not None \
            else list(self.base.pid_text)
        for pid in pids:
            d = dict(base.get(pid, {"kw_score": 0.0, "pos": 0.0, "count": 0.0,
                                     "assoc": 0.0, "hits": []}))
            s_title = th.get(pid, 0.0)
            d["s_title"] = s_title
            # 关键词通道总分 = v3 三项的加权 + 同一性项
            d["kw_score"] = (float(d.get("kw_score", 0.0))
                             + self.w_title * s_title)
            if s_title > 0:
                d["title_match"] = True
            out[pid] = d
        return out


# ============================================================================
# 关键词通道 (v4): 真实 BM25 全文路 + 文档同一性
# ============================================================================
class BM25LexicalPath:
    """
    v4 的 *真实全文路*: Okapi BM25 (k1=1.2, b=0.75), 与 Elasticsearch 默认
    参数一致, 与 RAGFlow 基线跑在 **同一语料、同一分词器** 上。

    为什么必须加这一路
    ------------------
    v3/v4 的"关键词通道"并不是全文检索: 它只统计关键词在文档里的 *位置*
    和 *次数*, 既没有 BM25 的词频饱和与文档长度归一, 也没有 IDF。因此它在
    检索能力上 **弱于** RAGFlow 的 ES/BM25 路 —— 也就是说, 原先的 HGRAG
    并非"RAGFlow + 图谱", 而是"向量 + 图谱 + 一个更弱的词法近似"。

    论文的立论是 *多路互补*, 而非 *图谱 > BM25*。要让这个立论可被检验,
    HGRAG 的词法路必须与 RAGFlow 的词法路 **同等强**; 否则 HGRAG 输给
    RAGFlow 只反映词法实现的差距, 反映不出图谱路的价值。

    加了这一路之后, "HGRAG 是否优于 RAGFlow" 才真正回答了
    "在同等词法能力下, 叠加 KG 激活扩散路能否带来增益"。

    文档同一性 (title identity)
    --------------------------
    BM25 对 *整个 chunk* (标题 + 正文) 一视同仁, 因此"问句点名了某篇文档"
    这一强先验被稀释。RAGFlow 靠 ES 的 title 字段 boost 拿到更高的 R@1
    (0.4328 vs HGRAG 0.3731)。这里显式地把该先验建模为独立信号 s_title:
    只看问句文本与语料标题的字面匹配, 不使用任何标注。
    """

    def __init__(self, passages: List[Dict[str, Any]],
                 w_title: float = 0.0, k1: float = 1.2, b: float = 0.75,
                 title_boost: float = 2.0,
                 es_index: Optional[Any] = None):
        self.k1 = k1
        self.b = b
        self.w_title = w_title
        self.title_boost = title_boost
        # 共享 ES 索引 (可选): 若提供, 则与 RAGFlow 基线查询 *同一个* 索引,
        # 从而让 "HGRAG vs RAGFlow" 的差值干净地归因到图谱路与融合方式。
        self.es_index = es_index

        self.pid_title = {p["pid"]: p.get("title", "") for p in passages}
        # chunk 文本 = 标题 + 正文 (与 RAGFlow 的 chunk 文本一致)
        self.pid_doc = {p["pid"]: f"{p.get('title','')}\n{p.get('text','')}"
                        for p in passages}
        # BM25 文档与 passage 一一对应 (本语料段落已 <= 512 token,
        # 故每段即一个 chunk), 因此 chunk 下标 = passage 下标
        self._chunk_pid: List[str] = [p["pid"] for p in passages]
        self.bm25 = RF.BM25Index([RF.tokenize(self.pid_doc[p["pid"]])
                                  for p in passages])

        # 标题索引 (文档同一性)
        self.pid_title_norm = {pid: _norm_title(t)
                               for pid, t in self.pid_title.items()}
        self.title_index: Dict[str, List[str]] = defaultdict(list)
        for pid, t in self.pid_title_norm.items():
            if t:
                self.title_index[t].append(pid)

    # ------------------------------------------------------------------ #
    def title_hits(self, keywords: Sequence[str]) -> Dict[str, float]:
        """
        找出"问句点名"的文档。

        三级匹配 (逐级放宽, 分数递减):
          1.0  标题与关键词完全一致
          0.6  关键词是标题的 *前缀* 且关键词长度 >= 4
                (例如关键词 "Scott Derrickson" 命中标题
                 "Scott Derrickson (director)")
        """
        hits: Dict[str, float] = {}
        for kw in keywords:
            k = _norm_title(kw)
            if not k or len(k) < 3:
                continue
            for pid in self.title_index.get(k, []):
                hits[pid] = max(hits.get(pid, 0.0), 1.0)
            if len(k) >= 4:
                for t, pids in self.title_index.items():
                    if t.startswith(k + " ") or t.startswith(k + "("):
                        for pid in pids:
                            hits[pid] = max(hits.get(pid, 0.0), 0.6)
        return hits

    # ------------------------------------------------------------------ #
    def score(self, query: str, keywords: Sequence[str],
              candidate_pids: Optional[Sequence[str]] = None
              ) -> Dict[str, Dict[str, Any]]:
        """
        对候选段落批量打分。

        返回每个 pid 的:
          s_bm25  —— 归一化到 [0,1] 的 BM25 分 (以候选池最高分为 1)
          s_title —— 文档同一性分 (问句点名了它)
          bm25    —— 原始 BM25 分
          hits    —— 命中的关键词
        """
        raw = self.bm25.score(RF.tokenize(query))
        # pid -> 最高 chunk 分 (同一 pid 只保留最好的一条)
        pid_raw: Dict[str, float] = {}
        for ci, sc in raw.items():
            pid = self._pid_of_chunk(ci)
            if pid is None:
                continue
            if sc > pid_raw.get(pid, -1.0):
                pid_raw[pid] = sc

        if self.es_index is not None:
            # 与 RAGFlow 基线查询 *同一个* ES 索引 (同分词器、同标题 boost)
            pid_raw = self.es_index.search_bm25(query, top_k=50, mode="hybrid")

        mx = max(pid_raw.values()) if pid_raw else 0.0
        th = self.title_hits(keywords) if self.w_title > 0 else {}

        pids = list(candidate_pids) if candidate_pids is not None \
            else list(self.pid_doc)
        out: Dict[str, Dict[str, Any]] = {}
        for pid in pids:
            b = pid_raw.get(pid, 0.0)
            s_bm25 = (b / mx) if mx > 0 else 0.0
            s_title = th.get(pid, 0.0)
            out[pid] = {
                "pid": pid,
                "s_bm25": s_bm25,
                "s_title": s_title,
                "bm25": b,
                "hits": [k for k in keywords
                         if _norm_title(k) and _norm_title(k) in
                         self.pid_title_norm.get(pid, "")],
            }
        return out

    # ------------------------------------------------------------------ #
    def _pid_of_chunk(self, ci: int) -> Optional[str]:
        return self._chunk_pid[ci] if 0 <= ci < len(self._chunk_pid) else None


# ============================================================================
# 融合检索器
# ============================================================================
class FusedRetrieverV4(H3.DualPathRetriever):
    """
    HGRAG v4 的双路融合检索器。

    在 v3 的 `DualPathRetriever` 基础上增加四个可配置项:

      * `norm_mode`  —— "rank" (默认) 或 "minmax", 控制两条主路的归一化方式;
      * `kw_scope`   —— "cand" (默认) / "pool" / "raw", 控制关键词通道的
                        归一化作用域与方式;
      * `design`     —— "sum" (默认) / "max" / "interleave", 控制融合算子;
      * `guarantee_mode` —— "off" / "paper" / "reserved"。

    其余流程与 v3 完全一致 (两路检索 -> 关键词打分 -> 保证集 -> 融合排序)。
    """

    def __init__(self, *args: Any, norm_mode: str = "rank",
                 kw_scope: str = "cand", design: str = "sum",
                 w_title: float = 0.0, lexical: str = "none",
                 w_bm25: float = 0.0, lex_norm: str = "max",
                 es_index: Optional[Any] = None,
                 **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.norm_mode = norm_mode
        self.kw_scope = kw_scope
        self.design = design
        self.w_title = w_title
        self.w_bm25 = w_bm25
        self.lex_norm = lex_norm        # "max" | "rank" | "minmax"
        self.lexical = lexical          # "none" | "bm25"
        # v4 关键词通道包装 v3 的 KeywordScorer (位置/数量/关联概念)
        self.kw = KeywordChannelV4(self.kw, self.passages, w_title=w_title)
        # v4 真实全文路 (Okapi BM25, 与 ES 默认参数一致)
        self.lex: Optional[BM25LexicalPath] = None
        if lexical == "bm25":
            self.lex = BM25LexicalPath(self.passages, w_title=w_title,
                                        es_index=es_index)

    # ------------------------------------------------------------------ #
    def _normalize_path(self, order: Sequence[str],
                        scores: Dict[str, float]) -> Dict[str, float]:
        if self.norm_mode == "rank":
            return rank_normalize(order, scores)
        return minmax_normalize(scores)

    def _normalize_kw(self, s_kw: Dict[str, float],
                      pool: set) -> Dict[str, float]:
        """关键词通道归一化。"""
        if self.kw_scope == "raw":
            # s_kw 本身已是 [0,1] 的有界加权分, 不再归一化
            return dict(s_kw)
        if self.kw_scope == "cand":
            sub = {p: s_kw[p] for p in pool if p in s_kw}
        else:  # "pool"
            sub = dict(s_kw)
        if self.norm_mode == "rank":
            # 排序必须是确定性的: 同分关键词文档的相对顺序不能依赖
            # set / dict 的迭代顺序 (后者受哈希随机化影响)。
            order = sorted(sub, key=lambda p: (-sub[p], p))
            return rank_normalize(order, sub)
        return minmax_normalize(sub)

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: int = 10,
                 keywords: Optional[Sequence[str]] = None
                 ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        t0 = time.perf_counter()

        # ---------- 路径 1: 稠密向量 ----------
        t_v0 = time.perf_counter()
        qv = self._query_vec(query)
        m_hits = self.milvus.search(qv, self.top_pool)
        t_v = time.perf_counter() - t_v0

        # ---------- 路径 2: KG 激活扩散 ----------
        t_g0 = time.perf_counter()
        g_hits = self.kg.retrieve(query, top_k=self.top_pool)
        t_g = time.perf_counter() - t_g0

        m_order = [h["pid"] for h in m_hits]
        g_order = [h["pid"] for h in g_hits]
        s_v = {h["pid"]: h["score"] for h in m_hits}
        s_g = {h["pid"]: h["score"] for h in g_hits}

        n_v = self._normalize_path(m_order, s_v)
        n_g = self._normalize_path(g_order, s_g)
        # 各路的 1-based 名次 (供 RRF 使用)
        rm = {p: i + 1 for i, p in enumerate(m_order)}
        rg = {p: i + 1 for i, p in enumerate(g_order)}

        # ---------- 路径 3: 关键词 ----------
        kws = [k for k in (keywords or []) if k and k.strip()]
        kw_detail: Dict[str, Dict[str, Any]] = {}
        s_kw: Dict[str, float] = {}
        if kws:
            kw_detail = self.kw.score(kws)
            s_kw = {pid: d["kw_score"] for pid, d in kw_detail.items()
                    if d["kw_score"] > 0}
        pool = set(s_v) | set(s_g)
        n_kw = self._normalize_kw(s_kw, pool) if s_kw else {}

        # ---------- 路径 4: 真实 BM25 全文路 (v4 新增) ----------
        lex_detail: Dict[str, Dict[str, Any]] = {}
        s_bm25: Dict[str, float] = {}
        n_bm25: Dict[str, float] = {}
        rb: Dict[str, int] = {}
        if self.lex is not None:
            lex_detail = self.lex.score(query, kws)
            s_bm25 = {pid: d["s_bm25"] for pid, d in lex_detail.items()
                      if d["s_bm25"] > 0}
            # BM25 的 1-based 名次 (供 RRF 使用)
            rb = {p: i + 1 for i, p in
                  enumerate(sorted(s_bm25, key=lambda x: (-s_bm25[x], x)))}
            # BM25 只在候选池内归一化, 避免长尾文档挤掉主路结果
            sub = {p: s_bm25[p] for p in (pool | set(s_bm25))
                   if p in s_bm25}
            if self.lex_norm == "max":
                # 与 RAGFlow 一致: BM25 分数除以候选池内最高分。
                # 不能用排名归一化: BM25 的命中集合很大 (任何与问句共享
                # 词元的文档都有分), 排名归一化会把这些长尾文档的名次
                # 压成近乎相等, 从而抹掉 BM25 真正的分差。
                mx = max(sub.values()) if sub else 0.0
                n_bm25 = ({p: v / mx for p, v in sub.items()} if mx > 0
                          else {})
            elif self.norm_mode == "rank":
                order = sorted(sub, key=lambda p: (-sub[p], p))
                n_bm25 = rank_normalize(order, sub)
            else:
                n_bm25 = minmax_normalize(sub)

        # ---------- 保证集 ----------
        if self.use_guarantee:
            guar_m = m_order[:self.guarantee_k]
            guar_g = g_order[:self.guarantee_k]
            guaranteed = list(dict.fromkeys(guar_m + guar_g))
        else:
            guar_m, guar_g, guaranteed = [], [], []

        # ---------- 融合 ----------
        all_pids = pool | set(s_kw) | set(s_bm25)
        fused: Dict[str, Dict[str, Any]] = {}
        for pid in all_pids:
            sv = n_v.get(pid, 0.0)
            sg = n_g.get(pid, 0.0)
            sk = n_kw.get(pid, 0.0)
            sb = n_bm25.get(pid, 0.0)
            st = float(lex_detail.get(pid, {}).get("s_title", 0.0))
            dual = 1.0 if (pid in s_v and pid in s_g) else 0.0
            if self.design == "max":
                # 尺度公平的融合: 主分取两路的 *最大* 归一化分 (而不是相加),
                # 辅以各路之和的少量加成。这样只在一条路里出现的好文档
                # 不会因为 "另一条路贡献 0" 而被系统性压低。
                score = (max(sv, sg) + 0.25 * (sv + sg) + self.w_kw * sk
                         + self.w_bm25 * sb + self.w_title * st
                         + self.eta * dual)
            elif self.design == "rrf3":
                # 三路 RRF (Reciprocal Rank Fusion): 各路按名次倒数加权和。
                # 天然处理 BM25 的长尾命中集合, 且不需要任何分数标定 ——
                # 这正是"向量 + 图 + 全文"三路异构信号最稳健的融合方式。
                K = 20.0
                rr = 0.0
                if pid in s_v:
                    rr += self.w_v / (K + rm.get(pid, 0))
                if pid in s_g:
                    rr += self.w_g / (K + rg.get(pid, 0))
                if pid in s_bm25:
                    rr += self.w_bm25 / (K + rb.get(pid, 0))
                score = rr + self.w_title * st + self.w_kw * sk
            else:
                score = (self.w_v * sv + self.w_g * sg + self.w_kw * sk
                         + self.w_bm25 * sb + self.w_title * st
                         + self.eta * dual)
            fused[pid] = {
                "pid": pid,
                "score": score,
                "s_v": sv, "s_g": sg, "s_kw": sk, "s_bm25": sb, "s_title": st,
                "dual": dual,
                "in_milvus": pid in s_v,
                "in_kg": pid in s_g,
                "in_kw": pid in s_kw,
                "in_bm25": pid in s_bm25,
                "guaranteed": pid in guaranteed,
                "guaranteed_from": (["milvus"] if pid in guar_m else [])
                                   + (["kg"] if pid in guar_g else []),
                "title": self.kg.pid_to_title.get(pid, ""),
                "kw_detail": kw_detail.get(pid, {}),
                "lex_detail": lex_detail.get(pid, {}),
            }

        # ---------- 输出 ----------
        # 关键: `all_pids` 必须保持 **确定性顺序**。Python 的 set 迭代顺序
        # 依赖字符串哈希随机化 (PYTHONHASHSEED), 每次进程启动都不同; 而下面
        # 的 sort 是 *稳定排序*, 同分文档的相对顺序完全由 `all_pids` 的
        # 迭代顺序决定 —— 于是同一次运行里, 两个分数相同的候选文档谁排前面
        # 是随机的。这会让超参寻优结果不可复现 (同一份 dev 上两次寻优选出
        # 不同的配置), 也让论文里的数字无法复现。
        # 因此这里用各路名次做确定性的 tie-break。
        def _tiebreak(pid: str) -> Tuple[int, int, int, str]:
            return (rm.get(pid, 10 ** 6), rg.get(pid, 10 ** 6),
                    rb.get(pid, 10 ** 6), pid)

        guar_set = set(guaranteed)
        if self.use_guarantee and self.guarantee_mode == "paper":
            # 论文式 Bottom-up: 保证集整体排在前部, 内部仍按融合分
            gitems = [fused[p] for p in guaranteed if p in fused]
            gitems.sort(key=lambda d: (-d["score"], _tiebreak(d["pid"])))
            ritems = [fused[p] for p in all_pids if p not in guar_set]
            ritems.sort(key=lambda d: (-d["score"], _tiebreak(d["pid"])))
            final = (gitems + ritems)[:top_k]
        elif self.use_guarantee and self.guarantee_mode == "reserved":
            # 保留槽位式: 保证项必须入选, 最终顺序完全由融合分决定
            gitems = [fused[p] for p in guaranteed if p in fused]
            ritems = [fused[p] for p in all_pids if p not in guar_set]
            ritems.sort(key=lambda d: (-d["score"], _tiebreak(d["pid"])))
            n_g = min(len(gitems), top_k)
            final = gitems[:n_g] + ritems[: top_k - n_g]
            final.sort(key=lambda d: (-d["score"], _tiebreak(d["pid"])))
        elif self.design == "interleave":
            # 轮转融合: 两条路交替出队, 保证每条路都能占满一半名额。
            # 动机: 加权求和会系统性压低 "只在一条路里出现" 的文档
            # (它只拿到 0.5 而双路文档拿到 1.0), 而这恰恰是双路互补
            # 最能体现价值的场景。
            def ranked(order: Sequence[str]) -> List[str]:
                seen = set()
                out = []
                for p in order:
                    if p in fused and p not in seen:
                        seen.add(p)
                        out.append(p)
                return out
            rm2, rg2 = ranked(m_order), ranked(g_order)
            merged: List[str] = []
            i = j = 0
            while len(merged) < top_k and (i < len(rm2) or j < len(rg2)):
                if i < len(rm2) and rm2[i] not in merged:
                    merged.append(rm2[i])
                if len(merged) < top_k and j < len(rg2) and rg2[j] not in merged:
                    merged.append(rg2[j])
                i += 1
                j += 1
            rest = [fused[p] for p in all_pids if p not in set(merged)]
            rest.sort(key=lambda d: (-d["score"], _tiebreak(d["pid"])))
            final = [fused[p] for p in merged] + rest
            final = final[:top_k]
        else:
            final = [fused[p] for p in all_pids if p in fused]
            final.sort(key=lambda d: (-d["score"], _tiebreak(d["pid"])))
            final = final[:top_k]

        stats = {
            "n_milvus": len(m_hits),
            "n_kg": len(g_hits),
            "n_kw": len(s_kw),
            "n_candidates": len(all_pids),
            "n_guaranteed": len([p for p in guaranteed if p in fused]),
            "guaranteed_milvus": guar_m,
            "guaranteed_kg": guar_g,
            "milvus_ranked": m_order[:top_k],
            "kg_ranked": g_order[:top_k],
            "bm25_ranked": (sorted(s_bm25, key=lambda x: (-s_bm25[x], x))[:top_k]
                            if s_bm25 else []),
            "t_milvus": round(t_v, 4),
            "t_kg": round(t_g, 4),
            "t_total": round(time.perf_counter() - t0, 4),
            "keywords": kws,
            "kw_matched_entities": self.kw.match_entities(kws) if kws else [],
            "norm_mode": self.norm_mode,
            "kw_scope": self.kw_scope,
        }
        return final, stats
