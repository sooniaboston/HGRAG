#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hgrag_fast.py  --  HGRAG 的低延迟实现 (逐位等价, 零精度损失)

动机
----
论文报告 HGRAG 平均 55.8 ms、RAGFlow 12.6 ms (4.4x)。逐行剖析 (profile_full.py)
表明, 这 55.8 ms 中有相当一部分是 **与算法无关的工程开销**, 且都可以在不改变
任何检索结果的前提下消除:

  P1  词法种子打分全表扫描       ~7.8 ms
      `_lexical_seed_score` 对全部 10,691 个实体名逐个做子串匹配与集合运算。
      -> 用 *bigram 倒排索引* 把候选集缩小到真正可能命中的少数。
  P2  title bonus 重复分词        ~4.0 ms
      每道题对全部 1,172 个标题重新分词 (78,658 次 sig_tokens / 67 题)。
      -> 标题 token 集合是静态的, 预计算一次。
  P3  实体->段落映射重复排序      ~0.3 ms
      每道题对每个激活实体重新 sorted()。
      -> 预计算。
  P4  向量路重复 embedding        ~0.5 ms
      `seed()` 里的词法打分循环对全部 6,728 个实体做 `eid in lex` 查找。
      -> 只对命中的实体写数组。
  P5  内存 BM25 算了又被丢弃      ~3.3 ms
      `BM25LexicalPath.score` 先算完整的内存 BM25, 然后在 es_index 存在时
      用 ES 结果把它 *整个覆盖*。前者是纯粹的浪费。
      -> 有 ES 索引时直接跳过。
  P6  三路串行                    ~8 ms
      稠密路 / 图谱路 / 词法路互不依赖, 却按顺序执行。词法路是 I/O 密集
      (ES 网络往返), 图谱路是 CPU 密集 (Python), 二者天然可重叠。
      -> 用线程池并发派发。

等价性
------
P1-P5 全部是 **等价变换**: 同一输入 -> 逐位相同的输出。
  * P1 的倒排索引只允许 "多算", 不允许 "漏算" (证明见 _lexical_seed_score);
  * P2/P3/P4 只把重复计算换成预计算, 数值完全一致;
  * P5 只是删掉一个被丢弃的结果;
  * P6 只改变各路的 *执行时机*, 每路的返回值不变。
因此本模块产出的检索结果与原实现 **逐条相同**, Recall/Precision 不可能有
任何变化 —— 这正是 "不明显降低精度" 的最强保证: 精度 *完全不变*。

fastpath_selftest.py 会逐条比对两条实现的输出以验证这一点。
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import hgrag3_core as H3  # noqa: E402
import hgrag4_fusion as F4  # noqa: E402
import hgrag4_ragflow as RF  # noqa: E402
from activation_graphrag import legal_normalize, sig_tokens  # noqa: E402


# ============================================================================
# P1: 词法种子打分的 bigram 倒排索引
# ============================================================================
class _LexicalSeedIndex:
    """
    为 `_lexical_seed_score` 建立倒排索引。

    原逻辑对每个 name_key 求两个分支:
      (a) `name_key in q`                      —— name_key 是规范化 query 的子串
      (b) `set(name_key.split()) & q_tokens`    —— 与显著词元有交集

    候选召回 (超集, 不漏):
      * (a) 若 name_key 是 q 的子串且 len(name_key)>=2, 则 name_key 的每个
        bigram 都是 q 的 bigram  -> 由 bigram 倒排索引召回。
      * (b) 若 t ∈ tokens(name_key) ∩ q_tokens, 因为 q_tokens ⊆ q.split(),
        所以 t 是 q 的一个词、也是 name_key 的一个词; t 的 bigram 同时属于
        q 与 name_key -> 同样由 bigram 倒排索引召回。
      * len(name_key) < 2 的极短名单独全表处理 (数量极少)。

    候选集是原实现可能命中集合的 **超集**, 因此对候选集跑原逻辑即可得到
    与原实现完全相同的结果 (只是更快)。
    """

    def __init__(self, name_index: Dict[str, str]):
        self.tokens: Dict[str, frozenset] = {}
        self.token_index: Dict[str, Tuple[str, ...]] = {}
        self.bigram_index: Dict[str, Tuple[str, ...]] = {}
        self.short: Tuple[str, ...] = ()

        tok_lists: Dict[str, List[str]] = defaultdict(list)
        bg_lists: Dict[str, List[str]] = defaultdict(list)
        short: List[str] = []
        for key in name_index:
            if not key:
                continue
            toks = frozenset(key.split())
            self.tokens[key] = toks
            if len(key) < 2:
                short.append(key)
                continue
            for t in toks:
                tok_lists[t].append(key)
            grams = {key[i:i + 2] for i in range(len(key) - 1)}
            for g in grams:
                bg_lists[g].append(key)

        self.token_index = {t: tuple(v) for t, v in tok_lists.items()}
        self.bigram_index = {g: tuple(v) for g, v in bg_lists.items()}
        self.short = tuple(short)


# ============================================================================
# P1-P4: 加速后的图谱路
# ============================================================================
class FastKGPathRetriever(H3.KGPathRetriever):
    """
    与 `H3.KGPathRetriever` 输出逐位相同, 但把四类重复计算换成预计算/索引。
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._lex_index = _LexicalSeedIndex(self.name_index)

        # P4: eid -> node_emb 中的行号 (可能有重复 eid, 故用列表)
        eid_pos: Dict[str, List[int]] = defaultdict(list)
        for i, eid in enumerate(self.node_ids):
            eid_pos[eid].append(i)
        self._eid_pos = dict(eid_pos)

        # P3: 实体 -> 段落 (静态, 预排序)
        self._ent_pids_sorted = {eid: tuple(sorted(pids))
                                 for eid, pids in self.ent_pids.items()}

        # P2: 段落 -> 标题显著词元 (静态)
        self._title_tokens = {pid: frozenset(sig_tokens(title))
                              for pid, title in self.pid_to_title.items()}

        # 段落顺序 (与 pid_emb 行顺序一致)
        self._pid_order = list(self.pid_to_title.keys())

    # ------------------------------------------------------------------ #
    def _lexical_seed_score(self, query: str) -> Dict[str, float]:
        q = legal_normalize(query)
        q_tokens = set(sig_tokens(query))
        idx = self._lex_index
        scores: Dict[str, float] = defaultdict(float)

        # ---- 候选超集 ----
        cand: Set[str] = set(idx.short)
        for t in q_tokens:
            for k in idx.token_index.get(t, ()):
                cand.add(k)
        if len(q) >= 2:
            for i in range(len(q) - 1):
                for k in idx.bigram_index.get(q[i:i + 2], ()):
                    cand.add(k)

        # ---- 对候选跑原逻辑 ----
        max_len = max(len(q), 1)
        for name_key in cand:
            eid = self.name_index[name_key]
            if name_key in q:
                v = min(1.0, 0.75 + 0.25 * len(name_key) / max_len)
                if v > scores[eid]:
                    scores[eid] = v
            else:
                ent_tokens = idx.tokens[name_key]
                if ent_tokens and q_tokens:
                    inter = ent_tokens & q_tokens
                    if inter:
                        v = len(inter) / len(ent_tokens | q_tokens) * 0.5
                        if v > scores[eid]:
                            scores[eid] = v
        return scores

    # ------------------------------------------------------------------ #
    def seed(self, query: str, qv: Optional[np.ndarray] = None
             ) -> List[Tuple[str, float]]:
        """
        与父类逐位相同, 但允许外部传入已算好的 query 向量 (省一次 embedding),
        并且只对命中的实体写词法分数组。
        """
        if self.node_emb is None:
            return []
        if qv is None:
            qv = self.embedder.encode([query], normalize=True)[0]
        sims = self.node_emb @ qv
        lex = self._lexical_seed_score(query)
        fused = self.seed_alpha * sims
        if lex:
            lex_arr = np.zeros_like(sims)
            for eid, v in lex.items():
                for p in self._eid_pos.get(eid, ()):
                    lex_arr[p] = v
            fused = fused + self.lexical_weight * lex_arr
        k = min(self.k_e, len(fused))
        if k <= 0:
            return []
        top = np.argpartition(-fused, kth=k - 1)[:k]
        top = sorted(top, key=lambda i: (-fused[i], self.node_ids[int(i)]))
        return [(self.node_ids[int(i)], float(max(fused[i], 0.0))) for i in top]

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: int = 10,
                 qv: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """与父类逐位相同, 但用预计算的 token 集合 / 排序结果。"""
        seeds = self.seed(query, qv=qv)
        act = self.spread(seeds)
        pid_score: Dict[str, float] = defaultdict(float)
        for eid, a in act.items():
            w = a * (self.ent_idf.get(eid, 1.0) if self.use_idf else 1.0)
            for pid in self._ent_pids_sorted.get(eid, ()):
                pid_score[pid] += w
        if self.norm_beta > 0:
            for pid in list(pid_score.keys()):
                cnt = max(1, self.pid_ent_count.get(pid, 1))
                pid_score[pid] /= (cnt ** (self.norm_beta / 2.0))
        if self.title_bonus > 0:
            q = legal_normalize(query)
            q_tokens = set(sig_tokens(query))
            for pid, t_tokens in self._title_tokens.items():
                if t_tokens and q_tokens and (t_tokens & q_tokens):
                    pid_score[pid] += self.title_bonus
        if self.dense_fallback and self.pid_emb is not None:
            if qv is None:
                qv = self.embedder.encode([query], normalize=True)[0]
            dsims = self.pid_emb @ qv
            for i, pid in enumerate(self._pid_order):
                if pid not in pid_score:
                    pid_score[pid] = self.dense_weight * float(dsims[i])
        if not pid_score:
            return []
        ranked = sorted(pid_score.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return [{"pid": p, "score": round(s, 6)} for p, s in ranked]


# ============================================================================
# P5: 加速后的词法路
# ============================================================================
class FastBM25LexicalPath(F4.BM25LexicalPath):
    """
    与 `BM25LexicalPath.score` 逐位相同。

    改动两处:
      * 有 ES 索引时 **不再计算内存 BM25** —— 原实现算完之后用 ES 结果把
        `pid_raw` 整个覆盖, 内存 BM25 的结果被丢弃 (纯浪费 ~3.3 ms)。
      * 关键词的标题归一化只算一次 (原实现对每个 (pid, keyword) 组合都
        重算 `_norm_title`)。
    """

    def score(self, query: str, keywords: Sequence[str],
              candidate_pids: Optional[Sequence[str]] = None
              ) -> Dict[str, Dict[str, Any]]:
        if self.es_index is not None:
            # 与 RAGFlow 基线查询 *同一个* ES 索引
            pid_raw = self.es_index.search_bm25(query, top_k=50, mode="hybrid")
        else:
            raw = self.bm25.score(RF.tokenize(query))
            pid_raw = {}
            for ci, sc in raw.items():
                pid = self._pid_of_chunk(ci)
                if pid is None:
                    continue
                if sc > pid_raw.get(pid, -1.0):
                    pid_raw[pid] = sc

        mx = max(pid_raw.values()) if pid_raw else 0.0
        th = self.title_hits(keywords) if self.w_title > 0 else {}

        pids = list(candidate_pids) if candidate_pids is not None \
            else list(self.pid_doc)
        # 关键词归一化只算一次
        norm_kws = [(k, _norm_title(k)) for k in keywords]

        out: Dict[str, Dict[str, Any]] = {}
        for pid in pids:
            b = pid_raw.get(pid, 0.0)
            s_bm25 = (b / mx) if mx > 0 else 0.0
            s_title = th.get(pid, 0.0)
            pt = self.pid_title_norm.get(pid, "")
            hits = [k for k, nk in norm_kws if nk and nk in pt]
            out[pid] = {
                "pid": pid,
                "s_bm25": s_bm25,
                "s_title": s_title,
                "bm25": b,
                "hits": hits,
            }
        return out


# ============================================================================
# P6: 并发派发的融合检索器
# ============================================================================
class FastFusedRetrieverV4(F4.FusedRetrieverV4):
    """
    与 `FusedRetrieverV4` 输出逐位相同。

    `parallel=True` 时把三路 (稠密 / 图谱 / 词法) 并发派发:
      * 词法路是 I/O 密集 (ES 网络往返), 会释放 GIL;
      * 图谱路是 Python 密集, 但其中的 numpy 矩阵乘也会释放 GIL;
      * 稠密路是 numpy 矩阵乘 + 排序。
    三路互不依赖, 因此并发执行只改变 *时序*, 不改变任何返回值。
    """

    def __init__(self, *args: Any, parallel: bool = True,
                 pool_workers: int = 3, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.parallel = parallel
        self._pool: Optional[ThreadPoolExecutor] = None
        if parallel:
            self._pool = ThreadPoolExecutor(
                max_workers=pool_workers,
                thread_name_prefix="hgrag-fast")

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: int = 10,
                 keywords: Optional[Sequence[str]] = None
                 ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        t0 = time.perf_counter()

        # ---------- query 向量只算一次, 三路共用 ----------
        qv = self._query_vec(query)

        # ---------- 并发派发三路 ----------
        if self.parallel and self._pool is not None:
            f_v = self._pool.submit(self.milvus.search, qv, self.top_pool)
            f_g = self._pool.submit(self.kg.retrieve, query, self.top_pool, qv)
            f_l = self._pool.submit(self._lex_score, query, keywords)
            m_hits = f_v.result()
            g_hits = f_g.result()
            lex_detail = f_l.result()
        else:
            m_hits = self.milvus.search(qv, self.top_pool)
            g_hits = self.kg.retrieve(query, self.top_pool, qv)
            lex_detail = self._lex_score(query, keywords)

        # ---------- 以下融合逻辑与父类逐字相同 ----------
        m_order = [h["pid"] for h in m_hits]
        g_order = [h["pid"] for h in g_hits]
        s_v = {h["pid"]: h["score"] for h in m_hits}
        s_g = {h["pid"]: h["score"] for h in g_hits}

        n_v = self._normalize_path(m_order, s_v)
        n_g = self._normalize_path(g_order, s_g)
        rm = {p: i + 1 for i, p in enumerate(m_order)}
        rg = {p: i + 1 for i, p in enumerate(g_order)}

        kws = [k for k in (keywords or []) if k and k.strip()]
        kw_detail: Dict[str, Dict[str, Any]] = {}
        s_kw: Dict[str, float] = {}
        if kws:
            kw_detail = self.kw.score(kws)
            s_kw = {pid: d["kw_score"] for pid, d in kw_detail.items()
                    if d["kw_score"] > 0}
        pool = set(s_v) | set(s_g)
        n_kw = self._normalize_kw(s_kw, pool) if s_kw else {}

        s_bm25: Dict[str, float] = {}
        n_bm25: Dict[str, float] = {}
        rb: Dict[str, int] = {}
        if self.lex is not None:
            s_bm25 = {pid: d["s_bm25"] for pid, d in lex_detail.items()
                      if d["s_bm25"] > 0}
            rb = {p: i + 1 for i, p in
                  enumerate(sorted(s_bm25, key=lambda x: (-s_bm25[x], x)))}
            sub = {p: s_bm25[p] for p in (pool | set(s_bm25))
                   if p in s_bm25}
            if self.lex_norm == "max":
                mx = max(sub.values()) if sub else 0.0
                n_bm25 = ({p: v / mx for p, v in sub.items()} if mx > 0
                          else {})
            elif self.norm_mode == "rank":
                order = sorted(sub, key=lambda p: (-sub[p], p))
                n_bm25 = F4.rank_normalize(order, sub)
            else:
                n_bm25 = F4.minmax_normalize(sub)

        if self.use_guarantee:
            guar_m = m_order[:self.guarantee_k]
            guar_g = g_order[:self.guarantee_k]
            guaranteed = list(dict.fromkeys(guar_m + guar_g))
        else:
            guar_m, guar_g, guaranteed = [], [], []

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
                score = (max(sv, sg) + 0.25 * (sv + sg) + self.w_kw * sk
                         + self.w_bm25 * sb + self.w_title * st
                         + self.eta * dual)
            elif self.design == "rrf3":
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
            }

        if self.use_guarantee and self.guarantee_mode == "paper":
            guar_items = [fused[p] for p in guaranteed if p in fused]
            rest_items = [fused[p] for p in all_pids
                          if p not in set(guaranteed) and p in fused]
            guar_items.sort(key=lambda d: -d["score"])
            rest_items.sort(key=lambda d: -d["score"])
            final = (guar_items + rest_items)[:top_k]
        elif self.use_guarantee and self.guarantee_mode == "reserved":
            guar_set = set(guaranteed)
            guar_items = [fused[p] for p in guaranteed if p in fused]
            rest_items = [fused[p] for p in all_pids
                          if p not in guar_set and p in fused]
            rest_items.sort(key=lambda d: -d["score"])
            n_guar = min(len(guar_items), top_k)
            final = guar_items[:n_guar] + rest_items[: top_k - n_guar]
            final.sort(key=lambda d: -d["score"])
        else:
            all_items = [fused[p] for p in all_pids if p in fused]
            all_items.sort(key=lambda d: -d["score"])
            final = all_items[:top_k]

        stats = {
            "n_milvus": len(m_hits),
            "n_kg": len(g_hits),
            "n_kw": len(s_kw),
            "n_candidates": len(all_pids),
            "milvus_ranked": m_order[:10],
            "kg_ranked": g_order[:10],
            "t_total": round(time.perf_counter() - t0, 4),
        }
        return final, stats

    # ------------------------------------------------------------------ #
    def _lex_score(self, query: str, keywords: Optional[Sequence[str]]):
        if self.lex is None:
            return {}
        kws = [k for k in (keywords or []) if k and k.strip()]
        return self.lex.score(query, kws)


# ============================================================================
# 工厂
# ============================================================================
def build_fast_retriever(corpus: Any, parallel: bool = True, **kwargs: Any
                         ) -> FastFusedRetrieverV4:
    """与 `hgrag4_experiments.build_retriever` 同参, 但返回低延迟实现。"""
    kg_params = dict(kwargs.pop("kg_params", None) or {})
    kg = FastKGPathRetriever(corpus.network, corpus.passages,
                             embedder=corpus.embedder, **kg_params)
    kw = H3.KeywordScorer(corpus.entities, corpus.relations, corpus.passages,
                          w_pos=kwargs.get("kw_pos", 0.35),
                          w_count=kwargs.get("kw_count", 0.35),
                          w_assoc=kwargs.get("kw_assoc", 0.30),
                          count_saturate=kwargs.get("count_saturate", 5),
                          assoc_top_m=kwargs.get("assoc_top_m", 10))
    es_index = kwargs.pop("es_index", None)
    if es_index is None:
        es_index = getattr(corpus, "shared_es", None)

    r = FastFusedRetrieverV4(
        milvus=corpus.milvus, kg=kg, kw_scorer=kw, passages=corpus.passages,
        embedder=corpus.embedder,
        w_v=kwargs.get("w_v", 0.4), w_g=kwargs.get("w_g", 0.5),
        w_kw=kwargs.get("w_kw", 0.2), eta=kwargs.get("eta", 0.0),
        guarantee_k=kwargs.get("guarantee_k", 0),
        use_guarantee=kwargs.get("use_guarantee", False),
        guarantee_mode=kwargs.get("guarantee_mode", "off"),
        kw_pos_weight=kwargs.get("kw_pos", 0.35),
        kw_count_weight=kwargs.get("kw_count", 0.35),
        kw_assoc_weight=kwargs.get("kw_assoc", 0.30),
        count_saturate=kwargs.get("count_saturate", 5),
        assoc_top_m=kwargs.get("assoc_top_m", 10),
        norm_mode=kwargs.get("norm_mode", "rank"),
        kw_scope=kwargs.get("kw_scope", "cand"),
        design=kwargs.get("design", "sum"),
        w_title=kwargs.get("w_title", 0.0),
        lexical=kwargs.get("lexical", "none"),
        w_bm25=kwargs.get("w_bm25", 0.0),
        lex_norm=kwargs.get("lex_norm", "max"),
        es_index=es_index,
        parallel=parallel,
    )
    # 用加速版词法路替换父类构造的慢版本
    if r.lex is not None:
        r.lex = FastBM25LexicalPath(corpus.passages,
                                    w_title=r.w_title,
                                    es_index=es_index)
    return r
