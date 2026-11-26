#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HGRAG v4 :: BM25 + Dense RRF 基线
====================================

为什么需要这个基线
--------------------
审稿人指出原稿 Table 1 缺少两个参照系:
  * dense-only ($w_v = 1.0$)  —— 我们已有 (milvus_only)
  * BM25-only                 —— 我们已有 (bm25_only)
  * BM25 + dense 的 RRF 融合  —— 之前缺失

RRF (Reciprocal Rank Fusion) 是工业界最常用的无分数标定融合方式, 与
RAGFlow 的 "分数线性加权" 形成对照:

    score(d) = sum_r  w_r / (K + rank_r(d)),   K = 60 (Cormack et al. 2009)

两者用 *同样的两路原始检索信号* (同一 ES 索引的 BM25 + 同一 embedding
模型的稠密向量), 只有 *融合算子* 不同。因此这个基线干净地回答:

    "HGRAG 的增益究竟来自多路召回 (多一个 KG 路 + 关键词通道),
     还是仅仅来自把 BM25 与 dense 融合在一起?"

若 RRF 已经很强, 则 HGRAG 的增益必须来自图谱路; 若 RRF 也很弱, 则
说明增益主要来自词法与语义的互补。

与其它基线的严格对齐
----------------------
  * 同一批 passage、同一批问题 (与 HGRAG / RAGFlow 完全相同)
  * 同一 Elasticsearch 索引 (hgrag4_shared_chunks, 同分词器、同标题 boost)
  * 同一 embedding 模型 (paraphrase-multilingual-MiniLM-L12-v2)
  * 同一评测口径 (文档级 Recall@k / Mean Rank)
  * 无任何调参: $K = 60$、$w_{bm25} = w_{vec} = 1.0$ 均为文献默认值

参考文献:
  G. V. Cormack, C. L. A. Clarke, and S. Buettcher,
  "Reciprocal rank fusion outperforms Condorcet and individual rank
  learning methods," SIGIR 2009.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import hgrag4_ragflow as RF

#: RRF 常数 K (Cormack et al. 2009 推荐的默认取值)
RRF_K = 60.0

#: 两路的权重 (默认均为 1.0 —— 文献默认值, 不做任何调参)
RRF_W_BM25 = 1.0
RRF_W_VEC = 1.0


class BM25DenseRRF:
    """
    BM25 + Dense 的 Reciprocal Rank Fusion 基线。

    两路候选:
      * 词法路: 与 RAGFlow / HGRAG 查询 *同一个* ES 索引的 BM25
      * 稠密路: 与 HGRAG 相同的 embedding 模型 + 同一段落向量

    融合: score(d) = w_bm25 / (K + rank_bm25(d)) + w_vec / (K + rank_vec(d))
    """

    def __init__(self, passages: Sequence[Dict[str, Any]],
                 embedder,
                 pid_emb: Optional[np.ndarray] = None,
                 shared_es: Optional[Any] = None,
                 es_url: Optional[str] = None,
                 top_pool: int = 50,
                 rrf_k: float = RRF_K,
                 w_bm25: float = RRF_W_BM25,
                 w_vec: float = RRF_W_VEC):
        self.embedder = embedder
        self.shared_es = shared_es
        self.es_url = es_url.rstrip("/") if es_url else None
        self.top_pool = top_pool
        self.rrf_k = float(rrf_k)
        self.w_bm25 = float(w_bm25)
        self.w_vec = float(w_vec)

        self.pid_to_title = {p["pid"]: p.get("title", "") for p in passages}
        self.pid_list = [p["pid"] for p in passages]

        # ---- 稠密路: 复用 HGRAG 的段落级向量 (同一 embedding 模型) ----
        if pid_emb is None:
            self.pid_emb = embedder.encode(
                [p.get("text", "") for p in passages], normalize=True)
        else:
            self.pid_emb = pid_emb

        # ---- 词法路: 共享 ES 索引 (首选) 或本地 BM25 ----
        self.bm25: Optional[RF.BM25Index] = None
        self.es_index = "rrf_chunks"
        self.es_client = None
        if shared_es is None:
            self.bm25 = RF.BM25Index(
                [RF.tokenize(p.get("text", "")) for p in passages])
        elif self.es_url:
            self._es_setup()

        self.pid_to_chunks: Dict[str, List[int]] = {}
        for i, p in enumerate(passages):
            self.pid_to_chunks.setdefault(p["pid"], []).append(i)

    # ------------------------------------------------------------------ #
    def _es_setup(self) -> None:
        """自建 ES 索引 (仅当没有共享索引时使用)。"""
        import requests

        sess = requests.Session()
        sess.delete(f"{self.es_url}/{self.es_index}", timeout=30)
        sess.put(f"{self.es_url}/{self.es_index}", timeout=30, json={
            "settings": {
                "index": {"number_of_shards": 1, "number_of_replicas": 0},
                "analysis": {"analyzer": {
                    "ragflow_en": {"type": "standard",
                                   "stopwords": "_english_"}}},
            },
            "mappings": {"properties": {
                "pid": {"type": "keyword"},
                "title": {"type": "text", "analyzer": "ragflow_en",
                          "fields": {"raw": {"type": "keyword"}}},
                "content": {"type": "text", "analyzer": "ragflow_en"},
            }},
        })
        bulk: List[str] = []
        for i, p in enumerate(self.pid_list):
            title = self.pid_to_title.get(p, "")
            bulk.append(json.dumps({"index": {"_index": self.es_index,
                                              "_id": str(i)}}))
            bulk.append(json.dumps({"pid": p, "title": title,
                                    "content": title}, ensure_ascii=False))
            if len(bulk) >= 1000:
                sess.post(f"{self.es_url}/_bulk", timeout=120,
                          data="\n".join(bulk) + "\n",
                          headers={"Content-Type": "application/x-ndjson"})
                bulk = []
        if bulk:
            sess.post(f"{self.es_url}/_bulk", timeout=120,
                      data="\n".join(bulk) + "\n",
                      headers={"Content-Type": "application/x-ndjson"})
        sess.post(f"{self.es_url}/{self.es_index}/_refresh", timeout=30)
        self.es_client = sess

    # ------------------------------------------------------------------ #
    def _bm25_scores(self, query: str) -> Dict[str, float]:
        """返回 {pid: bm25 分数} (共享 ES 优先)。"""
        if self.shared_es is not None:
            return self.shared_es.search_bm25(query, top_k=self.top_pool,
                                              mode="hybrid")
        if self.es_client is not None:
            query = {
                "size": self.top_pool,
                "query": {
                    "bool": {
                        "should": [
                            {"match": {"content": {"query": query,
                                                   "operator": "or"}}},
                            {"match": {"title": {"query": query,
                                                 "operator": "or",
                                                 "boost": 2.0}}},
                        ],
                        "minimum_should_match": 1,
                    }
                },
            }
            resp = self.es_client.post(
                f"{self.es_url}/{self.es_index}/_search", timeout=30,
                json=query)
            resp.raise_for_status()
            hits = resp.json().get("hits", {}).get("hits", [])
            out: Dict[str, float] = {}
            for h in hits:
                out[str(h["_source"]["pid"])] = float(h.get("_score", 0.0))
            return out
        raw = self.bm25.score(RF.tokenize(query))
        out = {}
        for ci, sc in raw.items():
            if 0 <= ci < len(self.pid_list):
                p = self.pid_list[ci]
                if sc > out.get(p, -1.0):
                    out[p] = sc
        return out

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: int = 10
                 ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        t0 = time.perf_counter()

        # ---------- 路 1: BM25 (共享 ES 索引) ----------
        t_k0 = time.perf_counter()
        bm = self._bm25_scores(query)
        t_k = time.perf_counter() - t_k0

        # ---------- 路 2: 稠密向量 ----------
        t_v0 = time.perf_counter()
        qv = self.embedder.encode([query], normalize=True)[0]
        vsims = self.pid_emb @ qv
        t_v = time.perf_counter() - t_v0

        # ---------- 各路排名 (1-based, 确定性 tie-break) ----------
        bm_order = sorted(bm, key=lambda p: (-bm[p], p))
        # 稠密路: 只保留 top_pool (稳定排序 + pid tie-break)
        idx = sorted(range(len(self.pid_list)),
                     key=lambda i: (-float(vsims[i]), self.pid_list[i]))
        vec_order = [self.pid_list[i] for i in idx[:self.top_pool]]
        rb = {p: i + 1 for i, p in enumerate(bm_order)}
        rv = {p: i + 1 for i, p in enumerate(vec_order)}

        # ---------- RRF 融合 ----------
        fused: Dict[str, float] = {}
        for p in set(rb) | set(rv):
            s = 0.0
            if p in rb:
                s += self.w_bm25 / (self.rrf_k + rb[p])
            if p in rv:
                s += self.w_vec / (self.rrf_k + rv[p])
            fused[p] = s

        ranked = sorted(fused, key=lambda p: (-fused[p], p))[:top_k]
        docs = [{"pid": p, "title": self.pid_to_title.get(p, ""),
                 "score": float(fused[p])} for p in ranked]

        stats = {
            "n_candidates": len(fused),
            "t_keyword": round(t_k, 4),
            "t_vector": round(t_v, 4),
            "t_total": round(time.perf_counter() - t0, 4),
        }
        return docs, stats

    def close(self) -> None:
        pass
