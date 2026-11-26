#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HGRAG v4 :: RAGFlow 基线 (同语料 / 同问题 / 同 embedding)
========================================================

为什么不是直接跑官方 RAGFlow
----------------------------
RAGFlow 官方发行版是一个完整服务栈: Elasticsearch (全文检索) + MinIO (对象存储)
+ MySQL/Redis + Python 后端 + 前端, 官方仅提供 docker-compose 部署。
本机环境无 Docker / WSL / Java / 管理员权限, 无法起完整服务。

因此本模块在 *检索算法层面* 忠实复刻 RAGFlow 的默认检索管线, 参数取自
官方文档 (ragflow.io/docs/retrieval_component, /configure_chunker_component):

  ┌ 切块 (Chunking) ─────────────────────────────────────────────┐
  │ 目标 512 token, 0% 重叠, 按分隔符切分 (\\n ! ? ; 。 ； ！ ？)     │
  │ 标题单独作为 chunk 头部                                        │
  └───────────────────────────────────────────────────────────────┘
  ┌ 索引 ─────────────────────────────────────────────────────────┐
  │ (a) 全文路: BM25 (与 RAGFlow 的 ES analyzer 同源)               │
  │ (b) 向量路: 余弦相似度 (与 HGRAG 共用同一 embedding 模型)        │
  └───────────────────────────────────────────────────────────────┘
  ┌ 检索 (RAGFlow 默认) ───────────────────────────────────────────┐
  │  hybrid_score = 0.7 * keyword_similarity                       │
  │               + 0.3 * vector_cosine_similarity                │
  │  similarity_threshold = 0.2   (低于该分数的 chunk 被过滤)        │
  │  无 rerank 模型 (RAGFlow 默认不启用)                            │
  └───────────────────────────────────────────────────────────────┘

若本机可用 Elasticsearch, `--use-elasticsearch` 会切到真实 ES/BM25;
否则用本地 BM25 实现 (Okapi BM25, k1=1.2, b=0.75, 与 ES 默认同参)。

为保证 *公平对照*, 本基线与 HGRAG 严格共享:
    * 同一语料 (passages.json)
    * 同一批问题 (splits.json 划分的 dev/test)
    * 同一 embedding 模型与同一 embedding 向量
    * 同一评测口径 (Recall@k / Precision@k / Mean Rank / 延迟)
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# RAGFlow 默认参数 (来源: 官方文档)
# ---------------------------------------------------------------------------
RF_VECTOR_WEIGHT = 0.3          # 向量相似度权重
RF_KEYWORD_WEIGHT = 0.7         # 关键词相似度权重 = 1 - vector_weight
RF_SIMILARITY_THRESHOLD = 0.2   # 相似度阈值
RF_TOP_N = 8                    # 送入 LLM 的 chunk 数
RF_CHUNK_TOKENS = 512           # 目标 chunk 长度
RF_CHUNK_OVERLAP = 0.0          # 重叠比例
RF_SEPARATORS = ["\n", "!", "?", ";", "。", "；", "！", "？"]


# ============================================================================
# 切块 (模拟 RAGFlow Token Chunker)
# ============================================================================
def _approx_tokens(text: str) -> int:
    """粗略 token 数: 英文按词, 中文按字, 其余按 4 字符/ token。"""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    words = len(re.findall(r"[A-Za-z0-9]+", text))
    other = max(0, len(text) - cjk - words)
    return cjk + words + other // 4


def chunk_passage(title: str, text: str,
                  max_tokens: int = RF_CHUNK_TOKENS,
                  separators: Sequence[str] = RF_SEPARATORS
                  ) -> List[Dict[str, Any]]:
    """
    把段落切成 RAGFlow 风格的 chunk。

    RAGFlow 的 chunk 文本形如 "title\\ncontent" (标题作为 chunk 的前缀),
    这样全文检索时标题词也能命中。
    """
    # 先按分隔符切
    parts: List[str] = []
    buf = text
    for sep in separators:
        if sep in buf:
            pieces = buf.split(sep)
            buf = pieces[0]
            parts.extend(p.strip() for p in pieces[1:] if p.strip())
    if buf.strip():
        parts.insert(0, buf.strip())
    if not parts:
        parts = [text.strip()]

    # 再按 token 上限合并
    chunks: List[str] = []
    cur = ""
    for p in parts:
        cand = (cur + " " + p).strip() if cur else p
        if cur and _approx_tokens(cand) > max_tokens:
            chunks.append(cur)
            cur = p
        else:
            cur = cand
    if cur:
        chunks.append(cur)

    return [{"title": title,
             "text": (f"{title}\n{c}" if title else c),
             "n_tokens": _approx_tokens(c)}
            for c in chunks]


# ============================================================================
# BM25 全文路 (Okapi BM25, 与 Elasticsearch 默认参数一致)
# ============================================================================
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> List[str]:
    """英文小写分词 + 中文单字切分 (对应 ES 的标准分析器 + CJK 单字)。"""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    """Okapi BM25。k1=1.2, b=0.75 与 Elasticsearch 默认一致。"""

    def __init__(self, docs: Sequence[Sequence[str]],
                 k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = max(1, len(docs))
        self.doc_len = [len(d) for d in docs]
        self.avgdl = sum(self.doc_len) / self.N if self.N else 0.0
        self.tf: List[Counter] = [Counter(d) for d in docs]
        df: Counter = Counter()
        for t in self.tf:
            df.update(t.keys())
        self.idf: Dict[str, float] = {
            w: math.log(1.0 + (self.N - n + 0.5) / (n + 0.5))
            for w, n in df.items()
        }
        self.postings: Dict[str, List[int]] = defaultdict(list)
        for i, t in enumerate(self.tf):
            for w in t:
                self.postings[w].append(i)

    def score(self, query_tokens: Sequence[str]) -> Dict[int, float]:
        scores: Dict[int, float] = defaultdict(float)
        # 按词元排序遍历: (a) 集合迭代顺序受哈希随机化影响;
        # (b) 浮点加法不满足结合律, 累加顺序不同会产生末位差异。
        for w in sorted(set(query_tokens)):
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i in self.postings.get(w, ()):
                f = self.tf[i][w]
                if f == 0:
                    continue
                denom = f + self.k1 * (
                    1.0 - self.b + self.b * self.doc_len[i] / max(self.avgdl, 1e-9))
                scores[i] += idf * (f * (self.k1 + 1.0)) / denom
        return scores


# ============================================================================
# RAGFlow 基线检索器
# ============================================================================
class RAGFlowBaseline:
    """
    RAGFlow 默认检索管线 (GraphRAG 关闭, 即纯 hybrid chunk retrieval)。

    "Use knowledge graph" 开关在 RAGFlow 中默认关闭, 且会 *极大* 增加延迟
    (官方文档: "greatly increasing retrieval time")。原论文的对照正是
    RAGFlow 的 "default GraphRAG configuration"; 这里两者都提供:
        use_knowledge_graph=False  (默认, 与 RAGFlow 出厂行为一致)
        use_knowledge_graph=True   (开启 RAGFlow 的知识图谱检索块)
    """

    def __init__(self, passages: Sequence[Dict[str, Any]],
                 embedder,
                 pid_emb: Optional[np.ndarray] = None,
                 vector_weight: float = RF_VECTOR_WEIGHT,
                 keyword_weight: float = RF_KEYWORD_WEIGHT,
                 threshold: float = RF_SIMILARITY_THRESHOLD,
                 top_pool: int = 50,
                 use_knowledge_graph: bool = False,
                 kg_chunks: Optional[Sequence[Dict[str, Any]]] = None,
                 es_url: Optional[str] = None,
                 shared_es: Optional[Any] = None):
        self.embedder = embedder
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.threshold = threshold
        self.top_pool = top_pool
        self.use_knowledge_graph = use_knowledge_graph
        self.es_url = es_url
        self.es_index = "ragflow_chunks"
        self.es_client = None
        # 与 HGRAG 共用的 ES 索引 (v4 新增) —— 保证两侧词法能力严格相同
        self.shared_es = shared_es

        # ---------- 切块 ----------
        self.chunks: List[Dict[str, Any]] = []
        for p in passages:
            for c in chunk_passage(p.get("title", ""), p.get("text", "")):
                c["pid"] = p["pid"]
                c["title"] = p.get("title", "")
                c["text_orig"] = p.get("text", "")
                self.chunks.append(c)
        self.chunk_pid = [c["pid"] for c in self.chunks]
        self.chunk_title = [c["title"] for c in self.chunks]
        self.chunk_text = [c["text"] for c in self.chunks]
        self.pid_to_title = {p["pid"]: p.get("title", "") for p in passages}

        # ---------- 全文索引 ----------
        self.bm25 = BM25Index([tokenize(t) for t in self.chunk_text])

        # ---------- 向量索引 ----------
        if pid_emb is None:
            self.chunk_emb = embedder.encode(self.chunk_text, normalize=True)
        else:
            # 复用 HGRAG 的段落级向量: chunk 向量 = 其所属段落的向量
            pid_index = {p["pid"]: i for i, p in enumerate(passages)}
            self.chunk_emb = np.stack(
                [pid_emb[pid_index[c["pid"]]] for c in self.chunks])

        # ---------- 知识图谱检索块 (可选) ----------
        self.kg_chunks = list(kg_chunks or [])

        # ---------- ES 后端 (可选) ----------
        if self.es_url:
            self._es_setup()

        # pid -> 该 pid 命中的最好 chunk 分数 (评测按 *文档* 计)
        self.pid_to_chunks: Dict[str, List[int]] = defaultdict(list)
        for i, c in enumerate(self.chunks):
            self.pid_to_chunks[c["pid"]].append(i)

    # ------------------------------------------------------------------ #
    # Elasticsearch 后端 (RAGFlow 实际使用的全文检索引擎)
    # ------------------------------------------------------------------ #
    def _es_setup(self) -> None:
        """在 ES 中建立 RAGFlow 风格的 chunk 索引。"""
        import requests

        sess = requests.Session()
        url = self.es_url.rstrip("/")
        # 删除旧索引 (幂等)
        sess.delete(f"{url}/{self.es_index}", timeout=30)
        sess.put(f"{url}/{self.es_index}", timeout=30, json={
            "settings": {
                "index": {"number_of_shards": 1, "number_of_replicas": 0},
                "analysis": {
                    "analyzer": {
                        "ragflow_en": {"type": "standard", "stopwords": "_english_"},
                        "ragflow_cjk": {"type": "standard",
                                        "tokenizer": "standard"},
                    }
                },
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "integer"},
                    "pid": {"type": "keyword"},
                    "title": {"type": "text", "analyzer": "ragflow_en",
                              "fields": {"raw": {"type": "keyword"}}},
                    "content": {"type": "text", "analyzer": "ragflow_en"},
                }
            },
        })
        # 批量写入
        bulk: List[str] = []
        for i, c in enumerate(self.chunks):
            bulk.append(json.dumps({"index": {"_index": self.es_index,
                                              "_id": str(i)}}))
            bulk.append(json.dumps({
                "chunk_id": i, "pid": c["pid"], "title": c["title"],
                "content": c["text"]}, ensure_ascii=False))
            if len(bulk) >= 1000:
                sess.post(f"{url}/_bulk", timeout=120,
                          data="\n".join(bulk) + "\n",
                          headers={"Content-Type": "application/x-ndjson"})
                bulk = []
        if bulk:
            sess.post(f"{url}/_bulk", timeout=120,
                      data="\n".join(bulk) + "\n",
                      headers={"Content-Type": "application/x-ndjson"})
        sess.post(f"{url}/{self.es_index}/_refresh", timeout=30)
        self.es_client = sess

    def _es_search(self, query: str, top_k: int
                   ) -> Dict[int, float]:
        """用 ES 跑 BM25 检索, 返回 {chunk_idx: bm25_score}。"""
        resp = self.es_client.post(
            f"{self.es_url.rstrip('/')}/{self.es_index}/_search",
            timeout=30,
            json={
                "size": top_k,
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
            })
        resp.raise_for_status()
        hits = resp.json().get("hits", {}).get("hits", [])
        out: Dict[int, float] = {}
        for h in hits:
            out[int(h["_id"])] = float(h.get("_score", 0.0))
        return out

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: int = 10
                 ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        t0 = time.perf_counter()

        # ---------- (a) 关键词路: BM25 ----------
        q_tokens = tokenize(query)
        t_k0 = time.perf_counter()
        if self.shared_es is not None:
            # 与 HGRAG 查询 *同一个* ES 索引 —— 两侧词法能力严格相同,
            # 差值干净地归因到图谱路与融合方式。
            bm = self.shared_es.search_bm25(query, top_k=self.top_pool,
                                            mode="hybrid")
        elif self.es_client is not None:
            bm = self._es_search(query, self.top_pool)
        else:
            bm = self.bm25.score(q_tokens)
        t_k = time.perf_counter() - t_k0

        # ---------- (b) 向量路: 余弦 ----------
        t_v0 = time.perf_counter()
        qv = self.embedder.encode([query], normalize=True)[0]
        vsims = self.chunk_emb @ qv
        t_v = time.perf_counter() - t_v0

        # ---------- 归一 + 混合 ----------
        # RAGFlow 用 ES 的 BM25 分数做 min-max 归一, 与向量余弦加权求和
        max_bm = max(bm.values()) if bm else 0.0
        max_v = float(np.max(vsims)) if len(vsims) else 0.0
        min_v = float(np.min(vsims)) if len(vsims) else 0.0
        v_range = max(max_v - min_v, 1e-9)

        scored: List[Dict[str, Any]] = []
        if self.shared_es is not None:
            # ---- 共享 ES 模式: BM25 以 pid 为键 ----
            bm_pid = bm
            # 向量路按 pid 取该 pid 的最好 chunk 余弦
            pid_vec: Dict[str, float] = {}
            for i, p in enumerate(self.chunk_pid):
                v = float(vsims[i])
                if v > pid_vec.get(p, -1e9):
                    pid_vec[p] = v
            for pid in sorted(set(bm_pid) | set(pid_vec)):
                kw_n = (bm_pid.get(pid, 0.0) / max_bm) if max_bm > 0 else 0.0
                v_n = float((pid_vec.get(pid, min_v) - min_v) / v_range)
                score = (self.keyword_weight * kw_n
                         + self.vector_weight * v_n)
                if score < self.threshold:
                    continue
                scored.append({"chunk_idx": -1, "pid": pid,
                               "title": self.pid_to_title.get(pid, ""),
                               "score": float(score),
                               "kw": float(kw_n), "vec": float(v_n)})
            n_cand = len(set(bm_pid) | set(pid_vec))
        else:
            cand = set(bm.keys())
            if len(vsims) > self.top_pool:
                cand |= set(np.argpartition(-vsims, self.top_pool)
                            [:self.top_pool].tolist())
            else:
                cand |= set(range(len(vsims)))
            n_cand = len(cand)
            for i in sorted(cand):
                kw_n = (bm.get(i, 0.0) / max_bm) if max_bm > 0 else 0.0
                v_n = float((vsims[i] - min_v) / v_range)
                score = (self.keyword_weight * kw_n
                         + self.vector_weight * v_n)
                if score < self.threshold:
                    continue
                scored.append({"chunk_idx": i, "pid": self.chunk_pid[i],
                               "title": self.chunk_title[i],
                               "score": float(score),
                               "kw": float(kw_n), "vec": float(v_n)})
        # 排序必须是确定性的: 下面的 sort 是稳定排序, 同分文档的相对顺序
        # 由入参顺序决定, 因此入参顺序必须固定 (否则每次运行结果会漂移)。
        scored.sort(key=lambda d: (-d["score"], str(d["pid"]),
                                   int(d["chunk_idx"])))

        # ---------- 知识图谱块 (可选) ----------
        if self.use_knowledge_graph and self.kg_chunks:
            for kc in self.kg_chunks:
                scored.append({"chunk_idx": -1, "pid": kc["pid"],
                               "title": kc.get("title", ""),
                               "score": float(kc.get("score", 0.0)),
                               "kw": 0.0, "vec": 0.0, "from_kg": True})
            scored.sort(key=lambda d: (-d["score"], str(d["pid"]),
                                       int(d["chunk_idx"])))

        # ---------- 去重到文档级 ----------
        # RAGFlow 的 top-N 是 chunk 级; 但 RAGFlow 在返回给用户时保留 chunk,
        # 我们在评测时按 *文档* 去重 (保留最高分 chunk), 与 HGRAG 的文档级
        # 输出对齐, 保证 recall 口径一致。
        seen: Dict[str, Dict[str, Any]] = {}
        for d in scored:
            if d["pid"] in seen:
                continue
            seen[d["pid"]] = d
            if len(seen) >= self.top_pool:
                break
        docs = sorted(seen.values(),
                      key=lambda d: (-d["score"], str(d["pid"])))[:top_k]

        stats = {
            "n_chunks": len(self.chunks),
            "n_candidates": n_cand,
            "n_above_threshold": len(scored),
            "t_keyword": round(t_k, 4),
            "t_vector": round(t_v, 4),
            "t_total": round(time.perf_counter() - t0, 4),
        }
        return docs, stats

    def close(self) -> None:
        pass
