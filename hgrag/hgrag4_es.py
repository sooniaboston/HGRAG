#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HGRAG v4 :: 共享 Elasticsearch 全文后端
=====================================

为什么需要这个模块
------------------
v4 首轮对照里, HGRAG 用本地 Okapi BM25, 而 RAGFlow 基线用真实
Elasticsearch。两者的分词器、索引结构、标题 boost 策略并不完全一致,
于是 "HGRAG vs RAGFlow" 的差值里混入了 **词法实现的差异**,
无法干净地回答论文真正关心的问题:

    在 *同等* 的词法检索能力下, 叠加 KG 激活扩散路能带来多少增益?

本模块让 HGRAG 与 RAGFlow **查询同一个 ES 索引** (同一分词器、同一
mappings、同一标题 boost), 只在融合方式上不同。这样两者的差值就
干净地归因到 "图谱路 + 多路融合" 上。

索引结构 (与 RAGFlow 的 chunk 索引一致):
    * content: title + 正文 (与 RAGFlow 的 chunk 文本一致)
    * title : 单独字段, 查询时 boost (与 RAGFlow 的 ES 查询一致)
    * pid   : keyword, 保留原文段落 id 以便回连

两套检索模式 (供消融):
    * "hybrid" (默认) —— RAGFlow 的原始查询:
        should: [ match(content, operator=or),
                  match(title,   operator=or, boost=2.0) ]
    * "content"      —— 只查 content, 不做标题 boost。
        用于消融 "标题 boost 贡献了多少 R@1"。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

import hgrag4_ragflow as RF

#: 默认 ES 查询模式
DEFAULT_ES_INDEX = "hgrag4_shared_chunks"

#: 标题字段 boost (与 RAGFlow 基线一致)
TITLE_BOOST = 2.0


class SharedESIndex:
    """
    HGRAG 与 RAGFlow 共用的 Elasticsearch 全文索引。

    **索引粒度 = chunk** (与 RAGFlow 完全一致: 用 RAGFlow 自己的
    `chunk_passage` 切分, 512 token / 0% 重叠 / 标题作 chunk 前缀),
    查询后两侧都按 *文档* 去重。这样两侧看到的是同一批 chunk、同一个
    分词器、同一套标题 boost, 差值只剩 "图谱路 + 融合方式"。

    用法:
        idx = SharedESIndex(es_url, passages)
        idx.build()                                   # 建索引 (幂等)
        idx.search_bm25("query", top_k=50)           # -> {pid: score}
        idx.close()
    """

    def __init__(self, es_url: str, passages: Sequence[Dict[str, Any]],
                 index_name: str = DEFAULT_ES_INDEX,
                 title_boost: float = TITLE_BOOST):
        self.es_url = es_url.rstrip("/")
        self.index_name = index_name
        self.title_boost = title_boost

        # ---------- 切块 (用 RAGFlow 自己的切分器, 保证粒度一致) ----------
        self.chunks: List[Dict[str, Any]] = []
        for p in passages:
            for c in RF.chunk_passage(p.get("title", ""), p.get("text", "")):
                c["pid"] = p["pid"]
                c["title"] = p.get("title", "")
                self.chunks.append(c)
        self.chunk_pid = [c["pid"] for c in self.chunks]
        self.pid_to_title = {p["pid"]: p.get("title", "") for p in passages}

        self._session = None
        self._built = False

    # ------------------------------------------------------------------ #
    def _sess(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    # ------------------------------------------------------------------ #
    def build(self) -> None:
        """建立 (或重建) 索引。幂等。"""
        sess = self._sess()
        sess.delete(f"{self.es_url}/{self.index_name}", timeout=30)
        sess.put(f"{self.es_url}/{self.index_name}", timeout=30, json={
            "settings": {
                "index": {"number_of_shards": 1, "number_of_replicas": 0},
                "analysis": {
                    "analyzer": {
                        "ragflow_en": {"type": "standard",
                                       "stopwords": "_english_"},
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
        bulk: List[str] = []
        for i, c in enumerate(self.chunks):
            bulk.append(json.dumps({"index": {"_index": self.index_name,
                                              "_id": str(i)}}))
            bulk.append(json.dumps({
                "chunk_id": i, "pid": c["pid"], "title": c["title"],
                "content": c["text"]}, ensure_ascii=False))
            if len(bulk) >= 1000:
                sess.post(f"{self.es_url}/_bulk", timeout=120,
                          data="\n".join(bulk) + "\n",
                          headers={"Content-Type": "application/x-ndjson"})
                bulk = []
        if bulk:
            sess.post(f"{self.es_url}/_bulk", timeout=120,
                      data="\n".join(bulk) + "\n",
                      headers={"Content-Type": "application/x-ndjson"})
        sess.post(f"{self.es_url}/{self.index_name}/_refresh", timeout=30)
        self._built = True

    # ------------------------------------------------------------------ #
    def search_bm25(self, query: str, top_k: int = 50,
                    mode: str = "hybrid") -> Dict[str, float]:
        """
        查询全文索引, 返回 {pid: 该 pid 的最好 chunk 分}。

        mode:
          "hybrid" —— content 匹配 OR title 匹配(boost=title_boost)
                      (与 RAGFlow 基线的 ES 查询完全一致)
          "content" —— 只匹配 content (无标题 boost), 用于消融
        """
        sess = self._sess()
        if mode == "content":
            should = [{"match": {"content": {"query": query,
                                             "operator": "or"}}}]
        else:
            should = [
                {"match": {"content": {"query": query, "operator": "or"}}},
                {"match": {"title": {"query": query, "operator": "or",
                                     "boost": self.title_boost}}},
            ]
        resp = sess.post(
            f"{self.es_url}/{self.index_name}/_search", timeout=30,
            json={"size": top_k,
                  "query": {"bool": {"should": should,
                                     "minimum_should_match": 1}}})
        resp.raise_for_status()
        hits = resp.json().get("hits", {}).get("hits", [])
        out: Dict[str, float] = {}
        for h in hits:
            pid = h["_source"]["pid"]
            sc = float(h.get("_score", 0.0))
            if sc > out.get(pid, -1.0):
                out[pid] = sc
        return out

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None


def try_connect(es_url: str, timeout: float = 5.0) -> Optional[SharedESIndex]:
    """尝试连接 ES; 连不上返回 None (调用方回退到本地 BM25)。"""
    try:
        import requests
        r = requests.get(es_url.rstrip("/"), timeout=timeout)
        if r.status_code == 200:
            return SharedESIndex(es_url, [])
        return None
    except Exception:
        return None
