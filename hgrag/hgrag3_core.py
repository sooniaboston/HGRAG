# -*- coding: utf-8 -*-
"""
HGRAG 3.0 —— 双路检索核心库
================================

实现论文 `access (3).tex` 描述的 HGRAG 架构, 并落实两项新增机制:

【A. 双路必返 (Dual-Path Guaranteed Top-k)】
    Milvus 稠密向量路与 KG 激活扩散路 *各自* 必须返回 top_k 项,
    进入保证集 G; 其余候选文档按融合分排序补足到最终 top_k。
    这比论文中的 "Bottom-up Top-k (N=2)" 更强:
    论文用 N=2 的小保证集, 本实现用可调 guarantee_k (默认 = 路径数 × 2)。

【B. 关键词检索打分 (Keyword-Aware Scoring)】
    1) 位置 + 数量加分:
       关键词在文档中出现的位置越靠前、次数越多, 加分越高。
           s_pos   = 1 / (1 + ln(1 + 首次出现位置))     (位置衰减)
           s_count = min(count / saturate, 1)            (数量饱和)
       基础分  s_kw = w_pos · s_pos + w_cnt · s_count
    2) 实体关联概念加权:
       若关键词是知识图谱中的实体, 则与该实体相关的 *概念*
       (共现邻居 / 图邻接实体) 按其关联权重 w_assoc 累加:
           s_assoc = Σ_{c ∈ concept(κ)} w(c) · [κ 或 c 出现在文档]
       文档最终关键词分  s_kw_total = s_kw + w_assoc · s_assoc

【C. 融合】
    s_hyb(d) = w_v · ŝ_v(d) + w_g · ŝ_g(d) + w_kw · ŝ_kw(d)
               + η · 1[双路命中]
    ŝ 表示 min-max 归一化分数。

【D. 索引后端】
    Milvus 后端使用 milvus-lite (嵌入式, HNSW 索引), 无需独立服务;
    同时提供纯 numpy 的精确检索作为 `--milvus-backend exact` 备选,
    用于在无 Milvus 环境下做可复现的对照实验。
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

LOG = logging.getLogger("hgrag3")


# ============================================================================
# 工具
# ============================================================================
def legal_normalize(name: str) -> str:
    """实体名归一化: 小写 + 去括号 + 去标点 + 压缩空白。"""
    if not name:
        return ""
    s = name.strip().lower()
    s = re.sub(r"\s*\([^)]*\)", " ", s)          # 去括号补充
    s = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", s)  # 去标点
    s = re.sub(r"\s+", " ", s).strip()
    return s


def sig_tokens(name: str) -> List[str]:
    """显著词元: 长度 >= 2 的词, 去掉停用词。"""
    stop = {"the", "a", "an", "of", "and", "or", "in", "on", "at", "to",
            "for", "by", "with", "from", "as", "is", "was", "were", "be",
            "been", "are", "that", "this", "it", "its"}
    out = []
    for t in legal_normalize(name).split():
        if len(t) >= 2 and t not in stop:
            out.append(t)
    return out


def minmax_norm(scores: Dict[str, float]) -> Dict[str, float]:
    """min-max 归一化到 [0,1]; 全部相等时返回 1.0。"""
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


# ============================================================================
# 1. Milvus 向量库 (milvus-lite / HNSW)
# ============================================================================
class MilvusVectorStore:
    """
    嵌入式 Milvus (milvus-lite) 向量库封装。

    - 索引: HNSW (M=16, efConstruction=200), metric = IP (内积, 向量已 L2 归一化
      故等价于余弦)
    - 动态字段保存 pid / title / text, 便于检索时取回原文
    """

    def __init__(self, db_path: str, collection: str = "hgrag_passages",
                 dim: int = 384, index_type: str = "HNSW",
                 m: int = 16, ef_construction: int = 200,
                 ef_search: int = 128):
        self.db_path = db_path
        self.collection = collection
        self.dim = dim
        self.index_type = index_type
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.client = None
        self.n_inserted = 0

    # ------------------------------------------------------------------ #
    def open(self) -> "MilvusVectorStore":
        from pymilvus import MilvusClient

        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        uri = self.db_path.replace("\\", "/")
        self.client = MilvusClient(uri=uri)
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            dimension=self.dim,
            primary_field_name="pid",
            id_type="string",
            max_length=64,
            vector_field_name="vector",
            metric_type="IP",
            auto_id=False,
            index_type=self.index_type,
            enable_dynamic_field=True,
            params={"M": self.m, "efConstruction": self.ef_construction},
        )
        LOG.info("Milvus 集合已建: %s (dim=%d, index=%s, M=%d)",
                 self.collection, self.dim, self.index_type, self.m)
        return self

    # ------------------------------------------------------------------ #
    def insert(self, pids: Sequence[str], vectors: np.ndarray,
               titles: Sequence[str], texts: Sequence[str]) -> None:
        assert len(pids) == len(titles) == len(texts) == vectors.shape[0]
        rows = []
        for i, pid in enumerate(pids):
            rows.append({
                "pid": str(pid),
                "vector": [float(x) for x in vectors[i]],
                "title": titles[i],
                "text": texts[i],
            })
        self.client.insert(self.collection, rows)
        self.n_inserted = len(rows)
        self.client.flush(self.collection)
        self.client.load_collection(self.collection)
        LOG.info("Milvus 插入 %d 条, 索引已加载", self.n_inserted)

    # ------------------------------------------------------------------ #
    def search(self, query_vec: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """返回 [{'pid','title','text','score'}, ...] 按分数降序。"""
        if self.client is None:
            raise RuntimeError("Milvus 未打开")
        hits = self.client.search(
            self.collection,
            [query_vec.tolist()],
            limit=top_k,
            output_fields=["title", "text"],
            search_params={"ef": self.ef_search},
        )[0]
        out = []
        for h in hits:
            # pymilvus 3.x: 主键在 h.id, 输出字段在 h.fields
            fields = getattr(h, "fields", None) or {}
            out.append({
                "pid": str(h.id),
                "title": fields.get("title", ""),
                "text": fields.get("text", ""),
                "score": float(h.distance),
            })
        return out

    def close(self) -> None:
        if self.client is not None:
            try:
                self.client.close()
            except Exception:  # noqa: BLE001
                pass
            self.client = None


class ExactVectorStore:
    """
    纯 numpy 精确检索 (IP 余弦), 作为 milvus 不可用时的等价对照。

    语料仅 991 段, 精确检索与 HNSW 的召回差异可忽略, 但延迟更高,
    因此正好用于量化 "HNSW 近似检索 vs 精确检索" 的代价。
    """

    def __init__(self, pids: Sequence[str], vectors: np.ndarray,
                 titles: Sequence[str], texts: Sequence[str]):
        self.pids = list(pids)
        self.vectors = vectors
        # titles/texts 允许为空列表 (调用方可能只关心 pid), 此时退化为空串
        self.titles = list(titles) if titles is not None else []
        self.texts = list(texts) if texts is not None else []
        self.index: Dict[str, int] = {p: i for i, p in enumerate(self.pids)}

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        sims = self.vectors @ query_vec
        k = min(top_k, len(sims))
        if k <= 0:
            return []
        idx = np.argpartition(-sims, kth=k - 1)[:k]
        # 确定性 tie-break: 同分向量的相对次序不能依赖 argpartition 的
        # 非稳定返回顺序, 否则两次运行的向量路结果会不同。
        idx = sorted(idx, key=lambda i: (-sims[i], self.pids[int(i)]))

        def _get(seq, i):
            return seq[i] if i < len(seq) else ""

        return [{"pid": self.pids[i], "title": _get(self.titles, i),
                 "text": _get(self.texts, i), "score": float(sims[i])}
                for i in idx]


# ============================================================================
# 2. 关键词检索打分器
# ============================================================================
class KeywordScorer:
    """
    关键词检索打分 (论文需求 2)。

    打分项:
      (a) 位置分 s_pos    —— 关键词首次出现位置越靠前越高
      (b) 数量分 s_count  —— 关键词出现次数越多越高 (带饱和)
      (c) 关联概念分 s_assoc —— 关键词若是实体, 其图邻接概念按权重累加
    """

    def __init__(self,
                 kg_entities: List[Dict[str, Any]],
                 kg_relations: List[Dict[str, Any]],
                 passages: List[Dict[str, Any]],
                 w_pos: float = 0.35,
                 w_count: float = 0.35,
                 w_assoc: float = 0.30,
                 count_saturate: int = 5,
                 assoc_hops: int = 1,
                 assoc_top_m: int = 10,
                 assoc_rel_types: Optional[Set[str]] = None):
        self.w_pos = w_pos
        self.w_count = w_count
        self.w_assoc = w_assoc
        self.count_saturate = max(1, count_saturate)
        self.assoc_hops = assoc_hops
        self.assoc_top_m = assoc_top_m

        self.passages = passages
        self.pid_text: Dict[str, str] = {}
        self.pid_lower: Dict[str, str] = {}
        for p in passages:
            self.pid_text[p["pid"]] = f"{p.get('title','')}. {p.get('text','')}"
            self.pid_lower[p["pid"]] = self.pid_text[p["pid"]].lower()

        # ---------- 实体索引 ----------
        self.ent_by_eid: Dict[str, Dict[str, Any]] = {}
        self.name_to_eids: Dict[str, List[str]] = defaultdict(list)
        for e in kg_entities:
            eid = e.get("eid")
            if not eid:
                continue
            self.ent_by_eid[eid] = e
            for nm in [e.get("name", "")] + list(e.get("aliases", [])):
                key = legal_normalize(nm)
                if key:
                    self.name_to_eids[key].append(eid)

        # ---------- 实体 -> 段落 ----------
        self.ent_pids: Dict[str, Set[str]] = defaultdict(set)
        for e in kg_entities:
            eid = e.get("eid")
            for pid in e.get("source_pids", []) or []:
                self.ent_pids[eid].add(pid)

        # ---------- 概念邻接 (带关联权重) ----------
        # 关联权重来自三元组类型的先验 + 端点信息量, 反映"概念关联强度"
        self.concepts: Dict[str, List[Tuple[str, float]]] = {}
        self._build_concepts(kg_relations, kg_entities, assoc_rel_types)

        # ---------- 段落 -> 出现的实体 (用于概念落地) ----------
        self.pid_ents: Dict[str, Set[str]] = defaultdict(set)
        for eid, pids in self.ent_pids.items():
            for pid in pids:
                self.pid_ents[pid].add(eid)

    # ------------------------------------------------------------------ #
    def _build_concepts(self, relations, entities, rel_filter) -> None:
        """
        构建 实体 -> [(概念实体, 关联权重)] 的邻接表。

        关联权重 w(c) 综合:
          - 关系类型的重要性 (is_a / part_of 视为强概念关联)
          - 端点实体的区分度 (IDF: 越罕见的概念越有信息量)
        """
        # 实体文档频率 -> IDF
        n_pass = max(1, len(self.passages))
        self.ent_idf: Dict[str, float] = {}
        for eid, pids in self.ent_pids.items():
            df = len(pids)
            self.ent_idf[eid] = math.log((1.0 + n_pass) / (1.0 + df)) + 1.0

        rel_w = {
            "is a": 1.0, "isa": 1.0, "is_a": 1.0, "subclass of": 1.0,
            "part of": 0.9, "part_of": 0.9, "has part": 0.85,
            "instance of": 0.9, "type of": 0.8,
            "directed by": 0.6, "written by": 0.6, "starred in": 0.55,
            "released in": 0.5, "located in": 0.6, "country": 0.6,
            "capital of": 0.6, "occupation": 0.7, "genre": 0.6,
            "related to": 0.35, "mentions": 0.3, "appears in": 0.4,
        }
        adj: Dict[str, Dict[str, float]] = defaultdict(dict)
        for r in relations:
            s, t = r.get("source"), r.get("target")
            if not s or not t or s not in self.ent_by_eid or t not in self.ent_by_eid:
                continue
            rtype = (r.get("type") or "").strip().lower()
            if rel_filter and rtype not in rel_filter:
                continue
            base = rel_w.get(rtype, 0.5)
            # 双向邻接, 权重按端点 IDF 调和
            idf_s = self.ent_idf.get(s, 1.0)
            idf_t = self.ent_idf.get(t, 1.0)
            w = base * (1.0 + 0.15 * min(idf_s, idf_t))
            adj[s][t] = max(adj[s].get(t, 0.0), w)
            adj[t][s] = max(adj[t].get(s, 0.0), w)

        for eid, nbrs in adj.items():
            # 确定性 tie-break: 同权重的邻居顺序必须固定 (见 retrieve() 注释)
            items = sorted(nbrs.items(),
                           key=lambda kv: (-kv[1], kv[0]))[: self.assoc_top_m]
            self.concepts[eid] = items

    # ------------------------------------------------------------------ #
    def match_entities(self, keywords: Sequence[str]) -> List[str]:
        """把关键词匹配到图谱实体 eid (支持精确名/别名匹配)。"""
        eids: List[str] = []
        for kw in keywords:
            key = legal_normalize(kw)
            if not key:
                continue
            for eid in self.name_to_eids.get(key, []):
                if eid not in eids:
                    eids.append(eid)
        return eids

    # ------------------------------------------------------------------ #
    def score_document(self, pid: str, keywords: Sequence[str],
                       kw_eids: Sequence[str]) -> Dict[str, Any]:
        """
        计算单个文档的关键词分。

        返回 {'kw_score', 'pos', 'count', 'assoc', 'hits'}。
        """
        text = self.pid_lower.get(pid, "")
        if not text or not keywords:
            return {"kw_score": 0.0, "pos": 0.0, "count": 0.0,
                    "assoc": 0.0, "hits": []}

        pos_parts: List[float] = []
        cnt_parts: List[float] = []
        hits: List[Dict[str, Any]] = []

        for kw in keywords:
            k = kw.lower().strip()
            if not k:
                continue
            first = text.find(k)
            if first < 0:
                continue
            n = text.count(k)
            # 位置分: 首次出现越靠前越高 (相对文档长度归一, 再取倒数衰减)
            rel_pos = first / max(len(text), 1)
            s_pos = 1.0 / (1.0 + 3.0 * rel_pos)
            # 数量分: 饱和
            s_cnt = min(n / self.count_saturate, 1.0)
            pos_parts.append(s_pos)
            cnt_parts.append(s_cnt)
            hits.append({"keyword": kw, "count": n, "first_pos": first,
                         "rel_pos": round(rel_pos, 4)})

        if not hits:
            s_pos_avg = s_cnt_avg = 0.0
        else:
            s_pos_avg = float(np.mean(pos_parts))
            s_cnt_avg = float(np.mean(cnt_parts))

        # ---------- 关联概念分 ----------
        # 关键词实体 e 的每个关联概念 c 权重 w(c);
        # 若文档包含 e 或 c, 则累加 w(c) (归一化到 [0,1])
        s_assoc = 0.0
        if kw_eids:
            doc_ents = self.pid_ents.get(pid, set())
            acc: Dict[str, float] = {}
            for eid in kw_eids:
                for c, w in self.concepts.get(eid, []):
                    if c in doc_ents or eid in doc_ents:
                        acc[c] = max(acc.get(c, 0.0), w)
            if acc:
                # 归一: 除以该关键词实体的最大概念权重
                max_w = max(
                    (w for eid in kw_eids
                     for _, w in self.concepts.get(eid, [])), default=1.0)
                s_assoc = sum(acc.values()) / (max_w * max(len(kw_eids), 1))
                s_assoc = min(s_assoc, 1.0)

        kw_score = (self.w_pos * s_pos_avg
                    + self.w_count * s_cnt_avg
                    + self.w_assoc * s_assoc)
        return {"kw_score": kw_score, "pos": s_pos_avg, "count": s_cnt_avg,
                "assoc": s_assoc, "hits": hits}

    # ------------------------------------------------------------------ #
    def score(self, keywords: Sequence[str],
              candidate_pids: Optional[Sequence[str]] = None
              ) -> Dict[str, Dict[str, Any]]:
        """对候选段落批量打分。"""
        if not keywords:
            return {}
        kw_eids = self.match_entities(keywords)
        pids = list(candidate_pids) if candidate_pids is not None else list(self.pid_text)
        out: Dict[str, Dict[str, Any]] = {}
        for pid in pids:
            if pid not in self.pid_text:
                continue
            out[pid] = self.score_document(pid, keywords, kw_eids)
        return out


# ============================================================================
# 3. 知识图谱激活扩散路
# ============================================================================
class KGPathRetriever:
    """
    KG 结构检索路: 复用 activation_graphrag 的激活扩散网络,
    做查询时激活扩散并按 IDF 加权聚合到段落。

    与论文的 R_g 差异: 论文用 TF-IDF + BM25 + 实体重叠 + PageRank +
    betweenness 的手工线性组合; 本实现用 Collins-Loftus 激活扩散,
    把图拓扑变成 *可微的激活流* 而非静态加权和。
    """

    def __init__(self, network: Dict[str, Any], passages: List[Dict[str, Any]],
                 embedder=None, seed_alpha: float = 0.9,
                 lexical_weight: float = 0.3,
                 seed_entities: int = 3, use_idf: bool = True,
                 idf_power: float = 1.0, norm_beta: float = 0.5,
                 dense_fallback: bool = True, dense_weight: float = 0.1,
                 title_bonus: float = 0.1, decay: float = 0.6,
                 max_hops: int = 2, firing_threshold: float = 0.001):
        self.network = network
        self.embedder = embedder
        self.seed_alpha = seed_alpha
        self.lexical_weight = lexical_weight
        self.k_e = seed_entities
        self.use_idf = use_idf
        self.idf_power = idf_power
        self.norm_beta = norm_beta
        self.dense_fallback = dense_fallback
        self.dense_weight = dense_weight
        self.title_bonus = title_bonus
        self.decay = decay
        self.max_hops = max_hops
        self.threshold = firing_threshold

        self.nodes = network["nodes"]
        self.adj: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for e in network["edges"]:
            self.adj[e["source"]].append((e["target"], e["weight"]))
            self.adj[e["target"]].append((e["source"], e["weight"]))

        self.ent_pids: Dict[str, Set[str]] = defaultdict(set)
        for n in self.nodes:
            for pid in n.get("source_pids", []):
                self.ent_pids[n["eid"]].add(pid)

        self.name_index: Dict[str, str] = {}
        for n in self.nodes:
            for nm in [n["name"]] + list(n.get("aliases", [])):
                key = legal_normalize(nm)
                if key and key not in self.name_index:
                    self.name_index[key] = n["eid"]

        self.pid_to_title = {p["pid"]: p.get("title", "") for p in passages}
        self.pid_text = {p["pid"]: f"{p.get('title','')}. {p.get('text','')}"
                         for p in passages}
        self.pid_index = {p["pid"]: i for i, p in enumerate(passages)}

        # 实体 IDF
        self.ent_idf: Dict[str, float] = {}
        if use_idf:
            n_pass = max(1, len({pid for s in self.ent_pids.values() for pid in s}))
            for eid, pids in self.ent_pids.items():
                df = len(pids)
                self.ent_idf[eid] = (math.log((1.0 + n_pass) / (1.0 + df)) + 1.0) ** idf_power

        # 段落激活实体数
        self.pid_ent_count: Dict[str, int] = defaultdict(int)
        for eid, pids in self.ent_pids.items():
            for pid in pids:
                self.pid_ent_count[pid] += 1

        # 节点 embedding (查询种子)
        self.node_ids = [n["eid"] for n in self.nodes]
        self.node_emb = None
        if embedder is not None:
            texts = [f"{n['name']}. {n.get('description','')}" for n in self.nodes]
            self.node_emb = embedder.encode(texts, normalize=True)
            # 段落 embedding (dense 回退)
            self.pid_emb = None
            if dense_fallback and passages:
                self.pid_emb = embedder.encode(
                    [self.pid_text[p["pid"]] for p in passages], normalize=True)
        else:
            self.pid_emb = None

    # ------------------------------------------------------------------ #
    def _lexical_seed_score(self, query: str) -> Dict[str, float]:
        q = legal_normalize(query)
        q_tokens = set(sig_tokens(query))
        scores: Dict[str, float] = defaultdict(float)
        for name_key, eid in self.name_index.items():
            if not name_key:
                continue
            if name_key in q:
                scores[eid] = max(scores[eid],
                                  min(1.0, 0.75 + 0.25 * len(name_key) / max(len(q), 1)))
            else:
                ent_tokens = set(name_key.split())
                if ent_tokens and q_tokens:
                    inter = ent_tokens & q_tokens
                    if inter:
                        scores[eid] = max(
                            scores[eid],
                            len(inter) / len(ent_tokens | q_tokens) * 0.5)
        return scores

    # ------------------------------------------------------------------ #
    def seed(self, query: str) -> List[Tuple[str, float]]:
        if self.node_emb is None:
            return []
        q = self.embedder.encode([query], normalize=True)[0]
        sims = self.node_emb @ q
        lex = self._lexical_seed_score(query)
        fused = self.seed_alpha * sims
        if lex:
            lex_arr = np.zeros_like(sims)
            for i, eid in enumerate(self.node_ids):
                if eid in lex:
                    lex_arr[i] = lex[eid]
            fused = fused + self.lexical_weight * lex_arr
        k = min(self.k_e, len(fused))
        if k <= 0:
            return []
        # 用 argpartition 取 top-k 后再排序; 但 argpartition 的返回顺序不保证
        # 稳定, 相同分数的实体在不同运行中可能被选入/剔除, 因此这里显式用
        # 分数与 eid 做确定性 tie-break。
        top = np.argpartition(-fused, kth=k - 1)[:k]
        top = sorted(top, key=lambda i: (-fused[i], self.node_ids[int(i)]))
        return [(self.node_ids[int(i)], float(max(fused[i], 0.0))) for i in top]

    # ------------------------------------------------------------------ #
    def spread(self, seeds: List[Tuple[str, float]]) -> Dict[str, float]:
        act: Dict[str, float] = defaultdict(float)
        for eid, s in seeds:
            act[eid] = max(act[eid], max(s, 0.0))
        if not act:
            return {}
        for _ in range(self.max_hops):
            nxt: Dict[str, float] = defaultdict(float)
            # 按 eid 排序遍历: 浮点加法不满足结合律, 累加顺序不同会得到
            # 末位不同的激活值, 进而让接近的分数在两次运行间互换次序。
            for u in sorted(act):
                au = act[u]
                if au < self.threshold:
                    continue
                nbrs = self.adj.get(u, [])
                if not nbrs:
                    continue
                total = sum(w for _, w in nbrs)
                if total <= 0:
                    continue
                for v, w in nbrs:
                    nxt[v] += au * (w / total) * self.decay
            for v in sorted(nxt):
                act[v] = max(act[v], nxt[v])
        return {k: v for k, v in act.items() if v > 0}

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        seeds = self.seed(query)
        act = self.spread(seeds)
        pid_score: Dict[str, float] = defaultdict(float)
        for eid, a in act.items():
            w = a * (self.ent_idf.get(eid, 1.0) if self.use_idf else 1.0)
            # `ent_pids[eid]` 是 set, 迭代顺序受哈希随机化影响; 而浮点加法
            # 不满足结合律, 累加顺序不同会得到末位不同的分数, 进而让
            # 分数接近的段落排序在两次运行间互换。这里排序后遍历以保证
            # 逐位可复现。
            for pid in sorted(self.ent_pids.get(eid, ())):
                pid_score[pid] += w
        if self.norm_beta > 0:
            for pid in list(pid_score.keys()):
                cnt = max(1, self.pid_ent_count.get(pid, 1))
                pid_score[pid] /= (cnt ** (self.norm_beta / 2.0))
        if self.title_bonus > 0:
            q = legal_normalize(query)
            q_tokens = set(sig_tokens(query))
            for pid, title in self.pid_to_title.items():
                t_tokens = set(sig_tokens(title))
                if t_tokens and q_tokens and (t_tokens & q_tokens):
                    pid_score[pid] += self.title_bonus
        if self.dense_fallback and self.pid_emb is not None:
            qv = self.embedder.encode([query], normalize=True)[0]
            dsims = self.pid_emb @ qv
            for i, pid in enumerate(self.pid_to_title):
                if pid not in pid_score:
                    pid_score[pid] = self.dense_weight * float(dsims[i])
        if not pid_score:
            return []
        # 排序必须是确定性的: `pid_score` 的插入顺序取决于 `act` 的遍历顺序,
        # 而 `act` 来自 dict 迭代 + `ent_pids` (set) 遍历, 二者都受
        # PYTHONHASHSEED 随机化影响。若只用分数做 key (稳定排序), 同分段落
        # 的相对次序就会随进程而变 —— 同一份数据两次运行得到不同的检索结果,
        # 令超参寻优不可复现。这里显式用 pid 作为最终的 tie-break。
        ranked = sorted(pid_score.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return [{"pid": p, "score": round(s, 6)} for p, s in ranked]


def assert_pid(pid: str) -> str:
    return pid


# ============================================================================
# 4. 双路融合检索器
# ============================================================================
class DualPathRetriever:
    """
    双路并行检索 + 必返保证 + 关键词打分 + 加权融合。

    流程:
      1. Milvus 路: 稠密向量 HNSW 检索 -> top_pool 条
      2. KG 路   : 激活扩散结构检索        -> top_pool 条
      3. 关键词路: 关键词位置/数量/关联概念打分 -> 候选并入
      4. 保证集  : Milvus top_g 与 KG top_g *必返* (各自独立)
      5. 融合    : s_hyb = w_v·ŝ_v + w_g·ŝ_g + w_kw·ŝ_kw + η·1[双路]
      6. 输出    : 保证集先按融合分排, 其余按融合分追加, 补足到 top_k
    """

    def __init__(self, milvus: Any, kg: KGPathRetriever,
                 kw_scorer: KeywordScorer,
                 passages: List[Dict[str, Any]],
                 embedder=None,
                 w_v: float = 0.4, w_g: float = 0.4, w_kw: float = 0.2,
                 eta: float = 0.05,
                 guarantee_k: int = 2,
                 top_pool: int = 20,
                 use_guarantee: bool = True,
                 guarantee_mode: str = "reserved",
                 kw_pos_weight: float = 0.35,
                 kw_count_weight: float = 0.35,
                 kw_assoc_weight: float = 0.30,
                 count_saturate: int = 5,
                 assoc_top_m: int = 10):
        """
        guarantee_mode:
          "off"      —— 关闭保证机制, 纯按融合分排序
          "paper"    —— 论文式: 保证集 G 先按融合分排, 其余追加
                       (保证项强制占据结果列表前部)
          "reserved" —— 保留槽位式: 保证项 *必须进入* top-k, 但最终顺序
                        仍完全由融合分决定 (不强行提前)
        """
        self.milvus = milvus
        self.kg = kg
        self.kw = kw_scorer
        self.passages = passages
        self.embedder = embedder
        self.w_v = w_v
        self.w_g = w_g
        self.w_kw = w_kw
        self.eta = eta
        self.guarantee_k = guarantee_k
        self.top_pool = top_pool
        self.use_guarantee = use_guarantee
        self.guarantee_mode = guarantee_mode
        self.kw_pos_weight = kw_pos_weight
        self.kw_count_weight = kw_count_weight
        self.kw_assoc_weight = kw_assoc_weight
        self.count_saturate = count_saturate
        self.assoc_top_m = assoc_top_m

    # ------------------------------------------------------------------ #
    def _query_vec(self, query: str) -> np.ndarray:
        return self.embedder.encode([query], normalize=True)[0]

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: int = 10,
                 keywords: Optional[Sequence[str]] = None
                 ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        t0 = time.perf_counter()

        # ---------- 路径 1: Milvus ----------
        qv = self._query_vec(query)
        t_v0 = time.perf_counter()
        m_hits = self.milvus.search(qv, self.top_pool)
        t_v = time.perf_counter() - t_v0

        # ---------- 路径 2: KG 激活扩散 ----------
        t_g0 = time.perf_counter()
        g_hits = self.kg.retrieve(query, top_k=self.top_pool)
        t_g = time.perf_counter() - t_g0

        s_v: Dict[str, float] = {h["pid"]: h["score"] for h in m_hits}
        s_g: Dict[str, float] = {h["pid"]: h["score"] for h in g_hits}
        n_v, n_g = minmax_norm(s_v), minmax_norm(s_g)

        # ---------- 路径 3: 关键词 ----------
        kws = [k for k in (keywords or []) if k and k.strip()]
        kw_detail: Dict[str, Dict[str, Any]] = {}
        s_kw: Dict[str, float] = {}
        if kws:
            candidates = set(s_v) | set(s_g)
            # 关键词路独立扫描全库 (保证能找到纯关键词命中的文档)
            kw_detail = self.kw.score(kws)
            s_kw = {pid: d["kw_score"] for pid, d in kw_detail.items()
                    if d["kw_score"] > 0}
            n_kw = minmax_norm(s_kw)
        else:
            n_kw = {}

        # ---------- 保证集 ----------
        if self.use_guarantee:
            guar_m = [h["pid"] for h in m_hits[:self.guarantee_k]]
            guar_g = [h["pid"] for h in g_hits[:self.guarantee_k]]
            guaranteed = list(dict.fromkeys(guar_m + guar_g))
        else:
            guar_m, guar_g, guaranteed = [], [], []

        # ---------- 融合 ----------
        all_pids = set(s_v) | set(s_g) | set(s_kw)
        fused: Dict[str, Dict[str, Any]] = {}
        for pid in all_pids:
            sv = n_v.get(pid, 0.0)
            sg = n_g.get(pid, 0.0)
            sk = n_kw.get(pid, 0.0)
            dual = 1.0 if (pid in s_v and pid in s_g) else 0.0
            score = (self.w_v * sv + self.w_g * sg + self.w_kw * sk
                     + self.eta * dual)
            fused[pid] = {
                "pid": pid,
                "score": score,
                "s_v": sv, "s_g": sg, "s_kw": sk,
                "dual": dual,
                "in_milvus": pid in s_v,
                "in_kg": pid in s_g,
                "in_kw": pid in s_kw,
                "guaranteed": pid in guaranteed,
                "guaranteed_from": ([ "milvus"] if pid in guar_m else [])
                                   + ([ "kg"] if pid in guar_g else []),
                "title": self.kg.pid_to_title.get(pid, ""),
                "kw_detail": kw_detail.get(pid, {}),
            }

        # ---------- 输出 ----------
        # 三种保证模式:
        #   off      —— 纯按融合分排序
        #   paper    —— 保证集整体排在前部 (论文 Bottom-up Top-k)
        #   reserved —— 保证项必须入选, 但最终顺序仍由融合分决定
        if self.use_guarantee and self.guarantee_mode == "paper":
            guar_items = [fused[p] for p in guaranteed if p in fused]
            rest_items = [fused[p] for p in all_pids
                          if p not in set(guaranteed) and p in fused]
            guar_items.sort(key=lambda d: -d["score"])
            rest_items.sort(key=lambda d: -d["score"])
            final = (guar_items + rest_items)[:top_k]
        elif self.use_guarantee and self.guarantee_mode == "reserved":
            # 保证项占住保留槽位, 其余槽位按融合分填充
            guar_set = set(guaranteed)
            guar_items = [fused[p] for p in guaranteed if p in fused]
            rest_items = [fused[p] for p in all_pids
                          if p not in guar_set and p in fused]
            rest_items.sort(key=lambda d: -d["score"])
            n_guar = min(len(guar_items), top_k)
            # 保留槽位放保证项, 剩余槽位用最高分的非保证项
            final = guar_items[:n_guar] + rest_items[: top_k - n_guar]
            # 最终顺序完全按融合分
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
            "n_guaranteed": len([p for p in guaranteed if p in fused]),
            "guaranteed_milvus": guar_m,
            "guaranteed_kg": guar_g,
            # 各路自身的排序结果 (前 top_k), 用于双路命中分布分析
            "milvus_ranked": [h["pid"] for h in m_hits[:top_k]],
            "kg_ranked": [h["pid"] for h in g_hits[:top_k]],
            "t_milvus": round(t_v, 4),
            "t_kg": round(t_g, 4),
            "t_total": round(time.perf_counter() - t0, 4),
            "keywords": kws,
            "kw_matched_entities": self.kw.match_entities(kws) if kws else [],
        }
        return final, stats
