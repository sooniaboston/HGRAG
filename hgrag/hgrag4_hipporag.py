#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HippoRAG 风格的 PPR 对照基线 (reviewer #2 concern #2)
======================================================

动机
----
审稿人 #2 要求: "You already cite HippoRAG's performance on publicly available
multi-hop benchmarks, so the comparison to at least one of those would improve
your paper substantially."

我们不能直接拿 HippoRAG 论文里的数字来对比, 原因是不可比:

  * HippoRAG 在 HotpotQA 上用 1,000 道 validation 题, 语料是"所有候选段落
    (supporting + distractor) 的并集", 规模远大于本文的 1,172 段/111 题;
  * HippoRAG 的 retriever backbone 是 Contriever / ColBERTv2, 本文用的是
    paraphrase-multilingual-MiniLM-L12-v2;
  * HippoRAG 的 PPR 作用在一个由 GPT-3.5 OpenIE 抽取的 *无 schema* KG 上,
    而本文的图谱是 union-find 归并 + veto 规则解析后的实体图。

因此我们复现 HippoRAG 的 **检索机制** (query node seeding + Personalized
PageRank + passage 级聚合), 而不是引用它的数字。这样比较的是"同一份语料、
同一批题、同一套 embedding 下, 图检索路径用 PPR 传播还是用 Collins-Loftus
激活扩散"。

实现要点 (对齐 HippoRAG 论文 §2.3 / §3.4)
------------------------------------------
1. **Query node seeding**: 用同一个 dense encoder 把 query 与 KG 节点名
   对比, 取 top-k 作为 query nodes (与 HippoRAG 的 PHR 检索编码器同构);
   再用 query 的词法匹配做一次实体名精确命中加成。
2. **Personalized PageRank**: 阻尼系数 alpha = 0.5, restart 向量 = query
   nodes 上的归一化种子分布 (HippoRAG 论文 §3.4 明确 alpha = 0.5)。
   用 power iteration 求解, 迭代到 L1 收敛或达到最大迭代数。
3. **Passage 聚合**: 把每个 passage 的 PPR 概率质量按"该 passage 出现在哪些
   节点的 source_pids 中"聚合, 与 HippoRAG "aggregate the output PPR node
   probability over the previously indexed passage nodes" 一致。
4. **只跑在 direct edges 上**: HippoRAG 的 KG 是 OpenIE 抽取的实体-关系图;
   本图的 `indirect` 边是 Collins-Loftus 扩散预计算出来的, 用它们做 PPR 就
   等于把本文的机制当成基线, 比较会失去意义。所以 PPR 只在 `direct` 边上
   求解 —— 即"只给 PPR, 不给激活扩散"。

因此本基线与 HGRAG 的图路共享: 同一个 KG、同样的 seeding、同一套 embedding、
同一个 passage 聚合口径。唯一被替换的是 **传播算子**: 幂次衰减扩散
(Collins-Loftus) vs Personalized PageRank。

用法
----
    python hgrag4_hipporag.py --kg-dir C:\\data\\zsk\\RAG\\kgv4 \
        --out-dir C:\\data\\zsk\\RAG\\hgrag4 --alpha 0.5 --seed-nodes 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import activation_graphrag as A  # noqa: E402
import hgrag3_core as H3  # noqa: E402
from hgrag4_experiments import Corpus  # noqa: E402
from hgrag4_splits import load_or_make_split, select  # noqa: E402

LOG = logging.getLogger("hgrag4.hipporag")

KG_DIR = r"C:\data\zsk\RAG\kgv4"


# ============================================================================
# 1. PPR 求解器
# ============================================================================
class PersonalizedPageRank:
    """
    在 KG 的 direct 边上求解 Personalized PageRank。

    转移矩阵按出边权重做行归一化 (与 HippoRAG 一致: 其 KG 边权来自 OpenIE
    计数, PPR 在该图上求解)。restart 向量 r = 种子分布 (query nodes)。

        p_{t+1} = (1 - alpha) * P^T p_t + alpha * r

    阻尼系数 alpha = 0.5 (HippoRAG 论文 §3.4: "the PPR damping factor at 0.5,
    which determines the probability that PPR will restart a random walk from
    the query nodes instead of continuing to explore the graph").
    """

    def __init__(self, nodes: List[Dict[str, Any]],
                 edges: List[Dict[str, Any]],
                 alpha: float = 0.5,
                 max_iter: int = 100,
                 tol: float = 1e-10):
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        self.eid_of: Dict[str, int] = {}
        self.node_ids: List[str] = []
        for n in nodes:
            eid = n["eid"]
            if eid in self.eid_of:
                continue
            self.eid_of[eid] = len(self.node_ids)
            self.node_ids.append(eid)
        self.n = len(self.node_ids)

        # 出边邻接 (行归一化)
        out: List[List[Tuple[int, float]]] = [[] for _ in range(self.n)]
        for e in edges:
            if e.get("kind") != "direct":
                continue  # 只用 direct 边: indirect 边是本方法自己的扩散产物
            s = self.eid_of.get(e["source"])
            t = self.eid_of.get(e["target"])
            if s is None or t is None:
                continue
            w = float(e.get("weight", 1.0))
            if w > 0:
                out[s].append((t, w))
        # 行归一化 + 记录列支持 (哪些节点有入边)
        self.out = out
        self.has_in = [False] * self.n
        self.out_total = [0.0] * self.n
        for s, nbrs in enumerate(out):
            tot = sum(w for _, w in nbrs)
            self.out_total[s] = tot
            if tot > 0:
                for _, w in nbrs:
                    self.out_total[s] = tot
                for t, _ in nbrs:
                    self.has_in[t] = True

        # 节点 -> passage
        self.pid_index: Dict[str, int] = {}
        self.titles: List[str] = []
        self.node_pids: List[List[int]] = [[] for _ in range(self.n)]
        for n in nodes:
            idx = self.eid_of[n["eid"]]
            for pid in n.get("source_pids", []) or []:
                if pid not in self.pid_index:
                    self.pid_index[pid] = len(self.titles)
                    self.titles.append(pid)
                self.node_pids[idx].append(self.pid_index[pid])

        # 矩阵形式 (稀疏 P^T): PPR 在稀疏图上求解, 且 scipy.sparse 的
        # 乘法比稠密 numpy 快两个数量级 (6728 节点的稠密矩阵每次 matvec
        # 要 45M 次乘加, 稀疏版只有 ~33K 条非零边)。
        import scipy.sparse as _sp
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        for s, nbrs in enumerate(out):
            tot = self.out_total[s]
            if tot <= 0:
                continue
            for t, w in nbrs:
                rows.append(t)
                cols.append(s)
                vals.append(w / tot)
        self.Pt = _sp.csr_matrix((vals, (rows, cols)),
                                 shape=(self.n, self.n), dtype=np.float64)

    # ------------------------------------------------------------------ #
    def solve(self, seeds: Sequence[Tuple[int, float]]) -> Dict[int, float]:
        """
        seeds: [(node_index, weight)] —— query nodes 及其种子权重。
        返回 {node_index: ppr_probability}。
        """
        if not seeds:
            return {}
        r = np.zeros(self.n, dtype=np.float64)
        tot = 0.0
        for idx, w in seeds:
            if idx < 0 or idx >= self.n or w <= 0:
                continue
            r[idx] += w
            tot += w
        if tot <= 0:
            return {}
        r /= tot

        p = r.copy()
        for _ in range(self.max_iter):
            p_new = (1.0 - self.alpha) * (self.Pt @ p) + self.alpha * r
            delta = float(np.abs(p_new - p).sum())
            p = p_new
            if delta < self.tol:
                break
        return {i: float(v) for i, v in enumerate(p) if v > 0}

    # ------------------------------------------------------------------ #
    def aggregate_passages(self, ppr: Dict[int, float]) -> Dict[str, float]:
        """
        把 PPR 节点概率聚合到 passage 上 (HippoRAG: "aggregate the output PPR
        node probability over the previously indexed passage nodes")。
        """
        pid_score: Dict[str, float] = {}
        for idx, prob in ppr.items():
            if prob <= 0:
                continue
            for pi in self.node_pids[idx]:
                pid = self.titles[pi]
                pid_score[pid] = pid_score.get(pid, 0.0) + prob
        return pid_score


# ============================================================================
# 2. HippoRAG 风格的检索器
# ============================================================================
class HippoRAGStyleRetriever:
    """
    HippoRAG 的检索路径: query node seeding + PPR + passage 聚合。

    与本文 KGPathRetriever 共享: 同一个 KG、同一套 embedding、同样的
    passage 聚合口径; 唯一区别是传播算子是 PPR 而非 Collins-Loftus 扩散。
    """

    def __init__(self, corpus: Corpus,
                 ppr: PersonalizedPageRank,
                 seed_nodes: int = 5,
                 alpha: float = 0.5,
                 lexical_weight: float = 0.3):
        self.corpus = corpus
        self.ppr = ppr
        self.k_e = int(seed_nodes)
        self.alpha = float(alpha)
        self.lexical_weight = float(lexical_weight)

        # 节点 embedding (与 KGPathRetriever 完全一致的方式)
        # 节点顺序严格取自 ppr.node_ids, 保证 embedding 行与 PPR 索引严格对齐
        self.node_ids = ppr.node_ids
        by_eid = {n["eid"]: n for n in corpus.entities}
        texts = []
        for e in self.node_ids:
            n = by_eid.get(e, {})
            texts.append(f"{n.get('name', '')}. {n.get('description', '')}")
        self.node_emb = corpus.embedder.encode(texts, normalize=True)

        # 词法实体名索引 (与 KGPathRetriever 同一套归一化)
        self.name_index: Dict[str, str] = {}
        for n in corpus.entities:
            for nm in [n["name"]] + list(n.get("aliases", []) or []):
                key = H3.legal_normalize(nm)
                if key and key not in self.name_index:
                    self.name_index[key] = n["eid"]

        self.pid_to_title = {p["pid"]: p.get("title", "") for p in corpus.passages}

    # ------------------------------------------------------------------ #
    def _lexical_seed_score(self, query: str) -> Dict[str, float]:
        q = H3.legal_normalize(query)
        q_tokens = set(H3.sig_tokens(query))
        scores: Dict[str, float] = {}
        for name_key, eid in self.name_index.items():
            if not name_key:
                continue
            if name_key in q:
                scores[eid] = max(scores.get(eid, 0.0),
                                  min(1.0, 0.75 + 0.25 * len(name_key) / max(len(q), 1)))
            else:
                ent_tokens = set(name_key.split())
                if ent_tokens and q_tokens:
                    inter = ent_tokens & q_tokens
                    if inter:
                        scores[eid] = max(
                            scores.get(eid, 0.0),
                            len(inter) / len(ent_tokens | q_tokens) * 0.5)
        return scores

    # ------------------------------------------------------------------ #
    def seed(self, query: str) -> List[Tuple[int, float]]:
        """query node 检索 (对应 HippoRAG 的 PHR 检索编码器)。"""
        q = self.corpus.embedder.encode([query], normalize=True)[0]
        sims = self.node_emb @ q
        lex = self._lexical_seed_score(query)
        fused = sims.copy()
        if lex:
            lex_arr = np.zeros_like(sims)
            for i, eid in enumerate(self.node_ids):
                if eid in lex:
                    lex_arr[i] = lex[eid]
            fused = fused + self.lexical_weight * lex_arr
        k = min(self.k_e, len(fused))
        if k <= 0:
            return []
        top = np.argpartition(-fused, kth=k - 1)[:k]
        # 确定性 tie-break
        top = sorted(top, key=lambda i: (-float(fused[i]), self.node_ids[int(i)]))
        out: List[Tuple[int, float]] = []
        for i in top:
            out.append((self.ppr.eid_of[self.node_ids[int(i)]],
                        float(max(fused[i], 0.0))))
        return out

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: int = 10) -> Tuple[List[Dict[str, Any]],
                                                              Dict[str, Any]]:
        t0 = time.perf_counter()
        seeds = self.seed(query)
        ppr = self.ppr.solve(seeds)
        pid_score = self.ppr.aggregate_passages(ppr)
        ranked = sorted(pid_score.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        res = [{"pid": p, "score": round(s, 8)} for p, s in ranked]
        t_total = time.perf_counter() - t0
        return res, {"t_total": t_total, "n_seeds": len(seeds),
                     "n_ppr_nodes": len(ppr)}


# ============================================================================
# 3. 评测 (与 evaluate_retriever 同一口径)
# ============================================================================
def evaluate_hipporag(retriever: HippoRAGStyleRetriever,
                       questions: List[Dict[str, Any]],
                       corpus: Corpus,
                       top_k: int = 10) -> Dict[str, Any]:
    """
    与 `evaluate_retriever` 完全相同的指标定义 (document-level Recall@k /
    Precision@k / mean rank / latency), 以保证与表中其它系统可比。
    """
    recalls = {1: [], 3: [], 5: [], 10: []}
    precisions = {1: [], 3: [], 5: [], 10: []}
    ranks: List[float] = []
    latencies: List[float] = []
    per_query: List[Dict[str, Any]] = []

    for qi, q in enumerate(questions):
        qid = q.get("id", str(qi))
        res, stats = retriever.retrieve(q["question"], top_k=max(top_k, 10))
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

        per_query.append({
            "id": qid, "question": q["question"], "gold": list(gold),
            "retrieved": titles[:10], "t": stats["t_total"],
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
        "per_query": per_query,
    }
    for k, v in recalls.items():
        out[f"per_question_recall@{k}"] = [round(x, 6) for x in v]
    out["per_question_score"] = [
        round(0.40 * recalls[1][i] + 0.30 * recalls[5][i] + 0.30 * recalls[10][i], 6)
        for i in range(len(questions))
    ]
    return out


# ============================================================================
# 4. 主流程
# ============================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="HippoRAG 风格 PPR 对照基线")
    p.add_argument("--kg-dir", default=KG_DIR)
    p.add_argument("--out-dir", default=r"C:\data\zsk\RAG\hgrag4")
    p.add_argument("--embed-model", default=A.DEFAULT_EMBED_MODEL)
    p.add_argument("--cache-dir", default=r"C:\data\zsk\RAG\hf_cache")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="PPR 阻尼系数 (HippoRAG 论文用 0.5)")
    p.add_argument("--seed-nodes", type=int, default=5,
                   help="query node 数量 k")
    p.add_argument("--dev-fraction", type=float, default=0.4)
    p.add_argument("--split-seed", type=int, default=20240917)
    p.add_argument("--stages", default="dev,test",
                   help="在哪个划分上评测: dev / test / dev,test")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    res_dir = os.path.join(args.out_dir, "results")
    os.makedirs(res_dir, exist_ok=True)

    LOG.info("载入语料: %s", args.kg_dir)
    corpus = Corpus(args.kg_dir, args.embed_model, args.cache_dir,
                    milvus_backend="exact")

    split = load_or_make_split(corpus.questions, args.kg_dir,
                               dev_fraction=args.dev_fraction,
                               seed=args.split_seed)
    dev_q = select(corpus.questions, split, "dev")
    test_q = select(corpus.questions, split, "test")

    LOG.info("建图 (direct edges only) + PPR 求解器")
    ppr = PersonalizedPageRank(corpus.entities, corpus.network["edges"],
                               alpha=args.alpha)
    LOG.info("PPR 图: %d 节点, %d 条 direct 边", ppr.n,
             sum(len(x) for x in ppr.out))

    retr = HippoRAGStyleRetriever(corpus, ppr, seed_nodes=args.seed_nodes,
                                  alpha=args.alpha)

    summary: Dict[str, Any] = {
        "baseline": "hipporag_style_ppr",
        "alpha": args.alpha,
        "seed_nodes": args.seed_nodes,
        "edge_kind": "direct only",
        "embed_model": args.embed_model,
        "n_nodes": ppr.n,
        "n_direct_edges": sum(len(x) for x in ppr.out),
        "source": "HippoRAG (Gutierrez et al., NeurIPS 2024) Sec. 2.3 / 3.4: "
                  "query-node seeding + Personalized PageRank (alpha=0.5) + "
                  "passage-level probability aggregation",
    }

    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    for name, qs in (("dev", dev_q), ("test", test_q)):
        if name not in stages:
            continue
        LOG.info("在 %s 集上评测 (%d 题)", name, len(qs))
        res = evaluate_hipporag(retr, qs, corpus, top_k=10)
        res["_evaluated_on"] = name
        summary[name] = {k: v for k, v in res.items() if k != "per_query"}
        LOG.info("%s: %s", name, {k: res[k] for k in
                                   ("recall@1", "recall@5", "recall@10")})
        with open(os.path.join(res_dir, f"exp14_hipporag_{name}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)

    with open(os.path.join(res_dir, "exp14_hipporag.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    LOG.info("写出: %s", os.path.join(res_dir, "exp14_hipporag.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
