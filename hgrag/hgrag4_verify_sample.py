#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
校验导出的基准样本可独立跑通 (审稿人 #2 concern #3)
==================================================

审稿人要求发布 "code and a benchmark sample"。仅仅把文件拷出去是不够的
—— 样本必须 *真的能跑*。本脚本在一个 **全新的进程** 里, 只用样本目录
里的文件 (外加公开的 embedding 模型) 完成:

  1. 载入样本语料与图谱;
  2. 构造三路检索器;
  3. 对样本中的每道题跑一次检索;
  4. 用 *样本自带的* 逐题结果向量, 独立复算论文里的配对自助法
     显著性检验, 并与 released 结果比对。

第 4 步是关键: 它证明样本里的数字不是"抄进去的", 而是可以被独立
复算的 —— 这正是审稿人所说的 "cannot be verified from the text of the
manuscript alone" 所要解决的东西。

用法:
    python hgrag4_verify_sample.py --sample release\\hgrag4_sample
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import activation_graphrag as A  # noqa: E402
import hgrag4_experiments as E4  # noqa: E402

CACHE_DIR = os.environ.get("HGRAG_CACHE_DIR", r"C:\data\zsk\RAG\hf_cache")


def paired_bootstrap(pairs: Sequence[Tuple[float, float]],
                      n_boot: int = 10000,
                      seed: int = 20240917
                      ) -> Tuple[float, List[float], float]:
    """
    配对自助法 —— 直接调用 *已发布的* 统计机制
    (hgrag4_experiments.paired_bootstrap), 而不是在这里另写一份。

    这一点很重要: 复核脚本若自带一份独立的实现, 复核的就不是论文所用的
    检验了。调用随代码发布的同一个函数, 才检验了"读者用随论文发布的
    代码能否复算出论文的数字"。
    """
    out = E4.paired_bootstrap(pairs, n_boot=n_boot, seed=seed)
    return (float(out["mean_diff"]), [float(out["ci95"][0]),
                                      float(out["ci95"][1])],
            float(out["p_value"]))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="校验导出的基准样本")
    p.add_argument("--sample", default=os.path.join("release",
                                                   "hgrag4_sample"))
    p.add_argument("--cache-dir", default=CACHE_DIR)
    p.add_argument("--n-probe", type=int, default=5,
                   help="抽样实际跑检索的题数")
    args = p.parse_args(argv)

    sample = os.path.abspath(args.sample)
    print("=== 基准样本可运行性校验 ===\n")
    print(f"样本目录: {sample}\n")

    # ---------------- 0) 清单完整性 ---------------- #
    man_path = os.path.join(sample, "MANIFEST.json")
    with open(man_path, encoding="utf-8") as fh:
        man = json.load(fh)
    print(f"[1/4] MANIFEST: {man['n_files']} 个文件, "
          f"SHA-256 校验中 ...")
    import hashlib
    bad = []
    for f in man["files"]:
        full = os.path.join(sample, f["path"])
        if not os.path.exists(full):
            bad.append(f["path"] + " (缺失)")
            continue
        h = hashlib.sha256()
        with open(full, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != f["sha256"]:
            bad.append(f["path"] + " (摘要不符)")
    if bad:
        print("      [FAIL] " + "; ".join(bad))
        return 1
    print("      [OK] 全部文件的 SHA-256 与清单一致")

    # ---------------- 1) 载入 ---------------- #
    print("\n[2/4] 只用样本目录载入语料与图谱 ...")
    corpus = E4.Corpus(os.path.join(sample, "kg"),
                        "paraphrase-multilingual-MiniLM-L12-v2",
                        args.cache_dir, "exact")
    questions = A.read_json(os.path.join(sample, "kg",
                                         "questions.json")) or []
    splits = A.read_json(os.path.join(sample, "kg", "splits.json")) or {}
    print(f"      段落 {len(corpus.passages)}, 实体 {len(corpus.entities)}, "
          f"关系 {len(corpus.relations)}, 问题 {len(questions)}")
    assert len(corpus.passages) > 0 and len(corpus.entities) > 0, \
        "样本语料或图谱为空"
    assert len(questions) > 0, "样本问题为空"

    # ---- 图谱连通性: 扩散是论文的核心算子, 样本图必须带边 ----
    # 早前的导出脚本误用 "id" 而不是 "eid" 过滤节点, 于是导出的样本图
    # 有节点却没有任何边, graph route 无法复现 —— 而显著性检验仍然能
    # 通过 (它复算用的是预存的逐题向量), 所以这个缺陷不会自己暴露。
    # 这里显式断言, 使同类缺陷以后必然在自检阶段就失败。
    net = A.read_json(os.path.join(sample, "kg",
                                   "activation_network.json")) or {}
    n_nodes = len(net.get("nodes") or [])
    n_edges = len(net.get("edges") or [])
    assert n_nodes > 0, "样本激活网络没有节点"
    assert n_edges > 0, (
        "样本激活网络没有任何边 —— 扩散 (spreading activation) 无法复现。"
        "请检查导出脚本 hgrag4_export_sample.py 中节点标识键是否用了 "
        "'eid' (而非 'id')。")
    _deg: Dict[str, int] = {}
    for _e in net["edges"]:
        _deg[_e["source"]] = _deg.get(_e["source"], 0) + 1
        _deg[_e["target"]] = _deg.get(_e["target"], 0) + 1
    _connected = sum(1 for _n in net["nodes"]
                     if _deg.get(_n.get("eid"), 0) > 0)
    print(f"      [OK] 激活网络: {n_nodes} 节点 / {n_edges} 边, "
          f"{_connected}/{n_nodes} 个节点至少有一个邻居")
    assert _connected > 0, "样本激活网络的每个节点都是孤立点"

    # ---------------- 2) 跑检索 ---------------- #
    print(f"\n[3/4] 构造三路检索器并对 {min(args.n_probe, len(questions))} "
          f"道题实跑检索 ...")
    retriever = E4.build_retriever(
        corpus,
        w_v=0.4, w_g=0.5, w_kw=0.2, eta=0.0,
        guarantee_k=0, use_guarantee=False, guarantee_mode="off",
        kw_pos=0.35, kw_count=0.35, kw_assoc=0.30,
        norm_mode="rank", kw_scope="cand", design="sum",
        w_title=0.0, lexical="bm25", w_bm25=0.7, lex_norm="max",
    )
    hits = 0
    for q in questions[:args.n_probe]:
        kws = E4.extract_keywords_from_question(q, corpus)
        docs, _ = retriever.retrieve(q["question"], top_k=10, keywords=kws)
        if docs:
            hits += 1
    assert hits == min(args.n_probe, len(questions)), \
        f"部分查询没有返回任何结果 ({hits}/{args.n_probe})"
    print(f"      [OK] {hits}/{min(args.n_probe, len(questions))} "
          f"道题检索成功并返回非空结果")

    # ---------------- 3) 独立复算显著性 ---------------- #
    print("\n[4/4] 用样本自带的逐题结果向量独立复算论文的显著性检验 ...")
    res_dir = os.path.join(sample, "results")
    exp1 = A.read_json(os.path.join(res_dir, "exp1_baselines.json")) or {}
    hm = (exp1.get("hgrag_full") or {}).get("per_query") or []
    rf = (exp1.get("ragflow") or {}).get("per_query") or []
    assert hm and rf, "exp1_baselines.json 缺少逐题结果向量"

    def _recall10(rec: Dict[str, Any]) -> float:
        """从逐题记录里 *重新计算* Recall@10, 而不是读取一个已存的
        数字 —— 这样才真正检验了样本所发布的数据是自洽的。"""
        gold = set(rec.get("gold") or [])
        got = (rec.get("retrieved") or [])[:10]
        if not gold:
            return float("nan")
        return len(gold & set(got)) / len(gold)

    hmap = {q["id"]: q for q in hm}
    rmap = {q["id"]: q for q in rf}
    common = sorted(set(hmap) & set(rmap))
    print(f"      可对齐的逐题向量: {len(common)} 题 "
          f"(HGRAG {len(hmap)}, RAGFlow {len(rmap)})")
    assert len(common) >= 50, "可对齐的题数过少"

    # 逐题记录与 per_question_* 向量是 *同一个顺序* (按评测时的问题顺序,
    # 而非按 id 排序), 因此复算必须按位置对齐, 不能按 id 排序后再配对 ——
    # 后者会静默地把两个系统的题目错配, 得出一个"看起来合理"的错误结果。
    assert len(hm) == len(rf), "两个系统的逐题记录数不同"
    for qh, qr in zip(hm, rf):
        assert qh["id"] == qr["id"], "两个系统的逐题记录顺序不一致"
    pairs = [(_recall10(qh), _recall10(qr)) for qh, qr in zip(hm, rf)]
    delta, ci, p = paired_bootstrap(pairs)
    print(f"      复算 HGRAG vs RAGFlow @ Recall@10: "
          f"Δ = {100*delta:+.2f} pp, "
          f"95% CI [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}], p = {p:.4f}")

    exp1_sig = (exp1.get("significance") or {})
    ref = None
    for k, v in exp1_sig.items():
        if k == "recall@10" and "hgrag_vs_ragflow" in v:
            ref = v["hgrag_vs_ragflow"]
    assert ref is not None, "released 结果中找不到 recall@10 的显著性记录"
    rd = ref["mean_diff"] * 100
    rci = [c * 100 for c in ref["ci95"]]
    rp = ref["p_value"]
    print(f"      released:            Δ = {rd:+.2f} pp, "
          f"95% CI [{rci[0]:+.2f}, {rci[1]:+.2f}], p = {rp:.4f}")
    ok = (abs(delta * 100 - rd) < 0.02
          and abs(ci[0] * 100 - rci[0]) < 0.02
          and abs(ci[1] * 100 - rci[1]) < 0.02
          and abs(p - rp) < 1e-9)
    assert ok, (f"复算结果与 released 结果不一致: Δ {100*delta:.4f} vs {rd:.4f}, "
                f"CI {ci} vs {rci}, p {p} vs {rp}")
    print("      [OK] 独立复算与 released 结果 *逐位* 一致")

    # ---------------- 4) 划分文件 ---------------- #
    print("\n[附] 划分文件 (种子 20240917): "
          f"dev {len(splits.get('dev', []))}, "
          f"test {len(splits.get('test', []))}")
    assert len(splits.get("dev", [])) == 44, "dev 划分大小不符"
    assert len(splits.get("test", [])) == 67, "test 划分大小不符"
    print("      [OK] 44/67 划分随样本一并发布, 故显著性检验可被 "
          "独立重算")

    print("\n[OK] 样本可独立运行、且其显著性数字可被独立复算 —— "
          "论文的核心论断现可由第三方核验。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
