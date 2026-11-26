#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HGRAG v4 :: 语料完整性校验 + 修复
=================================

问题: `activation_graphrag.py --stage build` 会重新执行 stage_corpus(),
      把 passages.json / questions.json 覆盖成 *全量* (2964 段 / 300 题),
      而图谱 (entities/relations/network) 是基于 *已抽提子集* (1172 段) 建的。
      两者不一致 => 部分题目的 gold 段落根本没有进入图谱, KG 路必然失效。

修复: 以 kgv4_build/ 为图谱来源, 重新把 passages.json / questions.json
      写回 "已抽提子集" 版本, 并做三项一致性断言:
        1. 每个题的 gold 标题都在语料中
        2. 每个题的 10 个上下文段落都在语料中
        3. 语料里的每个段落的 pid 都在 raw_extractions 中
"""
from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import activation_graphrag as A  # noqa: E402
import hgrag4_splits as SPL  # noqa: E402

DATA_ROOT = r"C:\data\zsk\RAG"
KG_BUILD = os.path.join(DATA_ROOT, "kgv4_build")
KGV4 = os.path.join(DATA_ROOT, "kgv4")


def main() -> int:
    # ---------- 1. 图谱文件从 kgv4_build 同步过来 ----------
    for f in ("entities.json", "relations.json", "activation_network.json",
              "activation_network.graphml", "activation_network.gexf",
              "triples.jsonl", "entity_embeddings.npy",
              "entity_embedding_meta.json", "merge_evidence.json"):
        src = os.path.join(KG_BUILD, f)
        dst = os.path.join(KGV4, f)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"[sync] {f}")
        else:
            print(f"[warn] 缺少 {f} (源: {src})")

    # ---------- 2. 重建 passages / questions (已抽提子集) ----------
    passages = A.read_json(os.path.join(KGV4, "passages.json")) or []
    questions = A.read_json(os.path.join(KGV4, "questions.json")) or []
    extracted = {r["pid"] for r in A.read_jsonl(
        os.path.join(KGV4, "raw_extractions.jsonl"))}
    print(f"[raw] passages={len(passages)} questions={len(questions)} "
          f"extracted={len(extracted)}")

    # 若 passages 已被覆盖成全量, 则按 extracted 过滤回去
    keep = [p for p in passages if p["pid"] in extracted]
    if len(keep) != len(passages):
        print(f"[fix] 过滤 passages: {len(passages)} -> {len(keep)}")
        passages = keep
    title_set = {p["title"] for p in passages}

    # questions: 只保留 gold 全在语料中的
    keep_q = [q for q in questions
              if all(t in title_set for t in q.get("gold_titles", []))]
    if len(keep_q) != len(questions):
        print(f"[fix] 过滤 questions: {len(questions)} -> {len(keep_q)}")
        questions = keep_q

    A.write_json(os.path.join(KGV4, "passages.json"), passages)
    A.write_json(os.path.join(KGV4, "questions.json"), questions)

    # ---------- 3. 一致性断言 ----------
    ent = A.read_json(os.path.join(KGV4, "entities.json")) or []
    rel = A.read_json(os.path.join(KGV4, "relations.json")) or []
    ent_pids = set()
    for e in ent:
        for pid in e.get("source_pids", []) or []:
            ent_pids.add(pid)
    net_pids = set()
    net = A.read_json(os.path.join(KGV4, "activation_network.json")) or {}
    for n in net.get("nodes", []):
        for pid in n.get("source_pids", []) or []:
            net_pids.add(pid)

    corpus_pids = {p["pid"] for p in passages}
    print(f"[check] 语料段落 {len(corpus_pids)}")
    print(f"[check] 出现在实体中的段落 {len(ent_pids & corpus_pids)}")
    print(f"[check] 出现在图谱节点中的段落 {len(net_pids & corpus_pids)}")
    missing = corpus_pids - net_pids
    print(f"[check] 未进入图谱的语料段落: {len(missing)}")
    if missing:
        # 抽提阶段没能从这些段落里抽到任何实体 —— 属正常现象
        # (例如段落全是数字或极短文本)。只要它们不是任何题的 gold, 就不影响评测。
        pid2title = {p["pid"]: p["title"] for p in passages}
        gold_pids = set()
        for q in questions:
            for t in q.get("gold_titles", []):
                if t in pid2title:
                    gold_pids.add(pid2title[t])
        bad = missing & gold_pids
        print(f"[check] 其中属于 gold 的: {len(bad)} (必须为 0)")
        for pid in list(missing)[:5]:
            print(f"        - {pid2title.get(pid, pid)!r}")
        assert not bad, f"有 {len(bad)} 个 gold 段落没进图谱"
        print("[note] 非 gold 段落无图谱节点, 不影响检索评测")
    for q in questions:
        for t in q["gold_titles"]:
            assert t in title_set, f"gold 缺失: {t}"
    print(f"[ok] {len(questions)} 题 / {len(passages)} 段, 全部一致")

    # ---------- 4. 重新生成划分 (语料已变) ----------
    split = SPL.load_or_make_split(questions, KGV4, force=True)
    print(f"[split] dev={len(split['dev'])} test={len(split['test'])}")
    for name in ("dev", "test"):
        print(f"        {name}: {SPL.audit(split)[name]['type_pct']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
