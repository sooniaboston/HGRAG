#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HGRAG v4 :: 语料定稿 (kgv4)
==========================

背景
----
本机 DeepSeek 账户余额在扩展到 300 题的过程中耗尽 (402 Insufficient Balance),
实体/关系抽提只完成了 1172 个段落。这直接限制了可用的图谱覆盖范围,
因此 v4 的语料必须建立在 *已完成抽提* 的段落子集上, 而不是全部 2964 段。

定稿规则 (保证 "同语料" 公平对照)
--------------------------------
1. 语料 corpus = 所有 *已完成抽提* 的段落 (当前 1172 段)。
2. 题目 questions = 其 10 个 distractor 上下文段落 *全部* 在语料中的题
   (当前 111 题)。这样每道题的 2 个 gold + 8 个 distractor 全部在库,
   检索难度结构与 HotpotQA distractor 设定保持一致, 且不存在
   "某个 gold 段落根本没有被索引" 这种人为失败。
3. 划分 dev/test 在上述题目上做 (见 hgrag4_splits.py)。

输出: C:\\data\\zsk\\RAG\\kgv4
  passages.json / questions.json  (语料与题目)
  raw_extractions.jsonl           (从 kg300 复制, 供 --stage build 使用)
"""
from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import activation_graphrag as A  # noqa: E402

DATA_ROOT = r"C:\data\zsk\RAG"
KG300 = os.path.join(DATA_ROOT, "kg300")
KGV4 = os.path.join(DATA_ROOT, "kgv4")
N_QUESTIONS = 300


def main() -> int:
    os.makedirs(KGV4, exist_ok=True)

    # ---------- 1. 抽提记录 ----------
    src_raw = os.path.join(KG300, "raw_extractions.jsonl")
    dst_raw = os.path.join(KGV4, "raw_extractions.jsonl")
    shutil.copy2(src_raw, dst_raw)
    extracted = {r["pid"] for r in A.read_jsonl(dst_raw)}
    print(f"[extract] 已抽提段落: {len(extracted)}")

    # ---------- 2. 完整语料 (300 题) ----------
    passages, questions = A.load_hotpotqa_corpus(
        DATA_ROOT, num_questions=N_QUESTIONS, split="validation")
    pid_of = {p.title: p.pid for p in passages}

    # ---------- 3. 每题的 10 个上下文段落是否都在已抽提集合内 ----------
    # 重新从原始 dataset 读 context, 因为 load_hotpotqa_corpus 不返回 context
    from datasets import load_from_disk
    ds = load_from_disk(os.path.join(DATA_ROOT, "datasets",
                                     "hotpotqa_distractor_validation"))
    data = ds if "question" in getattr(ds, "column_names", []) else ds["validation"]

    keep_idx = []
    for i in range(len(data)):
        titles = list(data[i]["context"]["title"])
        if all(pid_of.get(t) in extracted for t in titles if t):
            keep_idx.append(i)
    print(f"[questions] 上下文段落全部已抽提的题: {len(keep_idx)}")

    # ---------- 4. 语料 = 已抽提段落 ----------
    keep_pids = {p.pid for p in passages if p.pid in extracted}
    out_passages = [p.to_dict() for p in passages if p.pid in keep_pids]
    print(f"[passages] 语料段落: {len(out_passages)}")

    out_questions = [questions[i] for i in keep_idx]
    A.write_json(os.path.join(KGV4, "passages.json"), out_passages)
    A.write_json(os.path.join(KGV4, "questions.json"), out_questions)

    # ---------- 5. 覆盖率自检 ----------
    gold_missing = 0
    for q in out_questions:
        for t in q.get("gold_titles", []):
            if t not in {p["title"] for p in out_passages}:
                gold_missing += 1
    print(f"[check] gold 段落缺失数: {gold_missing}  (应为 0)")
    print(f"[ok] 输出目录: {KGV4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
