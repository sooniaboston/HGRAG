#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HGRAG v4 :: 语料扩展 (300 题) —— 为"验证集/测试集分离"准备底座
=========================================================

在 kg100 基础上扩展到 kg300:
  * 复用 kg100 已完成的 raw_extractions.jsonl (断点续传, 避免重复调用 LLM)
  * 只对新增段落做实体/关系抽提
  * 之后由 activation_graphrag.py --stage build 重建图谱

新目录: C:\\data\\zsk\\RAG\\kg300
"""
from __future__ import annotations

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import activation_graphrag as A  # noqa: E402

DATA_ROOT = r"C:\data\zsk\RAG"
KG100 = os.path.join(DATA_ROOT, "kg100")
KG300 = os.path.join(DATA_ROOT, "kg300")
N_QUESTIONS = 300


def main() -> int:
    os.makedirs(KG300, exist_ok=True)

    # ---------- 1. 复用已有抽提结果 ----------
    src_raw = os.path.join(KG100, "raw_extractions.jsonl")
    dst_raw = os.path.join(KG300, "raw_extractions.jsonl")
    if os.path.exists(src_raw) and not os.path.exists(dst_raw):
        shutil.copy2(src_raw, dst_raw)
        print(f"[reuse] 复制抽提结果 -> {dst_raw}")

    # ---------- 2. 构建 300 题语料 ----------
    passages, questions = A.load_hotpotqa_corpus(
        DATA_ROOT, num_questions=N_QUESTIONS, split="validation")
    print(f"[corpus] questions={len(questions)} passages={len(passages)}")

    # ---------- 3. 与 kg100 对比, 统计文本增量 ----------
    old_passages = A.read_json(os.path.join(KG100, "passages.json")) or []
    old_text = {p["pid"]: p.get("text", "") for p in old_passages}
    changed = same = 0
    for p in passages:
        o = old_text.get(p.pid)
        if o is None:
            continue
        if o == p.text:
            same += 1
        else:
            changed += 1
    print(f"[delta] 共有段落 {same + changed}: 文本未变 {same}, 文本有变 {changed}")

    # ---------- 4. 落盘 ----------
    A.write_json(os.path.join(KG300, "questions.json"), questions)
    A.write_json(os.path.join(KG300, "passages.json"),
                 [p.to_dict() for p in passages])

    # ---------- 5. 统计待抽提段落 ----------
    done = {r["pid"] for r in A.read_jsonl(dst_raw)}
    todo = [p for p in passages if p.pid not in done]
    print(f"[extract] 待抽提 {len(todo)} / {len(passages)} "
          f"(已完成 {len(done)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
