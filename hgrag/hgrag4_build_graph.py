#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HGRAG v4 :: 图谱构建 (用 *过滤后* 的语料)
=========================================

为什么要单独写一个构建脚本
--------------------------
`activation_graphrag.py --stage build` 的流程是:
    stage_corpus()  -> 重新写 passages.json / questions.json (全量 300 题)
    stage_build()   -> 用全量 passages 算 IDF / 共现关联

但 v4 的图谱只能建立在 *已抽提* 的 1172 个段落上 (DeepSeek 余额耗尽)。
若用全量 passages 算关联, IDF 分母与最终语料不一致, KG 路打分会偏。

本脚本:
    1. 读取 kgv4/passages.json (已过滤的 1172 段) 与 questions.json (111 题)
    2. 跳过 stage_corpus, 直接调 stage_build —— 用的是过滤后的 passages
    3. 输出到 kgv4_build/, 再由 hgrag4_verify_corpus.py 同步回 kgv4/
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import activation_graphrag as A  # noqa: E402

LOG = logging.getLogger("hgrag4.build")

KGV4 = r"C:\data\zsk\RAG\kgv4"
KGBUILD = r"C:\data\zsk\RAG\kgv4_build"


def main() -> int:
    p = A.build_argparser()
    # 与 kg100 调优后一致的超参
    argv = [
        "--data-root", r"C:\data\zsk\RAG",
        "--output-dir", KGBUILD,
        "--stage", "build",
        "--num-questions", "300",
        "--no-llm-arbitration",
        "--merge-threshold", "0.82",
        "--name-gate", "0.65",
        "--w-emb", "0.7", "--w-cooc", "0.2", "--w-kge", "0.1",
        "--assoc-top-k", "8", "--min-assoc", "0.35", "--kg-hops", "2",
        "--decay", "0.6", "--hops", "2",
        "--indirect-top-k", "12", "--indirect-min", "0.02",
        "--firing-threshold", "0.001", "--prune-per-row", "60",
        "--seed-entities", "3", "--seed-alpha", "0.9",
        "--lexical-weight", "0.3", "--norm-beta", "0.5",
        "--dense-weight", "0.1", "--title-bonus", "0.1",
        "--no-evaluate",
    ]
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    os.makedirs(KGBUILD, exist_ok=True)

    # ---------- 读取过滤后的语料 ----------
    raw_passages = A.read_json(os.path.join(KGV4, "passages.json")) or []
    questions = A.read_json(os.path.join(KGV4, "questions.json")) or []
    passages = [A.Passage(pid=p["pid"], title=p["title"], text=p["text"],
                          sentences=p.get("sentences", []) or [])
                for p in raw_passages]
    LOG.info("使用过滤后语料: %d 段落, %d 问题", len(passages), len(questions))

    # ---------- 复用抽提结果 ----------
    src_raw = os.path.join(KGV4, "raw_extractions.jsonl")
    dst_raw = os.path.join(KGBUILD, "raw_extractions.jsonl")
    if os.path.exists(src_raw) and not os.path.exists(dst_raw):
        import shutil
        shutil.copy2(src_raw, dst_raw)
        LOG.info("复制抽提结果 -> %s", dst_raw)

    # ---------- 构建 ----------
    pipeline = A.ActivationGraphRAGPipeline(args)
    pipeline.stage_build(passages, questions)
    LOG.info("构建完成 -> %s", KGBUILD)
    return 0


if __name__ == "__main__":
    sys.exit(main())
