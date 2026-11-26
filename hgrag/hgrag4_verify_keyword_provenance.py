#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HGRAG v4 :: 关键词约束集 K 的来源核验 (exp13)
=============================================

为什么需要这个脚本
------------------
论文正文 (para:keyword-provenance) 声称关键词约束集 K 是从 *问句文本*
抽取的、与标注无关的, 并给出了具体的覆盖统计 (66/67 非空, 44/67 与真值
相交, 16/67 覆盖全部真值, 平均真值覆盖度 0.448)。审稿人 #1 的意见 #6
直接问 "the keyword constraints K in the benchmark were obtained" ——
因此这些数字必须能从代码里复算出来, 而不是写死在正文里。

本脚本直接调用生产代码里的 `hgrag4_experiments.extract_keywords_from_question`,
在测试集划分 (seed 20240917) 上复算这些统计, 并额外做两项泄漏检查:

  1. 泄漏检查: 抽取函数是否读取了 gold_titles / answer / supporting_facts?
     -> 通过对 questions 逐条**抹掉**标注字段后重跑, 看 K 是否改变。
        若 K 不变, 说明抽取过程与标注无关 (这是泄漏检查的最强形式)。
  2. 对照统计: 若把 gold_titles 直接当作 K (即 v1 的做法), 覆盖率会变成
     多少? 用于量化"泄漏"与"合法抽取"之间的差别。

输出: results/exp13_keyword_provenance.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hgrag4_splits as SPL  # noqa: E402
from hgrag4_experiments import extract_keywords_from_question  # noqa: E402

#: 与正文一致的口径: K 的抽取只看问句文本
ANNOTATION_FIELDS = ("gold_titles", "answer", "supporting_facts", "type",
                      "level")


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="HGRAG v4 关键词约束来源核验")
    p.add_argument("--kg-dir", default=r"C:\data\zsk\RAG\kgv4")
    p.add_argument("--out-dir", default=r"C:\data\zsk\RAG\hgrag4")
    args = p.parse_args(argv)

    import logging
    logging.basicConfig(level=logging.WARNING)

    res_dir = os.path.join(args.out_dir, "results")
    os.makedirs(res_dir, exist_ok=True)

    passages = json.load(open(os.path.join(args.kg_dir, "passages.json"),
                              encoding="utf-8"))
    questions = json.load(open(os.path.join(args.kg_dir, "questions.json"),
                               encoding="utf-8"))
    entities = json.load(open(os.path.join(args.kg_dir, "entities.json"),
                              encoding="utf-8"))

    class _C:
        pass
    c = _C()
    c.titles = [p["title"] for p in passages]
    c.entities = entities

    split = SPL.load_or_make_split(questions, args.kg_dir,
                                   dev_fraction=SPL.DEV_FRACTION,
                                   seed=SPL.SPLIT_SEED, force=False)
    test_q = SPL.select(questions, split, "test")

    n = len(test_q)
    n_nonempty = n_intersect = n_all = 0
    cov_sum = 0.0
    n_changed_when_annotations_removed = 0
    n_gold_leak_cover = 0

    for q in test_q:
        k = extract_keywords_from_question(q, c)
        gold = set(q["gold_titles"])

        if k:
            n_nonempty += 1
        if gold & set(k):
            n_intersect += 1
        if gold and gold <= set(k):
            n_all += 1
        cov_sum += len(gold & set(k)) / max(len(gold), 1)

        # ---- 泄漏检查: 抹掉所有标注字段后, K 是否改变 ----
        stripped = {kk: vv for kk, vv in q.items()
                    if kk not in ANNOTATION_FIELDS}
        k2 = extract_keywords_from_question(stripped, c)
        if list(k2) != list(k):
            n_changed_when_annotations_removed += 1

        # ---- 对照: 直接用 gold_titles 作为 K (v1 的做法) ----
        if gold:
            n_gold_leak_cover += 1

    out: Dict[str, Any] = {
        "_evaluated_on": "test",
        "n_questions": n,
        "n_nonempty_K": n_nonempty,
        "n_K_intersects_gold": n_intersect,
        "n_K_contains_all_gold": n_all,
        "mean_gold_coverage_by_K": round(cov_sum / max(n, 1), 4),
        "n_questions_with_all_gold_available": n_gold_leak_cover,
        "leak_check_n_changed_when_annotations_removed":
            n_changed_when_annotations_removed,
        "leak_check_passed": n_changed_when_annotations_removed == 0,
        "annotation_fields_stripped": list(ANNOTATION_FIELDS),
        "note": ("K is extracted from question text only. The leak check "
                 "removes every annotation field from the question record and "
                 "re-runs the extractor; if K is unchanged for every "
                 "question, the extraction is provably annotation-free."),
    }
    path = os.path.join(res_dir, "exp13_keyword_provenance.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"[exp13] 已写出 {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
