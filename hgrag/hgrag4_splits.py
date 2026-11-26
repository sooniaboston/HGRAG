#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HGRAG v4 :: 数据划分协议 (验证集 / 测试集分离)
==============================================

v1 (已废弃, 结果见 `access (3).tex`) 的做法:
    同一批 100 道题既用于超参寻优 (exp6) 又用于最终报告 (exp1/exp3/exp4/exp5),
    属于 "在测试集上调参" —— 报告的 Recall 是乐观有偏的。

v4 的做法 (本模块):
    把 HotpotQA distractor validation 的前 N 题按 *固定随机种子* 划分为

        ├─ dev   (验证集, 用于一切超参选择)   —— 默认 40%
        └─ test  (测试集, 仅用于最终报告)     —— 默认 60%

    划分是 *确定性* 的 (seed 固定), 任何人都能复现同一份切分。
    划分文件写入 <kg-dir>/splits.json, 记录题目 id, 便于审计。

关键约束 (由 hgrag4_experiments.py 强制):
    * 调参 (exp6) 只看 dev 的指标;
    * 最终报告 (exp1/exp3/exp4/exp5) 只看 test 的指标;
    * 若某配置需要在 test 上评估, 必须显式传 --eval-split test,
      并在 summary.json 中标记 evaluated_on, 以便审计是否发生 test 泄漏。
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any, Dict, List, Sequence, Tuple

# 划分种子 —— 固定不变, 保证可复现
SPLIT_SEED = 20240917

# dev 占比 (其余为 test)
DEV_FRACTION = 0.4


def _question_key(q: Dict[str, Any], idx: int) -> str:
    """稳定的题目标识: 优先用 id, 否则用 index+文本哈希。"""
    qid = str(q.get("id") or "").strip()
    if qid:
        return qid
    h = hashlib.sha1(
        f"{q.get('question', '')}|{','.join(map(str, q.get('gold_titles', [])))}"
        .encode("utf-8")).hexdigest()[:16]
    return f"idx{idx:05d}_{h}"


def make_split(questions: Sequence[Dict[str, Any]],
               dev_fraction: float = DEV_FRACTION,
               seed: int = SPLIT_SEED) -> Dict[str, List[str]]:
    """
    确定性划分。

    做法: 先用 seed 打乱题目下标, 再按 dev_fraction 取前一段做 dev。
    之所以 "先打乱再切" 而不是 "直接取前 N 道", 是为了避免
    HotpotQA validation 的原始顺序带来的分布偏差 (原数据集按来源聚簇)。
    """
    keys = [_question_key(q, i) for i, q in enumerate(questions)]
    if len(set(keys)) != len(keys):
        raise ValueError("题目 id 不唯一, 无法划分")
    idx = list(range(len(keys)))
    random.Random(seed).shuffle(idx)
    n_dev = int(round(len(idx) * dev_fraction))
    n_dev = max(1, min(n_dev, len(idx) - 1)) if len(idx) > 1 else len(idx)
    dev = sorted(idx[:n_dev])
    test = sorted(idx[n_dev:])
    return {"dev": [keys[i] for i in dev], "test": [keys[i] for i in test]}


def load_or_make_split(questions: Sequence[Dict[str, Any]],
                       kg_dir: str,
                       dev_fraction: float = DEV_FRACTION,
                       seed: int = SPLIT_SEED,
                       force: bool = False) -> Dict[str, Any]:
    """
    读取或生成划分文件 <kg-dir>/splits.json。

    返回:
      {"dev": [qdict...], "test": [qdict...], "meta": {...}}
    """
    path = os.path.join(kg_dir, "splits.json")
    meta = {
        "seed": seed,
        "dev_fraction": dev_fraction,
        "n_total": len(questions),
        "split_file": path,
    }

    if os.path.exists(path) and not force:
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        keys = {_question_key(q, i): q for i, q in enumerate(questions)}
        dev = [keys[k] for k in blob["dev"] if k in keys]
        test = [keys[k] for k in blob["test"] if k in keys]
        missing = len(questions) - len(dev) - len(test)
        if missing:
            raise ValueError(
                f"划分文件与当前语料不匹配: 缺少 {missing} 道题 "
                f"(kg_dir={kg_dir}); 请用 --force-split 重新生成")
        meta.update(blob.get("meta", {}))
        meta["n_total"] = len(questions)
        meta["n_dev"] = len(dev)
        meta["n_test"] = len(test)
        return {"dev": dev, "test": test, "meta": meta}

    sp = make_split(questions, dev_fraction, seed)
    blob = {
        "dev": sp["dev"],
        "test": sp["test"],
        "meta": {**meta, "n_dev": len(sp["dev"]), "n_test": len(sp["test"])},
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, ensure_ascii=False, indent=2)
    meta["n_dev"] = len(sp["dev"])
    meta["n_test"] = len(sp["test"])
    dev_set = set(sp["dev"])
    test_set = set(sp["test"])
    dev_q, test_q = [], []
    for i, q in enumerate(questions):
        k = _question_key(q, i)
        if k in dev_set:
            dev_q.append(q)
        elif k in test_set:
            test_q.append(q)
    return {"dev": dev_q, "test": test_q, "meta": meta}


def select(questions: Sequence[Dict[str, Any]], split: Dict[str, Any],
           which: str) -> List[Dict[str, Any]]:
    """按 'dev' / 'test' 取出对应题目。"""
    if which not in ("dev", "test", "all"):
        raise ValueError(f"未知划分: {which}")
    if which == "all":
        return list(questions)
    return list(split[which])


def audit(split: Dict[str, Any]) -> Dict[str, Any]:
    """给划分做一次体检, 确认两半的题型分布没有明显失衡。"""
    out: Dict[str, Any] = {}
    for name in ("dev", "test"):
        qs = split[name]
        by_type: Dict[str, int] = {}
        by_level: Dict[str, int] = {}
        for q in qs:
            by_type[str(q.get("type", "?"))] = \
                by_type.get(str(q.get("type", "?")), 0) + 1
            by_level[str(q.get("level", "?"))] = \
                by_level.get(str(q.get("level", "?")), 0) + 1
        n = max(1, len(qs))
        out[name] = {
            "n": len(qs),
            "by_type": by_type,
            "by_level": by_level,
            "type_pct": {k: round(100.0 * v / n, 1) for k, v in by_type.items()},
        }
    return out
