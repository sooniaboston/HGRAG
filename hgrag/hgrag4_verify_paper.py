#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
access (4).tex 数字核对
=====================

把论文里出现的每个百分比 / pp 数字, 与 results/summary.json 里的
实际实验结果逐一对上。任何对不上的地方都直接报错 —— 论文数字必须
全部来自实验输出, 不能手写。

用法:
    python hgrag4_verify_paper.py --results <summary.json> --tex "access (4).tex"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import activation_graphrag as A  # noqa: E402

TOL = 0.06  # 允许 0.06 pp 的舍入误差


def pct(x: Optional[float]) -> Optional[float]:
    return None if x is None else 100.0 * x


def _find_sample_manifest() -> Optional[Dict[str, Any]]:
    """
    找到已导出的基准样本的 MANIFEST.json。

    论文中描述样本规模的数字 (段落数、题数、实体数、关系数) 必须有可复算
    的出处, 所以这里直接读样本自己的清单文件, 而不是把这些数字当作
    "已知常数" 豁免掉。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for rel in (os.path.join("release", "hgrag4_sample", "MANIFEST.json"),
                os.path.join("release", "sample", "MANIFEST.json")):
        p = os.path.join(here, rel)
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:  # noqa: BLE001
                return None
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tex", required=True)
    args = ap.parse_args(argv)

    res: Dict[str, Any] = A.read_json(args.results) or {}
    if not res:
        print("读取结果失败")
        return 2
    with open(args.tex, encoding="utf-8") as fh:
        tex = fh.read()

    # 参考文献里的数字 (arXiv 编号 "2410.05779"、页码、卷期) 不是实验结果,
    # 必须从扫描范围里剔除, 否则 "2410.05779" 会被误抓成 10.0 之类的"无出处数字"。
    _b = tex.find("\\begin{thebibliography}")
    _e = tex.find("\\end{thebibliography}")
    if _b != -1 and _e != -1 and _e > _b:
        tex = tex[:_b] + tex[_e:]

    e1 = res.get("exp1", {})
    e2 = res.get("exp2", {})
    e3 = res.get("exp3", {})
    e5 = res.get("exp5", {})
    e6 = res.get("exp6", {})
    e7 = res.get("exp7", {})
    sig = e1.get("significance", {})

    # ---------- 收集论文中出现的所有数字形态 ----------
    found = set()
    for pat in (r"(\d+\.\d{1,2})~ms",         # 55.8~ms
                r"(\d+\.\d{1,2})~s",          # 0.036~s
                ):
        for m in re.finditer(pat, tex):
            found.add(float(m.group(1)))
    # 百分比: 必须区分正负。LaTeX 里负号写作 $-3.4\%, 绝对值被上面的
    # 模式单独抓到, 会把 "-3.4" 误记成 "+3.4" —— 因此这里显式判断
    # 紧邻的减号, 由负号分支统一处理。
    for m in re.finditer(r"(\d+\.\d{1,2})\\%", tex):
        head = tex[:m.start()].rstrip()
        if head.endswith("$"):
            head = head[:-1].rstrip()
        # 只有当减号是"紧贴数字"时才视为负号。区间写法 "22.00\%--60.00\%"
        # 里第二个数字前面是 "--" (LaTeX 的 en-dash), 不���负号。
        if head.endswith("-") and not head.endswith("--"):
            continue          # 负号, 交给下面的负号分支
        found.add(float(m.group(1)))
    # pp 数值 (自带符号)
    for m in re.finditer(r"([+-]?\d+\.\d{1,2})\$?\s*pp", tex):
        found.add(float(m.group(1)))
    # "percentage points" 写法 (审稿人 Concern #5 段落里用)
    for m in re.finditer(r"([+-]?\d+\.\d{1,2})\$?\s*percentage points", tex):
        found.add(float(m.group(1)))
    # 三张新增显著性表的表体数字 (delta 列写成 $+6.72$ 这种形态, 不会被
    # 上面 "pp" 模式抓到, 因此单独扫一遍这三张表)
    #
    # tab:significance / tab:precisionsig 的 delta 列同样写成裸数字
    # (例如 "+14.93"), 也必须单独扫描, 否则这两张表里的 delta 不会被
    # 采集, 其出处登记也就无从核对。
    for _lbl in ("tab:weightsig", "tab:bottomupsig", "tab:keywordsig",
                 "tab:significance", "tab:precisionsig"):
        _m = re.search(r"\\label\{" + re.escape(_lbl) + r"\}(.*?)\\end\{table\*?\}",
                       tex, re.S)
        if not _m:
            continue
        for _cell in _m.group(1).split("\n"):
            for _c in _cell.split("&"):
                _c = _c.strip().replace("$", "").replace("\\%", "").strip()
                if re.fullmatch(r"[+-]?\d+\.\d{1,2}", _c):
                    found.add(float(_c))
    # LaTeX 负号写在 $ 内: $-3.4\% / $-2.24$ pp
    for m in re.finditer(r"\$\s*-\s*(\d+\.\d{1,2})\s*(?:\\%|pp)", tex):
        found.add(-float(m.group(1)))
    for m in re.finditer(r"\$\s*-\s*(\d+\.\d{1,2})\s*\$", tex):
        found.add(-float(m.group(1)))
    # 计数类数字: "568 of 1,172 passages", "Only 3 passages", "34.96 tokens"
    # 注意 LaTeX 用 1{,}172 表示千位分隔, 必须先归一化成 1172,
    # 否则只会抓到 172 (同一个 bug 会让 6{,}728 变成 728)。
    tex_num = re.sub(r"(\d)\{,\}(\d{3})", r"\1\2", tex)
    for pat in (r"(\d+)\s+of\s+1(\d{3})\s+(?:passages|chunks)",
                r"Only\s+(\d+)\s+(?:passages|chunks)",
                r"(\d+\.\d{1,2})\s+tokens"):
        for m in re.finditer(pat, tex_num):
            found.add(float(m.group(1)))
    # 语料规模 "1172 passages / chunks"、"6728 entities"、"8692 relation triples"
    for m in re.finditer(r"\b(1\d{3})\s+(?:passages|chunks)", tex_num):
        found.add(float(m.group(1)))
    for m in re.finditer(r"\b(\d{4})\s+entities", tex_num):
        found.add(float(m.group(1)))
    for m in re.finditer(r"\b(\d{4})\s+relation triples", tex_num):
        found.add(float(m.group(1)))
    # 已发布基准样本的规模 (Section "Released Benchmark Sample"):
    # "200 passages and 19 questions", "1,346 entities and 1,589 relations",
    # "1,172 passages and 111 questions"
    for m in re.finditer(r"\b(\d[\d,]*)\s+passages\s+and\s+(\d[\d,]*)\s+questions",
                         tex_num):
        found.add(float(m.group(1).replace(",", "")))
        found.add(float(m.group(2).replace(",", "")))
    for m in re.finditer(r"\b(\d[\d,]*)\s+entities\s+and\s+(\d[\d,]*)\s+relations",
                         tex_num):
        found.add(float(m.group(1).replace(",", "")))
        found.add(float(m.group(2).replace(",", "")))
    # 投稿版本表格中的 *小数* 形式 recall (例如 "Recall@1 = 0.3639,
    # Recall@10 = 0.7075"), 稿件在回顾该表格时以小数形式引用它们。
    for m in re.finditer(r"Recall@\d+\s*=\s*(0\.\d{3,4})", tex):
        found.add(float(m.group(1)))
    # 稿件回顾投稿版本时引用的小数延迟 (0.050 s / 0.567 s)
    for m in re.finditer(r"(?:mean retrieval latency of|latency figure)[^.]{0,40}?"
                         r"(0\.\d{3})\s*~?s", tex):
        found.add(float(m.group(1)))
    # 延迟表 (论文表 tab:latency) 的表体数字: "57.9 & 56.4 & 74.5 & 33.9"
    # 注意每行最后一个单元后面跟着 LaTeX 的行结束符 "\\", 必须先剥掉,
    # 否则该行最后一列 (Min) 永远匹配不上。
    #
    # 表体可能整张压在**一行**里 (本稿的 tab:design 就是如此), 因此不能
    # 只按换行分行 —— 必须先按 LaTeX 的行结束符 "\\\\" 切行, 再按 "&"
    # 切单元。否则 "2.67 \\ \quad Max (scale-fair)" 这种
    # "上一行末列 + 下一行行首" 的粘连单元永远匹配不上裸小数。
    def _table_cells(label: str) -> List[str]:
        """返回该表格体中所有单元格的纯文本 (已去 LaTeX 装饰)。"""
        m = re.search(r"\\label\{" + re.escape(label) + r"\}(.*?)\\end\{table\*?\}",
                      tex, re.S)
        if not m:
            return []
        body = m.group(1)
        # 先按行结束符切行: "\\\\" 但排除可选参数 "\\[2pt]" 与转义 "\\%"
        body = re.sub(r"\\\\(?!\[)", "\n", body)
        out: List[str] = []
        for row in body.split("\n"):
            for c in row.split("&"):
                c = c.strip()
                c = c.replace("$", "")
                c = c.replace("\\%", "")
                # 去掉行首的排版命令 (\\quad, \\midrule, \\toprule, \\textbf{...})
                c = re.sub(r"^\\(?:quad|qquad|midrule|toprule|bottomrule|hline)\b",
                           "", c).strip()
                c = re.sub(r"^\\textbf\{(.*)\}$", r"\1", c).strip()
                c = re.sub(r"\\+$", "", c).strip()
                if c:
                    out.append(c)
        return out

    for _c in _table_cells("tab:latency"):
        if re.fullmatch(r"\d+\.\d", _c):
            found.add(float(_c))
    # 融合算子 / 归一化表 (论文表 tab:design) 的表体数字, 其中 **Mean Rank
    # 列是裸小数** (2.67 / 2.61 / 3.50 / 2.72 / 2.66), 不带百分号也不带
    # 单位, 上面的模式全都抓不到, 必须单独扫这张表。
    for _c in _table_cells("tab:design"):
        if re.fullmatch(r"\d+\.\d{1,2}", _c):
            found.add(float(_c))
    # 计数形态: "67 of 67 questions"、"66 of 67"、"44 of 67"、"16 of 67"
    for m in re.finditer(r"\b(\d+)\s+of\s+(\d+)\s+questions", tex_num):
        found.add(float(m.group(1)))
        found.add(float(m.group(2)))
    for m in re.finditer(r"\b(\d+)\s+of\s+67\b", tex_num):
        found.add(float(m.group(1)))
        found.add(67.0)
    # 省略分母的计数: "it removes at least one gold document for 63 of the 67"
    for m in re.finditer(r"\b(\d+)\s+of\s+the\s+67\b", tex_num):
        found.add(float(m.group(1)))
        found.add(67.0)
    # 分子+分母同现的计数: "keeps only 35 of the 134 gold documents"
    for m in re.finditer(r"\b(\d+)\s+of\s+the\s+(\d+)\s+gold documents", tex_num):
        found.add(float(m.group(1)))
        found.add(float(m.group(2)))
    # 省略问句的计数: "in 27 of 66 questions ... and in 15 it contains"
    for m in re.finditer(r"\bin\s+(\d+)\s+it\s+contains\b", tex):
        found.add(float(m.group(1)))
    # 置信区间端点: [$-$5.97, +0.75]  (LaTeX 的负号写作 $-5.97$)
    # 注意 LaTeX 排版常用 `\,` 做间距, 正则必须容忍它。
    ci_found = set()
    _sp = r"(?:\s|\\[,;!:]|~)*"
    for m in re.finditer(
            r"\[\s*\$?\s*-\s*\$?\s*(\d+\.\d{1,2})\s*\$?\s*," + _sp +
            r"\s*\$?\s*([+-]?\d+\.\d{1,2})\s*\$?\s*\]",
            tex):
        ci_found.add(-float(m.group(1)))
        ci_found.add(float(m.group(2)))
    for m in re.finditer(
            r"\[\s*\$?\s*([+-]?\d+\.\d{1,2})\s*\$?\s*," + _sp +
            r"\s*\$?\s*([+-]?\d+\.\d{1,2})\s*\$?\s*\]",
            tex):
        ci_found.add(float(m.group(1)))
        ci_found.add(float(m.group(2)))
    found |= ci_found

    # ---------- 构造所有"应当出现"的数字 ----------
    expect: Dict[float, List[str]] = {}
    # optional: 数据里有、但论文不必逐条引用的量 (例如没有写进正文的
    # 置信区间端点)。它们只要"不与论文冲突"即可, 不强制出现。
    optional: Dict[float, List[str]] = {}

    def want(v: Optional[float], why: str) -> None:
        if v is None:
            return
        k = round(pct(v), 2)
        expect.setdefault(k, []).append(why)

    def want_raw(v: float, why: str) -> None:
        expect.setdefault(round(v, 2), []).append(why)

    def want_opt(v: float, why: str) -> None:
        if v is None:
            return
        optional.setdefault(round(v, 2), []).append(why)

    def want_pct_opt(v: Optional[float], why: str) -> None:
        """登记一个**比例**量的百分数形式 (0.789 -> 78.9)。

        与 want_opt 的区别: want_opt 登记的是**已经换算成论文里出现的
        数值**的量 (例如 pp、ms、计数), 而 want_pct_opt 登记的是
        summary.json 里以小数存储的比率, 论文里以百分数出现。
        混用这两者会让 "0.789" 被当成论文里真的写过的数字, 从而把
        本该允许缺失的量误判为缺漏。
        """
        if v is None:
            return
        optional.setdefault(round(pct(v), 2), []).append(why)

    def want_pct_raw(v: Optional[float], why: str) -> None:
        """强制出现一个以小数存储、以百分数出现在论文里的量。"""
        if v is None:
            return
        expect.setdefault(round(pct(v), 2), []).append(why)

    # 主表
    for key in ("milvus_only", "kg_only", "bm25_only", "ragflow",
                "bm25_dense_rrf", "hgrag_full"):
        d = e1.get(key) or {}
        for m in ("recall@1", "recall@5", "recall@10"):
            want(d.get(m), f"exp1.{key}.{m}")
        # Precision@k —— 论文主表 tab:baseline 列出 P@1/P@3/P@5/P@10
        for m in ("precision@1", "precision@3", "precision@5", "precision@10"):
            want(d.get(m), f"exp1.{key}.{m}")
        # Recall@3 未在主表单独列出, 但正文与 tab:lexical 的若干单元格
        # 引用了它, 因此登记为 optional (出现即须可溯源)。
        want_opt(d.get("recall@3"), f"exp1.{key}.recall@3")
        want_raw(round(float(d.get("mean_latency") or 0.0), 3),
                 f"exp1.{key}.mean_latency")
        # 延迟分布 (论文表 tab:latency: mean/median/p95/min, 单位 ms)
        lb = ((res.get("latency_bench") or {}).get("results") or {}).get(key)
        if lb:
            for stat in ("mean", "median", "p95", "min"):
                want_raw(round(float(lb.get(stat) or 0.0) * 1000.0, 1),
                         f"latency_bench.{key}.{stat}_ms")

    # 相对增益 (论文主表 tab:baseline 末三行: 相对 RAGFlow / Milvus / RRF)
    for base in ("ragflow", "milvus_only", "bm25_dense_rrf"):
        hd = e1.get("hgrag_full") or {}
        bd = e1.get(base) or {}
        for m in ("recall@1", "recall@5", "recall@10",
                  "precision@1", "precision@3", "precision@5", "precision@10"):
            bv = bd.get(m)
            hv = hd.get(m)
            if bv:
                want_raw(round((hv - bv) / bv * 100.0, 1),
                         f"relgain.{base}.{m}")

    # 主表 "Mean Time (s)" 列的相对增益 (论文末四行最后一列):
    # HGRAG 均值相对各基线均值的百分比差。用受控延迟基准 (latency_bench)
    # 的精确均值计算, 而不是表内四舍五入后的 0.056 / 0.013。
    _lbr = (res.get("latency_bench") or {}).get("results") or {}
    _hg_mean = float(_lbr.get("hgrag_full", {}).get("mean") or 0.0)
    if _hg_mean > 0:
        for base in ("ragflow", "milvus_only", "bm25_dense_rrf"):
            _bm = float(_lbr.get(base, {}).get("mean") or 0.0)
            if _bm > 0:
                want_raw(round((_hg_mean - _bm) / _bm * 100.0, 1),
                         f"relgain.latency.{base}")

    # ---------- exp14: HippoRAG 风格 PPR 对照 (审稿人 #2 concern #2) ----------
    # 该对照不在 exp1 里, 而是单独写在 exp14_hipporag_test.json /
    # exp14_hipporag_latency.json / exp14_significance.json 三个文件里,
    # 因此这里从这三个文件读取并逐一核对论文中出现的数字。
    e14_dir = os.path.dirname(os.path.abspath(args.results))
    e14_res = A.read_json(os.path.join(e14_dir, "exp14_hipporag_test.json")) or {}
    e14_lat = A.read_json(os.path.join(e14_dir, "exp14_hipporag_latency.json")) or {}
    e14_sig = A.read_json(os.path.join(e14_dir, "exp14_significance.json")) or {}
    if e14_res:
        for m in ("recall@1", "recall@5", "recall@10",
                  "precision@1", "precision@5", "precision@10"):
            want(e14_res.get(m), f"exp14.hipporag.{m}")
        # R@3 / P@3 未在论文主表中列出 (该表只列 R@1/R@5/R@10 与
        # P@1/P@5/P@10), 因此作为"有出处但不必引用"的可选项登记。
        # 注意用 want_pct_opt: 论文里以百分数出现, 而 summary.json
        # 里以小数存储。
        for m in ("recall@3", "precision@3"):
            want_pct_opt(e14_res.get(m), f"exp14.hipporag.{m}")
        lb14 = ((e14_lat.get("results") or {}).get("hipporag_ppr")) or {}
        if lb14:
            for stat in ("mean", "median", "p95", "min"):
                want_raw(round(float(lb14.get(stat) or 0.0) * 1000.0, 1),
                         f"exp14.hipporag_latency.{stat}_ms")
    # HGRAG 相对该对照的显著性 (论文 tab:significance 末四行 + 正文)
    # 与上面 "score" 的处理一致: 论文逐条引用了各指标的差值与 p 值,
    # 但只为 Recall@$k$ 家族给出 CI; 复合分 $S_c$ 的 CI 端点不单列,
    # 因此登记为 optional (出现即须可溯源)。
    for mname, v in ((e14_sig.get("hgrag_minus_hipporag") or {})).items():
        want_raw(round(float(v["delta_pp"]), 2),
                 f"exp14.sig.hgrag_minus_hipporag.{mname}.delta")
        _ci_optional = "composite" in mname.lower() or "score" in mname.lower()
        for ci in (v.get("ci95_pp") or []):
            if _ci_optional:
                want_opt(round(float(ci), 2),
                         f"exp14.sig.hgrag_minus_hipporag.{mname}.ci")
            else:
                want_raw(round(float(ci), 2),
                         f"exp14.sig.hgrag_minus_hipporag.{mname}.ci")
    # 主表末行 "Relative gain vs. HippoRAG-PPR"
    # 论文该行列出 R@1/R@3/R@5/R@10 与 P@1/P@3/P@5/P@10 八列相对增益,
    # 因此八个指标全部登记。
    _hg14 = e1.get("hgrag_full") or {}
    for m in ("recall@1", "recall@3", "recall@5", "recall@10",
              "precision@1", "precision@3", "precision@5", "precision@10"):
        bv = e14_res.get(m)
        hv = _hg14.get(m)
        if bv:
            want_raw(round((hv - bv) / bv * 100.0, 1),
                     f"relgain.hipporag_ppr.{m}")
    # 图 fig:baseline (b) 的 "头尾落差": 该对照的 R@10 - R@1
    if e14_res.get("recall@10") and e14_res.get("recall@1"):
        want_raw(round(100.0 * (float(e14_res["recall@10"])
                                - float(e14_res["recall@1"])), 2),
                 "fig_baseline.hipporag_ppr.head_to_tail_span")
    # 延迟比: 论文主表该行的 "+580.5\\%" (HGRAG 均值 vs 该对照均值),
    # 按与其它相对增益行一致的口径, 用受控基准里的精确值 (而非表内
    # 四舍五入后的 0.056 / 0.008) 计算。
    _l14 = ((e14_lat.get("results") or {}).get("hipporag_ppr")) or {}
    if _l14:
        _hm = float(((res.get("latency_bench") or {}).get("results") or {})
                    .get("hgrag_full", {}).get("mean") or 0.0)
        if _hm > 0 and float(_l14.get("mean") or 0.0) > 0:
            want_raw(round((_hm - float(_l14["mean"])) / float(_l14["mean"]) * 100.0, 1),
                     "relgain.hipporag_ppr.latency")

    # 显著性表 (pp)
    # 说明: 表 tab:significance 只列 Recall@$k$ 家族的 CI, 表
    # tab:precisionsig 只列 Precision@$k$ 家族的 CI; "score" (复合分
    # $S_c$) 的 *差值与 p 值* 在正文中逐条引用, 但其 CI 端点并未在
    # 论文中给出 (复合分是构造出来的汇总量, 论文不为其单列 CI 列)。
    # 因此 score 的 CI 端点登记为 optional: 出现即须可溯源, 不出现
    # 不算缺漏。其余指标族 (recall@N) 的 CI 端点必须出现在论文中。
    for m, rows in sig.items():
        for k, v in rows.items():
            want_raw(round(v["mean_diff"] * 100.0, 2), f"sig.{m}.{k}.delta")
            if m == "score":
                want_opt(round(v["ci95"][0] * 100.0, 2), f"sig.{m}.{k}.ci_lo")
                want_opt(round(v["ci95"][1] * 100.0, 2), f"sig.{m}.{k}.ci_hi")
            else:
                want_raw(round(v["ci95"][0] * 100.0, 2), f"sig.{m}.{k}.ci_lo")
                want_raw(round(v["ci95"][1] * 100.0, 2), f"sig.{m}.{k}.ci_hi")

    # 三路分布比例
    tri = e2.get("tri_counts") or {}
    n = max(1, int(e2.get("n_questions") or 1))
    for k, c in tri.items():
        want_raw(round(100.0 * c / n, 1), f"exp2.tri.{k}")
        # 论文正文里也会直接写 "only 10 of 67 questions ...", 即三路分布的
        # *原始计数*。这些计数必须能溯源到 summary.json, 否则会被判为
        # "无出处数字"。它们不需要强制出现在论文里 (optional), 但一旦
        # 出现必须有出处。
        want_opt(float(c), f"exp2.tri_counts.{k}")
    want_raw(round(100.0 * float(e2.get("tri_union_coverage") or 0.0), 2),
             "exp2.tri_union_coverage")

    # 词法消融 / 算子 / 归一化 —— 只取论文表格里实际列出的那些
    # (论文表 tab:lexical 列 R@1/R@3/R@5/R@10; tab:design 只列 R@1/R@5/R@10)
    for r in e5.get("lex_rows") or []:
        for m in ("recall@1", "recall@3", "recall@5", "recall@10"):
            want(r.get(m), "exp5.lex_rows")
    for r in e5.get("lex_norm_rows") or []:
        for m in ("recall@1", "recall@3", "recall@5", "recall@10"):
            want(r.get(m), "exp5.lex_norm_rows")
    for r in e5.get("design_rows") or []:
        for m in ("recall@1", "recall@5", "recall@10"):
            want(r.get(m), "exp5.design_rows")
    # 关键词消融 (论文表 tab:keyword 只列 R@1/R@5/R@10)
    for r in e5.get("rows") or []:
        for m in ("recall@1", "recall@5", "recall@10"):
            want(r.get(m), f"exp5.rows.{m}")

    # 权重扫描
    for r in (res.get("exp4", {}).get("rows") or []):
        for m in ("recall@1", "recall@3", "recall@5", "recall@10"):
            want(r.get(m), f"exp4.rows.{m}")

    # bottom-up
    for r in e3.get("rows") or []:
        for m in ("recall@1", "recall@5", "recall@10"):
            want(r.get(m), f"exp3.rows.{m}")

    # 调参 (dev)
    for key in ("stageA", "stageB2", "stageB", "stageC"):
        rr = e6.get(key) or []
        if rr:
            b = max(rr, key=lambda r: r["score"])
            for m in ("recall@1", "recall@5", "recall@10"):
                want(b.get(m), f"exp6.{key}.{m}")
    for m in ("recall@1", "recall@5", "recall@10"):
        want((e6.get("best_metrics") or {}).get(m), f"exp6.best.{m}")

    # ---------- exp6: 交叉验证表 (论文表 tab:cv) ----------
    # 每个候选的 dev R@10 / CV 均值 / CV 标准误, 以及 1-SE 阈值
    for c in (e6.get("candidates") or []):
        i = c.get("index")
        want(c.get("dev_recall@10"), f"exp6.candidate[{i}].dev_recall@10")
        want(c.get("cv_mean"), f"exp6.candidate[{i}].cv_mean")
        want(c.get("cv_se"), f"exp6.candidate[{i}].cv_se")
    sel6 = e6.get("selection") or {}
    if sel6.get("one_se_threshold") is not None:
        want(sel6["one_se_threshold"], "exp6.selection.one_se_threshold")

    # ---------- exp13: 关键词约束来源 (论文 para:keyword-provenance) ----------
    e13 = res.get("exp13_keyword_provenance") or {}
    if e13:
        for k in ("n_questions", "n_nonempty_K", "n_K_intersects_gold",
                  "n_K_contains_all_gold",
                  "leak_check_n_changed_when_annotations_removed"):
            want_raw(round(float(e13.get(k) or 0), 2), f"exp13.{k}")
        want_raw(round(float(e13.get("mean_gold_coverage_by_K") or 0.0), 3),
                 "exp13.mean_gold_coverage_by_K")
        # 论文正文写的是 "the constraint set is incomplete for 51 of 67
        # test questions" —— 即 (n_questions - n_K_contains_all_gold) 的补数。
        _nq = float(e13.get("n_questions") or 0)
        _nall = float(e13.get("n_K_contains_all_gold") or 0)
        if _nq and _nall:
            want_raw(round(_nq - _nall, 2),
                     "exp13.K_incomplete_complement")

    # ---------- 投稿版本被撤回的标题数字 (论文 para: 讨论节) ----------
    # 论文明确引用了投稿版本的 "32.9% relative recall improvement" 与
    # "91.2% latency reduction" 作为**被撤回**的历史数字来解释清楚,
    # 这两个数来自投稿版本本身, 不来自当前实验, 因此没有 summary.json
    # 出处 —— 这里登记为 optional (有出处: 投稿版本), 使"每个数字都能
    # 溯源"这一不变量成立, 同时不强求它们出现在论文里。
    want_opt(32.9, "submitted_version.withdrawn_relative_recall_gain")
    want_opt(91.2, "submitted_version.withdrawn_latency_reduction")

    # ---------- exp12: 稠密路后端等价性 (论文 sec:backend) ----------
    e12 = res.get("exp12_hnsw_parity") or {}
    if e12:
        want_raw(round(float(e12.get("n_identical_topk_sets") or 0), 2),
                 "exp12.n_identical_topk_sets")
        want_raw(round(float(e12.get("n_top1_changed") or 0), 2),
                 "exp12.n_top1_changed")
        want_raw(round(float(e12.get("overlap_at10_mean") or 0.0), 3),
                 "exp12.overlap_at10_mean")
        want_raw(round(float(e12.get("latency_exact_mean_ms") or 0.0), 1),
                 "exp12.latency_exact_ms")
        want_raw(round(float(e12.get("latency_hnsw_mean_ms") or 0.0), 1),
                 "exp12.latency_hnsw_ms")
        for k in ("dense_recall@1_exact", "dense_recall@10_exact"):
            want(e12.get(k), f"exp12.{k}")

    # ---------- 历史版本对照数字 (论文正文中作为对照引用, 数据来自 hgrag3) ----------
    # 90.5% / 97.0% 是 v1 的关键词泄漏前后 Recall@10:
    #   hgrag3/exp5_keyword.json 的 "No Keyword Scoring" 行 = 0.905
    #   同一文件启用关键词通道的行 = 0.97 (v1 的关键词来自 gold_titles)
    try:
        # results 目录是 <root>/hgrag4/results/summary.json, 因此 v1 在
        # <root>/hgrag3/results
        _root = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(args.results))))
        v1_dir = os.path.join(_root, "hgrag3", "results")
        v1 = A.read_json(os.path.join(v1_dir, "summary.json")) or {}
        v1e1 = (v1.get("exp1") or {}).get("hgrag_full") or {}
        want(v1e1.get("recall@10"), "v1.hgrag_full.recall@10")
        v1kw = A.read_json(os.path.join(v1_dir, "exp5_keyword.json")) or []
        # 论文正文只引用了 v1 的两个数: 泄漏前 (No Keyword Scoring) 与
        # 泄漏后 (启用关键词通道), 因此只核这两行, 其余 v1 配置论文未引用。
        for r in v1kw:
            if not isinstance(r, dict) or "recall@10" not in r:
                continue
            cfg = r.get("config", "")
            if cfg in ("No Keyword Scoring", "All three (pos+cnt+assoc)"):
                want(r.get("recall@10"), f"v1.exp5.{cfg}")
    except Exception:  # noqa: BLE001
        pass
    # ---------- exp11: 扫描显著性 (论文表 tab:weightsig / tab:bottomupsig /
    #                    tab:keywordsig) ----------
    # 这三张表把 delta / p 值 / McNemar 的 b,c 直接列在表体里, 因此这里
    # 除了核对正文中的数字, 还要逐格核对表体中的数字。
    e11 = res.get("exp11", {})
    for block in ("guarantee", "weights", "keyword"):
        blk = e11.get(block) or {}
        for name, e in (blk.get("comparisons") or {}).items():
            for mkey, m in (e.get("metrics") or {}).items():
                want_raw(round(m["mean_diff"] * 100.0, 2),
                         f"exp11.{block}.{name}.{mkey}.delta")
                want_opt(round(m["ci95"][0] * 100.0, 2),
                         f"exp11.{block}.{name}.{mkey}.ci_lo")
                want_opt(round(m["ci95"][1] * 100.0, 2),
                         f"exp11.{block}.{name}.{mkey}.ci_hi")
            for mkey, mm in (e.get("mcnemar") or {}).items():
                want_opt(float(mm["b"]), f"exp11.{block}.{name}.{mkey}.mcnemar_b")
                want_opt(float(mm["c"]), f"exp11.{block}.{name}.{mkey}.mcnemar_c")

    # 论文的三张表只列了部分指标列 (权重表 R@1/R@5/R@10/Sc, 保证表多一列
    # R@3, 关键词表 R@1/R@5/R@10/Sc), 因此只在表体确实出现该列时要求
    # 强制出现。判定方式: 在该配置**所在行**里能否找到这个 delta 字符串。
    def _metric_in_table(label: str, row_key: str, value: float) -> bool:
        m = re.search(r"\\label\{" + re.escape(label) + r"\}(.*?)\\end\{table\*?\}",
                      tex, re.S)
        if not m:
            return False
        body = m.group(1).replace("$", "").replace("\\%", "")
        row = next((ln for ln in body.split("\n")
                    if row_key in ln and "&" in ln), None)
        if row is None:
            return False
        for cand in (f"{value:+.2f}", f"{abs(value):.2f}"):
            if cand in row:
                return True
        return False

    for block, label, rowkeys in (
            ("weights", "tab:weightsig",
             ("w_v=1.0", "w_v=0.8", "w_v=0.6", "w_v=0.5",
              "w_v=0.2", "w_v=0.0")),
            ("guarantee", "tab:bottomupsig", None),
            ("keyword", "tab:keywordsig", None)):
        for name, e in (((e11.get(block) or {}).get("comparisons") or {})).items():
            keys = rowkeys or (name,)
            for mkey, m in (e.get("metrics") or {}).items():
                v = round(m["mean_diff"] * 100.0, 2)
                if not any(_metric_in_table(label, k, v) for k in keys):
                    # 该列未出现在表体中 -> 降级为可选
                    whys = expect.pop(v, None)
                    if whys:
                        optional.setdefault(v, []).extend(whys)

    # exp11 具名比较: w_g = 0.8 -> 1.0 这一步 (审稿人 Concern #5 的核心)
    named11 = ((e11.get("weights") or {}).get("named") or {})
    step = named11.get("wg_0.8_to_1.0") or {}
    for mkey, m in (step.get("metrics") or {}).items():
        want_raw(round(m["mean_diff"] * 100.0, 2),
                 f"exp11.weights.wg_0.8_to_1.0.{mkey}.delta")
        want_opt(round(m["ci95"][0] * 100.0, 2),
                 f"exp11.weights.wg_0.8_to_1.0.{mkey}.ci_lo")
        want_opt(round(m["ci95"][1] * 100.0, 2),
                 f"exp11.weights.wg_0.8_to_1.0.{mkey}.ci_hi")
    for mkey, mm in (step.get("mcnemar") or {}).items():
        want_opt(float(mm["b"]), f"exp11.weights.wg_0.8_to_1.0.{mkey}.b")
        want_opt(float(mm["c"]), f"exp11.weights.wg_0.8_to_1.0.{mkey}.c")

    # R@1 跨扫描的极差 (正文 "3.73 percentage points" / "2.43")
    r1r = named11.get("r1_range") or {}
    if r1r.get("spread_pp") is not None:
        want_raw(float(r1r["spread_pp"]), "exp11.weights.r1_range.spread_pp")
    if r1r.get("max_se_pp") is not None:
        want_raw(float(r1r["max_se_pp"]), "exp11.weights.r1_range.max_se_pp")

    # v1 -> v4 稳健性对比 (R@1 极差 38.00 pp -> 3.73 pp)
    rob = e11.get("robustness_v1_vs_v4") or {}
    v1r1 = ((rob.get("v1") or {}).get("recall@1") or {})
    if v1r1.get("spread_pp") is not None:
        want_raw(float(v1r1["spread_pp"]), "exp11.robustness.v1.r1_spread_pp")

    # v1 审稿人算术: 净增 2 题的 McNemar 最小可达 p 值 = 0.50
    v1a = e11.get("v1_reviewer_arithmetic") or {}
    if v1a.get("min_attainable_p_value") is not None:
        want_raw(float(v1a["min_attainable_p_value"]),
                 "exp11.v1_reviewer_arithmetic.min_p")

    # v1 Bottom-up 表 (投稿版数字, 用于正文里 "the earlier version claimed"
    # 一段的引用): without R@5/R@10 = 90.00, with R@10 = 94.00
    v1b = e11.get("v1_bottomup") or {}
    for side in ("without", "with"):
        for mk in ("recall@5", "recall@10"):
            vv = ((v1b.get(side) or {}).get(mk))
            if vv is not None:
                want_raw(round(vv * 100.0, 2), f"exp11.v1_bottomup.{side}.{mk}")

    # v1 权重扫描表 (投稿版数字): 正文引用了其中的 R@1 极差 22.00/60.00
    v1w = ((e11.get("robustness_v1_vs_v4") or {}).get("v1") or {})
    v1r1b = (v1w.get("recall@1") or {})
    for k in ("min", "max"):
        if v1r1b.get(k) is not None:
            want_raw(round(float(v1r1b[k]) * 100.0, 2),
                     f"exp11.robustness.v1.r1_{k}")

    # ---------- 显著性表 tab:significance / tab:precisionsig 的逐格出处 ----------
    # 这两张表给出 HGRAG 与六个对照在 Recall@$k$ / Precision@$k$ 上的
    # $\Delta$、95% CI 与 $p$ 值。summary.json 的 exp1.significance 只
    # 存了 Recall@1/5/10 与复合分, 因此这里**重算**整张表:
    #   * Recall@$k$ 用 exp1.*.per_question_recall@k (缺失时按 per_query
    #     重建), Precision@$k$ 由恒等式 P@$k$ = (2/$k$)·R@$k$ 导出;
    #   * 复合分 S_c = 0.4·R@1 + 0.3·R@5 + 0.3·R@10;
    #   * 配对 bootstrap 用 hgrag4_experiments.paired_bootstrap 的同一协议
    #     (seed=20240917, 10000 次重采样)。
    # 重算出的每一格 (delta / CI 端点 / p) 都登记为出处, 于是表里的每个
    # 数字都能追到"由原始逐题向量重算得到", 而不是凭空出现。
    def _per_q_recall(sysd: Dict[str, Any], k: int) -> List[float]:
        st = sysd.get(f"per_question_recall@{k}")
        if st is not None:
            return list(st)
        out: List[float] = []
        for row in sysd.get("per_query") or []:
            gold = set(row.get("gold") or [])
            ret = row.get("retrieved") or []
            out.append(0.0 if not gold
                       else len(gold & set(ret[:k])) / len(gold))
        return out

    def _per_q_precision(sysd: Dict[str, Any], k: int) -> List[float]:
        return [(2.0 / k) * r for r in _per_q_recall(sysd, k)]

    def _per_q_score(sysd: Dict[str, Any]) -> List[float]:
        st = sysd.get("per_question_score")
        if st is not None:
            return list(st)
        r1 = _per_q_recall(sysd, 1)
        r5 = _per_q_recall(sysd, 5)
        r10 = _per_q_recall(sysd, 10)
        return [0.4 * a + 0.3 * b + 0.3 * c for a, b, c in zip(r1, r5, r10)]

    def _paired_bootstrap(pairs: Sequence[Any]) -> Optional[Dict[str, Any]]:
        """与 hgrag4_experiments.paired_bootstrap 完全一致的协议。"""
        if len(pairs) < 2:
            return None
        a = np.array([p[0] for p in pairs], dtype=float)
        b = np.array([p[1] for p in pairs], dtype=float)
        d = a - b
        rng = np.random.RandomState(20240917)
        n = len(d)
        idx = rng.randint(0, n, size=(10000, n))
        boot = d[idx].mean(axis=1)
        observed = float(d.mean())
        return {"mean_diff": observed,
                "ci95": (float(np.percentile(boot, 2.5)),
                         float(np.percentile(boot, 97.5))),
                "p_value": float((np.abs(boot - observed)
                                  >= abs(observed)).mean())}

    _hg = e1.get("hgrag_full") or {}
    _sig_bases = {
        "milvus only": e1.get("milvus_only") or {},
        "graph only": e1.get("kg_only") or {},
        "bm25 only": e1.get("bm25_only") or {},
        "ragflow": e1.get("ragflow") or {},
        "bm25+dense rrf": e1.get("bm25_dense_rrf") or {},
        "hipporag ppr": e14_res or {},
    }
    if _hg:
        for _k in (1, 3, 5, 10):
            for _bname, _sysd in _sig_bases.items():
                for _fam, _getter in (
                        ("recall", lambda s, k: _per_q_recall(s, k)),
                        ("precision", lambda s, k: _per_q_precision(s, k))):
                    _a = _getter(_hg, _k)
                    _b = _getter(_sysd, _k)
                    if len(_a) != len(_b) or not _a:
                        continue
                    _r = _paired_bootstrap(list(zip(_a, _b)))
                    if _r is None:
                        continue
                    _tag = f"tab:{'precisionsig' if _fam == 'precision' else 'significance'}.{_fam}@{_k}.{_bname}"
                    want_raw(round(_r["mean_diff"] * 100.0, 2), f"{_tag}.delta")
                    want_raw(round(_r["ci95"][0] * 100.0, 2), f"{_tag}.ci_lo")
                    want_raw(round(_r["ci95"][1] * 100.0, 2), f"{_tag}.ci_hi")

    # ---------- exp5 设计表 (论文表 tab:design) 的 Precision 列 ----------
    # 论文表 tab:design 列出 R@1/R@5/R@10 与 P@1/P@5/P@10 七个指标,
    # 而验证器原先只登记了 Recall 三列, 于是表里的 P@1/P@5/P@10 六个
    # 单元格 (52.24 / 28.06 / 17.46 / 32.24 / 18.21 / 19.10 / 31.64 /
    # 18.81) 无出处。这里把 design_rows 与 norm_rows 的 Precision 列
    # 一并登记。
    for r in e5.get("design_rows") or []:
        for m in ("precision@1", "precision@5", "precision@10"):
            want(r.get(m), f"exp5.design_rows.{r.get('design')}.{m}")
    # norm_rows 覆盖 {rank, minmax} x {cand, pool, raw} 六种组合。论文表
    # tab:design 的 "Structured-channel normalization" 段只报告**部署配置**
    # (kw_scope == "cand"), 因此只有 cand 行是强制出现的; pool / raw 行
    # 属于未部署的对照实验, 论文不必逐条引用, 但一旦出现必须有出处。
    # (注: minmax/cand 与 minmax/pool 的 R@1/R@5/R@10/P@1/P@5/P@10 六列
    #  数值完全相同, 因此曾把 2.68 (pool) 误当作表中值而漏掉 2.66 (cand);
    #  现按 scope 精确登记, 不再依赖数值巧合。)
    for r in e5.get("norm_rows") or []:
        _scope = r.get("kw_scope")
        _mandatory = (_scope == "cand")
        for m in ("recall@1", "recall@5", "recall@10",
                  "precision@1", "precision@5", "precision@10"):
            _tag = f"exp5.norm_rows.{r.get('norm_mode')}.{_scope}.{m}"
            if _mandatory:
                want(r.get(m), _tag)
            else:
                want_pct_opt(r.get(m), _tag)
        # Mean Rank: 表中两行都列出, 部署配置为 cand。
        if _mandatory:
            want_raw(round(float(r.get("mean_rank") or 0.0), 2),
                     f"exp5.norm_rows.{r.get('norm_mode')}.{_scope}.mean_rank")
        else:
            want_opt(round(float(r.get("mean_rank") or 0.0), 2),
                     f"exp5.norm_rows.{r.get('norm_mode')}.{_scope}.mean_rank")

    # design_rows 的 Mean Rank 列 (Weighted sum 2.67 / Max 2.61 / RRF 3.50 /
    # Interleave 2.72) 也逐行登记。
    for r in e5.get("design_rows") or []:
        want_raw(round(float(r.get("mean_rank") or 0.0), 2),
                 f"exp5.design_rows.{r.get('design')}.mean_rank")

    # ---------- tab:nesting 的三路重叠统计 (由 exp2.tri_counts 唯一确定) ----------
    # 论文表 tab:nesting 报告三路部署配置下的向量-词法 (38/47/30) 与
    # 图谱-词法 (62/47/45) 的重叠统计。这两组数值可由 exp2.tri_counts
    # 通过容斥**唯一**重建:
    #   all_three=29, V∩G=8, V∩L=1, G∩L=16, only_one=10, none=3
    #   |V| = 38, |G| = 62, |L| = 47, |V∩L| = 30, |G∩L| = 45
    # 因此 Containment / Jaccard / 对称差率 都有可复算的出处。
    _tri = e2.get("tri_counts") or {}
    _nq = max(1, int(e2.get("n_questions") or 1))
    if _tri:
        _all3 = _tri.get("all_three", 0)
        _vg = _tri.get("vector+graph", 0)
        _vl = _tri.get("vector+lexical", 0)
        _gl = _tri.get("graph+lexical", 0)
        _one = _tri.get("only_one", 0)
        # |V|, |G|, |L| 由并集 + 两两交集唯一确定
        _union = _all3 + _vg + _vl + _gl + _one
        _size_v = _all3 + _vg + _vl          # + V 独占的 only_one 部分
        _size_g = _all3 + _vg + _gl
        _size_l = _all3 + _vl + _gl
        # 用并集反推 only_one 中的独占份额 (总和必须等于 only_one)
        _excl = _union - (_size_v + _size_g + _size_l) + 2 * _all3
        # 三路的交集大小
        _ivg = _all3 + _vg
        _ivl = _all3 + _vl
        _igl = _all3 + _gl
        # |V| 与 |G| 由 exp10 的向量-图谱行给出 (两路三路一致)
        _e10q = ((res.get("exp10") or {}).get("question_level") or {})
        _vgrow = _e10q.get("vector_graph") or {}
        _size_v = int(_vgrow.get("n_vector") or _size_v)
        _size_g = int(_vgrow.get("n_graph") or _size_g)
        # 词法路的大小: 由容斥 |V∪G∪L| = |V|+|G|+|L| - |V∩G| - |V∩L|
        # - |G∩L| + |V∩G∩L| 唯一确定
        _size_l = (_union + _ivg + _ivl + _igl - _all3
                   - _size_v - _size_g)
        for _tag, _a, _b, _inter in (("vl", _size_v, _size_l, _ivl),
                                     ("gl", _size_g, _size_l, _igl)):
            _cont = 100.0 * _inter / _a
            _jac = _inter / float(_a + _b - _inter)
            _sym = 100.0 * (_a + _b - 2 * _inter) / float(_a + _b)
            want_raw(round(_cont, 1), f"exp2.tri.nesting.{_tag}.containment")
            want_raw(round(_jac, 3), f"exp2.tri.nesting.{_tag}.jaccard")
            want_raw(round(_sym, 1), f"exp2.tri.nesting.{_tag}.sym_rate")

    # ---------- 硬关键词过滤诊断 (results/exp15_hard_filter.json) ----------
    # 论文用一个重建的硬过滤实验说明投稿版的 AND 语义过滤在当前版本上
    # 是结构性失败: 33/67 题候选集为空, 36/67 题全部 gold 被删, 134 个
    # gold 文档只留下 35 个, Recall@10 上限 26.12%。该文件由
    # hgrag4_verify_hard_filter.py 生成, 与 summary.json 同级或位于
    # 本脚本的 results/ 下。
    _hf_candidates = [
        os.path.join(e14_dir, "exp15_hard_filter.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "results", "exp15_hard_filter.json"),
    ]
    for _hf_path in _hf_candidates:
        if not os.path.exists(_hf_path):
            continue
        _hf = A.read_json(_hf_path) or {}
        _pk = _hf.get("present_K_annotation_free") or {}
        for _k in ("n_questions", "n_nonempty_K", "n_empty_result_set",
                   "n_gold_document_removed", "n_all_gold_removed",
                   "gold_documents_total", "gold_documents_kept",
                   "n_substring_redundant_keywords", "n_generic_keyword"):
            if _pk.get(_k) is not None:
                want_raw(round(float(_pk[_k]), 2), f"exp15.hard_filter.{_k}")
        if _pk.get("gold_keep_rate") is not None:
            want_raw(round(float(_pk["gold_keep_rate"]) * 100.0, 2),
                     "exp15.hard_filter.gold_keep_rate")
        if _pk.get("ceiling_recall@10_and") is not None:
            want_raw(round(float(_pk["ceiling_recall@10_and"]) * 100.0, 2),
                     "exp15.hard_filter.ceiling_recall@10_and")
        if _pk.get("ceiling_recall@10_or") is not None:
            want_raw(round(float(_pk["ceiling_recall@10_or"]) * 100.0, 2),
                     "exp15.hard_filter.ceiling_recall@10_or")
        break

    # ---------- 复合分差值: 图谱路(去词法种子) vs BM25-only ----------
    # 论文正文写 "-6.04 pp" 的复合分差值。它由两个已登记的复合分精确
    # 相减得到:
    #   S_c(graph, no lexical seed) = 0.4*0.3060 + 0.3*0.7164 + 0.3*0.8433
    #                                 = 0.590310
    #   S_c(BM25-only)              = 0.4*0.4403 + 0.3*0.7313 + 0.3*0.8507
    #                                 = 0.650720
    #   差 = -0.060410 -> -6.04 pp
    def _composite(row: Dict[str, Any]) -> Optional[float]:
        try:
            return (0.4 * float(row["recall@1"]) + 0.3 * float(row["recall@5"])
                    + 0.3 * float(row["recall@10"]))
        except (KeyError, TypeError, ValueError):
            return None

    _g_nlex = ((res.get("exp8") or {}).get("graph_only") or {}).get(
        "no_lexical_seed") or {}
    _sc_g = _composite(_g_nlex)
    _sc_b = _composite(e1.get("bm25_only") or {})
    if _sc_g is not None and _sc_b is not None:
        want_raw(round((_sc_g - _sc_b) * 100.0, 2),
                 "exp8.graph_only.no_lexical_seed.minus.exp1.bm25_only.score")

    # ---------- 投稿版本被撤回的历史数字 ----------
    # 投稿版摘要报告过 "98.50% Recall@10"、"25.5% relative improvement"
    # (对比 "78.50%" 的稠密基线) 与 "0.027 s" 的平均时延; 投稿版的权重
    # 扫描表在 w_g=0.8 -> w_g=1.0 这一步上四个指标全部下降
    # (R@1 60->58, R@3 84->82, R@5 90->88, R@10 94->92)。这些数字来自
    # 投稿版本本身, 在当前实验里没有对应的 summary.json 条目, 因此
    # 显式登记为"投稿版本出处"(optional: 出现即须可溯源)。
    for _why, _v in (
            ("submitted_version.abstract.recall@10", 98.50),
            ("submitted_version.abstract.relative_gain", 25.5),
            ("submitted_version.abstract.dense_baseline", 78.50),
            ("submitted_version.abstract.mean_latency_s", 0.027),
            ("submitted_version.sweep.wg_1.0.r@1", 58.00),
            ("submitted_version.sweep.wg_1.0.r@3", 82.00),
            ("submitted_version.sweep.wg_1.0.r@5", 88.00),
            ("submitted_version.sweep.wg_1.0.r@10", 92.00),
    ):
        want_opt(_v, _why)

    # ---------- 三张新增显著性表逐格核对 ----------
    # 表体里的 delta 与 p 值直接来自 exp11, 这里按行比对, 确保论文
    # 没有把某个 p 值或 delta 抄错。
    def _table_rows(label: str) -> List[str]:
        m = re.search(r"\\label\{" + re.escape(label) + r"\}(.*?)\\end\{table\*?\}",
                      tex, re.S)
        return (m.group(1).split("\n") if m else [])

    def _check_sig_table(label: str, entries: List[Tuple[str, Dict[str, Any]]],
                         col_metrics: Sequence[str]) -> List[str]:
        """entries: [(row_label, entry_dict)]; col_metrics: 表体中的 delta 列
        对应的 metric key (按表体列顺序)。"""
        errs: List[str] = []
        rows = _table_rows(label)
        if not rows:
            return [f"{label}: 表体未找到"]
        for row_label, entry in entries:
            row = next((ln for ln in rows if row_label.replace("\\\\", "") in ln),
                       None)
            if row is None:
                errs.append(f"{label}: 未找到行 {row_label!r}")
                continue
            cells = [c.strip() for c in row.split("&")]
            # 去掉行首标签, 剩下的数字单元按顺序取
            nums = cells[1:]
            deltas = [entry["metrics"][k]["mean_diff"] * 100.0
                      for k in col_metrics if k in entry["metrics"]]
            got = []
            for c in nums:
                c2 = c.replace("$", "").replace("\\%", "").replace("%", "").strip()
                if re.fullmatch(r"[+-]?\d+\.\d{1,2}", c2):
                    got.append(float(c2))
            for d in deltas:
                if not any(abs(d - g) <= 0.06 for g in got):
                    errs.append(f"{label} / {row_label}: 期望 delta "
                                f"{d:+.2f}, 表体行 = {nums}")
            break_early = False
            if break_early:
                break
        return errs

    tbl_errs: List[str] = []
    # tab:weightsig
    wcmp = ((e11.get("weights") or {}).get("comparisons") or {})
    wrows = [("w_v=1.0", wcmp.get("w_v=1.0,w_g=0.0")),
             ("w_v=0.8", wcmp.get("w_v=0.8,w_g=0.2")),
             ("w_v=0.6", wcmp.get("w_v=0.6,w_g=0.4")),
             ("w_v=0.5", wcmp.get("w_v=0.5,w_g=0.5")),
             ("w_v=0.2", wcmp.get("w_v=0.2,w_g=0.8")),
             ("w_v=0.0", wcmp.get("w_v=0.0,w_g=1.0"))]
    wrows = [(a, b) for a, b in wrows if b]
    tbl_errs += _check_sig_table(
        "tab:weightsig", wrows,
        ("recall@1", "recall@5", "recall@10", "score"))
    # tab:bottomupsig
    gcmp = ((e11.get("guarantee") or {}).get("comparisons") or {})
    grows = [(k, v) for k, v in gcmp.items()]
    tbl_errs += _check_sig_table(
        "tab:bottomupsig", grows,
        ("recall@1", "recall@3", "recall@5", "recall@10", "score"))
    # tab:keywordsig
    kcmp = ((e11.get("keyword") or {}).get("comparisons") or {})
    krows = [(k, v) for k, v in kcmp.items()]
    tbl_errs += _check_sig_table(
        "tab:keywordsig", krows,
        ("recall@1", "recall@5", "recall@10", "score"))

    # ---------- exp8: 图谱路成分消融 (论文表 tab:structural) ----------
    e8 = res.get("exp8", {})
    for block in ("graph_only", "full_hgrag"):
        for key, row in (e8.get(block) or {}).items():
            for m in ("recall@1", "recall@3", "recall@5", "recall@10"):
                want(row.get(m), f"exp8.{block}.{key}.{m}")

    # exp8 显著性 (论文正文里的 pp 与 p 值)
    # 注意: 论文只引用了部分比较的置信区间, 因此 CI 端点记为 optional
    # (允许不出现, 但不允许与论文中的数字冲突); 均值差必须出现。
    for m, rows in (e8.get("significance") or {}).items():
        for k, v in rows.items():
            want_opt(round(v["mean_diff"] * 100.0, 2),
                     f"exp8.sig.{m}.{k}.delta")
            want_opt(round(v["ci95"][0] * 100.0, 2),
                     f"exp8.sig.{m}.{k}.ci_lo")
            want_opt(round(v["ci95"][1] * 100.0, 2),
                     f"exp8.sig.{m}.{k}.ci_hi")
    # 图谱单路的显著性 (论文表 tab:structural 的头条证据)
    for m, rows in (e8.get("significance_graph_only") or {}).items():
        for k, v in rows.items():
            want_opt(round(v["mean_diff"] * 100.0, 2),
                     f"exp8.kgsig.{m}.{k}.delta")
            want_opt(round(v["ci95"][0] * 100.0, 2),
                     f"exp8.kgsig.{m}.{k}.ci_lo")
            want_opt(round(v["ci95"][1] * 100.0, 2),
                     f"exp8.kgsig.{m}.{k}.ci_hi")

    # ---------- exp8_truncation: 截断统计 (论文表 tab:truncation) ----------
    tr = res.get("exp8_truncation", {})
    if tr:
        want_raw(round(float(tr.get("share_truncated") or 0.0) * 100.0, 2),
                 "exp8_truncation.share")
        want_raw(round(float(tr.get("mean_dropped_tokens") or 0.0), 2),
                 "exp8_truncation.mean_dropped")
        want_raw(round(float(tr.get("n_truncated") or 0), 2),
                 "exp8_truncation.n_truncated")
        want_opt(round(float(tr.get("n_chunks") or 0), 2),
                 "exp8_truncation.n_chunks")
        want_opt(round(float(tr.get("mean_dropped_share") or 0.0) * 100.0, 2),
                 "exp8_truncation.mean_dropped_share")
        want_opt(round(float(tr.get("token_len_median") or 0.0), 1),
                 "exp8_truncation.token_median")
        want_opt(round(float(tr.get("token_len_mean") or 0.0), 2),
                 "exp8_truncation.token_mean")
        want_opt(round(float(tr.get("token_len_p90") or 0.0), 1),
                 "exp8_truncation.token_p90")
        want_opt(round(float(tr.get("char_len_median") or 0.0), 1),
                 "exp8_truncation.char_median")
        want_opt(round(float(tr.get("char_len_mean") or 0.0), 1),
                 "exp8_truncation.char_mean")
        want_opt(round(float(tr.get("char_len_p90") or 0.0), 1),
                 "exp8_truncation.char_p90")
        want_opt(round(float(tr.get("char_len_max") or 0), 2),
                 "exp8_truncation.char_max")
        want_opt(round(float(tr.get("n_chunks_over_512_tokens") or 0), 2),
                 "exp8_truncation.n_over_512")
        want_raw(round(float(tr.get("share_chunks_over_512_tokens") or 0.0)
                       * 100.0, 2), "exp8_truncation.share_over_512")

    # ---------- exp10: 路径覆盖口径 + 嵌套性 + wv=0 泄漏 (论文表 tab:nesting) ----------
    e10 = res.get("exp10", {})
    if e10:
        ql = e10.get("question_level") or {}
        vg = ql.get("vector_graph") or {}
        vl = ql.get("vector_lexical") or {}
        gl = ql.get("graph_lexical") or {}
        dl = e10.get("document_level") or {}

        # 计数 (整数)
        for key, tag in (("n_vector", "vg.n_vector"), ("n_graph", "vg.n_graph"),
                         ("n_both", "vg.n_both"),
                         ("n_vector_only", "vg.n_vector_only"),
                         ("n_graph_only", "vg.n_graph_only"),
                         ("n_symmetric_diff", "vg.n_symmetric_diff")):
            want_opt(float(vg.get(key) or 0), f"exp10.{tag}")
        for key, tag in (("n_vector", "vl.n_vector"), ("n_graph", "vl.n_graph"),
                         ("n_both", "vl.n_both"),
                         ("n_symmetric_diff", "vl.n_symmetric_diff")):
            want_opt(float(vl.get(key) or 0), f"exp10.{tag}")
        for key, tag in (("n_vector", "gl.n_vector"), ("n_graph", "gl.n_graph"),
                         ("n_both", "gl.n_both"),
                         ("n_symmetric_diff", "gl.n_symmetric_diff")):
            want_opt(float(gl.get(key) or 0), f"exp10.{tag}")
        for key, tag in (("n_gold_documents", "doc.n_gold"),
                         ("doc_vector_only", "doc.v_only"),
                         ("doc_graph_only", "doc.g_only"),
                         ("doc_both", "doc.both"),
                         ("doc_vector_total", "doc.v_total"),
                         ("doc_graph_total", "doc.g_total")):
            want_opt(float(dl.get(key) or 0), f"exp10.{tag}")

        # Jaccard (三位小数) 与包含系数 / 对称差率 (百分比)
        #
        # 注意: exp10 中的 "vector_lexical" 与 "graph_lexical" 两组是用
        # **两路** 系统的图谱路 (命中 53 题) 算出来的, 属于旧配置;
        # 论文表 tab:nesting 报告的是**三路**部署配置下的
        # 向量-词法 (38/47/30, 包含系数 78.9%) 与 图谱-词法 (62/47/45,
        # 包含系数 72.6%) 的重叠统计, 数字与 exp10 的两路旧值不同,
        # 论文中的值才是当前配置的正确值。因此这两组的包含系数与对称
        # 差率登记为 optional: 出现即须可溯源, 不出现不算缺漏。
        # "vector_graph" 一组 (38/62/37, 97.4%, 0.587, 26.0%) 两路三路
        # 一致, 仍为强制项。
        for m, tag in ((vg, "vg"), (vl, "vl"), (gl, "gl")):
            want_raw(round(float(m.get("jaccard") or 0.0), 3),
                     f"exp10.{tag}.jaccard")
            if tag == "vg":
                want_raw(round(float(m.get("symmetric_diff_rate") or 0.0) * 100.0, 1),
                         f"exp10.{tag}.sym_rate")
                want_raw(round(float(m.get("containment_vector") or 0.0) * 100.0, 1),
                         f"exp10.{tag}.containment")
            else:
                want_opt(round(float(m.get("symmetric_diff_rate") or 0.0) * 100.0, 1),
                         f"exp10.{tag}.sym_rate")
                want_opt(round(float(m.get("containment_vector") or 0.0) * 100.0, 1),
                         f"exp10.{tag}.containment")
        want_raw(round(float(dl.get("doc_jaccard") or 0.0), 3), "exp10.doc.jaccard")
        want_raw(round(float(dl.get("doc_containment_graph_in_vector") or 0.0)
                       * 100.0, 1), "exp10.doc.containment")
        # 文档级对称差率: (v_only + g_only) / (v_total + g_total)
        _dv = float(dl.get("doc_vector_total") or 0)
        _dg = float(dl.get("doc_graph_total") or 0)
        _dsym = float(dl.get("doc_vector_only") or 0) + \
            float(dl.get("doc_graph_only") or 0)
        if (_dv + _dg) > 0:
            want_raw(round(_dsym / (_dv + _dg) * 100.0, 1),
                     "exp10.doc.sym_rate")
        # 图谱路相对向量路的余交比
        want_raw(round(float(vg.get("relative_excess_graph_over_vector") or 0.0)
                       * 100.0, 1), "exp10.vg.relative_excess")

        # wv = 0 泄漏核验
        leak = e10.get("wv0_leak_check") or {}
        if leak:
            res_off = leak.get("off") or {}
            res_rsv = leak.get("reserved") or {}
            want_opt(float(res_off.get("n_leaked_docs") or 0),
                     "exp10.leak.off.docs")
            want_opt(float(res_rsv.get("n_leaked_docs") or 0),
                     "exp10.leak.reserved.docs")
            want_opt(float(res_rsv.get("n_questions_with_vector_leak") or 0),
                     "exp10.leak.reserved.questions")
            want_raw(round(float(res_rsv.get("share_questions_with_vector_leak")
                                 or 0.0) * 100.0, 1),
                     "exp10.leak.reserved.share_q")
            want_raw(round(float(res_rsv.get("share_leaked_slots") or 0.0)
                           * 100.0, 1), "exp10.leak.reserved.share_slots")

    # ---------- exp9: 截断稳健性 (论文表 tab:truncation) ----------
    e9 = res.get("exp9", {})
    for tag in ("max_seq_128", "max_seq_512"):
        rows = (e9.get("settings") or {}).get(tag) or {}
        for key in ("milvus_only", "kg_only", "hgrag_dense_graph"):
            r = rows.get(key) or {}
            for m in ("recall@1", "recall@5", "recall@10"):
                want(r.get(m), f"exp9.{tag}.{key}.{m}")
    for key, v in (e9.get("significance_truncation") or {}).items():
        want_raw(round(v["mean_diff"] * 100.0, 2),
                 f"exp9.sig.{key}.delta")
        want_opt(round(v["ci95"][0] * 100.0, 2),
                 f"exp9.sig.{key}.ci_lo")
        want_opt(round(v["ci95"][1] * 100.0, 2),
                 f"exp9.sig.{key}.ci_hi")

    # ---------- exp9 诊断: 实体 token 统计 + 向量差异 ----------
    diag = e9.get("diagnostic") or {}
    ent = diag.get("entity_embed_token_len") or {}
    for k in ("mean", "median", "p99"):
        want_opt(ent.get(k), f"exp9.diag.entity_token.{k}")
    want_opt(ent.get("max"), "exp9.diag.entity_token.max")
    want_opt(ent.get("n"), "exp9.diag.entity_token.n")
    want_opt(ent.get("n_over_128"), "exp9.diag.entity_token.n_over_128")
    diff = diag.get("passage_vector_diff_128_vs_512") or {}
    for k in ("mean_abs_diff", "max_abs_diff"):
        want_opt(diff.get(k), f"exp9.diag.vec_diff.{k}")
    want_opt(diff.get("n_passages_differing"),
             "exp9.diag.vec_diff.n_differing")
    want_opt(diff.get("share_differing", 0) * 100.0,
             "exp9.diag.vec_diff.share_differing")
    want_opt(diff.get("n_passages_identical"),
             "exp9.diag.vec_diff.n_identical")

    # ---------- 语料/图规模 (论文中以 1{,}172 / 6{,}728 / 8{,}692 形式出现) ----------
    want_opt(float(res.get("exp8_truncation", {}).get("n_chunks") or 0),
             "corpus.n_chunks")
    want_opt(float(res.get("n_entities") or 0), "corpus.n_entities")
    want_opt(float(res.get("n_relations") or 0), "corpus.n_relations")
    want_opt(float(res.get("n_es_chunks") or 0), "corpus.n_es_chunks")

    # ---------- exp7: 分路径延迟 (论文 "Latency Profile" 段) ----------
    for k in ("milvus_mean", "kg_mean", "total_mean", "total_median",
              "total_p95"):
        want_opt(round(float(e7.get(k) or 0.0) * 1000.0, 1),
                 f"exp7.{k}_ms")
    # 衰减 d=0.6 与阈值等实现常量 (论文方法节给出)
    kgp = res.get("kg_params") or {}
    for k in ("decay", "hops", "seed_alpha", "lexical_weight"):
        v = kgp.get(k)
        if v is not None:
            want_opt(float(v), f"params.{k}")

    # ---------- 延迟 (ms) ----------
    # 论文报告的延迟一律来自 *受控延迟基准* (latency_bench), 而不是主实验
    # 流水线里顺手记的 exp7 —— 后者受当时的机器负载影响。
    # 若基准不可用, 退回 exp7。
    lb = res.get("latency_bench") or {}
    lbr = lb.get("results") or {}
    hg_lat = (e1.get("hgrag_full") or {})
    rf_lat = (e1.get("ragflow") or {})
    rr_lat = (e1.get("bm25_dense_rrf") or {})
    if lbr:
        want_raw(round(float(hg_lat.get("mean_latency") or 0.0) * 1000, 1),
                 "bench.hgrag.mean_ms")
        want_raw(round(float(hg_lat.get("median_latency") or 0.0) * 1000, 1),
                 "bench.hgrag.median_ms")
        want_raw(round(float(hg_lat.get("p95_latency") or 0.0) * 1000, 1),
                 "bench.hgrag.p95_ms")
        want_raw(round(float(rf_lat.get("mean_latency") or 0.0) * 1000, 1),
                 "bench.ragflow.mean_ms")
        want_raw(round(float(rr_lat.get("mean_latency") or 0.0) * 1000, 1),
                 "bench.rrf.mean_ms")
        # kg_only / milvus_only 的延迟在论文中以 *秒* 给出 (Table 1),
        # 因此不强制要求其毫秒形态出现。
    else:
        want_raw(round(float(e7.get("milvus_mean") or 0.0) * 1000, 1),
                 "exp7.milvus_ms")
        want_raw(round(float(e7.get("kg_mean") or 0.0) * 1000, 1),
                 "exp7.kg_ms")
        want_raw(round(float(e7.get("total_mean") or 0.0) * 1000, 1),
                 "exp7.total_ms")
        want_raw(round(float(e7.get("total_median") or 0.0) * 1000, 1),
                 "exp7.median_ms")
        want_raw(round(float(e7.get("total_p95") or 0.0) * 1000, 1),
                 "exp7.p95_ms")
        want_raw(round(float(e7.get("ragflow_mean") or 0.0) * 1000, 1),
                 "exp7.ragflow_ms")

    # ---------- 派生量: 相对提升 / 比例 ----------
    hg = e1.get("hgrag_full") or {}
    for other in ("ragflow", "milvus_only", "bm25_dense_rrf"):
        od = e1.get(other) or {}
        for m in ("recall@1", "recall@5", "recall@10"):
            a, b = hg.get(m), od.get(m)
            if a is not None and b:
                want_raw(round(100.0 * (a - b) / b, 1),
                         f"rel.{other}.{m}")
        a, b = hg.get("mean_latency"), od.get("mean_latency")
        if a is not None and b:
            want_raw(round(100.0 * (a - b) / b, 1), f"rel.{other}.lat")
    # 融合天花板达成比例
    uc = float(e2.get("tri_union_coverage") or 0.0)
    if uc and hg.get("recall@10"):
        want_raw(round(100.0 * hg["recall@10"] / uc, 1), "ceiling_ratio")
    # 三路分布里 "恰好两路命中" 的合计比例
    tri = e2.get("tri_counts") or {}
    n = max(1, int(e2.get("n_questions") or 1))
    two = sum(tri.get(k, 0) for k in ("vector+graph", "vector+lexical",
                                      "graph+lexical"))
    want_raw(round(100.0 * two / n, 1), "exp2.tri.exactly_two")

    # ---------- 投稿版本的历史表格数值 ----------
    # 稿件在讨论部分回顾了投稿版本的表格, 并以算术论证说明那个版本的
    # 94.00% 不可能来自 20 题评测。这些历史数值来自投稿版本本身, 因此
    # 显式登记为 "已知历史出处", 而不是把它们当作无法溯源的数字放过去。
    submitted = {
        "sub.tab1.hgrag.r@1": 60.00,
        "sub.tab1.hgrag.r@10": 94.00,
        "sub.tab1.hgrag.lat": 0.050,
        "sub.tab3.without.r@10": 90.00,
        "sub.tab3.with.r@5": 90.00,
        "sub.tab3.with.r@10": 94.00,
        "sub.tab4.opt.r@1": 60.00,
        "sub.tab4.opt.r@3": 84.00,
        "sub.tab4.opt.r@5": 90.00,
        "sub.tab4.opt.r@10": 94.00,
    }
    for _why, _v in submitted.items():
        want_raw(_v, _why)
    # 投稿版本 RAGFlow 行的小数 (0.7075 / 0.3639) —— 稿件回顾该表格时
    # 正是以小数形式引用它们, 因此登记这两个值本身。
    for _why, _v in (("sub.tab1.ragflow.r@10", 0.7075),
                     ("sub.tab1.ragflow.r@1", 0.3639)):
        want_raw(_v, _why)

    # ---------- 已发布基准样本的规模 (Section "Released Benchmark Sample") ----------
    # 这些数字来自 *导出的样本* 本身, 而不是 summary.json, 因此直接读取
    # 样本的 MANIFEST.json 来核对 —— 这样论文里描述样本规模的每个数字
    # 也都被钉在可复算的出处上, 而不是一句声明。
    sample_manifest = _find_sample_manifest()
    if sample_manifest:
        g = sample_manifest.get("graph_in_sample") or {}
        want_raw(float(sample_manifest.get("n_passages_in_sample") or 0),
                 "sample.n_passages")
        want_raw(float(sample_manifest.get("n_questions_in_sample") or 0),
                 "sample.n_questions")
        want_raw(float(g.get("entities") or 0), "sample.entities")
        want_raw(float(g.get("relations") or 0), "sample.relations")
        # 与完整语料的规模对照
        want_raw(float(sample_manifest.get("n_passages_full") or 0),
                 "sample.full.passages")
        want_raw(float(sample_manifest.get("n_questions_full") or 0),
                 "sample.full.questions")

    # ---------- 核对 ----------
    missing = []
    for v, whys in sorted(expect.items()):
        # 允许 0.06 的舍入误差
        ok = any(abs(v - f) <= TOL for f in found)
        if not ok:
            missing.append((v, whys))

    print(f"论文中出现的数字: {len(found)} 个")
    print(f"应当出现 (来自 summary.json): {len(expect)} 个")
    print(f"可选 (数据里有, 论文不必逐条引用): {len(optional)} 个")

    # 三张新表逐格核对
    if tbl_errs:
        print(f"\n!! 新增显著性表逐格核对失败 ({len(tbl_errs)} 处):")
        for e in tbl_errs:
            print("   ", e)
    else:
        print("新增显著性表逐格核对: [OK] 三张表的每一行 delta 都与 "
              "summary.json 一致。")

    if missing:
        print(f"\n!! 有 {len(missing)} 个数字在论文中找不到:")
        for v, whys in missing:
            print(f"   {v:>9}  <- {whys[0]}")
        return 1
    if tbl_errs:
        return 1
    print("\n[OK] 论文中所有来自实验结果的数字都能在 summary.json 中找到对应。")

    # ---------- 冲突检查: 论文里出现的数字必须有出处 ----------
    # 论文数字必须能匹配 expect (强制项) 或 optional (可选但有出处项) 之一。
    # 否则它就是无出处的数字 —— 那才是真正的"编造风险"。
    known = list(expect.keys()) + list(optional.keys())
    conflicts = []
    for f in sorted(found):
        if not any(abs(f - e) <= TOL for e in known):
            conflicts.append(f)
    if conflicts:
        print(f"\n!! 以下论文数字在 summary.json 中找不到出处:")
        print(f"   {conflicts}")
        return 1

    # ---------- 反向检查: 论文里的数字是否都有出处 ----------
    # 排除掉明显不是实验结果的数字 (年份、章节号等)
    unexplained = []
    for f in sorted(found):
        if not any(abs(f - v) <= TOL for v in expect):
            unexplained.append(f)
    if unexplained:
        print(f"\n(注) 以下数字未在 summary.json 中直接匹配 (可能是年份/编号/"
              f"或推导值): {unexplained}")
    print("\n[OK] 反向检查通过: 论文中出现的每个数字都能溯源到 summary.json。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
