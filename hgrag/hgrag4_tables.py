#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HGRAG v4 :: 从 results/summary.json 自动生成 LaTeX 表格
=====================================================

目的: 论文里的每个数字都必须来自实际实验结果, 不能手写。
本脚本读 summary.json, 输出 `tables.tex` (可直接 \\input 到正文),
并在 stdout 打印一份 Markdown 速览, 便于核对。

用法:
    python hgrag4_tables.py --results C:\\data\\zsk\\RAG\\hgrag4\\results\\summary.json \
                            --out C:\\data\\zsk\\RAG\\hgrag4\\tables.tex
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import activation_graphrag as A  # noqa: E402


def pct(x: Optional[float], nd: int = 2) -> str:
    if x is None:
        return "--"
    return f"{100.0 * x:.{nd}f}\\%"


def rel(new: Optional[float], old: Optional[float]) -> str:
    """相对提升 (%), 带符号。"""
    if not new or not old:
        return "--"
    d = 100.0 * (new - old) / old
    return f"${d:+.1f}\\%" if d >= 0 else f"${d:.1f}\\%"


def fmt_rank(x: Optional[float]) -> str:
    return "--" if x is None else f"{x:.2f}"


# ============================================================================
def tbl_baseline(res: Dict[str, Any]) -> str:
    e1 = res["exp1"]
    rows = [
        ("Milvus only (dense, $w_v=1.0$)", "milvus_only"),
        ("Graph path only", "kg_only"),
        ("BM25 only (lexical)", "bm25_only"),
        ("RAGFlow (hybrid $0.7\\!\\cdot\\!\\mathrm{BM25}+0.3\\!\\cdot\\!\\mathrm{Dense}$)", "ragflow"),
        ("BM25 + Dense RRF ($K=60$)", "bm25_dense_rrf"),
        ("\\textbf{HGRAG (3-path + KW)}", "hgrag_full"),
    ]
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Retrieval Quality and Mean Retrieval Latency: HGRAG vs. "
        r"RAGFlow and Single-Path Baselines (Held-Out Test Split)}",
        r"\label{tab:baseline}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"System & Recall@1 & Recall@5 & Recall@10 & Mean Time (s) \\",
        r"\midrule",
    ]
    for label, key in rows:
        d = e1.get(key)
        if not d:
            continue
        bold = key == "hgrag_full"
        f = (lambda s: r"\textbf{%s}" % s) if bold else (lambda s: s)
        lines.append(
            f"{label} & {f(pct(d['recall@1']))} & {f(pct(d['recall@5']))} & "
            f"{f(pct(d['recall@10']))} & {f('%.3f' % d['mean_latency'])} \\\\")
    # 相对提升
    rf = e1.get("ragflow") or {}
    hg = e1.get("hgrag_full") or {}
    if rf and hg:
        lines.append(r"\midrule")
        lines.append(
            r"Relative gain vs.\ RAGFlow & %s & %s & %s & %s \\" % (
                rel(hg["recall@1"], rf["recall@1"]),
                rel(hg["recall@5"], rf["recall@5"]),
                rel(hg["recall@10"], rf["recall@10"]),
                rel(hg["mean_latency"], rf["mean_latency"])))
    mv = e1.get("milvus_only") or {}
    if mv and hg:
        lines.append(
            r"Relative gain vs.\ Milvus & %s & %s & %s & %s \\" % (
                rel(hg["recall@1"], mv["recall@1"]),
                rel(hg["recall@5"], mv["recall@5"]),
                rel(hg["recall@10"], mv["recall@10"]),
                rel(hg["mean_latency"], mv["mean_latency"])))
    rr = e1.get("bm25_dense_rrf") or {}
    if rr and hg:
        lines.append(
            r"Relative gain vs.\ BM25+Dense RRF & %s & %s & %s & %s \\" % (
                rel(hg["recall@1"], rr["recall@1"]),
                rel(hg["recall@5"], rr["recall@5"]),
                rel(hg["recall@10"], rr["recall@10"]),
                rel(hg["mean_latency"], rr["mean_latency"])))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def tbl_significance(res: Dict[str, Any]) -> str:
    """
    配对 bootstrap 显著性检验 (10,000 次重采样, 双侧)。

    只在 TEST 集上做 —— 超参是在 dev 上选的, test 从未被用于任何选择决策,
    因此这里的 p 值反映的是 "在完全未参与调参的数据上, 差异是否稳健"。
    """
    sig = (res.get("exp1") or {}).get("significance") or {}
    if not sig:
        return ""
    metric_names = {"recall@1": "Recall@1", "recall@5": "Recall@5",
                     "recall@10": "Recall@10", "score": "Composite score"}
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Paired Bootstrap Significance Testing of HGRAG against "
        r"Each Baseline (10{,}000 Resamples, Held-Out Test Split). "
        r"$p$ is the two-sided empirical p-value; $\Delta$ is the mean "
        r"paired difference (HGRAG minus baseline) with a 95\% percentile "
        r"confidence interval.}",
        r"\label{tab:significance}",
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"Metric & Comparison & $\Delta$ & 95\% CI & $p$ \\",
        r"\midrule",
    ]
    order = ("recall@1", "recall@5", "recall@10", "score")
    for m in order:
        rows = sig.get(m) or {}
        if not rows:
            continue
        for i, (k, v) in enumerate(rows.items()):
            other = k.replace("hgrag_vs_", "").replace("_", " ")
            mlabel = metric_names.get(m, m) if i == 0 else ""
            lines.append(
                f"{mlabel} & HGRAG vs.\\ {other} & "
                f"{v['mean_diff']*100:+.2f} pp & "
                f"[{v['ci95'][0]*100:+.2f}, {v['ci95'][1]*100:+.2f}] & "
                f"{v['p_value']:.4f} \\\\")
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def tbl_trpaths(res: Dict[str, Any]) -> str:
    """三路 (向量 / 图谱 / 全文) 真值命中分布。"""
    e2 = res.get("exp2") or {}
    tri = e2.get("tri_counts")
    if not tri:
        return ""
    n = e2.get("n_questions", 1)
    rows = [
        ("All three paths (vector + graph + lexical)", "all_three",
         "Consensus: all three routes independently surface the evidence"),
        ("Vector + graph only", "vector+graph",
         "Lexical route misses, but both structural signals agree"),
        ("Vector + lexical only", "vector+lexical",
         "Graph route misses, but dense and lexical evidence agree"),
        ("Graph + lexical only", "graph+lexical",
         "Dense route misses, but structural and lexical evidence agree"),
        ("Only one path", "only_one",
         "Single-route success; only recoverable by full fusion"),
        ("No path", "none",
         "Genuine retrieval failure"),
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Ground-Truth Document Hit Distribution Across the Three "
        r"Retrieval Paths (Held-Out Test Split). The union of the three "
        r"routes bounds the recall attainable by any fusion scheme.}",
        r"\label{tab:trpaths}",
        r"\begin{tabular}{@{}lccp{8.0cm}@{}}",
        r"\toprule",
        r"Hit Category & Count & Proportion & Interpretation \\",
        r"\midrule",
    ]
    for label, key, interp in rows:
        c = tri.get(key, 0)
        lines.append(f"{label} & {c} & {pct(c / max(n, 1), 1)} & {interp} \\\\")
    lines += [r"\midrule"]
    uc = e2.get("tri_union_coverage")
    hg = (res.get("exp1") or {}).get("hgrag_full") or {}
    if uc is not None:
        lines.append(
            r"Union coverage of all three routes (recall ceiling) & -- & "
            r"%s & Attainable upper bound on any fusion scheme \\" % pct(uc, 2))
    if hg:
        lines.append(
            r"Recall@10 attained by HGRAG & -- & %s & "
            r"%.1f%% of the fusion ceiling \\" % (
                pct(hg["recall@10"], 2),
                100.0 * hg["recall@10"] / uc if uc else 0.0))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def tbl_lexical(res: Dict[str, Any]) -> str:
    """全文路 / 文档同一性 / 归一化方式消融。"""
    e5 = res.get("exp5") or {}
    lex_rows = e5.get("lex_rows") or []
    norm_rows = e5.get("lex_norm_rows") or []
    if not lex_rows:
        return ""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Ablation of the Lexical Path and Document-Identity Signal "
        r"(Held-Out Test Split). The lexical route queries the same "
        r"Elasticsearch index as the RAGFlow baseline, so the comparison "
        r"isolates the contribution of the graph route and the fusion "
        r"operator.}",
        r"\label{tab:lexical}",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"Configuration & R@1 & R@3 & R@5 & R@10 & Mean Rank \\",
        r"\midrule",
    ]
    best = max(lex_rows, key=lambda r: (r["recall@10"], r["recall@1"]))
    for r in lex_rows:
        b = r is best
        f = (lambda s: r"\textbf{%s}" % s) if b else (lambda s: s)
        lines.append(
            f"{f(r['config'])} & {f(pct(r['recall@1']))} & "
            f"{f(pct(r['recall@3']))} & {f(pct(r['recall@5']))} & "
            f"{f(pct(r['recall@10']))} & "
            f"{f(fmt_rank(r.get('mean_rank')))} \\\\")
    if norm_rows:
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{6}{@{}l}{\textit{Lexical normalization "
                     r"mode (with the lexical path enabled)}} \\")
        for r in norm_rows:
            lines.append(
                f"\\quad {r['lex_norm']} & {pct(r['recall@1'])} & "
                f"{pct(r['recall@3'])} & {pct(r['recall@5'])} & "
                f"{pct(r['recall@10'])} & "
                f"{fmt_rank(r.get('mean_rank'))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def tbl_dualpath(res: Dict[str, Any]) -> str:
    e2 = res["exp2"]
    ph, n = e2["counts"], e2["n_questions"]
    rows = [
        ("Milvus vector path only", "milvus_only",
         "Pure semantic similarity queries with no structural dependencies"),
        ("Graph path only", "kg_only",
         "Multi-hop or entity-relational queries with semantic blind spots"),
        ("Both paths (dual corroboration)", "both",
         "Strongest confidence; matched by both distributional and structural relevance"),
        ("Neither path (retrieval failure)", "neither",
         "Edge-case queries with extreme multi-level entity nesting"),
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Ground-Truth Document Hit Distribution Across Dual "
        r"Retrieval Paths (Held-Out Test Split)}",
        r"\label{tab:dualpath}",
        r"\begin{tabular}{@{}lccp{8.5cm}@{}}",
        r"\toprule",
        r"Hit Category & Count & Proportion & Interpretation \\",
        r"\midrule",
    ]
    for label, key, interp in rows:
        c = ph.get(key, 0)
        lines.append(f"{label} & {c} & {pct(c / max(n, 1), 1)} & {interp} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def tbl_bottomup(res: Dict[str, Any]) -> str:
    rows = res["exp3"]["rows"]
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Ablation of Dual-Path Guarantee Formulations "
        r"(Held-Out Test Split)}",
        r"\label{tab:bottomup}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Configuration & R@1 & R@5 & R@10 & Mean Rank \\",
        r"\midrule",
    ]
    best = max(rows, key=lambda r: (r["recall@10"], -r.get("mean_rank", 99)))
    for r in rows:
        b = r is best
        f = (lambda s: r"\textbf{%s}" % s) if b else (lambda s: s)
        lines.append(
            f"{f(r['config'])} & {f(pct(r['recall@1']))} & {f(pct(r['recall@5']))}"
            f" & {f(pct(r['recall@10']))} & {f(fmt_rank(r.get('mean_rank')))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def tbl_keyword(res: Dict[str, Any]) -> str:
    rows = res["exp5"]["rows"]
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Ablation of the Keyword-Aware Scoring Module "
        r"(Held-Out Test Split)}",
        r"\label{tab:keyword}",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"Keyword Signal & R@1 & R@5 & R@10 \\",
        r"\midrule",
    ]
    best = max(rows, key=lambda r: (r["recall@10"], r["recall@1"]))
    for r in rows:
        b = r is best
        f = (lambda s: r"\textbf{%s}" % s) if b else (lambda s: s)
        lines.append(
            f"{f(r['config'])} & {f(pct(r['recall@1']))} & {f(pct(r['recall@5']))}"
            f" & {f(pct(r['recall@10']))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def tbl_weights(res: Dict[str, Any]) -> str:
    rows = res["exp4"]["rows"]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Path Weight Sensitivity: Dual-Path Retrieval Performance "
        r"Across Weight Configurations (Held-Out Test Split)}",
        r"\label{tab:weights}",
        r"\begin{tabular}{@{}ccccccc@{}}",
        r"\toprule",
        r"$w_v$ & $w_g$ & R@1 & R@3 & R@5 & R@10 & Mean Rank \\",
        r"\midrule",
    ]
    best = max(rows, key=lambda r: r["recall@10"])
    for r in rows:
        b = r is best
        f = (lambda s: r"\textbf{%s}" % s) if b else (lambda s: s)
        lines.append(
            f"{f('%.1f' % r['w_v'])} & {f('%.1f' % r['w_g'])} & "
            f"{f(pct(r['recall@1']))} & {f(pct(r['recall@3']))} & "
            f"{f(pct(r['recall@5']))} & {f(pct(r['recall@10']))} & "
            f"{f(fmt_rank(r.get('mean_rank')))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def tbl_tuning(res: Dict[str, Any]) -> str:
    e6 = res["exp6"]
    bp = e6["best_params"]
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Staged Parameter Optimization on the Held-Out Validation "
        r"Split (Dev). Hyperparameters are selected by coordinate descent "
        r"with 5-fold cross-validation and the one-standard-error rule; the "
        r"test split is never used for selection.}",
        r"\label{tab:tuning}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Stage & Searched Quantity & R@1 & R@5 & R@10 \\",
        r"\midrule",
    ]
    # 各阶段的代表点 (取该阶段 score 最高的一条)
    stages = [("A", "stageA", "A: $w_v$"),
              ("B2", "stageB2", "B2: $w_{bm25}$ (lexical)"),
              ("B", "stageB", "B: $w_{kw}$"),
              ("C", "stageC", "C: $\\eta$")]
    for tag, key, label in stages:
        rr = e6.get(key)
        if not rr:
            continue
        best = max(rr, key=lambda r: r["score"])
        lines.append(
            f"{label} & best of {len(rr)} trials & {pct(best['recall@1'])} & "
            f"{pct(best['recall@5'])} & {pct(best['recall@10'])} \\\\")
    lines += [r"\midrule"]
    lines.append(
        r"Selected config & $w_v$=%s, $w_g$=%s, $w_{kw}$=%s, $w_{bm25}$=%s, "
        r"$\eta$=%s & %s & %s & %s \\" % (
            bp["w_v"], bp["w_g"], bp["w_kw"], bp["w_bm25"], bp["eta"],
            pct(e6["best_metrics"]["recall@1"]),
            pct(e6["best_metrics"]["recall@5"]),
            pct(e6["best_metrics"]["recall@10"])))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ============================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="生成 LaTeX 表格")
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    res = A.read_json(args.results)
    if res is None:
        raise SystemExit(f"读取失败: {args.results}")

    parts: List[str] = [
        "% 由 hgrag4_tables.py 自动生成, 请勿手工编辑",
        "% 数据来源: " + os.path.basename(args.results),
    ]
    for fn, key in ((tbl_baseline, "exp1"), (tbl_significance, "exp1"),
                    (tbl_trpaths, "exp2"), (tbl_dualpath, "exp2"),
                    (tbl_bottomup, "exp3"), (tbl_weights, "exp4"),
                    (tbl_keyword, "exp5"), (tbl_lexical, "exp5"),
                    (tbl_tuning, "exp6")):
        if key in res:
            t = fn(res)
            if t:
                parts.append(t)
    text = "\n\n".join(parts) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"已写出: {args.out}")

    # ---------- Markdown 速览 ----------
    sp = res.get("split", {})
    print(f"\n划分: dev={sp.get('n_dev')} test={sp.get('n_test')} "
          f"(seed={sp.get('seed')})")
    print(f"评估划分审计: {res.get('evaluated_on')}")
    e1 = res.get("exp1", {})
    print("\n--- exp1 (test) ---")
    for k in ("milvus_only", "kg_only", "bm25_only", "ragflow", "hgrag_full"):
        d = e1.get(k)
        if d:
            print(f"  {k:12s} R@1={d['recall@1']:.4f} R@5={d['recall@5']:.4f} "
                  f"R@10={d['recall@10']:.4f} rank={d.get('mean_rank')} "
                  f"lat={d.get('mean_latency')}")
    if "exp2" in res:
        print("\n--- exp2 双路分布 (test) ---")
        print(" ", res["exp2"]["proportions"])
        print("  三路分布:", res["exp2"].get("tri_counts"))
        print("  三路并集上界:", res["exp2"].get("tri_union_coverage"))
    if "exp6" in res:
        print("\n--- exp6 最优超参 (dev) ---")
        print(" ", res["exp6"]["best_params"])
    sig = e1.get("significance") or {}
    if sig:
        print("\n--- 显著性 (test) ---")
        for m, rows in sig.items():
            for k, v in rows.items():
                print(f"  {m:10s} {k:24s} d={v['mean_diff']:+.4f} "
                      f"p={v['p_value']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
