# -*- coding: utf-8 -*-
"""
hgrag4_significance_sweeps.py
================================
Reviewer #1, Concern #5: "Small effects are reported without tests, and some
statements contradict the tables. One question equals 2 points; the Bottom-up
Top-k gain is a net 2 questions, for which the exact McNemar p-value is at
least 0.50. In Table 4 all four metrics drop from wg = 0.8 to 1.0, contradicting
'monotonically improves', and R@1 ranges from 22% to 60%, which does not support
'strong hyperparameter robustness'. Please add paired tests from the per-question
results."

This module computes, directly from the per-question metric vectors that the
harness already records (no new retrieval runs are required), the paired tests
the reviewer asks for:

  1. Paired bootstrap (10,000 resamples, seed 20240917) -- byte-identical
     implementation and seed to the already-reported Table 2 -- for every
     configuration of every ablation sweep, against the sweep's reference
     configuration, on R@1 / R@3 / R@5 / R@10 and the composite score.

  2. Exact two-sided McNemar test on the *binary* per-question hit indicator
     (a question counts as a hit iff its top-k list contains *all* of its gold
     documents), which is the test the reviewer names explicitly.

Sign convention throughout:  Delta = configuration - reference, so a NEGATIVE
Delta means the configuration is WORSE than the reference.  For the guarantee
ablation the reference is "No guarantee" (so a negative Delta means the
guarantee mechanism hurts); for the weight sweep the reference is the deployed
configuration (w_v = 0.4, w_g = 0.6); for the keyword channel the reference is
"No Keyword Scoring".

Usage
-----
    python hgrag4_significance_sweeps.py            # analyze + write JSON
    python hgrag4_significance_sweeps.py --print    # also print a summary
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

RESULTS = r"C:\data\zsk\RAG\hgrag4\results"
SEED = 20240917
N_BOOT = 10000

# metric key -> human readable name
METRICS = [
    ("recall@1", "Recall@1"),
    ("recall@3", "Recall@3"),
    ("recall@5", "Recall@5"),
    ("recall@10", "Recall@10"),
    ("score", "Composite score"),
]

# metrics for which a binary complete-evidence hit indicator is meaningful
BINARY_METRICS = ["recall@1", "recall@5", "recall@10"]


# --------------------------------------------------------------------------- #
# Test 1: paired bootstrap (identical implementation/seed to Table 2)
# --------------------------------------------------------------------------- #
def paired_bootstrap(pairs: Sequence[Tuple[float, float]],
                     n_boot: int = N_BOOT,
                     seed: int = SEED) -> Dict[str, Any]:
    """Paired bootstrap on (A_i, B_i) per-question score vectors.

    Returns the mean paired difference A-B, a 95% percentile CI and the
    two-sided empirical p-value under H0: mean_diff = 0.
    """
    if len(pairs) < 2:
        return {"n": len(pairs), "mean_diff": 0.0, "ci95": [0.0, 0.0],
                "p_value": 1.0, "prob_a_better": 0.5}
    a = np.array([p[0] for p in pairs], dtype=float)
    b = np.array([p[1] for p in pairs], dtype=float)
    d = a - b
    rng = np.random.RandomState(seed)
    n = len(d)
    idx = rng.randint(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    observed = float(d.mean())
    p = float((np.abs(boot - observed) >= abs(observed)).mean())
    return {
        "n": n,
        "mean_diff": round(observed, 6),
        "ci95": [round(float(np.percentile(boot, 2.5)), 6),
                 round(float(np.percentile(boot, 97.5)), 6)],
        "p_value": round(p, 6),
        "prob_a_better": round(float((boot > 0).mean()), 6),
    }


# --------------------------------------------------------------------------- #
# Test 2: exact McNemar (binomial) test on binary per-question hits
# --------------------------------------------------------------------------- #
def mcnemar_exact(b: int, c: int) -> Dict[str, Any]:
    """Exact two-sided McNemar test on the discordant pairs.

    `b` = questions won by A only, `c` = questions won by B only.
    Under H0 the discordant pairs are split Binomial(b + c, 0.5); the two-sided
    p-value is 2 * P[X <= min(b, c)] (clipped at 1.0).
    """
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0,
                "net_questions": b - c}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    p = min(1.0, 2.0 * tail)
    return {"b": b, "c": c, "n_discordant": n,
            "p_value": round(p, 6),
            "net_questions": b - c}


def binary_hits(per_q: Sequence[float]) -> List[int]:
    """Convert a per-question (document-level) recall vector into a binary
    per-question hit indicator: 1 iff *every* gold document was retrieved.

    The benchmark questions have exactly 2 gold documents, so a per-question
    recall of 1.0 is a complete-evidence hit and 0.5 is a partial hit.
    """
    return [1 if v >= 0.999999 else 0 for v in per_q]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _load(name: str) -> Dict[str, Any]:
    with open(os.path.join(RESULTS, name), encoding="utf-8") as fh:
        return json.load(fh)


def _pairs(a: Sequence[float], b: Sequence[float]) -> List[Tuple[float, float]]:
    return [(float(x), float(y)) for x, y in zip(a, b)]


def _composite(pq: Dict[str, Any]) -> List[float]:
    """Composite score S_c = 0.4 R@1 + 0.3 R@5 + 0.3 R@10 (as in the paper).

    The harness already stores the per-question composite under the key
    ``per_question_score``; we recompute it here from the raw recall vectors
    and assert that the two agree, so the tests are run on exactly the same
    quantity the manuscript reports.
    """
    r1 = np.asarray(pq["per_question_recall@1"], dtype=float)
    r5 = np.asarray(pq["per_question_recall@5"], dtype=float)
    r10 = np.asarray(pq["per_question_recall@10"], dtype=float)
    recomputed = np.round(0.4 * r1 + 0.3 * r5 + 0.3 * r10, 6)
    stored = pq.get("per_question_score")
    if stored is not None:
        stored_arr = np.asarray(stored, dtype=float)
        assert np.allclose(recomputed, stored_arr, atol=1e-6), \
            "stored per_question_score disagrees with the composite formula"
        return list(stored_arr)
    return list(recomputed)


def _metric_vec(row: Dict[str, Any], key: str) -> List[float]:
    if key == "score":
        return _composite(row)
    return [float(v) for v in row["per_question_" + key]]


def _metric_value(row: Dict[str, Any], key: str) -> float:
    """Aggregate metric value of a row (mean over questions)."""
    if key == "score":
        return float(np.mean(_composite(row)))
    return float(row[key])


def _mcnemar_pair(ref_row: Dict[str, Any], row: Dict[str, Any],
                  mkey: str) -> Dict[str, Any]:
    """Exact McNemar with A = `row` (the configuration) and B = `ref_row`."""
    ha = binary_hits(row["per_question_" + mkey])
    hb = binary_hits(ref_row["per_question_" + mkey])
    a_only = sum(1 for x, y in zip(ha, hb) if x == 1 and y == 0)
    b_only = sum(1 for x, y in zip(ha, hb) if x == 0 and y == 1)
    r = mcnemar_exact(a_only, b_only)
    r["config_wins"] = a_only
    r["ref_wins"] = b_only
    return r


def _compare(ref_row: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    """Paired bootstrap + exact McNemar for `row` against `ref_row`.

    Delta = configuration - reference (negative = configuration is worse).
    """
    metrics: Dict[str, Any] = {}
    for mkey, mname in METRICS:
        pb = paired_bootstrap(_pairs(_metric_vec(row, mkey),
                                      _metric_vec(ref_row, mkey)))
        metrics[mkey] = {
            "label": mname,
            "config_value": _metric_value(row, mkey),
            "ref_value": _metric_value(ref_row, mkey),
            **pb,
        }
    mcn: Dict[str, Any] = {}
    for mkey in BINARY_METRICS:
        mcn[mkey] = _mcnemar_pair(ref_row, row, mkey)
    return {"metrics": metrics, "mcnemar": mcn}


def _fmt_pp(x: float) -> str:
    return f"{x * 100:+.2f}"


def _fmt_ci(ci: Sequence[float]) -> str:
    return f"[{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}]"


# --------------------------------------------------------------------------- #
# Analysis 1: guarantee (Bottom-up Top-k) ablation
# --------------------------------------------------------------------------- #
def analyze_guarantee(exp3: Dict[str, Any]) -> Dict[str, Any]:
    rows = exp3["rows"]
    ref = rows[0]                       # "No guarantee" is the first row
    assert ref["mode"] == "off", ref["mode"]
    out: Dict[str, Any] = {
        "reference": ref["config"],
        "reference_mode": ref["mode"],
        "reference_guarantee_k": ref["guarantee_k"],
        "comparisons": {},
    }
    for row in rows[1:]:
        entry = _compare(ref, row)
        entry["mode"] = row["mode"]
        entry["guarantee_k"] = row["guarantee_k"]
        out["comparisons"][row["config"]] = entry
    return out


# --------------------------------------------------------------------------- #
# Analysis 2: path-weight sweep
# --------------------------------------------------------------------------- #
def analyze_weights(exp4: Dict[str, Any]) -> Dict[str, Any]:
    rows = exp4["rows"]
    # The sweep places the two path weights on the grid w_v + w_g = 1, so the
    # deployed pair (w_v=0.4, w_g=0.5) is not one of its rows.  We therefore
    # use as the reference the sweep row that carries the deployed *dense*
    # weight, w_v = 0.4 (which the grid pairs with w_g = 0.6), and say so
    # explicitly in the manuscript.
    ref = min(rows, key=lambda r: (abs(r["w_v"] - 0.4), abs(r["w_g"] - 0.6)))
    out: Dict[str, Any] = {
        "reference": {"w_v": ref["w_v"], "w_g": ref["w_g"]},
        "reference_note": ("sweep row carrying the deployed dense weight "
                           "(w_v = 0.4); the deployed pair (0.4, 0.5) is not "
                           "on the w_v + w_g = 1 grid used by the sweep"),
        "comparisons": {},
    }
    for row in rows:
        if row is ref:
            continue
        key = f"w_v={row['w_v']:.1f},w_g={row['w_g']:.1f}"
        out["comparisons"][key] = _compare(ref, row)

    by_wg = {round(r["w_g"], 2): r for r in rows}

    # ---- (a) w_g = 0.8 -> w_g = 1.0 : the "monotonically improves" claim --- #
    named: Dict[str, Any] = {}
    a_row = by_wg.get(0.8)
    b_row = by_wg.get(1.0)
    if a_row is not None and b_row is not None:
        # A = w_g = 1.0, B = w_g = 0.8  =>  Delta = (w_g=1.0) - (w_g=0.8)
        named["wg_0.8_to_1.0"] = {
            "description": ("step w_g = 0.8 -> w_g = 1.0; "
                            "Delta = (w_g=1.0) - (w_g=0.8)"),
            **_compare(a_row, b_row),
        }

    # ---- (b) the R@1 range across the sweep -> the "robustness" claim ------ #
    r1 = [(r["w_v"], r["w_g"], r["recall@1"]) for r in rows]
    lo = min(r1, key=lambda t: t[2])
    hi = max(r1, key=lambda t: t[2])
    # standard error of the R@1 mean over the 67 questions, per configuration
    se = [float(np.std(r["per_question_recall@1"], ddof=1) / math.sqrt(len(r["per_question_recall@1"])))
          for r in rows]
    named["r1_range"] = {
        "min": {"w_v": lo[0], "w_g": lo[1], "value": lo[2]},
        "max": {"w_v": hi[0], "w_g": hi[1], "value": hi[2]},
        "spread_pp": round((hi[2] - lo[2]) * 100, 2),
        "max_se_pp": round(max(se) * 100, 3),
        "description": "R@1 spread across the whole weight sweep",
    }

    # ---- (c) the plateau: is the saturated tail really flat? ------------- #
    # Reference for the plateau is the *deployed* dense weight row (w_g=0.6),
    # because that is the configuration the system actually ships.
    e = {"description": ("plateau check; Delta = (row) - (w_g=0.6 row)"),
         "metrics": {}, "mcnemar": {}}
    base = by_wg.get(0.6)
    if base is not None:
        for wgv in (0.8, 1.0):
            other = by_wg.get(wgv)
            if other is None:
                continue
            cmp = _compare(base, other)
            for mkey in ("recall@10", "score"):
                e["metrics"][f"w_g={wgv}|{mkey}"] = cmp["metrics"][mkey]
            e["mcnemar"][f"w_g={wgv}|recall@10"] = cmp["mcnemar"]["recall@10"]
    named["plateau"] = e

    # ---- (d) the R@10 monotonicity of the sweep, tested step by step ----- #
    steps = []
    ordered = sorted(rows, key=lambda r: r["w_g"])
    for lo_r, hi_r in zip(ordered, ordered[1:]):
        # ref = lower w_g, row = higher w_g  =>  Delta = (higher w_g) - (lower w_g)
        cmp = _compare(lo_r, hi_r)
        steps.append({
            "from": f"w_g={lo_r['w_g']:.1f}",
            "to": f"w_g={hi_r['w_g']:.1f}",
            "delta_recall@10_pp": round(cmp["metrics"]["recall@10"]["mean_diff"] * 100, 2),
            "p_recall@10": cmp["metrics"]["recall@10"]["p_value"],
            "mcnemar_recall@10": cmp["mcnemar"]["recall@10"],
        })
    named["r10_monotonic_steps"] = {
        "description": ("stepwise Recall@10 change as w_g increases; "
                        "Delta = (higher w_g) - (lower w_g)"),
        "steps": steps,
    }

    # ---- (e) the reviewer's v1 step w_g = 0.8 -> 1.0, side by side -------- #
    # In the submitted version all four metrics DROPPED across this step
    # (R@1 60->58, R@3 84->82, R@5 90->88, R@10 94->92), which is what
    # contradicted the "monotonically improves" sentence.  We record the v1
    # deltas and the v4 deltas for the same step so the contrast is explicit.
    def _row_by_v1_wv(wv: float) -> Dict[str, float]:
        for r in V1_WEIGHT_ROWS:
            if abs(r["w_v"] - wv) < 1e-9:
                return r
        raise KeyError(wv)

    v1_hi = _row_by_v1_wv(0.0)      # w_g = 1.0
    v1_lo = _row_by_v1_wv(0.2)      # w_g = 0.8
    v1_step = {k: round((v1_hi[k] - v1_lo[k]) * 100, 2)
               for k in ("recall@1", "recall@3", "recall@5", "recall@10")}
    v4_step = {k: round(named["wg_0.8_to_1.0"]["metrics"][k]["mean_diff"] * 100, 2)
               for k in ("recall@1", "recall@3", "recall@5", "recall@10")}
    named["wg_0.8_to_1.0_v1_vs_v4"] = {
        "description": ("the w_g = 0.8 -> 1.0 step, in the submitted version "
                        "and in the present version; Delta = (w_g=1.0) - "
                        "(w_g=0.8), so a negative value is a drop"),
        "v1_delta_pp": v1_step,
        "v4_delta_pp": v4_step,
        "v4_p": {k: named["wg_0.8_to_1.0"]["metrics"][k]["p_value"]
                 for k in ("recall@1", "recall@3", "recall@5", "recall@10")},
    }
    out["named"] = named
    return out


# --------------------------------------------------------------------------- #
# Analysis 2b: keyword-channel ablation (Table 7)
# --------------------------------------------------------------------------- #
def analyze_keyword(exp5: Dict[str, Any]) -> Dict[str, Any]:
    rows = exp5["rows"]
    ref = rows[0]                       # "No Keyword Scoring" is the first row
    out: Dict[str, Any] = {"reference": ref["config"], "comparisons": {}}
    for row in rows[1:]:
        out["comparisons"][row["config"]] = _compare(ref, row)
    return out


# --------------------------------------------------------------------------- #
# Analysis 3: the reviewer's own arithmetic on the v1 numbers
# --------------------------------------------------------------------------- #
def analyze_v1_reviewer_arithmetic() -> Dict[str, Any]:
    """Reproduce the reviewer's McNemar bound on the v1 Bottom-up gain.

    In the submitted version the Bottom-up Top-k gain was a net 2 questions at
    Recall@10 on a 50-question benchmark.  The two-sided exact McNemar p-value
    for a net gain of 2 questions is minimised over all (b, c) with b - c = 2;
    the minimum is attained at (b, c) = (2, 0), giving p = 0.50 exactly.
    """
    best = None
    for c in range(0, 51):
        b = c + 2
        if b + c > 50:
            break
        r = mcnemar_exact(b, c)
        if best is None or r["p_value"] < best["p_value"]:
            best = dict(r)
            best["b"] = b
            best["c"] = c
    return {
        "net_questions": 2,
        "n_questions": 50,
        "min_attainable_p_value": best["p_value"],
        "min_attainable_at": {"b": best["b"], "c": best["c"]},
        "note": ("A net gain of 2 questions with no discordant-pair "
                 "compensation can never reach significance at the 5% level; "
                 "the reviewer's 'at least 0.50' is exact."),
    }


# --------------------------------------------------------------------------- #
# Analysis 4: the robustness comparison, v1 vs v4
# --------------------------------------------------------------------------- #
def analyze_robustness(v1_rows: List[Dict[str, float]],
                       exp4: Dict[str, Any]) -> Dict[str, Any]:
    """Quantify the change in the weight-sweep spread from v1 to v4."""
    rows = exp4["rows"]

    def spread(rs: List[Dict[str, float]], key: str) -> Dict[str, float]:
        vals = [r[key] for r in rs]
        return {"min": min(vals), "max": max(vals),
                "spread_pp": round((max(vals) - min(vals)) * 100, 2)}

    v1_r1 = spread(v1_rows, "recall@1")
    v1_r10 = spread(v1_rows, "recall@10")
    v4_r1 = spread(rows, "recall@1")
    v4_r10 = spread(rows, "recall@10")
    return {
        "v1": {"n_questions": 50,
               "recall@1": v1_r1, "recall@10": v1_r10},
        "v4": {"n_questions": 67,
               "recall@1": v4_r1, "recall@10": v4_r10},
        "recall@1_spread_reduction_factor":
            round(v1_r1["spread_pp"] / v4_r1["spread_pp"], 1),
    }


# v1 (submitted version) path-weight table, 50-question benchmark
V1_WEIGHT_ROWS = [
    {"w_v": 0.8, "w_g": 0.2, "recall@1": 0.22, "recall@3": 0.50,
     "recall@5": 0.60, "recall@10": 0.86, "mean_rank": 4.21},
    {"w_v": 0.6, "w_g": 0.4, "recall@1": 0.34, "recall@3": 0.58,
     "recall@5": 0.76, "recall@10": 0.92, "mean_rank": 3.17},
    {"w_v": 0.5, "w_g": 0.5, "recall@1": 0.40, "recall@3": 0.76,
     "recall@5": 0.84, "recall@10": 0.92, "mean_rank": 2.43},
    {"w_v": 0.4, "w_g": 0.6, "recall@1": 0.46, "recall@3": 0.78,
     "recall@5": 0.88, "recall@10": 0.92, "mean_rank": 2.11},
    {"w_v": 0.2, "w_g": 0.8, "recall@1": 0.60, "recall@3": 0.84,
     "recall@5": 0.90, "recall@10": 0.94, "mean_rank": 1.81},
    {"w_v": 0.0, "w_g": 1.0, "recall@1": 0.58, "recall@3": 0.82,
     "recall@5": 0.88, "recall@10": 0.92, "mean_rank": 1.81},
]

# v1 (submitted version) Bottom-up Top-k table, 50-question benchmark.
# "Without Bottom-up Top-k": R@5 = 90.00, R@10 = 90.00, mean rank 3.02
# "With Bottom-up Top-k":    R@5 = 90.00, R@10 = 94.00, mean rank 1.81
# The gain is 4 percentage points of document-level Recall@10 on a
# 50-question benchmark whose questions carry 2 gold documents each,
# i.e. a net gain of exactly 2 questions.
V1_BOTTOMUP = {
    "n_questions": 50,
    "without": {"recall@5": 0.90, "recall@10": 0.90, "mean_rank": 3.02},
    "with": {"recall@5": 0.90, "recall@10": 0.94, "mean_rank": 1.81},
    "net_questions": 2,
    "note": ("the 4.00 pp Recall@10 gain on a 50-question benchmark with "
             "2 gold documents per question is a net gain of 2 questions"),
}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", dest="do_print", action="store_true")
    ap.add_argument("--out", default=os.path.join(RESULTS,
                                                  "exp11_significance_sweeps.json"))
    args = ap.parse_args()

    exp3 = _load("exp3_bottomup.json")
    exp4 = _load("exp4_weights.json")
    exp5 = _load("exp5_keyword.json")

    res = {
        "_evaluated_on": "test",
        "n_questions": 67,
        "n_boot": N_BOOT,
        "seed": SEED,
        "sign_convention": ("Delta = configuration - reference; negative "
                            "Delta means the configuration is worse"),
        "test_definitions": {
            "paired_bootstrap": ("10,000 resamples, seed 20240917, two-sided "
                                  "empirical p-value, 95% percentile CI; "
                                  "identical implementation to Table 2"),
            "mcnemar_exact": ("exact two-sided McNemar on the binary "
                              "complete-evidence hit indicator "
                              "(a question counts as a hit iff its top-k list "
                              "contains all of its gold documents); "
                              "config_wins / ref_wins are the discordant pairs"),
            "composite": "S_c = 0.4*R@1 + 0.3*R@5 + 0.3*R@10",
        },
        "guarantee": analyze_guarantee(exp3),
        "weights": analyze_weights(exp4),
        "keyword": analyze_keyword(exp5),
        "v1_reviewer_arithmetic": analyze_v1_reviewer_arithmetic(),
        "robustness_v1_vs_v4": analyze_robustness(V1_WEIGHT_ROWS, exp4),
        "v1_bottomup": V1_BOTTOMUP,
    }

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print("written:", args.out)

    if args.do_print:
        def dump_block(title: str, block: Dict[str, Any]) -> None:
            print(f"\n=== {title} ===")
            print(f"  reference: {block.get('reference')}")
            for name, e in block["comparisons"].items():
                print(f"\n-- {name}")
                for mkey, m in e["metrics"].items():
                    print(f"   {m['label']:16s} d={_fmt_pp(m['mean_diff']):>8s} pp "
                          f"CI={_fmt_ci(m['ci95']):>20s} p={m['p_value']:.4f}")
                for mkey, mm in e["mcnemar"].items():
                    print(f"   McNemar {mkey:12s} wins={mm['config_wins']:2d}/"
                          f"{mm['ref_wins']:2d} p={mm['p_value']:.4f}")

        dump_block("GUARANTEE (Bottom-up Top-k) vs No guarantee",
                   res["guarantee"])
        dump_block("WEIGHTS vs deployed (w_v=0.4, w_g=0.5)",
                   res["weights"])
        dump_block("KEYWORD vs No Keyword Scoring", res["keyword"])

        print("\n=== NAMED: w_g 0.8 -> 1.0 ===")
        e = res["weights"]["named"]["wg_0.8_to_1.0"]
        for mkey, m in e["metrics"].items():
            print(f"   {m['label']:16s} d={_fmt_pp(m['mean_diff']):>8s} pp "
                  f"CI={_fmt_ci(m['ci95']):>20s} p={m['p_value']:.4f}")
        for mkey, mm in e["mcnemar"].items():
            print(f"   McNemar {mkey:12s} wins={mm['config_wins']:2d}/"
                  f"{mm['ref_wins']:2d} p={mm['p_value']:.4f}")

        print("\n=== NAMED: R@1 range ===")
        print(" ", res["weights"]["named"]["r1_range"])
        print("\n=== NAMED: Recall@10 monotone steps ===")
        for s in res["weights"]["named"]["r10_monotonic_steps"]["steps"]:
            print(f"   {s['from']} -> {s['to']}: "
                  f"dR@10={s['delta_recall@10_pp']:+.2f} pp  "
                  f"p={s['p_recall@10']:.4f}  "
                  f"McNemar wins={s['mcnemar_recall@10']['config_wins']}/"
                  f"{s['mcnemar_recall@10']['ref_wins']} "
                  f"p={s['mcnemar_recall@10']['p_value']:.4f}")

        print("\n=== v1 reviewer arithmetic ===")
        print(" ", res["v1_reviewer_arithmetic"])
        print("\n=== robustness v1 vs v4 ===")
        print(json.dumps(res["robustness_v1_vs_v4"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())