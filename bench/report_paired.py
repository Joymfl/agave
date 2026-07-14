#!/usr/bin/env python3
"""Paired differences between two variants, with confidence intervals.

THIS IS THE TRUSTWORTHY COMPARISON. Every deploy is paired with its counterpart
from the same run, same condition, same program -- the two arms ran back to back
against the same machine and the same block schedule. Taking the difference
cancels the block quantisation that makes medians jump between modes, so this is
the only summary that survives a hostile reviewer.

A confidence interval that spans zero means "no measurable difference", which is
exactly what you should see for final_tx -- the control, identical RPC code on
both arms. If final_tx ever comes out significant, the rig is lying and nothing
else in the run can be trusted.

  ./bench/report_paired.py results.csv --a baseline --b tpu_next
"""

import argparse
import collections
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.lib import SPANS, conditions, load, programs


def paired(rows, metric, a, b, condition=None, program=None):
    """Deltas of (b - a), one per (run, condition, program) where both arms ran."""
    pairs = collections.defaultdict(dict)
    for r in rows:
        if r["metric"] != metric:
            continue
        if condition and r["netem"] != condition:
            continue
        if program and r["program"] != program:
            continue
        pairs[(r["run"], r["netem"], r["program"])][r["variant"]] = r["sec"]
    return [v[b] - v[a] for v in pairs.values() if a in v and b in v]


def ci95(deltas):
    n = len(deltas)
    if n < 2:
        return None
    mean = st.mean(deltas)
    sem = st.stdev(deltas) / (n**0.5)
    return mean, mean - 1.96 * sem, mean + 1.96 * sem, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--a", default="baseline", help="reference arm")
    ap.add_argument("--b", default="tpu_next", help="arm under test")
    ap.add_argument("--by", choices=["condition", "program", "overall"], default="condition")
    args = ap.parse_args()

    rows = load(args.results)
    metrics = [m for m in SPANS if any(r["metric"] == m for r in rows)]

    print(f"paired: {args.b} minus {args.a}   (positive = {args.b} slower)")
    print("a 95% CI that spans zero means no measurable difference\n")

    groups = {"overall": [None]}.get(args.by) or (
        conditions(rows) if args.by == "condition" else programs(rows)
    )

    for g in groups:
        label = g or "all conditions & programs"
        kw = {}
        if args.by == "condition" and g:
            kw["condition"] = g
        elif args.by == "program" and g:
            kw["program"] = g

        print(f"--- {label} ---")
        print(f"  {'span':<15} {'n':>4} {'delta':>9} {'95% CI':>22}  verdict")
        for m in metrics:
            d = paired(rows, m, args.a, args.b, **kw)
            r = ci95(d)
            if not r:
                continue
            mean, lo, hi, n = r
            sig = "SIGNIFICANT" if lo * hi > 0 else "no difference"
            star = "  <-- control" if m == "final_tx" else ""
            print(f"  {m:<15} {n:4} {mean:+9.3f}s [{lo:+8.3f},{hi:+8.3f}]  {sig}{star}")
        print()


if __name__ == "__main__":
    main()
