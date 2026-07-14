#!/usr/bin/env python3
"""Where the time goes: each span, per condition, per variant.

Reports the FLOOR (min) and the MEAN, deliberately not the median.

  floor -- the run with no retries. Stable, and the right number for mechanism
           claims: client_setup's floor is an exact multiple of RTT, which is how
           you read off the number of serial round trips.
  mean  -- averages over the block-quantised modes rather than picking one, so it
           does not jump the way a median does.

If you want a single number to compare the two arms, do not use this script --
use report_paired.py. This one is for seeing the shape.

  ./bench/report_spans.py results.csv
"""

import argparse
import collections
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.lib import BLOCK_QUANTISED, SPANS, chunk_counts, conditions, load, programs, variants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--program", help="restrict to one program")
    args = ap.parse_args()

    rows = load(args.results)
    progs = [args.program] if args.program else programs(rows)
    vs = variants(rows)
    chunks = chunk_counts(rows)

    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["program"], r["netem"], r["metric"], r["variant"])].append(r["sec"])

    metrics = [m for m in SPANS if any(r["metric"] == m for r in rows)]

    for p in progs:
        print(f"===== {p}  ({chunks.get(p, '?')} write messages) =====")
        for c in conditions(rows):
            print(f"  --- {c} ---")
            head = f"    {'span':<15}"
            for v in vs:
                head += f" {v + ' floor':>14} {v + ' mean':>14}"
            print(head)
            for m in metrics:
                line = f"    {m:<15}"
                for v in vs:
                    x = agg.get((p, c, m, v))
                    if not x:
                        line += f" {'-':>14} {'-':>14}"
                        continue
                    line += f" {min(x):13.3f}s {st.mean(x):13.3f}s"
                mark = "  *block-quantised" if m in BLOCK_QUANTISED else ""
                print(line + mark)
            print()

    print("* block-quantised: these wait on a slot, so the distribution is")
    print("  multimodal. Compare them with report_paired.py, not by eye.")


if __name__ == "__main__":
    main()
