#!/usr/bin/env python3
"""Success rate per condition and variant. Run this FIRST, always.

Latency of successful runs is only meaningful next to this number. A variant that
gives up quickly has a flattering latency distribution built out of the runs that
happened to survive, so a fast p99 next to a 60% success rate is worse than a slow
p99 next to 100%.

Outcomes:
  success       deployed, and the program is on chain and executable
  failure       the CLI exited non-zero
  not_deployed  the CLI exited 0 but no executable program exists on chain
  timeout       abandoned after --timeout seconds

  ./bench/report_success.py results.csv
"""

import argparse
import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    args = ap.parse_args()

    # deliberately NOT lib.load(): that drops failures, which are the point here
    rows = [r for r in csv.DictReader(open(args.results)) if r["metric"] == "wall"]
    if not rows:
        sys.exit("no `wall` rows — every deploy emits one, so this file looks wrong")

    total = collections.Counter(r["outcome"] for r in rows)
    print(f"{len(rows)} deploys: " + ", ".join(f"{k}={v}" for k, v in sorted(total.items())))
    print()

    cells = collections.defaultdict(collections.Counter)
    for r in rows:
        cells[(r["netem"], r["variant"], r["program"])][r["outcome"]] += 1

    print(f"{'condition':<12} {'variant':<10} {'program':<14} {'success':>8}  {'n':>4}  failures")
    for key in sorted(cells):
        c = cells[key]
        n = sum(c.values())
        ok = c.get("success", 0)
        bad = ", ".join(f"{k}={v}" for k, v in sorted(c.items()) if k != "success") or "-"
        flag = "" if ok == n else "   <-- "
        print(f"{key[0]:<12} {key[1]:<10} {key[2]:<14} {ok / n:7.1%}  {n:4}  {bad}{flag}")


if __name__ == "__main__":
    main()
