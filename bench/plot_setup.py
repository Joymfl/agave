#!/usr/bin/env python3
"""client_setup floor against RTT: how many serial round trips to build the transport.

Setup is a chain of request -> wait -> response. Each exchange costs one whole RTT
and you cannot have a fraction of one, so:

    setup_time = N * RTT + compute

Plot the FLOOR against RTT and the slope IS N; the intercept is the compute cost,
which should be ~0 if setup is purely network-bound. The floor (not the median) is
used because retries add whole extra round trips and make the distribution
multimodal.

CAVEAT for the writeup: this is the SERIAL DEPTH of the critical path, not the
number of requests. Three concurrent requests cost 1 RTT and would be counted as 1.

  ./bench/plot_setup.py results.csv --out plots/setup.png
"""

import argparse
import collections
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.lib import load, pick, variants

COLOR = {"baseline": "#1f77b4", "tpu_next": "#d62728"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", default="plots/setup.png")
    args = ap.parse_args()

    rows = [r for r in load(args.results) if r["metric"] == "client_setup" and r["rtt_s"] > 0]
    if not rows:
        sys.exit("no client_setup rows with delay — need a run with a delay profile")

    floors = collections.defaultdict(list)
    for r in rows:
        floors[(r["variant"], r["rtt_s"])].append(r["sec"])

    rtts = sorted({k[1] for k in floors})
    fig, ax = plt.subplots(figsize=(7, 4.6))

    print(f"{'variant':<10} {'RTT':>7} {'floor':>8} {'round trips':>12}")
    for v in variants(rows):
        ys = [min(floors[(v, t)]) for t in rtts]
        n = ys[-1] / rtts[-1]
        ax.plot([t * 1000 for t in rtts], ys, marker="o",
                color=COLOR.get(v, None), label=f"{v} — {n:.0f} round trips")
        for t, y in zip(rtts, ys):
            print(f"{v:<10} {t * 1000:6.0f}ms {y:7.3f}s {y / t:12.2f}")
            ax.annotate(f"{y / t:.1f}×RTT", (t * 1000, y), textcoords="offset points",
                        xytext=(6, -3), fontsize=7, color=COLOR.get(v, "0.3"))

    # least squares through the origin-anchored points, to show the fit is integer
    for v in variants(rows):
        ys = [min(floors[(v, t)]) for t in rtts]
        n = len(rtts)
        sx, sy = sum(rtts), sum(ys)
        sxx = sum(t * t for t in rtts)
        sxy = sum(t * y for t, y in zip(rtts, ys))
        slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        icept = (sy - slope * sx) / n
        print(f"  fit {v:<9}: setup = {slope:.2f} × RTT + {icept * 1000:+.1f}ms")

    ax.set_title("Transport setup is a fixed number of serial round trips")
    ax.set_xlabel("round-trip time (ms)")
    ax.set_ylabel("client_setup, floor (s)")
    ax.set_xlim(0, max(rtts) * 1000 * 1.15)
    ax.set_ylim(0, None)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
