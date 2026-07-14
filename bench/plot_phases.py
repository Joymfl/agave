#!/usr/bin/env python3
"""Where a deploy's time goes: setup, send, confirm, final tx — stacked, per variant.

Uses the MEAN, which averages over the block-quantised modes instead of picking
one the way a median does. The stack is drawn against the send staircase so the
reader can see how much of the send phase is `tokio::time::sleep` and how much is
anything else: transaction N sleeps N * SEND_INTERVAL before it is sent, so with
1866 messages the last one waits 18.65s before it leaves the client.

  ./bench/plot_phases.py results.csv --condition d25_l1 --out plots/phases.png
"""

import argparse
import collections
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.lib import chunk_counts, load, programs, variants

SEND_INTERVAL_S = 0.010
# drawn bottom-up, in the order they happen
STACK = [
    ("client_setup", "setup", "#4c72b0"),
    ("send_phase", "send", "#dd8452"),
    ("confirm_phase", "confirm", "#c44e52"),
    ("final_tx", "final tx", "#55a868"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--condition", help="netem label; default = the slowest one")
    ap.add_argument("--out", default="plots/phases.png")
    args = ap.parse_args()

    rows = load(args.results)
    if not any(r["metric"] == "send_phase" for r in rows):
        sys.exit("no send_phase rows — this run predates the send/confirm split")

    if args.condition:
        cond = args.condition
    else:
        tot = collections.defaultdict(list)
        for r in rows:
            if r["metric"] == "total_deploy":
                tot[r["netem"]].append(r["sec"])
        cond = max(tot, key=lambda c: st.mean(tot[c]))

    rows = [r for r in rows if r["netem"] == cond]
    progs, vs = programs(rows), variants(rows)
    chunks = chunk_counts(rows)

    mean = collections.defaultdict(list)
    for r in rows:
        mean[(r["program"], r["variant"], r["metric"])].append(r["sec"])

    fig, axes = plt.subplots(1, len(progs), figsize=(4.4 * len(progs), 4.4), squeeze=False)

    for ax, p in zip(axes[0], progs):
        bottoms = [0.0] * len(vs)
        for metric, label, colour in STACK:
            vals = [
                st.mean(mean[(p, v, metric)]) if mean.get((p, v, metric)) else 0.0
                for v in vs
            ]
            ax.bar(range(len(vs)), vals, bottom=bottoms, color=colour,
                   label=label if p == progs[0] else None)
            bottoms = [b + v for b, v in zip(bottoms, vals)]

        staircase = (chunks.get(p, 0) - 1) * SEND_INTERVAL_S
        ax.axhline(staircase, color="black", linestyle="--", linewidth=1.2)
        ax.annotate(f"(N−1)×10ms = {staircase:.1f}s", (len(vs) - 0.5, staircase),
                    textcoords="offset points", xytext=(-4, 4), ha="right", fontsize=7)

        ax.set_title(f"{p}  ({chunks.get(p, '?')} msgs)", fontsize=10)
        ax.set_xticks(range(len(vs)))
        ax.set_xticklabels(vs, fontsize=8)
        ax.set_ylabel("seconds (mean)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"Where a deploy spends its time — {cond}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}\n")

    for p in progs:
        staircase = (chunks.get(p, 0) - 1) * SEND_INTERVAL_S
        print(f"{p} ({chunks.get(p)} msgs, staircase {staircase:.2f}s)")
        for v in vs:
            parts = " ".join(
                f"{lbl}={st.mean(mean[(p, v, m)]):6.2f}s" if mean.get((p, v, m)) else f"{lbl}=  -   "
                for m, lbl, _ in STACK
            )
            print(f"  {v:<10} {parts}")
        print()


if __name__ == "__main__":
    main()
