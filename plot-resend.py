#!/usr/bin/env python3
"""One figure for the resend finding.

At 200ms RTT and ZERO packet loss, v3 (tpu-client-next) has to resend far more
transactions than v2 (ConnectionCache) before they confirm. Nothing is being
dropped by the network -- the loss is entirely on the client side, because
`send_transactions_in_batch` returns Ok on a local enqueue and the send/confirm
engine believes the transaction was delivered when it was not.

Zero loss is the point of the figure: it removes the network as an explanation.

  ./plot-resend.py --results results-resend.csv --condition d100_l0
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLOR = {"baseline": "#1f77b4", "tpu_next": "#d62728"}
LABEL = {"baseline": "v2 (ConnectionCache)", "tpu_next": "v3 (tpu-client-next)"}

PANELS = [
    ("resend_rounds", "resend rounds", "rounds", 1),
    ("resent_txs", "transactions resent", "transactions", 1),
    ("confirm_phase", "time spent confirming", "seconds", 1e6),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results-resend.csv")
    ap.add_argument("--condition", default="d100_l0")
    ap.add_argument("--out", default="plots/resend.png")
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    df = df[(df["outcome"] == "success") & (df["netem"] == args.condition)]
    if df.empty:
        sys.exit(f"no successful rows for condition {args.condition!r}")

    programs = sorted(df["program"].unique(),
                      key=lambda p: df[df["program"] == p]["program_bytes"].iloc[0])
    variants = ["baseline", "tpu_next"]
    x = range(len(programs))
    width = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, (metric, title, ylabel, scale) in zip(axes, PANELS):
        for i, v in enumerate(variants):
            # Median across runs. These are counts of discrete events, so the
            # median is the right summary -- unlike the latency spans, there is no
            # round-trip quantisation to make it jump between modes.
            vals = [
                df[(df["program"] == p) & (df["variant"] == v) & (df["metric"] == metric)]["us"].median() / scale
                for p in programs
            ]
            bars = ax.bar([xi + (i - 0.5) * width for xi in x], vals, width,
                          color=COLOR[v], label=LABEL[v])
            fmt = "%.2f" if scale > 1 else "%.0f"
            ax.bar_label(bars, fmt=fmt, fontsize=8, padding=2)

        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(x))
        ax.set_xticklabels(programs, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].legend(fontsize=9)
    fig.suptitle(
        f"`solana program deploy` — client-side transaction loss at {args.condition}"
        "  (200ms RTT, 0% packet loss)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")

    print("\nmedians:")
    for metric, title, _, scale in PANELS:
        print(f"  {title}")
        for p in programs:
            row = [
                df[(df["program"] == p) & (df["variant"] == v) & (df["metric"] == metric)]["us"].median() / scale
                for v in variants
            ]
            print(f"    {p:<12} v2={row[0]:>8.2f}   v3={row[1]:>8.2f}   x{row[1] / row[0]:.1f}")


if __name__ == "__main__":
    main()
