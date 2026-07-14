#!/usr/bin/env python3
"""Plots for bench-deploy.py output. Three figures, one per question.

  1-qos.png        Does it still work, and how slow does it get, as the link
                   degrades? Success rate + p50/p99, across both sweeps.
  2-latency.png    What does the distribution actually look like? Histogram and
                   CDF, so the tail is visible rather than summarised away.
  3-spans.png      Where does the time go? client_setup vs write_chunks, which is
                   where the two transports diverge in opposite directions.

Delay and loss are read from the recorded netem spec, so whatever sweep you ran
is what gets drawn. Conditions whose spec is not a plain delay/loss (gemodel)
have no single x value, so they are excluded from the swept curves in figure 1
but still appear in figures 2 and 3.

  ./plot.py --results results.csv --outdir plots
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DELAY_RE = re.compile(r"delay\s+([\d.]+)ms")
LOSS_RE = re.compile(r"loss\s+(?:random\s+)?([\d.]+)%")

# `wall` is the subprocess clock: the one metric present for every run and every
# binary, instrumented or not.
WALL = "wall"
SPANS = ["client_setup", "write_chunks", "final_tx"]

# Colour by variant, consistently across all three figures.
COLOR = {"baseline": "#1f77b4", "tpu_next": "#d62728"}


def parse_spec(spec):
    """(delay_ms, loss_pct) per hop, or (None, None) for a non-simple spec."""
    if not isinstance(spec, str) or not spec.strip():
        return 0.0, 0.0
    spec = spec.strip()
    if "gemodel" in spec or "state" in spec:
        return None, None
    d = DELAY_RE.search(spec)
    lo = LOSS_RE.search(spec)
    return (float(d.group(1)) if d else 0.0, float(lo.group(1)) if lo else 0.0)


def load(path):
    df = pd.read_csv(path)
    if "netem" not in df.columns:
        sys.exit("results have no `netem` column — regenerate with the current harness")
    if "netem_spec" not in df.columns:
        df["netem_spec"] = ""
    parsed = df["netem_spec"].apply(parse_spec)
    df["delay_ms"] = [p[0] for p in parsed]
    df["loss_pct"] = [p[1] for p in parsed]
    df["sec"] = df["us"] / 1e6
    return df


def color(v, i):
    return COLOR.get(v, f"C{i}")


def fig_qos(df, programs, variants, out):
    """Success rate and latency against each swept axis.

    p50 and p99 share a panel: the gap between them IS the tail. Success rate is
    annotated rather than given its own axis -- when it is 100% everywhere, a
    dedicated panel is a flat line that says nothing, but its value still has to
    be visible or the latency curves read as more trustworthy than they are.
    """
    wall = df[df["metric"] == WALL]
    ok = wall[wall["outcome"] == "success"]

    sweeps = [
        ("delay_ms", "loss_pct", "one-way delay (ms)"),
        ("loss_pct", "delay_ms", "packet loss per hop (%)"),
    ]
    fig, axes = plt.subplots(
        len(programs), 2, figsize=(12, 3.6 * len(programs)), squeeze=False
    )

    for r, program in enumerate(programs):
        for c, (x, held, xlabel) in enumerate(sweeps):
            ax = axes[r][c]
            # Hold the other factor at whichever value has the most x points --
            # that is the sweep the user actually ran along this axis.
            cand = ok[(ok["program"] == program) & ok[x].notna()]
            if cand.empty:
                ax.set_visible(False)
                continue
            held_val = cand.groupby(held)[x].nunique().idxmax()
            sub = cand[cand[held] == held_val]

            for i, v in enumerate(variants):
                g = sub[sub["variant"] == v].groupby(x)["sec"]
                if g.ngroups < 2:
                    continue
                p50, p99 = g.quantile(0.50), g.quantile(0.99)
                ax.plot(p50.index, p50.values, marker="o", color=color(v, i), label=f"{v} p50")
                ax.plot(
                    p99.index, p99.values, marker="^", linestyle="--",
                    color=color(v, i), alpha=0.75, label=f"{v} p99",
                )

            cell = wall[(wall["program"] == program) & (wall[held] == held_val)]
            rate = (cell["outcome"] == "success").mean() if len(cell) else float("nan")
            ax.text(
                0.03, 0.95, f"success {rate:.0%} (n={len(cell)})",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8),
            )

            ax.set_title(f"{program} — vs {xlabel.split(' (')[0]} ({held} = {held_val:g})", fontsize=10)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("time to deploy (s)")
            ax.grid(True, alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(fontsize=8)

    fig.suptitle("Deploy QoS: latency and success rate as the link degrades", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_latency(df, programs, variants, out):
    """Distribution shape: histogram (left) and CDF (right) at the worst condition.

    The CDF is the one to read percentiles off: find 0.99 on y, go across, drop
    down to x. The histogram is there because a CDF hides multimodality -- two
    clusters of runs look like one smooth curve.
    """
    ok = df[(df["metric"] == WALL) & (df["outcome"] == "success")]
    # Worst = the condition with the highest median wall time; that is where any
    # difference between the transports has the best chance of being visible.
    worst = ok.groupby("netem")["sec"].median().idxmax()
    sub_all = ok[ok["netem"] == worst]

    fig, axes = plt.subplots(len(programs), 2, figsize=(12, 3.4 * len(programs)), squeeze=False)

    for r, program in enumerate(programs):
        sub = sub_all[sub_all["program"] == program]
        hax, cax = axes[r][0], axes[r][1]

        for i, v in enumerate(variants):
            vals = sub[sub["variant"] == v]["sec"].sort_values().values
            if not len(vals):
                continue
            hax.hist(vals, bins=15, alpha=0.55, color=color(v, i), label=v)
            # i+1 so the slowest run sits at 1.0 rather than just below it.
            pct = [(k + 1) / len(vals) for k in range(len(vals))]
            cax.plot(vals, pct, marker=".", markersize=4, color=color(v, i), label=v)

        hax.set_title(f"{program} — distribution at {worst}", fontsize=10)
        hax.set_xlabel("time to deploy (s)")
        hax.set_ylabel("deploys")
        hax.grid(True, alpha=0.3)

        cax.set_title(f"{program} — CDF at {worst}", fontsize=10)
        cax.set_xlabel("time to deploy (s)")
        cax.set_ylabel("fraction completed in ≤ x")
        cax.set_ylim(0, 1.02)
        cax.axhline(0.99, color="grey", linewidth=0.6, linestyle=":")
        cax.grid(True, alpha=0.3)
        if r == 0:
            hax.legend(fontsize=8)
            cax.legend(fontsize=8)

    fig.suptitle(f"Latency distribution at the harshest condition ({worst})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# The CLI paces its write transactions: transaction N sleeps N * SEND_INTERVAL
# before it is sent (send_and_confirm_transactions_in_parallel.rs, SEND_INTERVAL).
# For a large program this staircase dominates the whole deploy, so any span plot
# that omits it invites the reader to attribute the time to the network.
SEND_INTERVAL_S = 0.010


def fig_spans(df, programs, variants, out):
    """Where the time actually goes.

    Uses the FLOOR (min), not the median. Under delay every span quantises into
    whole round trips, so the distributions are multimodal and a median just
    reports whichever mode held the 50th sample -- it jumps between modes and
    produces impossible artefacts (spans getting *faster* with more delay). The
    floor is the no-retry best case and is stable.

    Left:   setup cost in ROUND TRIPS. Flat lines mean a fixed number of
            exchanges, which is the thing that actually differs between the two
            connection layers.
    Middle: the write phase against the client's own send staircase. The gap
            between the bars is all the network ever contributes.
    Right:  the deploy time budget, so the staircase's share is unmissable.
    """
    ok = df[(df["outcome"] == "success") & df["delay_ms"].notna() & (df["delay_ms"] > 0)]
    if ok.empty or "client_setup" not in set(ok["metric"]):
        print("no span rows with delay in results — skipping spans figure")
        return

    fig, axes = plt.subplots(len(programs), 3, figsize=(14, 3.4 * len(programs)), squeeze=False)

    for r, program in enumerate(programs):
        p = ok[ok["program"] == program]
        held = p.groupby("loss_pct")["delay_ms"].nunique().idxmax()
        p = p[p["loss_pct"] == held]
        chunks = int(df[(df["program"] == program) & (df["metric"] == "write_chunks")]["count"].iloc[0])
        staircase = chunks * SEND_INTERVAL_S

        # --- setup, in round trips ---
        ax = axes[r][0]
        for i, v in enumerate(variants):
            g = p[(p["variant"] == v) & (p["metric"] == "client_setup")]
            floor = g.groupby("delay_ms")["sec"].min()
            rtt = floor.index / 1000.0 * 2  # one-way ms -> RTT seconds
            ax.plot(floor.index, floor.values / rtt, marker="o", color=color(v, i), label=v)
        ax.set_title(f"{program} — transport setup", fontsize=10)
        ax.set_xlabel("one-way delay (ms)")
        ax.set_ylabel("round trips to connect")
        ax.set_ylim(0, 8)
        ax.grid(True, alpha=0.3)
        if r == 0:
            ax.legend(fontsize=8)

        # --- write phase vs the staircase ---
        ax = axes[r][1]
        for i, v in enumerate(variants):
            g = p[(p["variant"] == v) & (p["metric"] == "write_chunks")]
            floor = g.groupby("delay_ms")["sec"].min()
            ax.plot(floor.index, floor.values, marker="o", color=color(v, i), label=v)
        ax.axhline(
            staircase, color="black", linestyle="--", linewidth=1.2,
            label=f"client sleep: {chunks}×10ms = {staircase:.1f}s",
        )
        ax.set_title(f"{program} — write phase ({chunks} chunks)", fontsize=10)
        ax.set_xlabel("one-way delay (ms)")
        ax.set_ylabel("seconds (floor)")
        ax.set_ylim(0, max(staircase * 1.25, p[p["metric"] == "write_chunks"]["sec"].max() * 1.1))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

        # --- budget ---
        ax = axes[r][2]
        worst = p["delay_ms"].max()
        w = p[p["delay_ms"] == worst]
        labels, bottoms = [], []
        for i, v in enumerate(variants):
            setup = w[(w["variant"] == v) & (w["metric"] == "client_setup")]["sec"].min()
            final = w[(w["variant"] == v) & (w["metric"] == "final_tx")]["sec"].min()
            write = w[(w["variant"] == v) & (w["metric"] == "write_chunks")]["sec"].min()
            residual = max(write - staircase, 0.0)
            ax.bar(i, staircase, color="0.25", label="client sleep" if r == 0 and i == 0 else None)
            ax.bar(i, residual, bottom=staircase, color="0.6",
                   label="write: network+confirm" if r == 0 and i == 0 else None)
            ax.bar(i, setup, bottom=staircase + residual, color=color(v, i),
                   label="setup" if r == 0 and i == 0 else None)
            ax.bar(i, final, bottom=staircase + residual + setup, color=color(v, i), alpha=0.45,
                   label="final tx" if r == 0 and i == 0 else None)
            labels.append(v)
            bottoms.append(staircase + residual + setup + final)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"{program} — budget @ {worst:g}ms ({staircase / max(bottoms):.0%} is sleep)", fontsize=10)
        ax.set_ylabel("seconds")
        ax.grid(True, axis="y", alpha=0.3)
        if r == 0:
            ax.legend(fontsize=7)

    fig.suptitle(
        "Where the time goes: setup is the only place the transports differ; "
        "the write phase is a client-side sleep",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.csv")
    ap.add_argument("--outdir", default="plots")
    args = ap.parse_args()

    df = load(args.results)
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    programs = sorted(df["program"].unique(), key=lambda p: df[df["program"] == p]["program_bytes"].iloc[0])
    variants = sorted(df["variant"].unique())

    wall = df[df["metric"] == WALL]
    print("success rate:")
    for (prog, netem, v), g in wall.groupby(["program", "netem", "variant"]):
        rate = (g["outcome"] == "success").mean()
        if rate < 1.0:
            print(f"  {prog:<14} {netem:<10} {v:<10} {rate:6.1%}  (n={len(g)})")
    if (wall["outcome"] == "success").all():
        print(f"  all {len(wall)} deploys succeeded, every condition, both variants")

    fig_qos(df, programs, variants, outdir / "1-qos.png")
    fig_latency(df, programs, variants, outdir / "2-latency.png")
    fig_spans(df, programs, variants, outdir / "3-spans.png")


if __name__ == "__main__":
    main()
