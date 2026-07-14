#!/usr/bin/env python3
"""Plots for bench-deploy.py output.

Three kinds of figure, one file per (program, view):

  success-*   success rate against the swept axis. This is the headline QoS
              result: does the deploy still work as the link degrades?
  p99-*       p99 time-to-deploy against the swept axis. Only meaningful read
              next to the success plot -- a variant that fails fast has a
              flattering p99 built from the runs that happened to survive.
  cdf-*       the full latency distribution at one fixed condition.

Delay and loss are derived from the recorded netem spec rather than from profile
names, so the sweeps drawn are whatever you actually ran. A condition whose spec
is not a plain delay/loss (gemodel, say) still gets a CDF but is left out of the
swept curves -- there is no single x value to put it at.

For a single merged 3-figure summary instead, use bench/plot_summary.py.

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

# `wall` is the subprocess clock, so it is the one metric present for every run
# and every binary, instrumented or not. Swap for total_deploy to plot the spans.
LATENCY_METRIC = "wall"


def parse_spec(spec):
    """(delay_ms per hop, loss_pct per hop), or (None, None) if not a simple spec."""
    if not isinstance(spec, str) or not spec.strip():
        return 0.0, 0.0
    spec = spec.strip()
    if "gemodel" in spec or "state" in spec or "rate" in spec:
        return None, None
    delay = DELAY_RE.search(spec)
    loss = LOSS_RE.search(spec)
    return (
        float(delay.group(1)) if delay else 0.0,
        float(loss.group(1)) if loss else 0.0,
    )


def success_rates(df):
    wall = df[df["metric"] == LATENCY_METRIC].copy()
    wall["ok"] = wall["outcome"].eq("success")
    return wall.groupby(["netem", "program", "variant"], as_index=False).agg(
        rate=("ok", "mean"),
        n=("ok", "size"),
        delay_ms=("delay_ms", "first"),
        loss_pct=("loss_pct", "first"),
    )


def tail_latencies(df):
    ok = df[(df["outcome"] == "success") & (df["metric"] == LATENCY_METRIC)]
    if ok.empty:
        return ok
    return ok.groupby(["netem", "program", "variant"], as_index=False).agg(
        p99=("us", lambda s: s.quantile(0.99) / 1e6),
        p50=("us", lambda s: s.quantile(0.50) / 1e6),
        delay_ms=("delay_ms", "first"),
        loss_pct=("loss_pct", "first"),
    )


def swept_plot(table, program, x_col, held_col, held_val, y_col, ylabel, title, path):
    sub = table[
        (table["program"] == program)
        & (table[held_col] == held_val)
        & table[x_col].notna()
    ]
    if sub[x_col].nunique() < 2:
        return False

    fig, ax = plt.subplots()
    for variant, g in sub.groupby("variant"):
        g = g.sort_values(x_col)
        ax.plot(g[x_col], g[y_col], marker="o", label=variant)

    ax.set_title(title)
    ax.set_xlabel(
        "one-way delay (ms)" if x_col == "delay_ms" else "packet loss per hop (%)"
    )
    ax.set_ylabel(ylabel)
    if y_col == "rate":
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(1.0, color="grey", linewidth=0.5, linestyle=":")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    return True


def cdf_plot(df, rates, program, netem, path):
    """Y is the fraction of deploys finishing in X seconds or less. To read p99:
    find 0.99 on Y, go across to the curve, drop down to X."""
    sub = df[
        (df["program"] == program)
        & (df["netem"] == netem)
        & (df["metric"] == LATENCY_METRIC)
        & (df["outcome"] == "success")
    ]
    if sub.empty:
        return False

    fig, ax = plt.subplots()
    for variant, g in sub.groupby("variant"):
        values = sorted(g["us"] / 1e6)
        # i+1 so the slowest run lands at 1.0 rather than just below it.
        pct = [(i + 1) / len(values) for i in range(len(values))]
        r = rates[
            (rates["netem"] == netem)
            & (rates["program"] == program)
            & (rates["variant"] == variant)
        ]
        # Failed runs are absent from this curve entirely, so the success rate
        # belongs on the plot or the curve reads as better than it is.
        label = variant
        if not r.empty:
            label = f"{variant} ({r.iloc[0]['rate']:.0%} ok, n={int(r.iloc[0]['n'])})"
        ax.plot(values, pct, marker=".", markersize=3, label=label)

    ax.set_title(f"{program} — {netem}")
    ax.set_xlabel("time to deploy (s)")
    ax.set_ylabel("fraction of deploys completed in ≤ x")
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.csv")
    ap.add_argument("--outdir", default="plots")
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    if "netem" not in df.columns:
        sys.exit("results have no `netem` column — regenerate with the current harness")
    if "netem_spec" not in df.columns:
        df["netem_spec"] = ""

    parsed = df["netem_spec"].apply(parse_spec)
    df["delay_ms"] = [p[0] for p in parsed]
    df["loss_pct"] = [p[1] for p in parsed]

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    rates = success_rates(df)
    tails = tail_latencies(df)

    print("success rate:")
    for _, r in rates.sort_values(["program", "netem", "variant"]).iterrows():
        print(
            f"  {r['program']:<14} {r['netem']:<10} {r['variant']:<10} "
            f"{r['rate']:6.1%}  (n={int(r['n'])})"
        )

    for program in sorted(df["program"].unique()):
        # Sweep A: delay stepped, loss held. Sweep B: loss stepped, delay held.
        # Both are derived from the data, so whatever you ran is what gets drawn.
        for x_col, held_col in (("delay_ms", "loss_pct"), ("loss_pct", "delay_ms")):
            for held_val in sorted(rates[held_col].dropna().unique()):
                tag = f"{held_col}{held_val:g}"
                axis = x_col.replace("_ms", "").replace("_pct", "")
                held_desc = f"{held_col.replace('_', ' ')} = {held_val:g}"
                swept_plot(
                    rates, program, x_col, held_col, held_val,
                    "rate", "success rate",
                    f"{program} — success rate vs {axis} ({held_desc})",
                    outdir / f"success-{program}-by-{axis}-{tag}.png",
                )
                if not tails.empty:
                    swept_plot(
                        tails, program, x_col, held_col, held_val,
                        "p99", "p99 time to deploy (s)",
                        f"{program} — p99 vs {axis} ({held_desc})",
                        outdir / f"p99-{program}-by-{axis}-{tag}.png",
                    )

        for netem in sorted(df["netem"].unique()):
            cdf_plot(df, rates, program, netem, outdir / f"cdf-{program}-{netem}.png")

    # Success rate as a bar chart over conditions. The swept success-* curves
    # above only cover conditions with a delay/loss axis value; categorical ones
    # (gemodel, rate) are excluded from them by design, and this figure is the
    # success view that covers every condition regardless of shape.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bench.plot_success import build as success_bars

    success_bars(args.results, outdir / "success-by-condition.png", quiet=True)

    # Merged views into a subdir: same data, one png per question, for side-by-side
    # reading instead of flipping between individual figures.
    from bench.plot_summary import fig_latency, fig_qos, fig_spans

    merged = outdir / "merged"
    merged.mkdir(exist_ok=True)
    df["sec"] = df["us"] / 1e6
    progs = sorted(
        df["program"].unique(),
        key=lambda p: df[df["program"] == p]["program_bytes"].iloc[0],
    )
    variants = sorted(df["variant"].unique())
    fig_qos(df, progs, variants, merged / "1-qos.png")
    fig_latency(df, progs, variants, merged / "2-latency.png")
    fig_spans(df, progs, variants, merged / "3-spans.png")


if __name__ == "__main__":
    main()
