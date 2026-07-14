#!/usr/bin/env python3
"""Transport-scoped plots for the n=20 sweep (results.csv).

Only the transaction-sending path is in scope, i.e. the parts that change when
the CLI deploy switches from the old TPU client to tpu-client-next:

  client_setup   bringing the transport up (websocket + leader cache + socket)
  write_chunks   signing, sending and confirming every write chunk

Everything else in the deploy -- the prologue before `total_deploy` starts, and
`final_tx` -- goes over RPC in both arms and is not evidence about the transport.

THE STAIRCASE. Inside `write_chunks`, chunk N sleeps N * SEND_INTERVAL (10ms)
before it is sent, so the phase carries a constant floor of (chunks-1) * 10ms:
1.07s for token, 5.00s for token-2022, 18.65s for jupiter-v6. That floor is the
client's own choice and it is identical in both arms -- for jupiter it is ~93% of
the phase. Every figure here is therefore drawn TWICE, side by side:

  with staircase     send_total = client_setup + write_chunks
                     what the deploy actually costs; the staircase is real time a
                     user waits, so this is the honest end-to-end view.
  staircase removed  send_work  = send_total - (chunks - 1) * SEND_INTERVAL
                     the same runs with the constant sleep subtracted, which is
                     the only view where a transport difference is visible rather
                     than diluted 14:1 by a sleep neither arm controls.

Read them together: the left panel says how much it matters, the right says what
is happening.

X AXES ARE ROUND-TRIP. netem is applied to both ends of the veth pair, so a
recorded `delay 25ms` is a 50ms RTT and `loss 1%` is 1-(0.99^2) = 2% of round
trips lost. The profile names understate the impairment by 2x; the axes here do
not.

  ./plot-send.py --results results.csv --outdir plots
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DELAY_RE = re.compile(r"delay\s+([\d.]+)ms")
LOSS_RE = re.compile(r"loss\s+(?:random\s+)?([\d.]+)%")

SEND_INTERVAL_MS = 10.0

COLOR = {"baseline": "#4c72b0", "tpu_next": "#c44e52"}
LABEL = {"baseline": "TPU client (baseline)", "tpu_next": "tpu-client-next"}

# (column, human name) for the two views every figure is drawn in.
VIEWS = [
    ("send_total", "with staircase"),
    ("send_work", "staircase removed"),
]


def parse_spec(spec):
    """(rtt_ms, round_trip_loss_pct) -- doubled, because netem is on both ends."""
    if not isinstance(spec, str) or not spec.strip():
        return 0.0, 0.0
    d = DELAY_RE.search(spec)
    lo = LOSS_RE.search(spec)
    delay = float(d.group(1)) if d else 0.0
    loss = float(lo.group(1)) if lo else 0.0
    return 2.0 * delay, 100.0 * (1.0 - (1.0 - loss / 100.0) ** 2)


def load(path):
    df = pd.read_csv(path)
    df = df[df.outcome == "success"]

    # One row per deploy, with the spans we care about pivoted into columns.
    wide = df.pivot_table(
        index=["run", "netem", "netem_spec", "variant", "program"],
        columns="metric",
        values="us",
        aggfunc="first",
    ).reset_index()

    # `count` on the write_chunks row is the chunk count: the real independent
    # variable behind "small/medium/large", and what sets the staircase floor.
    chunks = df[df.metric == "write_chunks"].groupby("program")["count"].max().rename("chunks")
    wide = wide.merge(chunks, on="program")

    missing = {"client_setup", "write_chunks"} - set(wide.columns)
    if missing:
        raise SystemExit(f"results.csv has no {missing} rows -- wrong instrumentation revision")

    wide["staircase"] = (wide.chunks - 1) * SEND_INTERVAL_MS * 1000.0
    wide["send_total"] = (wide.client_setup + wide.write_chunks) / 1e6
    wide["send_work"] = (wide.client_setup + wide.write_chunks - wide.staircase) / 1e6
    wide["wall_s"] = wide.wall / 1e6

    rtt_loss = wide.netem_spec.fillna("").apply(parse_spec)
    wide["rtt_ms"] = [x[0] for x in rtt_loss]
    wide["loss_pct"] = [x[1] for x in rtt_loss]
    return wide


def order_programs(df):
    """Smallest first, so a row of panels reads as increasing chunk count."""
    return list(df.groupby("program").chunks.max().sort_values().index)


def pct(s, p):
    return np.percentile(s, p) if len(s) else np.nan


def hist_panel(ax, sub, col):
    lo, hi = sub[col].min(), sub[col].max()
    pad = max((hi - lo) * 0.1, 0.05)
    bins = np.linspace(lo - pad, hi + pad, 16)
    for variant, g in sub.groupby("variant"):
        c = COLOR.get(variant, "grey")
        ax.hist(g[col], bins=bins, alpha=0.55, color=c, label=LABEL.get(variant, variant))
        ax.axvline(g[col].median(), color=c, ls="--", lw=1.2)
    ax.set_xlabel("send cost (s)")
    ax.legend(fontsize=7, frameon=False)


def ecdf_panel(ax, sub, col):
    for variant, g in sub.groupby("variant"):
        c = COLOR.get(variant, "grey")
        v = np.sort(g[col].values)
        ax.step(v, np.arange(1, len(v) + 1) / len(v), color=c, lw=1.8,
                label=LABEL.get(variant, variant))
    ax.set_xlabel("send cost (s)")
    ax.set_ylabel("fraction of deploys")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)


def fig_distribution(df, outdir):
    """What the sending cost looks like run to run, in both views.

    Drawn at the sweep's crossing point (25ms/1% per hop = 50ms RTT, 2%
    round-trip loss), the one condition both sweeps share -- pooling all eight
    conditions would smear four different networks into one blob.

    Columns: histogram and ECDF with the staircase, then the same two with it
    removed. The x scales differ between the pairs on purpose: that difference in
    scale IS the point -- it is how much of the phase was never about the network.
    """
    cell = df[df.netem == "d25_l1"]
    programs = order_programs(cell)
    fig, axes = plt.subplots(len(programs), 4, figsize=(19, 3.2 * len(programs)), squeeze=False)

    for row, prog in enumerate(programs):
        sub = cell[cell.program == prog]
        chunks = int(sub.chunks.iloc[0])
        n = len(sub) // sub.variant.nunique()

        for view, (col, name) in enumerate(VIEWS):
            hist_ax, cdf_ax = axes[row][view * 2], axes[row][view * 2 + 1]
            hist_panel(hist_ax, sub, col)
            ecdf_panel(cdf_ax, sub, col)
            hist_ax.set_ylabel(f"deploys (n={n}/arm)")
            hist_ax.set_title(f"{prog} ({chunks} chunks) — histogram, {name}", loc="left", fontsize=9)
            cdf_ax.set_title(f"{prog} — ECDF, {name}", loc="left", fontsize=9)

    fig.suptitle(
        "Sending cost at 50ms RTT / 2% round-trip loss — left pair includes the 10ms-per-chunk "
        "send staircase, right pair has it subtracted out",
        fontsize=11, y=1.0,
    )
    fig.tight_layout()
    out = outdir / "send-1-distribution.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


def sweep_panel(ax, sub, xcol, col, xlabel):
    """p50 line with a p50..p95 band, per variant."""
    for variant, g in sub.groupby("variant"):
        c = COLOR.get(variant, "grey")
        agg = (
            g.groupby(xcol)[col]
            .agg(p50=lambda s: pct(s, 50), p95=lambda s: pct(s, 95))
            .reset_index()
            .sort_values(xcol)
        )
        ax.plot(agg[xcol], agg.p50, "o-", color=c, lw=1.8, ms=4, label=LABEL.get(variant, variant))
        ax.fill_between(agg[xcol], agg.p50, agg.p95, color=c, alpha=0.15, lw=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("send cost (s)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)


def fig_sweeps(df, outdir):
    """Send cost against the two things we swept: RTT and loss.

    Each sweep holds the other axis fixed, so the x value is the only thing that
    moved. Shaded band is p50..p95 -- how much worse a bad run gets, which is the
    part a median hides. Each sweep is drawn in both views, side by side.
    """
    programs = order_programs(df)
    fig, axes = plt.subplots(len(programs), 4, figsize=(19, 3.2 * len(programs)), squeeze=False)

    # RTT sweep holds loss at 1% per hop; loss sweep holds delay at 25ms per hop.
    sweeps = [
        (df[df.netem.isin(["d00_l1", "d25_l1", "d50_l1", "d100_l1"])],
         "rtt_ms", "round-trip time (ms)", "vs RTT, loss held at 2%"),
        (df[df.netem.isin(["d25_l0", "d25_l1", "d25_l3", "d25_l5"])],
         "loss_pct", "round-trip packet loss (%)", "vs loss, RTT held at 50ms"),
    ]

    for row, prog in enumerate(programs):
        chunks = int(df[df.program == prog].chunks.iloc[0])
        for i, (sweep, xcol, xlabel, what) in enumerate(sweeps):
            for view, (col, name) in enumerate(VIEWS):
                ax = axes[row][i * 2 + view]
                sweep_panel(ax, sweep[sweep.program == prog], xcol, col, xlabel)
                ax.set_title(f"{prog} ({chunks} chunks) — {what}, {name}", loc="left", fontsize=9)

    fig.suptitle(
        "Sending cost vs network conditions (line = p50, band = p50..p95, n=20/point) — "
        "each sweep shown with the send staircase and with it subtracted out",
        fontsize=11, y=1.0,
    )
    fig.tight_layout()
    out = outdir / "send-2-sweeps.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


def table(df):
    """Numbers behind the figures. `wall` is included so the send-cost delta can
    be read as a fraction of what a user actually waits for."""
    rows = []
    for (netem, prog, variant), g in df.groupby(["netem", "program", "variant"]):
        rows.append({
            "program": prog,
            "netem": netem,
            "rtt_ms": g.rtt_ms.iloc[0],
            "loss_pct": round(g.loss_pct.iloc[0], 1),
            "variant": variant,
            "n": len(g),
            "staircase": round(g.staircase.iloc[0] / 1e6, 2),
            "total_p50": round(pct(g.send_total, 50), 2),
            "work_p50": round(pct(g.send_work, 50), 2),
            "work_p95": round(pct(g.send_work, 95), 2),
            "wall_p50": round(pct(g.wall_s, 50), 2),
        })
    t = pd.DataFrame(rows).sort_values(["program", "rtt_ms", "loss_pct", "variant"])

    piv = t.pivot_table(index=["program", "netem", "rtt_ms", "loss_pct"], columns="variant",
                        values=["work_p50", "wall_p50"])
    piv["work_delta"] = (piv[("work_p50", "tpu_next")] - piv[("work_p50", "baseline")]).round(2)
    piv["wall_delta"] = (piv[("wall_p50", "tpu_next")] - piv[("wall_p50", "baseline")]).round(2)

    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print("\nper-cell medians (seconds):\n")
        print(t.to_string(index=False))
        print("\ntpu_next - baseline (seconds; negative = tpu-client-next faster):\n")
        print(piv[["work_delta", "wall_delta"]].to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.csv")
    ap.add_argument("--outdir", default="plots")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    df = load(Path(args.results))

    fig_distribution(df, outdir)
    fig_sweeps(df, outdir)
    table(df)


if __name__ == "__main__":
    main()
