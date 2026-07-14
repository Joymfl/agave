#!/usr/bin/env python3
"""The two findings that survive scrutiny, as one figure for the PR.

LEFT — the send phase is a client-side sleep.
  sign_all_messages_and_send sleeps `index * SEND_INTERVAL` before sending each
  chunk (SEND_INTERVAL = 10ms). The futures run concurrently, so the phase ends
  when the last one finishes: (N-1) * 10ms. Measured send_phase sits on that line
  to within a few milliseconds, identically on both transports. Deploy throughput
  is therefore capped at 100 chunks/sec by a constant, not by the network.

RIGHT — setup is a fixed number of serial round trips.
  Plotting the FLOOR (min) of client_setup against RTT gives an integer slope and
  a ~zero intercept: setup is pure network waiting, N dependent exchanges deep.
  v2 needs 6, v3 needs 4. The floor is used because these distributions are
  multimodal -- retries add whole round trips, and a median jumps between modes.

  ./plot-pr.py --spans results-resend.csv --sweep results.csv
"""

import argparse
import collections
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DELAY_RE = re.compile(r"delay\s+([\d.]+)ms")
SEND_INTERVAL_MS = 10.0

COLOR = {"baseline": "#1f77b4", "tpu_next": "#d62728"}
LABEL = {"baseline": "v2 (ConnectionCache)", "tpu_next": "v3 (tpu-client-next)"}


def load(path):
    rows = [r for r in csv.DictReader(open(path)) if r["outcome"] == "success"]
    for r in rows:
        m = DELAY_RE.search(r.get("netem_spec") or "")
        r["delay_ms"] = float(m.group(1)) if m else 0.0
        r["sec"] = int(r["us"]) / 1e6
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", default="results-resend.csv", help="run with send_phase spans")
    ap.add_argument("--sweep", default="results.csv", help="the delay sweep, for client_setup")
    ap.add_argument("--out", default="plots/pr-findings.png")
    args = ap.parse_args()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    # ---------- LEFT: send_phase == the staircase ----------
    spans = load(args.spans)
    chunks = {r["program"]: int(r["count"]) for r in spans if r["metric"] == "write_chunks"}
    send = collections.defaultdict(list)
    for r in spans:
        if r["metric"] == "send_phase":
            send[(r["program"], r["variant"])].append(r["sec"])

    programs = sorted(chunks, key=lambda p: chunks[p])
    ns = [chunks[p] for p in programs]

    # the prediction, drawn as a line through the origin
    xs = [0, max(ns) * 1.05]
    ax1.plot(xs, [(x - 1) * SEND_INTERVAL_MS / 1000 for x in xs], color="black",
             linestyle="--", linewidth=1.3, label="(N−1) × 10ms  (the sleep alone)")

    for v in ["baseline", "tpu_next"]:
        ys = [sorted(send[(p, v)])[len(send[(p, v)]) // 2] for p in programs]
        ax1.plot(ns, ys, marker="o", markersize=9, linestyle="none",
                 color=COLOR[v], alpha=0.75, label=LABEL[v])

    for p, n in zip(programs, ns):
        ys = sorted(send[(p, "baseline")])
        ax1.annotate(p, (n, ys[len(ys) // 2]), textcoords="offset points",
                     xytext=(6, -12), fontsize=8, color="0.3")

    ax1.set_title("The send phase is a client-side sleep", fontsize=11)
    ax1.set_xlabel("write chunks (N)")
    ax1.set_ylabel("send_phase (s)")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # ---------- RIGHT: client_setup floor vs RTT ----------
    sweep = load(args.sweep)
    setup = collections.defaultdict(list)
    for r in sweep:
        if r["metric"] == "client_setup" and r["delay_ms"] > 0:
            setup[(r["variant"], 2 * r["delay_ms"] / 1000.0)].append(r["sec"])

    rtts = sorted({k[1] for k in setup})
    for v in ["baseline", "tpu_next"]:
        floors = [min(setup[(v, rtt)]) for rtt in rtts]
        n = floors[-1] / rtts[-1]
        ax2.plot([r * 1000 for r in rtts], floors, marker="o", color=COLOR[v],
                 label=f"{LABEL[v]} — {n:.0f} round trips")
        for rtt, f in zip(rtts, floors):
            ax2.annotate(f"{f / rtt:.1f}×RTT", (rtt * 1000, f), textcoords="offset points",
                         xytext=(6, -3), fontsize=7, color=COLOR[v])

    ax2.set_title("Transport setup is a fixed number of serial round trips", fontsize=11)
    ax2.set_xlabel("round-trip time (ms)")
    ax2.set_ylabel("client_setup, floor (s)")
    ax2.set_xlim(0, max(rtts) * 1000 * 1.15)
    ax2.set_ylim(0, None)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("`solana program deploy`: where the time actually goes", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}\n")

    print("LEFT — send_phase vs (N-1)*10ms")
    for p, n in zip(programs, ns):
        pred = (n - 1) * SEND_INTERVAL_MS / 1000
        for v in ["baseline", "tpu_next"]:
            vals = sorted(send[(p, v)])
            med = vals[len(vals) // 2]
            print(f"  {p:<12} {v:<9} N={n:<5} predicted {pred:7.3f}s  measured {med:7.3f}s"
                  f"  residual {med - pred * 1:.3f}s")
    print("\nRIGHT — client_setup floor / RTT")
    for v in ["baseline", "tpu_next"]:
        for rtt in rtts:
            f = min(setup[(v, rtt)])
            print(f"  {v:<9} RTT {rtt * 1000:5.0f}ms  floor {f:6.3f}s  = {f / rtt:5.2f} RTT")


if __name__ == "__main__":
    main()
