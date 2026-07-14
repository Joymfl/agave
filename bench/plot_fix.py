#!/usr/bin/env python3
"""The channel-size fix, across all programs.

Three arms, same 200ms-RTT link:
  baseline      v2 (ConnectionCache)
  v3_chan2      v3 with the current default worker_channel_size = 2
  v3_chan4096   v3 with the channel raised past the message count

LEFT  -- transactions dropped inside tpu-client-next (quic_sent shortfall).
         chan2 loses them under latency; chan4096 does not. v2 has no such stat,
         so it is absent by construction, not zero.
RIGHT -- total deploy time, mean. This is the payoff: with the drops gone, v3's
         real 2-round-trip setup advantage stops being eaten by resends, and it
         goes from slower than v2 to faster.

  ./bench/plot_fix.py results-chan.csv --condition d100_l1 --out plots/fix.png
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
from bench.lib import load, programs

# Known nice labels; any variant not listed uses its raw name. Palette cycles.
NICE = {
    "baseline": "v2 (ConnectionCache)",
    "tpu_next": "v3 (tpu-client-next)",
    "v3_chan2": "v3, channel=2 (default)",
    "v3_chan4096": "v3, channel=4096",
}
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
# baseline first so it is always the reference bar; the rest keep CSV order.
_PREFERRED = ["baseline", "tpu_next", "v3_chan2", "v3_chan4096"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--title", default="Transactions dropped vs total deploy time",
                    help="figure suptitle")
    ap.add_argument("--condition", default="d100_l1")
    ap.add_argument("--out", default="plots/fix.png")
    args = ap.parse_args()

    rows = [r for r in load(args.results) if r["netem"] == args.condition]
    if not rows:
        sys.exit(f"no successful rows for condition {args.condition!r}")
    progs = programs(rows)

    # Auto-detect the arms present, preferred names first, then anything else.
    seen = {r["variant"] for r in rows}
    arms = [a for a in _PREFERRED if a in seen] + sorted(seen - set(_PREFERRED))
    NICE_ = {a: NICE.get(a, a) for a in arms}
    COLOR = {a: PALETTE[i % len(PALETTE)] for i, a in enumerate(arms)}
    global NICE
    NICE = NICE_

    # drops: quic_sent shortfall, per deploy
    per = collections.defaultdict(dict)
    for r in rows:
        k = (r["program"], r["variant"], r["run"])
        if r["metric"] == "quic_sent":
            per[k]["sent"] = r["value"]
            per[k]["msgs"] = r["count"]
        elif r["metric"] == "resent_txs":
            per[k]["resent"] = r["value"]
        elif r["metric"] == "write_chunks":
            per[k]["msgs"] = r["count"]
    drops = collections.defaultdict(list)
    for (prog, var, run), d in per.items():
        if "sent" in d and "resent" in d:
            drops[(prog, var)].append(d["msgs"] + d["resent"] - d["sent"])

    # total deploy time, per deploy
    total = collections.defaultdict(list)
    for r in rows:
        if r["metric"] == "total_deploy":
            total[(r["program"], r["variant"])].append(r["sec"])

    fig, (ax_d, ax_t) = plt.subplots(1, 2, figsize=(13, 4.6))
    x = range(len(progs))
    w = 0.8 / len(arms)

    # ---- LEFT: drops ----
    for i, a in enumerate(arms):
        vals = [st.median(drops[(p, a)]) if drops.get((p, a)) else None for p in progs]
        xs = [j + (i - (len(arms) - 1) / 2) * w for j in x]
        present = [(xx, v) for xx, v in zip(xs, vals) if v is not None]
        if not present:
            continue  # v2 has no quic_sent
        bars = ax_d.bar([xx for xx, _ in present], [v for _, v in present], w,
                        color=COLOR[a], label=NICE[a])
        ax_d.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)

    ax_d.set_title(f"Transactions dropped inside tpu-client-next ({args.condition})", fontsize=11)
    ax_d.set_ylabel("dropped (quic_sent shortfall)")
    ax_d.set_xticks(list(x))
    ax_d.set_xticklabels(progs, fontsize=9)
    ax_d.legend(fontsize=8)
    ax_d.grid(True, axis="y", alpha=0.3)
    ax_d.set_axisbelow(True)

    # ---- RIGHT: total deploy time ----
    for i, a in enumerate(arms):
        vals = [st.mean(total[(p, a)]) if total.get((p, a)) else 0 for p in progs]
        xs = [j + (i - (len(arms) - 1) / 2) * w for j in x]
        bars = ax_t.bar(xs, vals, w, color=COLOR[a], label=NICE[a])
        ax_t.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)

    ax_t.set_title(f"Total deploy time, mean ({args.condition})", fontsize=11)
    ax_t.set_ylabel("seconds")
    ax_t.set_xticks(list(x))
    ax_t.set_xticklabels(progs, fontsize=9)
    ax_t.legend(fontsize=8)
    ax_t.grid(True, axis="y", alpha=0.3)
    ax_t.set_axisbelow(True)

    fig.suptitle(args.title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}\n")

    print(f"{'program':<12} {'arm':<24} {'dropped':>8} {'total (s)':>10}")
    for p in progs:
        for a in arms:
            d = f"{st.median(drops[(p, a)]):.0f}" if drops.get((p, a)) else "-"
            t = f"{st.mean(total[(p, a)]):.2f}" if total.get((p, a)) else "-"
            print(f"{p:<12} {NICE[a]:<24} {d:>8} {t:>10}")
        print()


if __name__ == "__main__":
    main()
