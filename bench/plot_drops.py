#!/usr/bin/env python3
"""tpu-client-next silently discards transactions under latency.

Counts come from the crate's OWN stat (`SendTransactionStats::successfully_sent`,
read through ClientBuilder::metric_reporter) -- transactions actually written to a
QUIC stream. Compare it against what the CLI handed over:

    handed over = write messages + resent_txs
    reached QUIC = quic_sent
    shortfall    = discarded inside tpu-client-next

The shortfall is not network loss. It is `NonblockingBroadcaster::send_to_workers`
dropping the batch on WorkersCacheError::FullChannel, logging at debug!, and
returning Ok(()). Nothing in SendTransactionStats counts it, and the caller is told
the transaction was delivered.

Only v3 has this stat; the baseline arm has no equivalent hook, so it is absent by
design rather than missing.

  ./bench/plot_drops.py results-quic.csv --out plots/drops.png
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
from bench.lib import conditions, load, programs

HANDED = "#4c72b0"
SENT = "#55a868"
DROPPED = "#c44e52"


def collect(rows):
    """(condition, program) -> list of (handed_over, reached_quic) per deploy."""
    per = collections.defaultdict(dict)
    for r in rows:
        if r["variant"] != "tpu_next":
            continue
        key = (r["run"], r["netem"], r["program"])
        if r["metric"] == "quic_sent":
            per[key]["sent"] = r["value"]
            per[key]["msgs"] = r["count"]
        elif r["metric"] == "resent_txs":
            per[key]["resent"] = r["value"]
        elif r["metric"] == "write_chunks":
            per[key]["msgs"] = r["count"]

    out = collections.defaultdict(list)
    for (run, cond, prog), d in per.items():
        if "sent" not in d or "resent" not in d:
            continue
        out[(cond, prog)].append((d["msgs"] + d["resent"], d["sent"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", default="plots/drops.png")
    args = ap.parse_args()

    rows = load(args.results)
    data = collect(rows)
    if not data:
        sys.exit("no quic_sent rows — rebuild the tpu_next arm with the metric_reporter")

    progs = programs(rows)
    conds = conditions(rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    # ---- left: handed over vs reached QUIC ----
    ax = axes[0]
    labels, handed, sent = [], [], []
    for c in conds:
        for p in progs:
            if (c, p) not in data:
                continue
            labels.append(f"{p}\n{c}")
            handed.append(st.median([x[0] for x in data[(c, p)]]))
            sent.append(st.median([x[1] for x in data[(c, p)]]))

    x = range(len(labels))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], handed, w, color=HANDED, label="handed to tpu-client-next")
    b2 = ax.bar([i + w / 2 for i in x], sent, w, color=SENT, label="actually written to QUIC")
    ax.bar_label(b1, fmt="%.0f", fontsize=7, padding=2)
    ax.bar_label(b2, fmt="%.0f", fontsize=7, padding=2)
    ax.set_title("Transactions submitted vs transactions sent", fontsize=11)
    ax.set_ylabel("transactions")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    # ---- right: the shortfall, as a percentage ----
    ax = axes[1]
    for i, c in enumerate(conds):
        pcts = []
        for p in progs:
            v = data.get((c, p))
            pcts.append(st.median([(h - s) / h * 100 for h, s in v]) if v else 0.0)
        bars = ax.bar([j + (i - (len(conds) - 1) / 2) * 0.38 for j in range(len(progs))],
                      pcts, 0.38, color=DROPPED if len(conds) == 1 else None,
                      alpha=1.0 if i else 0.45, label=c)
        ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=2)

    ax.set_title("Silently discarded inside tpu-client-next", fontsize=11)
    ax.set_ylabel("% of submitted transactions dropped")
    ax.set_xticks(range(len(progs)))
    ax.set_xticklabels(progs, fontsize=9)
    ax.legend(fontsize=8, title="network")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.suptitle(
        "tpu-client-next drops transactions under latency and reports them as sent",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}\n")

    # The percentage is the median of the PER-DEPLOY percentages, matching the
    # chart. Dividing median(handed) by median(sent) would give a different number,
    # because a deploy that drops a lot also resends a lot and so hands over more.
    print(f"{'condition':<10} {'program':<12} {'handed':>8} {'to QUIC':>8} {'dropped %':>10}   per-deploy %")
    for c in conds:
        for p in progs:
            v = data.get((c, p))
            if not v:
                continue
            h = st.median([x[0] for x in v])
            s = st.median([x[1] for x in v])
            pcts = sorted((hh - ss) / hh * 100 for hh, ss in v)
            each = ", ".join(f"{x:.1f}" for x in pcts)
            print(f"{c:<10} {p:<12} {h:8.0f} {s:8.0f} {st.median(pcts):9.1f}%   [{each}]  n={len(v)}")


if __name__ == "__main__":
    main()
