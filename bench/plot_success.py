#!/usr/bin/env python3
"""Success rate as a bar chart, per condition x variant, one panel per program.

For categorical conditions (gemodel, rate limits) that have no delay/loss value
to sweep along -- plot.py's success-* curves exclude them by design, and this is
the figure that covers them instead.

Bars are annotated with successes/attempts, not just the percentage: at small n
the raw counts ARE the honest message. 8/9 and 89% read very differently, and
only one of them stops the reader from over-trusting a three-run cell.

  ./bench/plot_success.py results-burst2.csv --out plots-burst2/success.png
"""

import argparse
import collections
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

COLOR = {"baseline": "#1f77b4", "tpu_next": "#d62728"}


def build(results_path, out_path, quiet=False):
    """Generate the success-rate bar figure. Importable so plot.py can emit it as
    part of its standard output set."""
    # NOT bench.lib.load(): that drops failures, which are the entire point here.
    rows = [r for r in csv.DictReader(open(results_path)) if r["metric"] == "wall"]
    if not rows:
        sys.exit("no `wall` rows — every deploy emits one, so this file looks wrong")

    sizes = {r["program"]: int(r["program_bytes"]) for r in rows}
    programs = sorted(sizes, key=sizes.get)
    variants = sorted({r["variant"] for r in rows})
    conditions = sorted({r["netem"] for r in rows})

    cells = collections.defaultdict(lambda: [0, 0])  # (cond, prog, var) -> [ok, n]
    for r in rows:
        c = cells[(r["netem"], r["program"], r["variant"])]
        c[1] += 1
        if r["outcome"] == "success":
            c[0] += 1

    fig, axes = plt.subplots(
        1, len(programs), figsize=(4.4 * len(programs), 4.2), squeeze=False, sharey=True
    )

    width = 0.8 / len(variants)
    for ax, program in zip(axes[0], programs):
        for i, v in enumerate(variants):
            xs, ys, labels = [], [], []
            for j, cond in enumerate(conditions):
                ok, n = cells.get((cond, program, v), (0, 0))
                if n == 0:
                    continue
                xs.append(j + (i - (len(variants) - 1) / 2) * width)
                ys.append(ok / n)
                labels.append(f"{ok}/{n}")
            bars = ax.bar(xs, ys, width, color=COLOR.get(v, f"C{i}"), label=v)
            ax.bar_label(bars, labels=labels, fontsize=7, padding=2)

        ax.set_title(program, fontsize=10)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, fontsize=8, rotation=20)
        ax.set_ylim(0, 1.12)
        ax.axhline(1.0, color="grey", linewidth=0.6, linestyle=":")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("success rate")
    axes[0][0].legend(fontsize=8, loc="lower left")

    fig.suptitle("Deploy success rate by network condition", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    if not quiet:
        for cond in conditions:
            for program in programs:
                for v in variants:
                    ok, n = cells.get((cond, program, v), (0, 0))
                    if n:
                        print(f"{cond:<10} {program:<12} {v:<10} {ok}/{n}  ({ok / n:.0%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", default="plots/success.png")
    args = ap.parse_args()
    build(args.results, args.out)


if __name__ == "__main__":
    main()
