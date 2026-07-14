"""Shared loading and parsing for bench-deploy.py output. No analysis here.

The CSV is long-format: one row per (run, condition, variant, program, metric).
`us` holds microseconds for the timing spans and a raw count for the counters
(resend_rounds, resent_txs) -- so never treat `us` as a duration without checking
which metric you are looking at. COUNTERS lists the exceptions.
"""

import csv
import re

# spans, in the order they happen during a deploy.
# resend_time is a SUBSET of confirm_phase: the part spent inside the transport
# resending, as opposed to waiting for the 1Hz confirmation poller. So
#     confirm_phase = resend_time + (time waiting for confirmations)
SPANS = [
    "client_setup", "send_phase", "resend_time", "confirm_phase",
    "write_chunks", "final_tx", "total_deploy",
]
COUNTERS = ["resend_rounds", "resent_txs"]

DELAY_RE = re.compile(r"delay\s+([\d.]+)ms")
LOSS_RE = re.compile(r"loss\s+(?:random\s+)?([\d.]+)%")

# Spans that wait on a block are BIMODAL: a transaction either makes the next
# slot or it does not, so timings land in whole-block clusters with a gap between
# them. A median of those jumps between clusters and invents differences that are
# not there. Use paired differences (report_paired.py) for anything in this list.
BLOCK_QUANTISED = {"confirm_phase", "final_tx", "write_chunks", "total_deploy"}


def parse_netem(spec):
    """(delay_ms, loss_pct) per hop. (None, None) if the spec is not simple
    delay/loss -- gemodel and rate have no single value to place on an axis."""
    if not isinstance(spec, str) or not spec.strip():
        return 0.0, 0.0
    spec = spec.strip()
    if "gemodel" in spec or "state" in spec or "rate" in spec:
        return None, None
    d = DELAY_RE.search(spec)
    lo = LOSS_RE.search(spec)
    return (float(d.group(1)) if d else 0.0, float(lo.group(1)) if lo else 0.0)


def load(path, successes_only=True):
    """Returns a list of dicts with the raw columns plus:
        sec       -- us / 1e6 (meaningless for COUNTERS)
        value     -- int(us), the honest raw number
        delay_ms  -- one-way delay, per hop
        loss_pct  -- packet loss, per hop
        rtt_s     -- round trip = 2 * delay, in seconds
    """
    rows = []
    for r in csv.DictReader(open(path)):
        if successes_only and r["outcome"] != "success":
            continue
        delay, loss = parse_netem(r.get("netem_spec"))
        r["value"] = int(r["us"])
        r["sec"] = r["value"] / 1e6
        r["delay_ms"] = delay
        r["loss_pct"] = loss
        r["rtt_s"] = (2 * delay / 1000.0) if delay else 0.0
        r["run"] = int(r["run"])
        r["count"] = int(r["count"])
        rows.append(r)
    return rows


def programs(rows):
    """Program names, smallest first."""
    sizes = {r["program"]: int(r["program_bytes"]) for r in rows}
    return sorted(sizes, key=sizes.get)


def variants(rows):
    return sorted({r["variant"] for r in rows})


def conditions(rows):
    return sorted({r["netem"] for r in rows})


def chunk_counts(rows):
    """Write messages per program. NOT the ELF divided by chunk size: the CLI
    skips chunks that already match the (zero-filled) buffer, so an ELF with a lot
    of padding produces far fewer messages than its size suggests."""
    return {r["program"]: r["count"] for r in rows if r["metric"] == "write_chunks"}


def pick(rows, **kw):
    """rows filtered by exact column matches: pick(rows, metric='final_tx', variant='v2')"""
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]
