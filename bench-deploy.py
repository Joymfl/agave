#!/usr/bin/env python3
"""Benchmark `solana program deploy` across program sizes, CLI variants, and
network conditions.

Usage:
  # clean link, no impairment
  ./bench-deploy.py --variant tpu_next=target/release/solana \
                    --variant baseline=/path/to/master/target/release/solana \
                    --runs 100

  # under netem (requires: sudo ./netns-setup.sh up)
  ./bench-deploy.py --variant tpu_next=... --variant baseline=... \
                    --netns val --rpc-host 10.10.0.2 \
                    --netem clean= \
                    --netem loss1="loss random 1%" \
                    --netem loss3="loss random 3%" \
                    --runs 100
"""

import argparse
import csv
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROGRAM_DIR = SCRIPT_DIR / "program-deploy-test"

RPC_PORT = 8899

# Matches `info!(target: "deploy_metric", "<span>,<us>,<count>")`, after
# env_logger's "[timestamp LEVEL target]" prefix.
SPAN_RE = re.compile(
    # resend_rounds/resent_txs are counts, not microseconds -- they land in the
    # `us` column, which is a lie the plotter must not treat as a duration.
    r"\b(total_deploy|client_setup|write_chunks|final_tx"
    r"|send_phase|confirm_phase|first_send|resend_time|quic_sent"
    r"|resend_rounds|resent_txs),(\d+),(\d+)\s*$",
    re.MULTILINE,
)

FIELDS = [
    "run",
    "netem",
    "netem_spec",
    "variant",
    "program",
    "program_bytes",
    "outcome",
    "metric",
    "us",
    "count",
]


def die(msg):
    sys.exit(f"error: {msg}")


def sh(cmd, check=True):
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


class Cluster:
    """Where the validator lives and how we reach it.

    With --netns the validator runs inside a network namespace and the CLI stays
    in the root namespace, so their traffic crosses the veth pair and can be
    impaired. Without it, everything is on loopback and no impairment is possible
    (see --netem, which refuses to run without a namespace).
    """

    def __init__(self, rpc_host, netns, veth_host, veth_peer):
        self.rpc_host = rpc_host
        self.netns = netns
        self.veth_host = veth_host
        self.veth_peer = veth_peer
        self.rpc_url = f"http://{rpc_host}:{RPC_PORT}"
        self.ws_url = f"ws://{rpc_host}:{RPC_PORT + 1}"

    def validator_cmd(self, ledger, mint_pubkey):
        cmd = []
        if self.netns:
            # `ip netns exec` needs root; drop straight back to the invoking user
            # so the ledger directory isn't left owned by root.
            cmd += ["sudo", "ip", "netns", "exec", self.netns]
            cmd += ["sudo", "-u", os.environ.get("USER", "nobody")]
        cmd += [
            shutil.which("solana-test-validator"),
            "--reset",
            "--quiet",
            "--ledger", str(ledger),
            "--rpc-port", str(RPC_PORT),
            "--bind-address", self.rpc_host,
            "--mint", mint_pubkey,
        ]
        return cmd

    def apply_netem(self, spec):
        """Attach (or clear) a netem qdisc on BOTH ends of the veth pair.

        netem only shapes traffic *leaving* an interface, so a qdisc on one end
        gives a one-way impairment. Both ends gives a symmetric link. Note that
        `delay 25ms` on each side is a 50ms round trip, and 1% loss each way is
        ~2% of round trips lost -- say which you mean when you report numbers.
        """
        if not self.netns:
            return
        ends = [
            (["sudo"], self.veth_host),
            (["sudo", "ip", "netns", "exec", self.netns], self.veth_peer),
        ]
        for prefix, dev in ends:
            if spec:
                sh(prefix + ["tc", "qdisc", "replace", "dev", dev, "root", "netem", *spec.split()])
            else:
                # No qdisc to delete on the first clean profile; that is fine.
                sh(prefix + ["tc", "qdisc", "del", "dev", dev, "root"], check=False)

    def rpc(self, method, params, timeout=5):
        req = urllib.request.Request(
            self.rpc_url,
            data=json.dumps(
                {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def healthy(self):
        try:
            return self.rpc("getHealth", [], timeout=2).get("result") == "ok"
        except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
            return False

    def program_is_deployed(self, program_pubkey):
        """A zero exit code is not proof the program landed. The CLI can exit 0
        while nothing executable exists on chain, so ask the cluster directly.

        Reads at `confirmed`, not the default `finalized`: the deploy confirms at
        confirmed, so a finalized read right after it races the cluster and
        reports a perfectly good deploy as missing.
        """
        try:
            value = self.rpc(
                "getAccountInfo",
                [program_pubkey, {"encoding": "base64", "commitment": "confirmed"}],
            )["result"]["value"]
        except (urllib.error.URLError, OSError, KeyError, TypeError, json.JSONDecodeError):
            return False
        return bool(value) and value.get("executable") is True


def keygen(path):
    """Fresh keypair. Every deploy needs a new program id, otherwise the second
    deploy of a program is an *upgrade*, which is a different code path."""
    sh(["solana-keygen", "new", "--no-bip39-passphrase", "--silent", "-f", "-o", str(path)])
    return sh(["solana-keygen", "pubkey", str(path)]).stdout.strip()


def start_validator(cluster, ledger, mint_pubkey, timeout=90):
    """Fresh validator per run. `--mint` funds the fee payer at genesis, which
    keeps an airdrop round-trip out of the measured region."""
    # If something is already answering on the RPC port, our new validator will
    # fail to bind and every deploy will silently be routed to the *stale* one --
    # which was minted for a different fee payer, so every deploy fails with
    # "insufficient funds". Refuse to start rather than generate a night of
    # confidently wrong data.
    if cluster.healthy():
        die(
            "a validator is already serving on this RPC port; the harness would "
            "deploy against it instead of a fresh one.\n"
            "  sudo pkill -f solana-test-validator"
        )

    proc = subprocess.Popen(
        cluster.validator_cmd(ledger, mint_pubkey),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            die("solana-test-validator exited during startup")
        if cluster.healthy():
            return proc
        time.sleep(0.25)
    stop_validator(proc, cluster, ledger)
    die(f"validator not healthy after {timeout}s")


def stop_validator(proc, cluster, ledger):
    """Kill the validator and *verify* it is gone.

    Under --netns the validator is launched via `sudo ip netns exec ... sudo -u`,
    so proc.terminate() only signals the outer sudo, which does not reliably
    propagate through the nested one. The validator survives, keeps the RPC port,
    and the next run deploys against it. Match on the ledger path, which is unique
    to this harness invocation, so we never kill an unrelated validator.
    """
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    if cluster.netns:
        pattern = f"solana-test-validator.*{ledger}"
        sh(["sudo", "pkill", "-TERM", "-f", pattern], check=False)

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if not cluster.healthy():
            return
        time.sleep(0.5)

    if cluster.netns:
        sh(["sudo", "pkill", "-KILL", "-f", f"solana-test-validator.*{ledger}"], check=False)
        time.sleep(2)
    if cluster.healthy():
        die("validator would not die; refusing to continue with a stale one serving")


def deploy(cluster, binary, program_so, fee_payer, program_keypair, program_pubkey, timeout):
    """Returns (outcome, wall_us, spans). `spans` is empty for an uninstrumented
    binary, which is expected if you point a variant at a stock CLI."""
    env = {**os.environ, "RUST_LOG": "deploy_metric=info"}
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [
                # --url/--ws/--keypair are global flags: they must precede the
                # subcommand or clap rejects them.
                str(binary),
                "--url", cluster.rpc_url,
                "--ws", cluster.ws_url,
                "--keypair", str(fee_payer),
                "program", "deploy",
                "--program-id", str(program_keypair),
                str(program_so),
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        # Under heavy loss a deploy can hang indefinitely. Give up, record it as a
        # failure, and move on -- one stuck deploy must not stall the whole sweep.
        # A timeout is a real QoS outcome, not an error in the harness.
        wall_us = int((time.monotonic() - start) * 1_000_000)
        print(f"    TIMEOUT after {timeout}s")
        return "timeout", wall_us, []

    wall_us = int((time.monotonic() - start) * 1_000_000)

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        print(f"    FAILED: {tail[-1] if tail else 'no output'}")
        return "failure", wall_us, []

    if not cluster.program_is_deployed(program_pubkey):
        print("    FAILED: exited 0 but no executable program on chain")
        return "not_deployed", wall_us, []

    spans = [(n, int(us), int(c)) for n, us, c in SPAN_RE.findall(proc.stderr)]
    return "success", wall_us, spans


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    attempts, failures = defaultdict(int), defaultdict(int)
    for r in rows:
        if r["metric"] != "wall":
            continue
        key = (r["netem"], r["variant"], r["program"])
        attempts[key] += 1
        if r["outcome"] != "success":
            failures[key] += 1

    print("\nsuccess rate:")
    for key in sorted(attempts):
        n, bad = attempts[key], failures[key]
        print(f"  {key[0]:<10} {key[1]:<10} {key[2]:<14} {(n - bad) / n:6.1%}  ({n - bad}/{n})")

    lat = defaultdict(list)
    for r in rows:
        if r["outcome"] == "success":
            lat[(r["netem"], r["variant"], r["program"], r["metric"])].append(r["us"])

    print("\nlatency of successful runs (seconds):")
    print(
        f"  {'netem':<10} {'variant':<10} {'program':<14} {'metric':<14}"
        f" {'p50':>7} {'p95':>7} {'p99':>7} {'sd':>7} {'n':>4}"
    )
    for key in sorted(lat):
        vals = sorted(lat[key])

        def pct(p):
            return vals[min(int(len(vals) * p / 100), len(vals) - 1)] / 1e6

        sd = statistics.stdev(vals) / 1e6 if len(vals) > 1 else 0.0
        print(
            f"  {key[0]:<10} {key[1]:<10} {key[2]:<14} {key[3]:<14}"
            f" {pct(50):7.3f} {pct(95):7.3f} {pct(99):7.3f} {sd:7.3f} {len(vals):4}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", required=True, metavar="NAME=PATH",
                    help="CLI binary to benchmark; repeat for each arm")
    ap.add_argument("--netem", action="append", metavar="NAME=SPEC", default=None,
                    help="network condition, e.g. loss3='loss random 3%%'. "
                         "Empty SPEC means no impairment. Repeat to sweep. Requires --netns.")
    ap.add_argument("--netns", default=None, help="namespace the validator runs in")
    ap.add_argument("--rpc-host", default="127.0.0.1", help="address the validator binds to")
    ap.add_argument("--veth-host", default="veth-host")
    ap.add_argument("--veth-peer", default="veth-val")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--seed", type=int, default=0, help="seeds the variant interleaving")
    ap.add_argument("--timeout", type=int, default=300,
                    help="seconds before a single deploy is abandoned and recorded as a timeout")
    args = ap.parse_args()

    for tool in ("solana-test-validator", "solana-keygen"):
        if not shutil.which(tool):
            die(f"{tool} not found on PATH")

    variants = {}
    for spec in args.variant:
        if "=" not in spec:
            die(f"--variant expects NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        path = Path(path).resolve()
        if not path.is_file():
            die(f"binary for variant {name!r} not found: {path}")
        variants[name] = path

    # Impairment without a namespace would have to go on `lo`, which carries the
    # validator's own internal traffic as well as ours -- you would be measuring a
    # crippled validator, not a bad client link. Refuse rather than produce
    # confident nonsense.
    profiles = {"none": ""}
    if args.netem:
        if not args.netns:
            die("--netem requires --netns; impairing loopback would also impair the "
                "validator's own traffic. Run: sudo ./netns-setup.sh up")
        profiles = {}
        for spec in args.netem:
            if "=" not in spec:
                die(f"--netem expects NAME=SPEC, got {spec!r}")
            name, netem_spec = spec.split("=", 1)
            profiles[name] = netem_spec.strip()

    if args.netns:
        if args.rpc_host == "127.0.0.1":
            die("--netns given but --rpc-host is still 127.0.0.1; "
                "point it at the validator's namespace address (e.g. 10.10.0.2)")
        if subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode != 0:
            die("passwordless sudo needed to drive `ip netns` and `tc`; run `sudo -v` first")

    cluster = Cluster(args.rpc_host, args.netns, args.veth_host, args.veth_peer)

    programs = sorted(PROGRAM_DIR.glob("*.so"))
    if not programs:
        die(f"no .so files in {PROGRAM_DIR} - run ./program-deploy-test/fetch-corpus.sh")

    print("programs:")
    for so in programs:
        print(f"  {so.stem:<14} {so.stat().st_size:>9,} bytes")
    print("variants:")
    for name, path in variants.items():
        print(f"  {name:<14} {path}")
    print("network conditions:")
    for name, spec in profiles.items():
        print(f"  {name:<14} {spec or '(no impairment)'}")

    rng = random.Random(args.seed)
    rows = []
    total = args.runs * len(profiles) * len(programs) * len(variants)
    done = 0

    try:
        with tempfile.TemporaryDirectory(prefix="bench-deploy-") as tmp:
            tmp = Path(tmp)
            fee_payer = tmp / "fee-payer.json"
            mint_pubkey = keygen(fee_payer)

            # `run` is the outermost loop so every condition is sampled evenly
            # across the whole benchmark. Blocking by condition would let machine
            # drift masquerade as an effect of the network.
            for run in range(args.runs):
                for profile, spec in profiles.items():
                    cluster.apply_netem(spec)

                    for so in programs:
                        # Interleave the arms inside each cell so drift cannot
                        # accumulate against a single variant.
                        order = list(variants.items())
                        rng.shuffle(order)

                        for name, binary in order:
                            done += 1
                            print(f"[{done}/{total}] run {run} {profile} {so.stem} {name}")

                            program_keypair = tmp / f"prog-{run}-{profile}-{so.stem}-{name}.json"
                            program_pubkey = keygen(program_keypair)

                            ledger = tmp / "ledger"
                            validator = start_validator(cluster, ledger, mint_pubkey)
                            try:
                                outcome, wall_us, spans = deploy(
                                    cluster, binary, so, fee_payer,
                                    program_keypair, program_pubkey, args.timeout,
                                )
                            finally:
                                stop_validator(validator, cluster, ledger)

                            base = {
                                "run": run,
                                "netem": profile,
                                "netem_spec": spec,
                                "variant": name,
                                "program": so.stem,
                                "program_bytes": so.stat().st_size,
                                "outcome": outcome,
                            }
                            # The wall row exists for every run, success or not. It
                            # is the only metric comparable across an instrumented
                            # and an uninstrumented binary.
                            rows.append({**base, "metric": "wall", "us": wall_us, "count": 0})
                            for span, us, count in spans:
                                rows.append({**base, "metric": span, "us": us, "count": count})

                            # Rewritten every run so a long benchmark can be
                            # interrupted without losing what it already measured.
                            write_csv(args.out, rows)
    finally:
        cluster.apply_netem("")  # leave the link clean whatever happened

    summarize(rows)
    print(f"\nwrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
