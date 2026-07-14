#!/usr/bin/env bash
# Creates (or destroys) the namespace + veth pair that bench-deploy.py drives.
#
# The validator runs inside the namespace; the CLI stays in the root namespace.
# Traffic between them crosses the veth pair, which is the only place netem is
# applied. The validator's own localhost traffic stays on the namespace's private
# `lo` and is left completely alone -- which is the entire reason for going to
# this trouble instead of putting netem on the root `lo`.
#
#   sudo ./netns-setup.sh up
#   sudo ./netns-setup.sh down
set -euo pipefail

NS=${NS:-val}
VETH_HOST=${VETH_HOST:-veth-host}
VETH_PEER=${VETH_PEER:-veth-val}
HOST_IP=${HOST_IP:-10.10.0.1}
PEER_IP=${PEER_IP:-10.10.0.2}
PREFIX=${PREFIX:-24}

case "${1:-}" in
up)
    ip netns add "$NS"
    ip link add "$VETH_HOST" type veth peer name "$VETH_PEER"
    ip link set "$VETH_PEER" netns "$NS"

    ip addr add "$HOST_IP/$PREFIX" dev "$VETH_HOST"
    ip link set "$VETH_HOST" up

    ip netns exec "$NS" ip addr add "$PEER_IP/$PREFIX" dev "$VETH_PEER"
    ip netns exec "$NS" ip link set "$VETH_PEER" up
    ip netns exec "$NS" ip link set lo up

    echo "namespace '$NS' up: client $HOST_IP <-> validator $PEER_IP"
    ping -c2 -W2 "$PEER_IP" >/dev/null && echo "link ok"
    echo
    echo "now run bench-deploy.py with:  --netns $NS --rpc-host $PEER_IP"
    ;;
down)
    ip netns del "$NS" 2>/dev/null || true
    ip link del "$VETH_HOST" 2>/dev/null || true
    echo "namespace '$NS' down"
    ;;
*)
    echo "usage: sudo $0 {up|down}" >&2
    exit 1
    ;;
esac
