#!/usr/bin/env bash
# Dumps the deploy-benchmark program corpus from mainnet.
#
# Three sizes, per the plan in anza-xyz/agave#7744. What actually drives the
# deploy path is the write-chunk count, and that is a function of these byte
# sizes, so small/medium/large is shorthand rather than the variable itself.
set -euo pipefail

cd "$(dirname "$0")"
URL=${URL:-https://api.mainnet-beta.solana.com}

dump() {
  local name=$1 program_id=$2
  if [[ -f "$name.so" ]]; then
    echo "$name.so exists, skipping"
    return
  fi
  echo "dumping $name ($program_id)"
  solana program dump -u "$URL" "$program_id" "$name.so"
}

dump token       TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA
dump token-2022  TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb
dump jupiter-v6  JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4

echo
ls -l ./*.so
echo
echo "Deploy each once against a local validator before benchmarking: a dumped"
echo "mainnet ELF still has to pass read_and_verify_elf under the current feature"
echo "set, and you want to find that out now rather than 300 runs in."
