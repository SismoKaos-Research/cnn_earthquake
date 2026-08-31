#!/usr/bin/env bash
# Keeps both TDVMS queue slots full.
#
# Submission must not be chained onto a monitor or a foreground command: a
# 15-minute monitor timeout once killed the queue refill before it ran, and the
# campaign sat idle with zero requests in flight while waiting for emails that
# were never going to arrive. This runs standalone and is safe to re-run --
# `next` refuses to double-claim, so an address that is already busy is a no-op.
set -uo pipefail
cd "$(dirname "$0")/.."
for e in "$@"; do
    echo "=== $e  $(date +%H:%M:%S) ==="
    uv run python scripts/afad_campaign.py next --email "$e" 2>&1 | tail -2
done
uv run python scripts/afad_campaign.py status 2>&1 | tail -6
