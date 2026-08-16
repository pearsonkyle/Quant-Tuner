#!/usr/bin/env bash
# Wait for the KLD PROCESS to exit, not for a GPU-free poll: two-pass evaluation moves a
# ~51 GB model off the card between corpora, so free VRAM dips mid-run and a memory poll
# would fire early and collide with it.
set -uo pipefail
KLDPID=2048974
echo "waiting for KLD pid $KLDPID to exit …"
while kill -0 "$KLDPID" 2>/dev/null; do sleep 30; done
echo "KLD finished"
sleep 30
exec bash /workspace/Quant-Tuner/scripts/exp060_test_graft.sh
