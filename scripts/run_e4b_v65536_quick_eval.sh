#!/usr/bin/env bash
# Quick agentic + MTG eval for a stage-1 checkpoint, ~25 min per arm.
#
# Sized for ITERATION, not for the record. The trims that buy the speed:
#
#   * 36 of the 110 tool-call sessions, chosen round-robin across sources.
#     This is not merely a subset -- the full set is 63% nemotron-swe, and the
#     round-robin cut is better balanced across the 11 tool families, so the
#     blended number actually means something. What it loses is precision:
#     108 scored turns puts the standard error near 4 points, so read a small
#     move as "not yet resolved", not as "no effect".
#   * max_turns 3 per session rather than 8.
#   * MTG decisions batched 8-up (decode is bandwidth-bound; see
#     LocalGemma4Client.generate_batch).
#
# For a number worth quoting, run the full set instead:
#   --holdout out/e4b-v65536/eval/toolcall_holdout.jsonl --max-turns-per-session 4
# which is ~1.5 h per arm.
#
# Everything runs in-process against the training venv (flash_attn is ABI-pinned
# to its torch), so the GPU must be free -- run it at a training stop.
#
# THREE arms by default, in pipeline order:
#
#   vanilla (unmodified)   google/gemma-4-E4B-it-qat-q4_0-unquantized -- what
#                          the pipeline started from. Its chat template is
#                          byte-identical to ours (sha1 82a71fd41798), so the
#                          tool-call wire format is the same and the parser
#                          reads its output unchanged. Read this arm as "what
#                          the whole pipeline bought", and read it knowing the
#                          holdout is drawn from OUR corpus: part of any gap is
#                          our training, part is our corpus's own conventions.
#   stage 0 final          the frozen base every adapter sits on -- "what
#                          stage 1 bought".
#   checkpoint-N           the adapter under test.
#
# Add --include-pruned-base to either eval for a fourth arm that separates the
# damage pruning did from the repair stage 0 made.
#
#   ./scripts/run_e4b_v65536_quick_eval.sh <ckpt-dir> [<ckpt-dir> ...]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/workspace/venvs/gemma4/bin/python
STAGE0=/workspace/models/gemma4-e4b-stage0-32k-v65536/final
OUT="$REPO/out/e4b-v65536/eval"
STAMP="$(date +%Y%m%d-%H%M)"
mkdir -p "$OUT"

if [ $# -lt 1 ]; then
    echo "usage: $0 <checkpoint-dir> [<checkpoint-dir> ...]" >&2
    exit 2
fi
ADAPTERS=("$@")
for a in "${ADAPTERS[@]}"; do
    [ -f "$a/adapter_model.safetensors" ] || { echo "not an adapter: $a" >&2; exit 2; }
done

if nvidia-smi --query-gpu=memory.used --format=csv,noheader | grep -qvE '^\s*[0-9]{1,4} MiB'; then
    echo "WARNING: the GPU is not idle. Training must be stopped first." >&2
    nvidia-smi --query-gpu=memory.used --format=csv,noheader >&2
    exit 1
fi

echo "=== 1/3  tool-call accuracy (quick: 36 sessions, 3 turns each) ==="
PYTHONPATH="$REPO/src" "$PY" "$REPO/scripts/eval_toolcall_local.py" \
    --holdout "$OUT/toolcall_holdout_quick.jsonl" \
    --adapters "${ADAPTERS[@]}" \
    --include-vanilla \
    --max-turns-per-session 3 --max-tokens 1536 --max-len 65536 \
    --out "$OUT/toolcall_${STAMP}.json"

echo "=== 2/3  MTG gameplay decisions (68 rows, exact match on move_index) ==="
PYTHONPATH="$REPO/src" "$PY" "$REPO/scripts/eval_mtg_gameplay.py" \
    --holdout /workspace/mtg-gameplay-heldout.jsonl \
    --adapters "${ADAPTERS[@]}" \
    --include-vanilla \
    --batch-size 8 --max-len 32768 \
    --out "$OUT/mtg_gameplay_${STAMP}.json"

echo "=== 3/3  MTG instruct, bits-per-byte (300 rows, no generation) ==="
"$PY" /workspace/LLM-Training-Kit/scripts/native/paired_checkpoints.py \
    --holdout /workspace/mtg-instruct-heldout.jsonl \
    --max-len 32768 --include-vanilla --adapters "${ADAPTERS[@]}" \
    --out "$OUT/mtg_instruct_bpb_${STAMP}.json"

echo
echo "results in $OUT (stamp $STAMP):"
ls -1 "$OUT" | grep "$STAMP" | sed 's/^/  /'
