#!/usr/bin/env bash
# Delete the intermediates an export leaves behind, keeping only the Q2_0 deliverable.
#
#     bash scripts/prune_export_intermediates.sh TAG [TAG...]
#     DRY_RUN=1 bash scripts/prune_export_intermediates.sh sft32k
#
# `export_qat` writes three things per tag:
#
#     out/exp-057/model_<tag>/                    ~31 GB   HF checkpoint, conversion input
#     out/exp-057/Ternary-Bonsai-8B-<tag>-F16.gguf ~19 GB   conversion output, quantize input
#     out/exp-057/Ternary-Bonsai-8B-<tag>-Q2_0.gguf ~2.1 GB THE DELIVERABLE
#
# So each export costs ~52 GB to produce a 2.1 GB artifact. The chain exports four times
# (the ablation plus three curriculum rounds) — ~208 GB of intermediates against ~130 GB
# of free disk, i.e. it fills the disk during round 2 and takes the run with it.
#
# Nothing outside export.py references either intermediate: grep for `model_<tag>` and
# `-F16.gguf` finds only the export script's own definitions. They are regenerable from
# the latents at any time, and for a NATIVELY TERNARY model the F16 is a lossless
# container of w = s*c rather than a higher-fidelity reference, so keeping it buys nothing
# a re-export could not.
#
# REFUSES to prune a tag with no Q2_0 — that means the export did not finish, and the
# intermediates are the only thing standing between you and re-running it.
set -euo pipefail
cd "$(dirname "$0")/.."

EXP_DIR="${EXP_DIR:-out/exp-057}"
[ $# -ge 1 ] || { echo "usage: prune_export_intermediates.sh TAG [TAG...]"; exit 2; }

total=0
for tag in "$@"; do
    q2="$EXP_DIR/Ternary-Bonsai-8B-${tag}-Q2_0.gguf"
    if [ ! -f "$q2" ]; then
        echo "[prune] $tag: no Q2_0 at $q2 — export unfinished, keeping intermediates"
        continue
    fi
    for victim in "$EXP_DIR/model_${tag}" "$EXP_DIR/Ternary-Bonsai-8B-${tag}-F16.gguf"; do
        [ -e "$victim" ] || continue
        sz=$(du -sm "$victim" 2>/dev/null | cut -f1)
        if [ "${DRY_RUN:-0}" = "1" ]; then
            echo "[prune] would remove $victim (${sz} MB)"
        else
            rm -rf "$victim"
            echo "[prune] removed $victim (${sz} MB)"
        fi
        total=$((total + sz))
    done
done

echo "[prune] $([ "${DRY_RUN:-0}" = "1" ] && echo "would free" || echo "freed") $((total / 1024)) GB"
df -h /workspace | tail -1
