#!/usr/bin/env bash
# Post-build audit of the exp-060 AWQ IQ2_M rung.
#
# Everything here is cheap and read-only except step 1. Run it before any eval:
# each check guards a failure whose ONLY symptom is a mediocre benchmark number
# hours later, which is indistinguishable from "AWQ just didn't help".
set -euo pipefail
cd /workspace/Quant-Tuner

WS=out/exp-060-32k-awq
QUANT="$WS/gguf/IQ2_M-awq-best-hybrid_custom-mtp.gguf"
TMPL=data/chat_templates/qwen3_8_safe_v2.jinja
NAME=Qwen3.8-27B

[ -f "$QUANT" ] || { echo "FATAL: no quant at $QUANT" >&2; exit 1; }

echo "=== 1. GGUF metadata (chat template + general.name, one rewrite) ==="
# The build that produced this file ran before general_name existed in the
# pipeline, so it baked the template only. set_metadata is idempotent per
# (template, name) via its stamp — a re-run after both were baked is a no-op.
PYTHONPATH=src .venv/bin/python - <<EOF
from pathlib import Path
from quant_tuner.quantize import gguf
q, t = Path("$QUANT"), Path("$TMPL")
stamp = gguf.template_stamp(q, t, "$NAME")
if stamp.exists():
    print(f"  already baked ({stamp.name}) — skipping")
else:
    gguf.set_metadata(q, template=t, general_name="$NAME",
                      log=Path("$WS/logs/chat-template.log"))
    print("  baked")
EOF

echo
echo "=== 2. Metadata reads back correctly ==="
PYTHONPATH=vendor/llama.cpp/gguf-py .venv/bin/python - <<EOF 2>/dev/null
from gguf import GGUFReader
r = GGUFReader("$QUANT")
def s(k):
    f = r.fields.get(k)
    return bytes(f.parts[f.data[0]]).decode() if f else None
tmpl = s("tokenizer.chat_template")
want = open("$TMPL").read()
print(f"  general.name        : {s('general.name')!r}")
print(f"  chat_template bytes : {len(tmpl) if tmpl else 0}")
print(f"  matches repo .jinja : {tmpl == want}")
assert s("general.name") == "$NAME", "general.name not applied"
assert tmpl == want, "chat template not applied"
EOF

echo
echo "=== 3. MTP draft head really got the Q8_0 pin (handoff 5.4) ==="
# llama-quantize silently accepts a --tensor-type pattern matching NOTHING, so
# an unpinned head shows up only as a mediocre draft acceptance rate.
PYTHONPATH=vendor/llama.cpp/gguf-py .venv/bin/python - <<EOF 2>/dev/null
from collections import Counter
from gguf import GGUFReader
r = GGUFReader("$QUANT")
blk64 = [t for t in r.tensors if t.name.startswith("blk.64.")]
types = Counter(t.tensor_type.name for t in blk64)
print(f"  blk.64 tensors: {len(blk64)}  types: {dict(types)}")
assert len(blk64) == 15, f"expected 15 MTP tensors, got {len(blk64)}"
assert types.get("Q8_0", 0) == 8, f"expected 8 Q8_0 tensors, got {types.get('Q8_0', 0)}"
print("  PASS: 15 tensors, 8 x Q8_0")
EOF

echo
echo "=== 4. imatrix coverage: who did llama-quantize quantize blind? (handoff 5.6) ==="
# Expected: token_embd.weight (an embedding lookup, never collectable) and the
# blk.64.* MTP head (outside the forward pass, mitigated by the Q8_0 pin).
# ANYTHING ELSE in this list is a bug.
LOG="$WS/logs/quantize-IQ2_M.log"
if [ -f "$LOG" ]; then
  grep -o "did not find weights for [^ ]*" "$LOG" | awk '{print $NF}' | sort -u \
    | awk '{ if ($0 == "token_embd.weight" || $0 ~ /^blk\.64\./) print "  expected : " $0;
             else { print "  UNEXPECTED: " $0; bad++ } }
           END { exit (bad > 0) }' \
    || { echo "  FAIL: unexpected tensors quantized without imatrix" >&2; exit 1; }
  echo "  PASS: only the two expected families"
else
  echo "  WARN: $LOG not found" >&2
fi

echo
echo "=== 5. Size / bpw vs the opponent ==="
OPP=out/exp-060-32k/iq2_m/Qwen3.8-27B-IQ2_M.gguf
for f in "$OPP" "$QUANT"; do
  printf "  %-56s %s\n" "$(basename "$f")" "$(du -h "$f" | cut -f1)"
done

echo
echo "All audits passed."
