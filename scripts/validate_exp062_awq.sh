#!/usr/bin/env bash
# Post-build audit of the exp-062 AWQ rungs (IQ2_M + IQ3_M).
#
# Everything here is cheap and read-only. Run it before any eval: each check
# guards a failure whose ONLY symptom is a mediocre benchmark number hours later,
# which is indistinguishable from "AWQ just didn't help".
set -uo pipefail
cd /workspace/Quant-Tuner

TMPL=data/chat_templates/qwen3_8_safe_v2.jinja
NAME=Qwen3.8-27B
RC=0

audit_rung() {
  local ws="$1" quant="$2" opp="$3"
  echo
  echo "##############################################################"
  echo "# $(basename "$quant")"
  echo "##############################################################"

  if [ ! -f "$quant" ]; then
    echo "  SKIP: not built yet ($quant)"
    return 0
  fi

  echo "=== 1. Metadata reads back correctly (template + general.name) ==="
  # The pipeline bakes these after llama-quantize. If the bake silently no-op'd,
  # llama-server would render with whatever the F16 carried — and the four
  # template bugs (reasoning_effort=high -> HTTP 400 among them) come back.
  PYTHONPATH=vendor/llama.cpp/gguf-py .venv/bin/python - "$quant" "$TMPL" "$NAME" <<'EOF' 2>/dev/null || RC=1
import sys
from gguf import GGUFReader
quant, tmpl_path, name = sys.argv[1:4]
r = GGUFReader(quant)
def s(k):
    f = r.fields.get(k)
    return bytes(f.parts[f.data[0]]).decode() if f else None
tmpl = s("tokenizer.chat_template")
want = open(tmpl_path).read()
print(f"  general.name        : {s('general.name')!r}")
print(f"  chat_template bytes : {len(tmpl) if tmpl else 0}")
print(f"  matches repo .jinja : {tmpl == want}")
assert s("general.name") == name, "general.name not applied"
assert tmpl == want, "chat template not applied"
print("  PASS")
EOF

  echo
  echo "=== 2. MTP draft head really got the Q8_0 pin ==="
  # llama-quantize silently accepts a --tensor-type pattern matching NOTHING, so
  # an unpinned head shows up only as a mediocre draft acceptance rate.
  PYTHONPATH=vendor/llama.cpp/gguf-py .venv/bin/python - "$quant" <<'EOF' 2>/dev/null || RC=1
import sys
from collections import Counter
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
blk64 = [t for t in r.tensors if t.name.startswith("blk.64.")]
types = Counter(t.tensor_type.name for t in blk64)
print(f"  blk.64 tensors: {len(blk64)}  types: {dict(types)}")
assert len(blk64) == 15, f"expected 15 MTP tensors, got {len(blk64)}"
assert types.get("Q8_0", 0) == 8, f"expected 8 Q8_0 tensors, got {types.get('Q8_0', 0)}"
print("  PASS: 15 tensors, 8 x Q8_0")
EOF

  echo
  echo "=== 3. imatrix coverage: who did llama-quantize quantize blind? ==="
  # Expected: token_embd.weight (an embedding lookup, never collectable) and the
  # blk.64.* MTP head (outside the forward pass, mitigated by the Q8_0 pin).
  # ANYTHING ELSE in this list is a bug.
  local qlog
  qlog=$(ls "$ws"/logs/quantize-*.log 2>/dev/null | head -1)
  if [ -n "$qlog" ] && [ -f "$qlog" ]; then
    if grep -o "did not find weights for [^ ]*" "$qlog" | awk '{print $NF}' | sort -u \
      | awk '{ if ($0 == "token_embd.weight" || $0 ~ /^blk\.64\./) print "  expected  : " $0;
               else { print "  UNEXPECTED: " $0; bad++ } }
             END { exit (bad > 0) }'; then
      echo "  PASS: only the two expected families"
    else
      echo "  FAIL: unexpected tensors quantized without imatrix" >&2
      RC=1
    fi
  else
    echo "  WARN: no quantize log under $ws/logs/"
  fi

  echo
  echo "=== 4. Size / bpw vs the shipped opponent ==="
  if [ -f "$opp" ]; then
    printf "  %-46s %s\n" "$(basename "$opp") (shipped)" "$(du -h "$opp" | cut -f1)"
  else
    printf "  %-46s %s\n" "$(basename "$opp") (shipped)" "not on disk — pull from HF to compare"
  fi
  printf "  %-46s %s\n" "$(basename "$quant") (new)" "$(du -h "$quant" | cut -f1)"
}

audit_rung out/exp-062-awq-iq2m \
  out/exp-062-awq-iq2m/gguf/IQ2_M-awq-best-hybrid_custom-mtp.gguf \
  out/exp-060-32k/iq2_m/Qwen3.8-27B-IQ2_M.gguf

audit_rung out/exp-062-awq-iq3m \
  out/exp-062-awq-iq3m/gguf/IQ3_M-awq-best-hybrid_custom-mtp.gguf \
  out/exp-060-32k/iq3_m/Qwen3.8-27B-IQ3_M.gguf

audit_rung out/exp-062-awq-iq4xs \
  out/exp-062-awq-iq4xs/gguf/IQ4_XS-awq-best-hybrid_custom-mtp.gguf \
  out/exp-060-32k/iq4_xs/Qwen3.8-27B-IQ4_XS.gguf

echo
if [ "$RC" -eq 0 ]; then echo "All audits passed."; else echo "AUDITS FAILED (rc=$RC)" >&2; fi
exit "$RC"
