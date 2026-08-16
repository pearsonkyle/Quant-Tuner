"""Verify a built calibration corpus carries tool schemas and reasoning correctly.

These are the two properties a quant most easily loses and that a naive count most easily
lies about:

* **Reasoning.** Chat templates keep reasoning only on a render's FINAL assistant turn and
  scrub it from history — and they emit an *empty* ``<think></think>`` on that turn when
  none is supplied. A bare ``</think>`` count therefore reports healthy coverage on a corpus
  with almost no real reasoning (measured on a real build: 2 genuine blocks out of 4,291
  available). So this counts NON-EMPTY blocks specifically.
* **Tool schemas.** The packer dedups schemas (``tool_schema_quota``): full schemas render
  in the first window per unique schema set, every other window gets a name-only stub. Both
  must be present — all-stub means the quant never sees a real schema, all-full means the
  boilerplate is eating the calibration budget.

Also re-runs ``universal.scan_special_tokens`` on the bytes as written, which is what
guarantees ``<|im_start|>`` / ``<tool_call>`` reach llama-imatrix as single control ids
rather than shattering into ordinary BPE pieces.

    PYTHONPATH=src .venv/bin/python scripts/exp060_verify_cal_corpus.py \\
        --corpus out/exp-060-32k/corpora/corpus.cal.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.data import universal

THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path,
                   default=REPO / "out/exp-060-32k/corpora/corpus.cal.txt")
    p.add_argument("--model", default="Qwen/Qwen3.8-27B")
    a = p.parse_args()

    # newline="" — universal newlines would rewrite the \r in agent tool output.
    with a.corpus.open(newline="") as f:
        text = f.read()

    print(f"corpus: {a.corpus}  ({len(text):,} chars)\n")

    # --- reasoning --------------------------------------------------------------------
    blocks = THINK_BLOCK.findall(text)
    nonempty = [b for b in blocks if b.strip()]
    print("REASONING")
    print(f"  <think> opens          : {text.count('<think>'):,}")
    print(f"  </think> closes        : {text.count('</think>'):,}")
    print(f"  matched blocks         : {len(blocks):,}")
    print(f"  NON-EMPTY blocks       : {len(nonempty):,}   <-- the number that matters")
    if nonempty:
        lens = sorted(len(b.strip()) for b in nonempty)
        print(f"  block chars med/p90/max: {lens[len(lens)//2]:,} / "
              f"{lens[int(0.9*(len(lens)-1))]:,} / {lens[-1]:,}")
        print(f"  reasoning chars total  : {sum(lens):,} "
              f"({100*sum(lens)/len(text):.2f}% of corpus)")

    # --- tool calls -------------------------------------------------------------------
    calls = TOOL_CALL_BLOCK.findall(text)
    parsed_ok = 0
    named: set[str] = set()
    for c in calls:
        try:
            obj = json.loads(c.strip())
        except Exception:  # noqa: BLE001
            continue
        parsed_ok += 1
        if isinstance(obj, dict) and obj.get("name"):
            named.add(str(obj["name"]))
    print("\nTOOL CALLS")
    print(f"  <tool_call> blocks     : {len(calls):,}")
    print(f"  parse as JSON          : {parsed_ok:,} "
          f"({100*parsed_ok/max(1,len(calls)):.1f}%)")
    print(f"  distinct tool names    : {len(named):,}")
    print(f"  <tool_response> markers: {text.count('<tool_response>'):,}")
    if named:
        print(f"  sample names           : {', '.join(sorted(named)[:12])}")

    # --- tool schemas -----------------------------------------------------------------
    # A rendered schema carries a JSON Schema parameters object; a stub does not.
    full_schema = text.count('"parameters"')
    typed_fn = text.count('"type": "function"') + text.count('"type":"function"')
    tools_hdr = text.count("# Tools")
    print("\nTOOL SCHEMAS")
    print(f"  '# Tools' sections     : {tools_hdr:,}")
    print(f"  '\"type\": \"function\"'   : {typed_fn:,}")
    print(f"  '\"parameters\"' objects : {full_schema:,}   <-- full schemas (stubs lack these)")

    # --- special tokens ---------------------------------------------------------------
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    scan = universal.scan_special_tokens(text, tok)
    print("\nSPECIAL TOKENS (must each round-trip to exactly ONE id)")
    bad = []
    for name, info in sorted((scan.get("tokens") or scan).items()
                             if isinstance(scan.get("tokens") or scan, dict) else []):
        if isinstance(info, dict) and "n_ids" in info:
            ok = info["n_ids"] == 1
            if not ok:
                bad.append(name)
            print(f"  {name:24s} count={info.get('count', '?'):>8} n_ids={info['n_ids']} "
                  f"{'ok' if ok else 'SHATTERED'}")
    if not bad:
        print("  all present special tokens tokenize to a single id")
    else:
        print(f"  !! SHATTERED: {bad}")

    print("\nraw scan_special_tokens:")
    print(json.dumps(scan, indent=2)[:1500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
