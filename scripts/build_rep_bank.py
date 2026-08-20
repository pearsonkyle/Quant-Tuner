"""Extract a bank of REAL (tool_call, tool_response) pairs + task statements from the
training corpus, for repetition-steering contexts.

anchor8 proved the synthetic contexts don't transfer: the escalation was inverted on
the training distribution (0.27->0.22 over k=1..5, below vanilla) while the real
episode still looped 56x — the real loop state (real commands, real tracebacks/empty
results, real task framing) looks nothing like the synthetic bank. This builds the
contexts from the corpus itself. The eval instance (dask) is excluded so the mimic
stays held out.

    PYTHONPATH=src .venv/bin/python scripts/build_rep_bank.py \
        out/exp-058/fixed/corpus_ourssft_32768.pt --out out/exp-058/kd/rep_bank.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_tuner.qat.stop_probe import SENTENCE as PROBE_SENTENCE  # noqa: E402
from quant_tuner.qat.stop_probe import USER as PROBE_USER  # noqa: E402

CALL_RE = re.compile(r"<tool_call>\n.*?\n</tool_call>", re.S)
RESP_RE = re.compile(r"<tool_response>\n(.*?)\n</tool_response>", re.S)
USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)
EXCLUDE = ("dask", PROBE_SENTENCE, PROBE_USER)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--model-dir", default="out/exp-057/model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", type=int, default=200)
    ap.add_argument("--tasks", type=int, default=60)
    ap.add_argument("--max-call", type=int, default=600)
    ap.add_argument("--max-resp", type=int, default=800)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    blob = torch.load(args.corpus, weights_only=False, mmap=True)
    ids_all = blob["ids"]

    pairs: list[tuple[str, str]] = []
    tasks: list[str] = []
    seen_c: set[str] = set()
    seen_t: set[str] = set()
    for w in range(ids_all.shape[0]):
        text = tok.decode(ids_all[w], skip_special_tokens=False)
        calls = [(m.group(0), m.end()) for m in CALL_RE.finditer(text)]
        for call, end in calls:
            if len(call) > args.max_call or any(x in call for x in EXCLUDE):
                continue
            m = RESP_RE.search(text, end)
            if not m or m.start() - end > 80:          # response must follow the call
                continue
            resp = m.group(1)[:args.max_resp]
            if any(x in resp for x in EXCLUDE) or call in seen_c:
                continue
            seen_c.add(call)
            pairs.append((call, resp))
        for m in USER_RE.finditer(text):
            t = m.group(1).strip()
            if (150 < len(t) < 1200 and not t.startswith("<tool_response>")
                    and not any(x in t for x in EXCLUDE) and t not in seen_t):
                seen_t.add(t)
                tasks.append(t[:1000])
        if len(pairs) > args.pairs * 20 and len(tasks) > args.tasks * 5:
            break

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    rng.shuffle(tasks)
    bank = {"pairs": pairs[:args.pairs], "tasks": tasks[:args.tasks],
            "corpus": str(args.corpus), "seed": args.seed}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(bank, indent=1))
    print(f"[bank] {len(bank['pairs'])} pairs, {len(bank['tasks'])} tasks "
          f"-> {args.out} (from {len(seen_c)} unique calls scanned)", flush=True)


if __name__ == "__main__":
    main()
