#!/usr/bin/env python3
"""Long-context retrieval check against an OpenAI-compatible server.

Deliverable 5 of the W4A16 handoff: the point of calibrating at 32K is
long-context behaviour, so it has to be *tested*, not assumed. This is the same
probe that validated the gemma-4 W4A16 checkpoint (a rare token retrieved with
exact spelling), generalized to sweep several depths.

The needle is a **rare, hyphenated, digit-bearing token sequence** on purpose:
that is precisely what a damaged or over-quantized output head garbles first
(the osoi5 checkpoint returned "Pineple" for "PINEAPPLE-7742"). A model that
returns the right *concept* with the wrong *spelling* has failed this test.

Haystack text is drawn from a real eval corpus rather than repeated filler, so
the distractor distribution matches what the model was calibrated on.

    .venv-vllm/bin/python scripts/longctx_needle.py \
        --base-url http://127.0.0.1:18080/v1 \
        --model out/exp-060-w4a16-32k/checkpoint \
        --haystack out/exp-060-32k/corpora/corpus.eval.broad.txt \
        --target-tokens 30000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

NEEDLE_TEMPLATE = "The {label} authorization code is {secret}. Remember it exactly."
DEFAULT_SECRETS = [
    ("quarterly", "PINEAPPLE-7742"),
    ("archival", "ZEPHYR-3391-QX"),
    ("emergency", "MARMALADE-8156"),
]


def _post(url: str, payload: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def build_prompt(haystack: str, needle: str, depth: float) -> str:
    """Insert ``needle`` at ``depth`` (0.0 = start, 1.0 = end) on a line boundary."""
    lines = haystack.splitlines()
    at = max(0, min(len(lines), int(len(lines) * depth)))
    return "\n".join(lines[:at] + [needle] + lines[at:])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:18080/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--haystack", required=True, type=Path)
    ap.add_argument("--target-tokens", type=int, default=30000)
    ap.add_argument(
        "--chars-per-token",
        type=float,
        default=3.6,
        help="rough ratio used to size the haystack before sending",
    )
    ap.add_argument("--depths", type=float, nargs="*", default=[0.1, 0.5, 0.9])
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out", type=Path, default=None, help="optional JSON results path")
    args = ap.parse_args()

    text = args.haystack.read_text(encoding="utf-8")
    want_chars = int(args.target_tokens * args.chars_per_token)
    while len(text) < want_chars:
        text += "\n" + text
    text = text[:want_chars]

    results = []
    for depth, (label, secret) in zip(args.depths, DEFAULT_SECRETS, strict=False):
        needle = NEEDLE_TEMPLATE.format(label=label, secret=secret)
        haystack = build_prompt(text, needle, depth)
        question = (
            f"What is the {label} authorization code mentioned in the document? "
            "Reply with the code only, copied exactly."
        )
        started = time.time()
        resp = _post(
            f"{args.base_url.rstrip('/')}/chat/completions",
            {
                "model": args.model,
                "messages": [
                    {"role": "user", "content": f"{haystack}\n\n{question}"},
                ],
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
            },
        )
        answer = resp["choices"][0]["message"]["content"] or ""
        usage = resp.get("usage", {})
        exact = secret in answer
        # Case-insensitive, punctuation-tolerant near-miss: right concept, wrong spelling.
        loose = bool(re.search(re.escape(secret).replace(r"\-", r"[-\s]?"), answer, re.I))
        results.append(
            {
                "depth": depth,
                "label": label,
                "secret": secret,
                "prompt_tokens": usage.get("prompt_tokens"),
                "exact": exact,
                "loose": loose,
                "answer": answer.strip()[:300],
                "elapsed_s": round(time.time() - started, 1),
            }
        )
        status = "EXACT" if exact else ("SPELLING-MISS" if loose else "FAIL")
        print(
            f"depth {depth:>4} | {usage.get('prompt_tokens')} prompt tok | "
            f"{status} | {answer.strip()[:80]!r}",
            flush=True,
        )

    n_exact = sum(r["exact"] for r in results)
    print(f"\n{n_exact}/{len(results)} exact retrievals")
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")
    sys.exit(0 if n_exact == len(results) else 1)


if __name__ == "__main__":
    main()
