#!/usr/bin/env python3
"""Write the card's MTP section from measured results.

Same contract as the other splicers: replaces everything between `<!-- MTP_TABLE -->` and
the next `---`, is idempotent, and asserts rather than silently appending. The section
wording is chosen by the DATA, not by hand — a working head, a head that starts but never
drafts, and a head that will not start at all are three different claims, and the card
should never overstate whichever one we actually got.

    PYTHONPATH=src .venv/bin/python scripts/fill_mtp_section.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MARKER = "<!-- MTP_TABLE -->"
# NOT "---": markdown table separator rows (|---:|---:|) contain it, so a bare "---"
# sentinel matches INSIDE the table this script just wrote, truncates at the wrong point
# and leaves the previous version of the section behind. Anchor on the next heading.
# Anchored on the <details> that follows the MTP section. NOT "---" (markdown table
# separators contain it) and NOT "## How it was made" (that heading now lives inside a
# <details> summary, so it is no longer a line-start heading).
END = "<details>"


def render(res: dict) -> str:
    rows = res.get("rows", [])
    base = res.get("baseline_tok_s")
    live = [r for r in rows if r.get("started") and r.get("draft")]

    if not rows:
        return "*pending*"

    if not any(r.get("started") for r in rows):
        ns = ", ".join(str(r["n"]) for r in rows)
        return (
            f"**Not available.** Tested at `num_speculative_tokens` {ns} — none start:\n\n"
            "```\nNotImplementedError: Unsupported speculative method: 'mtp'\n```\n\n"
            "The draft head is present and intact; vLLM's MTP detection simply does not "
            "recognise this checkpoint. For speculative decoding today use the "
            "[GGUF ladder](https://huggingface.co/pearsonkyle/Qwen3.8-27B-imatrix-MTP-GGUF) "
            "(`--spec-type draft-mtp --spec-draft-n-max 1`)."
        )

    if not live:
        return (
            "**Starts but never drafts.** The server accepts `speculative_config` and comes "
            "up, but emits zero draft tokens, so there is no speed-up. Serve without "
            "`--speculative-config`; the head costs 0.79 GiB and currently does nothing."
        )

    best = max(live, key=lambda r: r.get("speedup") or 0)
    lines = [
        "The trained MTP draft head ships **inside** the checkpoint at bf16 — no second "
        "file to download.\n",
        "```bash",
        "vllm serve pearsonkyle/Qwen3.8-27B-GPTQ-W4A16 \\",
        "    --max-model-len 32768 \\",
        f'    --speculative-config \'{{"method":"qwen3_5_mtp","num_speculative_tokens":{best["n"]}}}\'',
        "```\n",
        f"| draft-n | decode tok/s | vs baseline | acceptance |",
        "|---:|---:|---:|---:|",
        f"| off *(baseline)* | {base:.1f} | 1.00× | — |",
    ]
    for r in rows:
        if not r.get("started"):
            lines.append(f"| {r['n']} | did not start | — | — |")
            continue
        acc = f"{r['acceptance']:.1%}" if r.get("acceptance") is not None else "—"
        lines.append(f"| {r['n']} | {r['tok_s']:.1f} | {r['speedup']:.2f}× | {acc} |")

    gain = best.get("speedup") or 0
    if gain < 1.05:
        lines.append(
            f"\n**Enabled it is not worth it here:** best is {gain:.2f}× at "
            f"{best['acceptance']:.1%} acceptance. Qwen3.8 exposes one nextn layer, so the "
            "ceiling is low; serve without it unless you measure a win on your own traffic."
        )
    else:
        lines.append(
            f"\n**Best setting: `num_speculative_tokens: {best['n']}`** — {gain:.2f}× decode at "
            f"{best['acceptance']:.1%} acceptance. Qwen3.8 has **one** nextn layer, so a deeper "
            "draft re-runs that head on its own guess and per-token acceptance drops — but it "
            "still nets **more accepted tokens per step**, which is what throughput follows. "
            "Optimise tokens-gained-per-step, not acceptance rate."
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--card", type=Path,
                    default=Path("out/exp-060-w4a16-32k/release/README.md"))
    ap.add_argument("--results", type=Path,
                    default=Path("out/exp-060-w4a16-32k/results/graft_validation.json"))
    a = ap.parse_args()

    res = json.loads(a.results.read_text()) if a.results.exists() else {}
    body = render(res)

    text = a.card.read_text()
    if MARKER not in text:
        raise SystemExit(f"marker {MARKER!r} not found in {a.card}")
    head, rest = text.split(MARKER, 1)
    if END not in rest:
        raise SystemExit(f"end sentinel {END!r} not found after the marker — refusing to splice")
    tail = rest[rest.index(END):]
    a.card.write_text(f"{head}{MARKER}\n\n{body}\n\n{tail}")

    written = a.card.read_text()
    n = written.count(MARKER)
    if n != 1:
        raise SystemExit(f"post-write check failed: {n} markers, expected 1")
    # The section must contain exactly one table header — a duplicate means the sentinel
    # matched in the wrong place and stale content survived.
    dup = written.count("| draft-n | decode tok/s |")
    if dup > 1:
        raise SystemExit(f"post-write check failed: {dup} MTP tables, expected <=1")
    print(f"spliced MTP section ({'pending' if body == '*pending*' else 'measured'})")


if __name__ == "__main__":
    main()
