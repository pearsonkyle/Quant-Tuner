#!/usr/bin/env python3
"""Export the universal SFT corpus as chat-template-ready conversations.

The packed ``.pt`` / window ``jsonl.gz`` that ``build_sft_qat_corpus.py`` produces are
token ids: already templated, already windowed, and not reversible back to messages. For
ordinary SFT you want the conversations themselves, so a trainer can apply its own chat
template.

    python scripts/export_sft_chat_jsonl.py \\
        --sft out/corpora/qwen3-universal-v2/sft.jsonl.gz \\
        --split train --out out/exp-058/sft_chat_train.jsonl.gz

One line per conversation::

    {"id": ..., "source": "logs-agents", "split": "train",
     "messages": [...], "tools": [...] | null, "meta": {...},
     "n_messages": 41, "n_tool_calls": 18, "n_chars": 92014}

Two things this does that a plain copy of ``sft.jsonl.gz`` does not:

**1. Every row is rendered through a real chat template before it is written.** A row that
raises is a row that would blow up mid-training run, so the failure surfaces here. Strict
templates reject conversations that look fine as data — Qwen3.6's official template raises
"No user query found" for a window with no user turn — and that is exactly the class of
problem worth catching before a fine-tune, not during one. ``--on-error skip`` drops and
reports them instead of failing the export.

**2. ``tools`` is materialized.** The QAT path reconstructs schemas from observed calls when
a source carries none; doing that here means the consumer needs no helper from this repo to
render tool-calling conversations correctly.

``--render`` additionally stores the templated string in a ``text`` field. Off by default:
it roughly doubles the file and pins it to one model's template, which defeats the point of
shipping messages.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.qat.corpus import (  # noqa: E402
    CHAT_TEMPLATE,
    MODEL,
    load_tokenizer,
    reconstruct_tools,
)

_COUNTS = ("n_tool_calls", "n_tool_results", "n_reasoning")


def trim_to_terminal(messages: list[dict]) -> list[dict]:
    """Drop trailing messages until the conversation ends on a complete assistant answer.

    The terminating stop token is the *stop decision* — it is what teaches a model to
    finish rather than keep going, and omitting it from the trainable span is what caused
    the iter-2/3 looping this repo has already been bitten by. A conversation that ends on
    a tool result, an unanswered user turn, or an assistant turn still holding pending
    ``tool_calls`` has no stop decision at its tail: the last thing the model sees is a
    context that demands continuation. Trimming back to the last complete answer is what
    makes the final ``<|im_end|>`` mean "this is where you stop".

    Returns a possibly-empty list; the caller decides whether what remains is usable.
    """
    out = list(messages)
    while out:
        last = out[-1]
        if (last.get("role") == "assistant"
                and (last.get("content") or "").strip()
                and not last.get("tool_calls")):
            break
        out.pop()
    return out


def stop_suffix(tok) -> str:
    """The token a finished conversation must end on, for the render check."""
    return tok.eos_token or "<|im_end|>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", type=Path, required=True, help="sft.jsonl(.gz) from data.universal")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="train",
                    help="'train' (default) keeps the eval holdouts held out; 'all' for every row")
    ap.add_argument("--source", action="append", default=None,
                    help="restrict to these sources (repeatable)")
    ap.add_argument("--model", type=Path, default=MODEL, help="tokenizer dir for the render check")
    ap.add_argument("--chat-template", type=Path, default=CHAT_TEMPLATE)
    ap.add_argument("--render", action="store_true",
                    help="also store the templated string in a `text` field")
    ap.add_argument("--on-error", choices=["fail", "skip"], default="skip",
                    help="what to do with a conversation the template rejects")
    ap.add_argument("--no-ensure-terminal", action="store_true",
                    help="keep conversations that do not end on a complete assistant "
                         "answer (they carry no stop decision — see trim_to_terminal)")
    args = ap.parse_args()

    tok = load_tokenizer(args.model, args.chat_template)

    opener = gzip.open if args.sft.suffix == ".gz" else open
    with opener(args.sft, "rt") as fh:
        rows = [json.loads(ln) for ln in fh if ln.strip()]
    if args.split != "all":
        rows = [r for r in rows if r.get("split") == args.split]
    if args.source:
        rows = [r for r in rows if r.get("source") in set(args.source)]
    if not rows:
        sys.exit("no rows matched --split/--source")

    kept: collections.Counter[str] = collections.Counter()
    trimmed: collections.Counter[str] = collections.Counter()
    dropped: dict[str, collections.Counter[str]] = {
        k: collections.Counter() for k in ("template", "no_terminal", "no_user")}
    chars = collections.Counter()
    stop = stop_suffix(tok)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(args.out, "wt") as out:
        for r in rows:
            src = r.get("source", "?")
            msgs = r.get("messages") or []

            if not args.no_ensure_terminal:
                kept_msgs = trim_to_terminal(msgs)
                if len(kept_msgs) != len(msgs):
                    trimmed[src] += 1
                msgs = kept_msgs
                # A conversation needs something to answer and an answer to train on.
                # Strict templates (Qwen3.6's official one) reject the first case outright.
                if not any(m.get("role") == "user" for m in msgs) or not msgs:
                    dropped["no_user" if msgs else "no_terminal"][src] += 1
                    continue

            # Match what the QAT path renders with, so this export and that corpus agree
            # on which schemas each conversation was conditioned on.
            tools = r.get("tools") or reconstruct_tools(msgs) or None
            try:
                text = tok.apply_chat_template(msgs, tools=tools, tokenize=False,
                                               add_generation_prompt=False)
            except Exception as exc:
                if args.on_error == "fail":
                    sys.exit(f"chat template rejected {src}:{r.get('id')} — "
                             f"{type(exc).__name__}: {exc}\n"
                             f"re-run with --on-error skip to drop and report these")
                dropped["template"][src] += 1
                continue

            # The whole point of the trim: the render has to actually end on the stop
            # token, or the row teaches continuation instead of termination.
            if not text.rstrip().endswith(stop):
                dropped["no_terminal"][src] += 1
                continue

            rec = {"id": r.get("id"), "source": src, "split": r.get("split"),
                   "messages": msgs, "tools": tools, "meta": r.get("meta") or {},
                   "n_messages": len(msgs), "n_chars": len(text)}
            rec |= {k: r.get(k) for k in _COUNTS if r.get(k) is not None}
            if args.render:
                rec["text"] = text
            out.write(json.dumps(rec) + "\n")
            kept[src] += 1
            chars[src] += len(text)

    mb = args.out.stat().st_size / 1024**2
    print(f"[chat-export] {sum(kept.values())} conversations -> {args.out} ({mb:.1f} MiB gz)")
    print(f"[chat-export] template: {args.chat_template}")
    print(f"[chat-export] every row verified to render and end on {stop!r}")
    for src in sorted(kept):
        print(f"    {src:20s} {kept[src]:6d} convs  {chars[src]:12,d} templated chars"
              + (f"  ({trimmed[src]} trimmed to a terminal answer)" if trimmed[src] else ""))
    for why, counts in dropped.items():
        if counts:
            print(f"[chat-export] dropped ({why}): {dict(counts)} = {sum(counts.values())}",
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
