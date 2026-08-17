#!/usr/bin/env python3
"""Read a packed training window the way the model sees it.

    python scripts/inspect_corpus_window.py out/exp-058/corpus_ourssft_32768.pt --window 0
    python scripts/inspect_corpus_window.py CORPUS --audit          # structural checks only

Every other tool here reports statistics ABOUT the corpus. This prints the corpus, with
the supervised positions marked, because the statistics cannot show a malformed turn
boundary, a control token that tokenized as prose, or a supervised span that starts in the
wrong place — and those are exactly the faults that train a model to do something other
than what was intended while every aggregate looks healthy.

`[[...]]` marks a SUPERVISED target: the model is scored on producing that token. Plain
text is context it reads but is not scored on. The distinction is the whole design of the
corpus, so seeing it directly is the point.

--audit runs the structural checks instead: control tokens are single ids, every assistant
turn is closed, supervision starts after the assistant header and includes the closing
<|im_end|>, and no supervised span leaks into a user or tool turn.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from quant_tuner.qat.corpus import load_tokenizer  # noqa: E402

IGNORE = -100
CONTROL = ["<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>",
           "<tool_response>", "</tool_response>", "<think>", "</think>"]


def render(ids, lbl, tok, start: int, n: int) -> str:
    """Decode a span, bracketing supervised targets."""
    out, buf, sup = [], [], None
    for i in range(start, min(start + n, len(ids))):
        # A position is supervised when the model is scored on producing ids[i], which is
        # lbl[i] != IGNORE under the corpus's own alignment.
        s = lbl[i] != IGNORE
        if s != sup:
            if buf:
                text = tok.decode(buf)
                out.append(f"[[{text}]]" if sup else text)
            buf, sup = [], s
        buf.append(ids[i])
    if buf:
        text = tok.decode(buf)
        out.append(f"[[{text}]]" if sup else text)
    return "".join(out)


def audit(path: Path, tok, max_windows: int | None) -> int:
    blob = torch.load(path, weights_only=False)
    ids_all, lbl_all = blob["ids"], blob["labels"]
    if max_windows:
        ids_all, lbl_all = ids_all[:max_windows], lbl_all[:max_windows]

    cid = {t: tok.convert_tokens_to_ids(t) for t in CONTROL}
    bad_single = [t for t, i in cid.items()
                  if i is None or i < 0 or len(tok.encode(t, add_special_tokens=False)) != 1]
    im_start, im_end = cid["<|im_start|>"], cid["<|im_end|>"]

    print(f"corpus       {path.name}")
    print(f"windows      {ids_all.shape[0]} x {ids_all.shape[1]}")
    print("control ids  " + ", ".join(f"{t}={i}" for t, i in cid.items()))
    if bad_single:
        print(f"  !! NOT single ids: {bad_single}")
    else:
        print("  all control tokens are single ids")

    # role of each turn = the token(s) right after <|im_start|>
    roles: collections.Counter = collections.Counter()
    sup_by_role: collections.Counter = collections.Counter()
    tok_by_role: collections.Counter = collections.Counter()
    unclosed = 0
    sup_starts_at: collections.Counter = collections.Counter()
    sup_includes_stop = 0
    sup_spans = 0
    leaked = 0

    for w in range(ids_all.shape[0]):
        ids = ids_all[w].tolist()
        lbl = lbl_all[w].tolist()
        i, role, turn_open = 0, None, False
        span_open = False
        while i < len(ids):
            if ids[i] == im_start:
                role_txt = tok.decode(ids[i + 1:i + 4]).split("\n")[0].strip()
                role = role_txt or "?"
                roles[role] += 1
                turn_open = True
                # where does supervision begin relative to the header?
                j = i + 1
                while j < len(ids) and lbl[j] == IGNORE and ids[j] != im_end:
                    j += 1
                if j < len(ids) and lbl[j] != IGNORE:
                    sup_starts_at[role] += 1
            elif ids[i] == im_end:
                if turn_open:
                    turn_open = False
                else:
                    unclosed += 1
                if role and lbl[i] != IGNORE:
                    sup_includes_stop += 1
            if lbl[i] != IGNORE:
                tok_by_role[role] += 1
                if role not in ("assistant",):
                    leaked += 1
                if not span_open:
                    sup_spans += 1
                    sup_by_role[role] += 1
                    span_open = True
            else:
                span_open = False
            i += 1

    print()
    print("turns by role:")
    for r, n in roles.most_common():
        print(f"  {r:<12} {n:>8,} turns   {tok_by_role.get(r,0):>12,} supervised tokens"
              f"   {sup_by_role.get(r,0):>8,} spans")
    print()
    print(f"supervised spans           {sup_spans:,}")
    print(f"spans closing on <|im_end|> {sup_includes_stop:,} "
          f"({100*sup_includes_stop/max(1,sup_spans):.1f}% of spans)")
    print(f"supervised tokens in NON-assistant turns  {leaked:,}"
          f"{'   <-- LEAK' if leaked else '   (none, correct)'}")
    print(f"stray <|im_end|> with no open turn        {unclosed:,}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--chars", type=int, default=4000)
    ap.add_argument("--tokens", type=int, default=1200)
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--max-windows", type=int, default=None)
    a = ap.parse_args()

    tok = load_tokenizer()
    if a.audit:
        return audit(a.corpus, tok, a.max_windows)

    blob = torch.load(a.corpus, weights_only=False)
    ids = blob["ids"][a.window].tolist()
    lbl = blob["labels"][a.window].tolist()
    print(f"# window {a.window} of {blob['ids'].shape[0]}, tokens {a.start}.."
          f"{a.start + a.tokens}  ([[x]] = supervised)")
    print(render(ids, lbl, tok, a.start, a.tokens)[:a.chars])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
