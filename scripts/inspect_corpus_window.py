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
turn is closed, supervision starts after the assistant header and includes the closing stop
token, and no supervised span leaks into a user or tool turn.

The checks are per chat family and `--model` selects it. On gemma-4 the whole tool exchange
lives inside ONE model turn, so "in a model turn" stops implying "supervised" and the audit
gains the check that actually matters there: supervised tokens inside a
`<|tool_response>...<tool_response|>` span. Qwen needs no such check — a tool result is its
own turn and the role check already covers it.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from quant_tuner.qat.corpus import load_tokenizer  # noqa: E402
from quant_tuner.qat.dialect import detect as detect_dialect  # noqa: E402

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

    carryover = 0
    for w in range(ids_all.shape[0]):
        ids = ids_all[w].tolist()
        lbl = lbl_all[w].tolist()
        # A window starts mid-conversation: everything before its first <|im_start|> is
        # the tail of a turn that began in the PREVIOUS window. Counting those as
        # role-less supervised tokens reported a 4,078-token "leak" on a corpus with none,
        # and a false alarm here is worse than no check — it trains you to ignore it.
        first = next((k for k, t in enumerate(ids) if t == im_start), len(ids))
        carryover += sum(1 for k in range(first) if lbl[k] != IGNORE)
        i, role, turn_open = first, None, False
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
          f"{'   <-- REAL LEAK' if leaked else '   (none, correct)'}")
    print(f"carry-over from the previous window       {carryover:,}"
          f"   (expected: packed windows start mid-turn)")
    print(f"stray <|im_end|> with no open turn        {unclosed:,}"
          f"   (up to one per window is the same carry-over)")
    return 0


GEMMA4_CONTROL = ["<|turn>", "<turn|>", "<|tool>", "<tool|>", "<|tool_call>",
                  "<tool_call|>", "<|tool_response>", "<tool_response|>",
                  "<|channel>", "<channel|>"]


def audit_gemma4(path: Path, tok, max_windows: int | None) -> int:
    """The gemma-4 audit. Same questions as :func:`audit`, one extra and much sharper.

    gemma renders a whole tool exchange inside ONE ``model`` turn, so "is this token in a
    model turn" no longer implies "should the model be scored on it" — the tool RESULTS are
    in there too. **Supervised tokens inside a ``<|tool_response>…<tool_response|>`` span is
    therefore the check that matters**, and it is the one the Qwen audit has no notion of:
    on Qwen a tool result is its own turn and the role check already catches it.
    """
    blob = torch.load(path, weights_only=False)
    ids_all, lbl_all = blob["ids"], blob["labels"]
    if max_windows:
        ids_all, lbl_all = ids_all[:max_windows], lbl_all[:max_windows]

    cid = {t: tok.convert_tokens_to_ids(t) for t in GEMMA4_CONTROL}
    bad_single = [t for t, i in cid.items()
                  if i is None or i < 0 or len(tok.encode(t, add_special_tokens=False)) != 1]
    t_open, t_close = cid["<|turn>"], cid["<turn|>"]
    tr_open, tr_close = cid["<|tool_response>"], cid["<tool_response|>"]
    role_ids = {tok.convert_tokens_to_ids(r): r for r in ("model", "user", "system")}

    print(f"corpus       {path.name}   dialect gemma4")
    print(f"windows      {ids_all.shape[0]} x {ids_all.shape[1]}")
    print("control ids  " + ", ".join(f"{t}={i}" for t, i in cid.items()))
    print(f"  !! NOT single ids: {bad_single}" if bad_single
          else "  all control tokens are single ids")

    roles: collections.Counter = collections.Counter()
    tok_by_role: collections.Counter = collections.Counter()
    leaked = in_response = sup_stop = sup_spans = carryover = unclosed = 0
    sup_tool_call = 0

    for w in range(ids_all.shape[0]):
        ids, lbl = ids_all[w].tolist(), lbl_all[w].tolist()
        # As in the Qwen audit: a packed window starts mid-conversation, so supervised
        # tokens before its first <|turn> are the tail of the previous window's turn, not
        # a leak. Counting them as one would be a false alarm, and a false alarm here
        # trains you to ignore the check.
        first = next((k for k, t in enumerate(ids) if t == t_open), len(ids))
        carryover += sum(1 for k in range(first) if lbl[k] != IGNORE)

        # ...but the tool-response check MUST still cover that region, and on gemma that
        # is not a detail: one model turn holds an entire tool exchange, so a 32k window
        # routinely opens thousands of tokens deep inside one. Measured on the first
        # gemma corpus, 52% of all supervised tokens were carry-over — checking only from
        # `first` would have audited less than half of it. Scan from 0, inferring the
        # opening state: if the first tool-response marker seen is a CLOSING one, the
        # window began inside a response.
        nxt = next((t for t in ids if t in (tr_open, tr_close)), None)
        resp = nxt == tr_close
        for i in range(first):
            if ids[i] == tr_open:
                resp = True
            if resp and lbl[i] != IGNORE:
                in_response += 1
            if ids[i] == tr_close:
                resp = False

        role, turn_open, resp_open, span_open = None, False, resp, False
        for i in range(first, len(ids)):
            t = ids[i]
            if t == t_open:
                role = role_ids.get(ids[i + 1] if i + 1 < len(ids) else -1, "?")
                roles[role] += 1
                turn_open, resp_open = True, False
            elif t == tr_open:
                resp_open = True
            if lbl[i] != IGNORE:
                tok_by_role[role] += 1
                if role != "model":
                    leaked += 1
                if resp_open:
                    in_response += 1
                if t in (cid["<|tool_call>"], cid["<tool_call|>"]):
                    sup_tool_call += 1
                if not span_open:
                    sup_spans += 1
                    span_open = True
            else:
                span_open = False
            if t == tr_close:
                resp_open = False
            if t == t_close:
                if turn_open:
                    turn_open = False
                else:
                    unclosed += 1
                if lbl[i] != IGNORE:
                    sup_stop += 1

    print("\nturns by role:")
    for r, n in roles.most_common():
        print(f"  {str(r):<12} {n:>8,} turns   {tok_by_role.get(r,0):>12,} supervised tokens")
    print(f"\nsupervised spans                          {sup_spans:,}")
    print(f"spans closing on <turn|>                  {sup_stop:,} "
          f"({100*sup_stop/max(1,sup_spans):.1f}% of spans)")
    print(f"supervised tool-call markers              {sup_tool_call:,}"
          f"   (the model DOES emit these — expected > 0)")
    print(f"supervised tokens inside a TOOL RESPONSE  {in_response:,}"
          f"{'   <-- REAL LEAK, the mask is wrong' if in_response else '   (none, correct)'}")
    print(f"supervised tokens in NON-model turns      {leaked:,}"
          f"{'   <-- REAL LEAK' if leaked else '   (none, correct)'}")
    print(f"carry-over from the previous window       {carryover:,}"
          f"   (expected: packed windows start mid-turn)")
    print(f"stray <turn|> with no open turn           {unclosed:,}"
          f"   (up to one per window is the same carry-over)")
    return 1 if (in_response or leaked or sup_stop == 0) else 0


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
    ap.add_argument("--model", default=None,
                    help="tokenizer the corpus was built with (dir or HF repo id). "
                         "Default: the unpacked Bonsai model + prism chat template. The "
                         "audit dispatches on the family this resolves to.")
    a = ap.parse_args()

    tok = (load_tokenizer(a.model, None) if a.model else load_tokenizer())
    if a.audit:
        dialect = detect_dialect(tok)
        if dialect.name == "gemma4":
            return audit_gemma4(a.corpus, tok, a.max_windows)
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
