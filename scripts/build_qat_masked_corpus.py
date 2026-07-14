"""iter-2 QAT corpus: turn-aware, assistant-masked, packed to a big window.

Baseline (exp-057 v1) trained on flattened text with UNIFORM loss over 1024-token
windows — tool-call tokens (the thing we want to improve) were weighted the same as
system/user/wiki boilerplate. This builds the fix:

  * Uses the logtrain TRAIN slice only (seed-42 split; disjoint from the test/holdout
    slices that feed the PPL and agentic tool-call evals — no eval contamination).
  * Renders each session with the model's real chat template, tokenizes with
    offset mapping, and MASKS loss to assistant-generated tokens only (the
    `<|im_start|>assistant … <|im_end|>` spans, which include tool_calls). Non-
    assistant tokens get label -100.
  * Packs the per-session (ids,label) stream into WINDOW-token windows (default 8192,
    ~P99 of a single turn is 5.7k, so a window holds a full tool turn + context).
  * Optionally mixes wiki windows with FULL loss for anti-forgetting.
  * Shuffles all windows (seed 42) and saves tokenized tensors -> one .pt.

    PYTHONPATH=src .venv/bin/python scripts/build_qat_masked_corpus.py \
        --window 8192 --wiki-tokens 300000 --out out/exp-058/masked_corpus.pt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch

from quant_tuner.data import split

MODEL = REPO / "out" / "exp-057" / "model"
CHAT_TEMPLATE = REPO / "out" / "exp-057" / "chat_template.jinja"
LOGTRAIN = REPO / "logtrain.jsonl"
WIKI = REPO / "out" / "exp-001" / "wiki" / "wiki.test.raw"

# assistant span in Qwen render: from "<|im_start|>assistant\n" to the next "<|im_end|>"
_ASST_RE = re.compile(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", re.DOTALL)


def parse_session(s: dict) -> list[dict]:
    return [json.loads(m) if isinstance(m, str) else m for m in s["messages"]]


def masked_ids_for_session(msgs: list[dict], tok) -> tuple[list[int], list[int]]:
    """Return (ids, labels) with labels = ids on assistant tokens, -100 elsewhere."""
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    # char spans of assistant *content* (group 1 = between the header and <|im_end|>)
    spans = [(m.start(1), m.end(1)) for m in _ASST_RE.finditer(text)]
    labels = [-100] * len(ids)
    si = 0
    for j, (a, b) in enumerate(offs):
        if a == b:  # special/zero-width
            continue
        while si < len(spans) and spans[si][1] <= a:
            si += 1
        if si < len(spans) and a >= spans[si][0] and b <= spans[si][1]:
            labels[j] = ids[j]
    return ids, labels


def pack(stream_ids: list[int], stream_lbl: list[int], window: int) -> list[dict]:
    n = (len(stream_ids) // window) * window
    out = []
    for i in range(0, n, window):
        out.append({"ids": stream_ids[i:i+window], "labels": stream_lbl[i:i+window]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=8192)
    ap.add_argument("--wiki-tokens", type=int, default=300_000)
    ap.add_argument("--out", type=Path, default=REPO / "out" / "exp-058" / "masked_corpus.pt")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.chat_template = CHAT_TEMPLATE.read_text()

    sessions = [json.loads(l) for l in LOGTRAIN.read_text().splitlines()]
    splits = split.split_sessions(sessions, seed=42)
    train = splits["train"]
    print(f"[build] logtrain: {len(sessions)} sessions -> train slice {len(train)} "
          f"(test={len(splits['test'])} holdout={len(splits['holdout'])} excluded)", flush=True)

    # --- masked tool windows (concat all train sessions, then chunk) ---------
    ids_stream: list[int] = []
    lbl_stream: list[int] = []
    tot_asst = 0
    for k, s in enumerate(train):
        ids, lbl = masked_ids_for_session(parse_session(s), tok)
        ids_stream += ids
        lbl_stream += lbl
        tot_asst += sum(1 for x in lbl if x != -100)
        if (k + 1) % 25 == 0:
            print(f"[build]   {k+1}/{len(train)} sessions, {len(ids_stream):,} tokens", flush=True)
    tool_windows = pack(ids_stream, lbl_stream, args.window)
    frac = 100 * tot_asst / max(1, len(ids_stream))
    print(f"[build] tool: {len(ids_stream):,} tokens ({frac:.0f}% assistant-masked) "
          f"-> {len(tool_windows)} windows of {args.window}", flush=True)

    # --- wiki windows (FULL loss, anti-forgetting) ---------------------------
    wiki_windows: list[dict] = []
    if args.wiki_tokens > 0 and WIKI.exists():
        wids = tok(WIKI.read_text(), add_special_tokens=False)["input_ids"][:args.wiki_tokens]
        wiki_windows = pack(wids, list(wids), args.window)  # labels = ids (all count)
        print(f"[build] wiki: {len(wids):,} tokens -> {len(wiki_windows)} full-loss windows", flush=True)

    windows = tool_windows + wiki_windows
    # deterministic shuffle (seed 42) without Random(): index by a fixed permutation
    import random
    rng = random.Random(42)
    rng.shuffle(windows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ids_t = torch.tensor([w["ids"] for w in windows], dtype=torch.long)
    lbl_t = torch.tensor([w["labels"] for w in windows], dtype=torch.long)
    torch.save({"ids": ids_t, "labels": lbl_t, "window": args.window,
                "tool_windows": len(tool_windows), "wiki_windows": len(wiki_windows),
                "assistant_frac": frac / 100}, args.out)
    print(f"[build] saved {ids_t.shape[0]} windows [{ids_t.shape}] -> {args.out}")
    print(f"[build] masked-token share overall: "
          f"{100*(lbl_t != -100).float().mean():.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
