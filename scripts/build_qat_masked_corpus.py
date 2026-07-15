"""iter-2 QAT corpus: turn-aware, assistant-masked, packed to a big window.

Baseline (exp-057 v1) trained on flattened text with UNIFORM loss over 1024-token
windows — tool-call tokens (the thing we want to improve) were weighted the same as
system/user/wiki boilerplate. This builds the fix:

  * Uses the logtrain TRAIN slice only (seed-42 split; disjoint from the test/holdout
    slices that feed the PPL and agentic tool-call evals — no eval contamination).
    ``--split test`` builds the same corpus from the TEST slice instead — feed that
    to the trainer's ``--val-corpus`` as the held-out masked-CE validation set.
  * Renders each session with the model's real chat template, tokenizes with
    offset mapping, and MASKS loss to assistant-generated tokens (the
    `<|im_start|>assistant … <|im_end|>` spans, which include tool_calls) — PLUS
    the terminating `<|im_end|>` itself. Without the terminator the model gets
    zero gradient toward *ending its turn* (no position in the corpus had
    `<|im_end|>` as a CE target — the mechanistic cause of the looping pathology
    seen in iter-2/iter-3). Non-assistant tokens get label -100.
  * Renders the session's REAL tool schemas (``messages[0]["tools"]`` in the
    logtrain export, resolved via ``data.split.session_tools``); only sessions
    without stored schemas fall back to reconstructing name→arg-key stubs.
  * Optionally truncates giant tool outputs (``--max-tool-tokens``, head+tail)
    to raise trainable-token density — throughput is token-bound, so masked
    tool-output tokens are the main wall-time waste.
  * Packs the per-session (ids,label) stream into WINDOW-token windows (default
    4096 — the MPS hard max; 8192 hits the MPSGraph INT_MAX attention limit).
  * Optionally mixes wiki windows with FULL loss for anti-forgetting.
  * Shuffles all windows (seed 42) and saves tokenized tensors -> one .pt with a
    content fingerprint (the trainer's --resume refuses a mismatched corpus).

    PYTHONPATH=src .venv/bin/python scripts/build_qat_masked_corpus.py \
        --window 4096 --wiki-tokens 300000 --max-tool-tokens 1024 \
        --out out/exp-058/masked_corpus_4096_v2.pt
"""

from __future__ import annotations

import argparse
import hashlib
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
IM_END = "<|im_end|>"


def parse_session(s: dict) -> list[dict]:
    return [json.loads(m) if isinstance(m, str) else m for m in s["messages"]]


def reconstruct_tools(msgs: list[dict]) -> list[dict]:
    """FALLBACK ONLY: build per-session OpenAI-function schemas from the tools the
    session actually calls (name -> observed argument keys). The logtrain export
    stores the REAL schemas on the system message (``messages[0]["tools"]``) —
    resolved via ``split.session_tools`` — so this stub path should only fire for
    log formats that genuinely carry no schema block."""
    import collections
    argk: dict[str, set] = collections.defaultdict(set)
    for m in msgs:
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function", tc)
            name = fn.get("name") or tc.get("name")
            if not name:
                continue
            a = fn.get("arguments") or tc.get("arguments")
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except Exception:
                    a = {}
            if isinstance(a, dict):
                argk[name] |= {k for k in a if isinstance(k, str) and k.isidentifier()}
    return [{"type": "function", "function": {
        "name": n, "description": f"{n} tool",
        "parameters": {"type": "object",
                       "properties": {k: {"type": "string"} for k in sorted(ks)}}}}
            for n, ks in argk.items()]


def truncate_tool_messages(
    msgs: list[dict], tok, max_tool_tokens: int
) -> tuple[list[dict], int, int]:
    """Head+tail truncate ``role=="tool"`` string contents to ``max_tool_tokens``.

    Tool outputs carry their signal at both ends (command echo / final result +
    exit status); the middle of a 10k-token dump is masked context the model
    pays full forward/backward for. Returns (msgs, n_truncated, tokens_saved).
    Truncation happens BEFORE the chat-template render so spans/offsets stay
    consistent — they are computed on the final rendered text.
    """
    if max_tool_tokens <= 0:
        return msgs, 0, 0
    out: list[dict] = []
    n_trunc = 0
    saved = 0
    for m in msgs:
        c = m.get("content")
        if m.get("role") == "tool" and isinstance(c, str):
            ids = tok(c, add_special_tokens=False)["input_ids"]
            if len(ids) > max_tool_tokens:
                head = tok.decode(ids[: max_tool_tokens // 2])
                tail = tok.decode(ids[-(max_tool_tokens - max_tool_tokens // 2):])
                dropped = len(ids) - max_tool_tokens
                m = {**m, "content": f"{head}\n…[truncated {dropped} tokens]…\n{tail}"}
                n_trunc += 1
                saved += dropped
        out.append(m)
    return out, n_trunc, saved


def masked_ids_for_session(
    msgs: list[dict], tok, tools: list | None = None
) -> tuple[list[int], list[int]]:
    """Return (ids, labels) with labels = ids on assistant tokens, -100 elsewhere.

    The labeled span runs from the first content character after the
    ``<|im_start|>assistant\\n`` header through the END of the terminating
    ``<|im_end|>`` (``m.end(0)``): the terminator is a trained target — it is the
    stop/EOS decision — while the header stays context (at inference it is part
    of the generation prompt, never generated). The token after ``<|im_end|>``
    (the turn-separating newline) stays -100.

    Tool schemas render in a system turn (# Tools block), so the assistant-span
    mask naturally excludes them — schemas are context, only the assistant's
    tool_calls/text (+ terminator) are trained."""
    if tools is None:
        tools = reconstruct_tools(msgs)
    text = tok.apply_chat_template(msgs, tools=tools or None, tokenize=False,
                                   add_generation_prompt=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    # char spans: assistant *content* plus the terminating <|im_end|> (m.end(0))
    spans = [(m.start(1), m.end(0)) for m in _ASST_RE.finditer(text)]
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


def pack(stream_ids: list[int], stream_lbl: list[int], window: int,
         min_trainable: int = 8, min_density: float = 0.0) -> list[dict]:
    """Chunk the stream into windows, DROPPING windows with < ``min_trainable``
    non-(-100) label tokens or below ``min_density`` trainable fraction. An
    all-masked window (a 4096 chunk landing entirely in a long tool output —
    18.7% of them at window 4096 pre-truncation) has no gradient and yields a
    NaN CE loss (0/0); near-empty windows waste a full ~40s forward/backward on
    a few tokens of signal. ``min_density`` is the tunable wall-time lever —
    pick it from the density histogram this builder prints."""
    n = (len(stream_ids) // window) * window
    out = []
    for i in range(0, n, window):
        lbl = stream_lbl[i:i + window]
        n_train = sum(1 for x in lbl if x != -100)
        if n_train >= min_trainable and n_train / window >= min_density:
            out.append({"ids": stream_ids[i:i + window], "labels": lbl})
    return out


def corpus_fingerprint(ids_t: torch.Tensor, lbl_t: torch.Tensor) -> str:
    h = hashlib.sha256()
    h.update(str(tuple(ids_t.shape)).encode())
    h.update(ids_t.numpy().tobytes())
    h.update(lbl_t.numpy().tobytes())
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=4096,
                    help="tokens per window; 4096 is the MPS hard max (8192 hits the "
                         "MPSGraph INT_MAX attention limit)")
    ap.add_argument("--wiki-tokens", type=int, default=300_000)
    ap.add_argument("--split", choices=["train", "test"], default="train",
                    help="logtrain slice to build from; 'test' builds the masked-CE "
                         "validation corpus for the trainer's --val-corpus")
    ap.add_argument("--max-tool-tokens", type=int, default=0,
                    help="head+tail truncate role=tool contents to N tokens "
                         "(0 = off); raises trainable density / cuts wall-time")
    ap.add_argument("--min-density", type=float, default=0.0,
                    help="drop windows whose trainable-token fraction is below this "
                         "(on top of the >=8-token floor)")
    ap.add_argument("--out", type=Path, default=REPO / "out" / "exp-058" / "masked_corpus.pt")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.chat_template = CHAT_TEMPLATE.read_text()

    sessions = [json.loads(line) for line in LOGTRAIN.read_text().splitlines()]
    splits = split.split_sessions(sessions, seed=42)
    picked = splits[args.split]
    print(f"[build] logtrain: {len(sessions)} sessions -> {args.split} slice {len(picked)} "
          f"(train={len(splits['train'])} test={len(splits['test'])} "
          f"holdout={len(splits['holdout'])})", flush=True)

    # --- masked tool windows (concat all sessions of the slice, then chunk) ---
    ids_stream: list[int] = []
    lbl_stream: list[int] = []
    tot_asst = 0
    n_real_schema = 0
    n_stub_schema = 0
    n_trunc_msgs = 0
    trunc_saved = 0
    for k, s in enumerate(picked):
        msgs = parse_session(s)
        msgs, n_tr, saved = truncate_tool_messages(msgs, tok, args.max_tool_tokens)
        n_trunc_msgs += n_tr
        trunc_saved += saved
        tools = split.session_tools(s, msgs)
        if tools:
            n_real_schema += 1
        else:
            tools = reconstruct_tools(msgs)
            n_stub_schema += 1
        ids, lbl = masked_ids_for_session(msgs, tok, tools=tools)
        ids_stream += ids
        lbl_stream += lbl
        tot_asst += sum(1 for x in lbl if x != -100)
        if (k + 1) % 25 == 0:
            print(f"[build]   {k+1}/{len(picked)} sessions, {len(ids_stream):,} tokens", flush=True)
    tool_windows = pack(ids_stream, lbl_stream, args.window,
                        min_density=args.min_density)
    frac = 100 * tot_asst / max(1, len(ids_stream))
    print(f"[build] tool: {len(ids_stream):,} tokens ({frac:.0f}% assistant-masked) "
          f"-> {len(tool_windows)} windows of {args.window} "
          f"(min_density={args.min_density})", flush=True)
    print(f"[build] schemas: {n_real_schema} sessions with REAL stored schemas, "
          f"{n_stub_schema} fell back to reconstructed stubs", flush=True)
    if args.max_tool_tokens > 0:
        print(f"[build] tool-output truncation: {n_trunc_msgs} messages, "
              f"{trunc_saved:,} masked tokens dropped", flush=True)

    # --- wiki windows (FULL loss, anti-forgetting) ---------------------------
    wiki_windows: list[dict] = []
    if args.wiki_tokens > 0 and WIKI.exists():
        wids = tok(WIKI.read_text(), add_special_tokens=False)["input_ids"][:args.wiki_tokens]
        wiki_windows = pack(wids, list(wids), args.window)  # labels = ids (all count)
        print(f"[build] wiki: {len(wids):,} tokens -> {len(wiki_windows)} full-loss windows", flush=True)

    windows = tool_windows + wiki_windows
    if not windows:
        sys.exit("[build] no windows survived packing — check --min-density / the slice")
    # deterministic shuffle (seed 42) without Random(): index by a fixed permutation
    import random
    rng = random.Random(42)
    rng.shuffle(windows)

    ids_t = torch.tensor([w["ids"] for w in windows], dtype=torch.long)
    lbl_t = torch.tensor([w["labels"] for w in windows], dtype=torch.long)

    # --- audit: density histogram + the stop-token regression tripwire -------
    dens = ((lbl_t != -100).float().mean(dim=1))
    deciles = torch.quantile(dens, torch.linspace(0, 1, 11))
    print("[build] window trainable-density deciles: "
          + " ".join(f"{d:.2f}" for d in deciles.tolist()), flush=True)
    im_end_id = tok.convert_tokens_to_ids(IM_END)
    n_im_end_targets = (0 if im_end_id is None
                        else int(((lbl_t == im_end_id) & (lbl_t != -100)).sum()))
    print(f"[build] labeled {IM_END} targets: {n_im_end_targets}", flush=True)
    if tool_windows and im_end_id is not None:
        assert n_im_end_targets > 0, (
            f"no labeled {IM_END} targets — the stop-token masking bug is back "
            "(spans must extend through m.end(0))")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fp = corpus_fingerprint(ids_t, lbl_t)
    torch.save({"ids": ids_t, "labels": lbl_t, "window": args.window,
                "tool_windows": len(tool_windows), "wiki_windows": len(wiki_windows),
                "assistant_frac": frac / 100, "split": args.split,
                "min_density": args.min_density, "max_tool_tokens": args.max_tool_tokens,
                "im_end_targets": n_im_end_targets, "fingerprint": fp}, args.out)
    print(f"[build] saved {ids_t.shape[0]} windows [{ids_t.shape}] fingerprint={fp} -> {args.out}")
    print(f"[build] masked-token share overall: "
          f"{100 * (lbl_t != -100).float().mean():.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
