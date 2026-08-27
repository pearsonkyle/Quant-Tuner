#!/usr/bin/env python3
"""What does the corpus actually teach about WHERE to stop?

    python scripts/analyze_stop_context.py out/exp-058/corpus_*_32768.pt

The sft32k_sw1 ablation ruled out the loss weighting: --stop-weight 6.0 and 1.0 both
leave P(<|im_end|> | completed sentence) at ~0.95 against the shipped model's 0.009. A 6x
change in the weight moved the diagnostic by 0.02, so the stop decision is being taught by
the DATA, and this measures that directly instead of inferring it from a trained model.

The quantity that matters is conditional, not marginal. "One stop per 176 trainable
tokens" says stops are rare overall; it says nothing about how often a stop follows the
specific context the model is asked about at inference. If, in the corpus, a labeled
target that follows a sentence-ending period IS the stop token most of the time, then a
model reporting p=0.95 there has learned the corpus correctly — and the fix is the corpus,
not the objective.

Reported per source, because the corpora differ in exactly the way that should matter:
ultrachat has no tool calls at all (an assistant turn IS a block of prose, so ending after
a sentence is nearly always right), while the agentic corpora continue past a sentence
into a tool call. If the conditional rate tracks that split, it predicts which curriculum
rounds break termination before 33 h of training finds out.

CPU only, reads packed corpora — safe to run beside a training job.
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
# Sentence-final punctuation as it appears at the END of a decoded token. The probe's
# diagnostic prompt ends "...understand the bug." so the period is what we condition on;
# ! and ? are the same decision.
SENT_END = (".", "!", "?", '."', ".'", '.)', ".`")


def classify_context(piece: str) -> str | None:
    """What kind of position is this, from the PREVIOUS token alone."""
    s = piece.rstrip()
    if not s:
        return "whitespace"
    if s.endswith(SENT_END):
        return "sentence_end"
    if s.endswith((",", ";", ":")):
        return "clause_end"
    return None


def analyze(path: Path, tok, im_end: int, max_windows: int | None) -> dict:
    blob = torch.load(path, weights_only=False)
    ids, lbl = blob["ids"], blob["labels"]
    if max_windows:
        ids, lbl = ids[:max_windows], lbl[:max_windows]

    vocab_pieces: dict[int, str] = {}

    def piece(i: int) -> str:
        if i not in vocab_pieces:
            vocab_pieces[i] = tok.decode([i])
        return vocab_pieces[i]

    ctx_total: collections.Counter = collections.Counter()
    ctx_stop: collections.Counter = collections.Counter()
    n_trainable = 0
    n_stop = 0
    # What DOES follow a sentence end, when it is not a stop?
    after_sent: collections.Counter = collections.Counter()
    # THE PROBE'S ACTUAL SITUATION. The diagnostic prompt is one sentence into a fresh
    # assistant turn, not an arbitrary sentence end 3,000 tokens deep in a trajectory.
    # Pooling all sentence ends answers a different question than the probe asks, and the
    # two can differ by an order of magnitude if short turns usually end after one
    # sentence. Turn offset is measured from the last <|im_end|> in the raw stream, which
    # is where the previous turn closed.
    OFFSET_BUCKETS = ((0, 32), (32, 128), (128, 512), (512, 1 << 30))
    bucket_total: collections.Counter = collections.Counter()
    bucket_stop: collections.Counter = collections.Counter()

    for w in range(ids.shape[0]):
        row_ids = ids[w].tolist()
        row_lbl = lbl[w].tolist()
        last_end = 0
        for i in range(1, len(row_ids)):
            if row_ids[i - 1] == im_end:
                last_end = i
            t = row_lbl[i]
            if t == IGNORE:
                continue
            n_trainable += 1
            is_stop = t == im_end
            n_stop += is_stop
            kind = classify_context(piece(row_ids[i - 1]))
            if kind is None:
                continue
            ctx_total[kind] += 1
            ctx_stop[kind] += is_stop
            if kind == "sentence_end":
                off = i - last_end
                for lo, hi in OFFSET_BUCKETS:
                    if lo <= off < hi:
                        bucket_total[(lo, hi)] += 1
                        bucket_stop[(lo, hi)] += is_stop
                        break
                if not is_stop:
                    after_sent[piece(t)[:24]] += 1

    return {
        "path": path.name,
        "windows": int(ids.shape[0]),
        "trainable": n_trainable,
        "stops": n_stop,
        "marginal": n_stop / max(1, n_trainable),
        "ctx_total": dict(ctx_total),
        "ctx_stop": dict(ctx_stop),
        "after_sentence_top": after_sent.most_common(8),
        "bucket_total": {f"{lo}-{hi}": c for (lo, hi), c in bucket_total.items()},
        "bucket_stop": {f"{lo}-{hi}": c for (lo, hi), c in bucket_stop.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpora", nargs="+", type=Path)
    ap.add_argument("--max-windows", type=int, default=None,
                    help="sample the first N windows (default: all)")
    a = ap.parse_args()

    tok = load_tokenizer()
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    print(f"stop token <|im_end|> = {im_end}\n")

    reports = [analyze(p, tok, im_end, a.max_windows) for p in a.corpora]

    print(f"{'corpus':<34}{'trainable':>12}{'stops':>10}{'marginal':>11}"
          f"{'P(stop|sentence end)':>22}{'lift':>8}")
    print("-" * 97)
    for r in reports:
        n = r["ctx_total"].get("sentence_end", 0)
        s = r["ctx_stop"].get("sentence_end", 0)
        cond = s / max(1, n)
        lift = cond / max(1e-12, r["marginal"])
        print(f"{r['path']:<34}{r['trainable']:>12,}{r['stops']:>10,}"
              f"{r['marginal']:>11.4f}{cond:>16.4f} ({n:,}){lift:>7.1f}x")

    print("\nP(stop | sentence end) BY POSITION IN THE ASSISTANT TURN")
    print("  The probe asks about the first bucket: one sentence into a fresh turn.")
    order = ["0-32", "32-128", "128-512", "512-1073741824"]
    names = {"0-32": "<32 tok in", "32-128": "32-128", "128-512": "128-512",
             "512-1073741824": ">512"}
    print(f"  {'corpus':<32}" + "".join(f"{names[k]:>18}" for k in order))
    for r in reports:
        cells = []
        for k in order:
            n = r["bucket_total"].get(k, 0)
            s_ = r["bucket_stop"].get(k, 0)
            cells.append(f"{s_/n:.4f} ({n//1000}k)" if n else "—")
        print(f"  {r['path']:<32}" + "".join(f"{c:>18}" for c in cells))

    print("\nPer-context conditional stop rate:")
    for r in reports:
        print(f"\n  {r['path']}")
        for kind in ("sentence_end", "clause_end", "whitespace"):
            n = r["ctx_total"].get(kind, 0)
            if not n:
                continue
            s = r["ctx_stop"].get(kind, 0)
            print(f"    {kind:<14} {s:>9,}/{n:<11,} = {s/n:.4f}")
        print("    when a sentence end is NOT followed by a stop, what follows:")
        tot = sum(c for _, c in r["after_sentence_top"]) or 1
        for pc, c in r["after_sentence_top"]:
            print(f"      {pc!r:<26} {c:>8,} ({100*c/tot:.0f}% of non-stop)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
