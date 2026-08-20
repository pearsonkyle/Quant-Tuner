"""How much of the supervised corpus is verbatim command repetition?

Root-cause candidate for the loop escalation (anchor7 mimic, 59x): agent trajectories
contain legitimately repeated commands, and masked CE rewards "copy an earlier action"
as an easy win — a shortcut a capacity-starved ternary student may over-adopt.
This measures it: the share of supervised (labeled) tokens inside <tool_call> spans
whose command text verbatim-matches an EARLIER tool call in the same window,
per source. If the share is large, a per-token CE down-weight on copy spans (the
--stop-weight move, inverted) attacks the cause; if small, the corpus is exonerated
and the escalation is a pure training dynamic.

    PYTHONPATH=src .venv/bin/python scripts/measure_copy_spans.py \
        out/exp-058/fixed/corpus_ourssft_32768.pt
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--model-dir", default="out/exp-057/model")
    ap.add_argument("--windows", type=int, default=None, help="limit (default: all)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    open_id = tok.convert_tokens_to_ids("<tool_call>")
    close_id = tok.convert_tokens_to_ids("</tool_call>")
    assert open_id is not None and open_id >= 0, "<tool_call> not a single token"

    blob = torch.load(args.corpus, weights_only=False, mmap=True)
    ids_all, lbl_all = blob["ids"], blob["labels"]
    src_names = blob.get("source_names")
    win_src = blob.get("window_source")
    n_win = ids_all.shape[0] if args.windows is None else min(args.windows,
                                                              ids_all.shape[0])
    tot_sup = defaultdict(int)      # supervised tokens per source
    tot_call = defaultdict(int)     # supervised tokens inside any tool-call span
    tot_copy = defaultdict(int)     # ... inside a span verbatim-matching an earlier call
    n_calls = defaultdict(int)
    n_copies = defaultdict(int)

    for w in range(n_win):
        ids = ids_all[w]
        lbl = lbl_all[w]
        src = (src_names[int(win_src[w])]
               if src_names is not None and win_src is not None else "all")
        tot_sup[src] += int((lbl != -100).sum())
        seen: set[str] = set()
        pos = (ids == open_id).nonzero(as_tuple=True)[0].tolist()
        closes = (ids == close_id).nonzero(as_tuple=True)[0].tolist()
        for p in pos:
            q = next((c for c in closes if c > p), None)
            if q is None:
                continue
            span_ids = ids[p:q + 1]
            sup = int((lbl[p:q + 1] != -100).sum())
            if sup == 0:
                continue                      # a call in unsupervised context (history)
            text = tok.decode(span_ids, skip_special_tokens=False)
            n_calls[src] += 1
            tot_call[src] += sup
            if text in seen:
                n_copies[src] += 1
                tot_copy[src] += sup
            seen.add(text)
        if (w + 1) % 100 == 0:
            print(f"[copy] {w + 1}/{n_win} windows", flush=True)

    def row(src: str) -> str:
        s_, c_, k_ = tot_sup[src], tot_call[src], tot_copy[src]
        return (f"[copy] {src:28s} supervised={s_:>9,}  in-call={c_:>9,} "
                f"({100 * c_ / max(1, s_):5.1f}%)  verbatim-copy={k_:>8,} "
                f"({100 * k_ / max(1, s_):5.2f}% of sup, "
                f"{100 * k_ / max(1, c_):5.1f}% of calls)  "
                f"calls={n_calls[src]:,} copies={n_copies[src]:,}")

    for src in sorted(tot_sup):
        print(row(src), flush=True)
    for d in (tot_sup, tot_call, tot_copy, n_calls, n_copies):
        d["TOTAL"] = sum(v for k, v in d.items() if k != "TOTAL")
    print(row("TOTAL"), flush=True)


if __name__ == "__main__":
    main()
