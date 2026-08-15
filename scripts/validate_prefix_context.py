#!/usr/bin/env python
"""Prove prefix-context is EXACT on the real model before spending 30 h training with it.

The claim `--trained-tail` rests on: splitting a window into (no_grad prefix, gradient
tail) must not change what the tail's loss is. If it does, every long-window result is
measuring a different objective than the short-window results it will be compared with.

Three ways that claim fails silently — the loss still falls in all of them, which is why
this script exists rather than a glance at the training curve:

  * the prefix is dropped entirely (transformers nulls `past_key_values` under gradient
    checkpointing) → the tail trains on no context;
  * the causal mask misaligns (torch's `is_causal` anchors top-LEFT when kv_len > q_len)
    → the tail sees only the prefix's head;
  * `position_ids` restart at 0 → RoPE puts the tail on top of the prefix.

Each shows up here as a loss mismatch against the full-window reference. The unit tests
pin the same properties on a 2-layer toy; this runs the 36-layer model with the real
tokenizer and the real corpus, which is what a run actually uses.

    PYTHONPATH=src python scripts/validate_prefix_context.py \\
        --corpus out/exp-058/sft_corpus_universal_8064.pt --tail 4096 --windows 4

Exit code 1 on any mismatch. The window must be small enough that the FULL-gradient
reference still fits — that is the point of validating at 8064 and trusting 32768.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from quant_tuner.qat.attention import enable_chunked_sdpa  # noqa: E402
from quant_tuner.qat.train import MODEL, masked_forward, prefix_window  # noqa: E402

#: fp32 CE over ~thousands of positions through 36 layers; exact equality is not the bar,
#: but anything above this is a real difference, not accumulation order.
TOL = 2e-4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, default=MODEL)
    ap.add_argument("--tail", type=int, default=4096)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tol", type=float, default=TOL)
    args = ap.parse_args()

    enable_chunked_sdpa()
    blob = torch.load(args.corpus, weights_only=False)
    ids_all, lbl_all = blob["ids"], blob["labels"]
    window = int(ids_all.shape[1])
    n_prefix = window - args.tail
    if n_prefix <= 0:
        sys.exit(f"[validate] tail {args.tail} >= window {window}: nothing to validate")

    from transformers import AutoModelForCausalLM
    print(f"[validate] loading {args.model_dir}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.float32)
    model = model.to(args.device).eval()
    model.config.use_cache = False

    print(f"[validate] window {window} = {n_prefix} prefix + {args.tail} tail; "
          f"tol {args.tol}", flush=True)
    worst, checked, failures = 0.0, 0, 0
    for i in range(min(args.windows, ids_all.shape[0])):
        ids = ids_all[i:i + 1].to(args.device)
        lbl = lbl_all[i:i + 1].to(args.device)
        # The reference scores exactly the targets a prefix split keeps, with the WHOLE
        # window in the graph — same target set, full context, no split.
        tail_only = lbl.clone()
        tail_only[0, :n_prefix + 1] = -100
        if not bool((tail_only[0, 1:] != -100).any()):
            print(f"  window {i}: no target in the tail — skipped", flush=True)
            continue
        with torch.no_grad():
            ref, _, ref_idx = masked_forward(model, ids, tail_only, need_logits=False)
            with prefix_window(model, ids, n_prefix):
                got, _, got_idx = masked_forward(model, ids, lbl, need_logits=False,
                                                 n_prefix=n_prefix)
        same_targets = torch.equal(ref_idx.cpu(), got_idx.cpu())
        delta = abs(float(got) - float(ref))
        worst = max(worst, delta)
        checked += 1
        ok = same_targets and delta <= args.tol
        failures += not ok
        print(f"  window {i}: full {float(ref):.6f}  prefix {float(got):.6f}  "
              f"delta {delta:.2e}  targets {int(got_idx.numel())}"
              f"{'' if same_targets else ' TARGET-SET MISMATCH'}  {'ok' if ok else 'FAIL'}",
              flush=True)
        if args.device == "mps":
            torch.mps.empty_cache()

    if not checked:
        sys.exit("[validate] no window had a target in the tail — nothing was validated")
    print(f"\n[validate] {checked - failures}/{checked} windows match "
          f"(worst delta {worst:.2e}, tol {args.tol})")
    if failures:
        print("[validate] FAILED — do not start a long run; the tail is not seeing the "
              "prefix the way a full-window forward would.")
        return 1
    print("[validate] prefix-context is exact on this model. Safe to train long windows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
