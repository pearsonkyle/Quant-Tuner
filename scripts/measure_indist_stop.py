#!/usr/bin/env python
"""P(stop) at REAL stop positions in held-out windows — the middle ground between the
synthetic probe (one fixed minimal context) and the agent trajectory (end-to-end, slow).

The anchor runs hold corpus stop positions (an ~ 0.01) while the PROBE control
collapses; this measures whether the model still stops in realistic contexts, i.e.
whether the probe's minimal 2-turn context is the outlier or the canary.

    python scripts/measure_indist_stop.py --corpus VAL.pt --latents CKPT.pt [--windows 8]

Reports, per model (vanilla + each checkpoint): mean/median P(stop) over all val
positions whose TARGET is the stop token, split by whether a tool-call block closed
within the previous 64 tokens (after-tool-call vs prose-end stops).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from quant_tuner.qat._device import resolve_backend  # noqa: E402
from quant_tuner.qat.attention import enable_fp32_gqa_repeat  # noqa: E402
from quant_tuner.qat.train import wrap_model  # noqa: E402

STOP_ID = 151645
TOOL_CLOSE = "</tool_call>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True, help="val corpus .pt")
    ap.add_argument("--model-dir", type=Path,
                    default=Path("out/exp-057/model"))
    ap.add_argument("--latents", type=Path, action="append", default=[],
                    help="trained_latents.pt checkpoint(s); vanilla always measured")
    ap.add_argument("--windows", type=int, default=8)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    backend = resolve_backend("auto")
    dev = backend.name
    if dev == "cuda":
        # fp32 + GQA on CUDA dispatches to the math kernel and materializes
        # [heads, S, S] — 128 GiB at a 32768 window. Same patch train.py applies.
        enable_fp32_gqa_repeat()
    tok = AutoTokenizer.from_pretrained(str(args.model_dir))
    close_ids = tok(TOOL_CLOSE, add_special_tokens=False).input_ids

    blob = torch.load(args.corpus, weights_only=False)
    ids_all, lbl_all = blob["ids"], blob["labels"]
    n = min(args.windows, ids_all.shape[0])

    model = AutoModelForCausalLM.from_pretrained(str(args.model_dir),
                                                 dtype=torch.float32).to(dev)
    model.eval()
    wrap_model(model, 36)          # ternarized forward = what the export serves
    for p in model.parameters():
        p.requires_grad_(False)

    def measure(tag: str) -> None:
        after_tool, prose = [], []
        for w in range(n):
            ids = ids_all[w:w + 1].to(dev)
            tgt = lbl_all[w:w + 1][:, 1:][0]
            keep = (tgt == STOP_ID).nonzero(as_tuple=True)[0]
            if keep.numel() == 0:
                continue
            with torch.no_grad():
                h = model.model(input_ids=ids).last_hidden_state[0, keep, :]
                lg = model.lm_head(h).float()
                p_stop = torch.softmax(lg, dim=-1)[:, STOP_ID]
            id_list = ids[0].tolist()
            for j, pos in enumerate(keep.tolist()):
                ctx = id_list[max(0, pos - 64):pos + 1]
                is_tool = any(ctx[i:i + len(close_ids)] == close_ids
                              for i in range(len(ctx) - len(close_ids) + 1))
                (after_tool if is_tool else prose).append(float(p_stop[j]))
        for name, vals in (("after_tool_call", after_tool), ("prose_end", prose)):
            if vals:
                t = torch.tensor(vals)
                print(f"[indist] {tag:>12} {name:>15}: n={len(vals):4d} "
                      f"mean={t.mean():.4f} median={t.median():.4f} "
                      f"p10={t.quantile(0.1):.4f}", flush=True)

    measure("vanilla")
    for ck in args.latents:
        sd = torch.load(ck, map_location="cpu", weights_only=False, mmap=True)
        latents = sd["latents"]
        loaded = 0
        for name_, mod in model.named_modules():
            key = f"{name_}.linear.weight"
            alt = key.replace(".linear.weight", ".weight")
            src = latents.get(key, latents.get(alt))
            if src is not None and hasattr(mod, "linear"):
                mod.linear.weight.copy_(src.to(dev))
                loaded += 1
        print(f"[indist] loaded {loaded} latent tensors from {ck}", flush=True)
        measure(Path(ck).parent.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
