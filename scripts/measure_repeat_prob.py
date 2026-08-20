"""P(verbatim command repeat) as a function of k identical rounds in context.

The anchor7 mimic looped 59x while its in-training rp= telemetry read 0.0000 for the
whole run: RepBatch's k=1 contexts (first repeat opportunity) never put the model above
the 0.5 hinge cap, so no gradient ever flowed. This measures where the repeat
probability actually lives — per k, vanilla vs trained latents — so the anchor8 hinge
(cap, k schedule) is set from data instead of guessed.

    PYTHONPATH=src .venv/bin/python scripts/measure_repeat_prob.py \
        --latents out/exp-058/kd32b-full-anchor7/trained_latents.pt --k-max 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_tuner.qat.steer import RepBatch  # noqa: E402


def span_logp(model, batch: RepBatch, device: str) -> torch.Tensor:
    """[m] mean per-token logp of the teacher-forced repeated command."""
    with torch.no_grad():
        logits = model(input_ids=batch.ids.to(device),
                       attention_mask=batch.attn.to(device)).logits.float()
    logp = torch.log_softmax(logits, dim=-1)
    out = []
    for i in range(batch.ids.shape[0]):
        lo, hi = int(batch.span[i, 0]), int(batch.span[i, 1])
        tgt = batch.ids[i, lo:hi].to(device)
        lp = logp[i, lo - 1:hi - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        out.append(lp.mean())
    return torch.stack(out).cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="out/exp-057/model")
    ap.add_argument("--latents", default=None, help="trained_latents.pt (omit = vanilla only)")
    ap.add_argument("--k-max", type=int, default=5)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from quant_tuner.qat.attention import enable_fp32_gqa_repeat
    from quant_tuner.qat.train import wrap_model

    if args.device == "cuda":
        enable_fp32_gqa_repeat()
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.float32).to(args.device)
    model.eval()
    wrap_model(model, 36)          # ternarized forward = what the export serves

    batches = {k: RepBatch.build(tok, n=args.n, seed=args.seed, k=k)
               for k in range(1, args.k_max + 1)}

    def report(tag: str) -> None:
        for k, b in batches.items():
            lp = span_logp(model, b, args.device)
            p = lp.exp()
            print(f"[rep] {tag:24s} k={k}  mean_p={p.mean():.4f}  "
                  f"max_p={p.max():.4f}  rows>0.5: {(p > 0.5).sum().item()}/{len(p)}  "
                  f"rows>0.2: {(p > 0.2).sum().item()}/{len(p)}", flush=True)

    report("vanilla")
    if args.latents:
        sd = torch.load(args.latents, map_location="cpu", weights_only=False, mmap=True)
        latents = sd["latents"]
        n = 0
        with torch.no_grad():
            for name_, mod in model.named_modules():
                key = f"{name_}.linear.weight"
                src = latents.get(key, latents.get(key.replace(".linear.weight", ".weight")))
                if src is not None and hasattr(mod, "linear"):
                    mod.linear.weight.copy_(src.to(args.device))
                    n += 1
        print(f"[rep] loaded {n} latent tensors from {args.latents}", flush=True)
        report(Path(args.latents).parent.name)


if __name__ == "__main__":
    main()
