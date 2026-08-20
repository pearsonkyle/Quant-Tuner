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
    """[m] mean per-token logp of the teacher-forced repeated command.

    Trunk first, lm_head only at span positions — full [m, L, vocab] logits are
    13+ GiB fp32 at k=25 depth and OOM silently inside a piped run (same class as
    the repetition_losses fix)."""
    with torch.no_grad():
        hidden = model.model(input_ids=batch.ids.to(device),
                             attention_mask=batch.attn.to(device)).last_hidden_state
        out = []
        for i in range(batch.ids.shape[0]):
            lo, hi = int(batch.span[i, 0]), int(batch.span[i, 1])
            lg = model.lm_head(hidden[i, lo - 1:hi - 1]).float()
            tgt = batch.ids[i, lo:hi].to(device)
            lp = torch.log_softmax(lg, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            out.append(lp.mean())
    return torch.stack(out).cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="out/exp-057/model")
    ap.add_argument("--latents", default=None, help="trained_latents.pt (omit = vanilla only)")
    ap.add_argument("--k-max", type=int, default=5)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--bank", default=None,
                    help="real-material bank (build_rep_bank.py); default synthetic")
    ap.add_argument("--tag", default=None,
                    help="series name for --json-out (default: latents parent dir)")
    ap.add_argument("--json-out", default=None,
                    help="rep_measure.json to create/update — each measured model is "
                         "written into .series so the run report can plot the curves")
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

    bank = None
    if args.bank:
        import json
        bank = json.loads(Path(args.bank).read_text())
    batches = {k: RepBatch.build(tok, n=args.n, seed=args.seed, k=k, bank=bank)
               for k in range(1, args.k_max + 1)}

    def report(tag: str) -> None:
        curve = {}
        for k, b in batches.items():
            lp = span_logp(model, b, args.device)
            p = lp.exp()
            curve[str(k)] = round(float(p.mean()), 4)
            print(f"[rep] {tag:24s} k={k}  mean_p={p.mean():.4f}  "
                  f"max_p={p.max():.4f}  rows>0.5: {(p > 0.5).sum().item()}/{len(p)}  "
                  f"rows>0.2: {(p > 0.2).sum().item()}/{len(p)}", flush=True)
        if args.json_out:
            import json
            jp = Path(args.json_out)
            data = json.loads(jp.read_text()) if jp.exists() else {"series": {}}
            data.setdefault("series", {})[tag] = curve
            jp.write_text(json.dumps(data, indent=1))

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
        report(args.tag or Path(args.latents).parent.name)


if __name__ == "__main__":
    main()
