"""Rank decoder layers by their gradient signal on the masked tool-call objective.

Instead of blindly training the "last 18" layers, measure which layers actually
carry the learning signal for OUR task: run masked forward+backward over a sample
of the corpus and accumulate, per decoder layer, the L2 norm of the gradient w.r.t.
its ternary latents. High grad-norm = the loss wants to move that layer most =
training it has the most leverage. (STE makes grad-w.r.t.-latent == grad-w.r.t.-the
ternarized weight, so this is the true per-layer update pressure.)

No optimizer states, so full-36 fits (~60 GiB): model 32 + grads 28.

    PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python \
        scripts/exp058_layer_importance.py --corpus out/exp-058/masked_corpus_4096.pt --windows 24
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch

from quant_tuner.qat.ternary import TernaryLinear

MODEL = REPO / "out" / "exp-057" / "model"


def wrap_all(model) -> None:
    def swap(mod):
        for name, child in list(mod.named_children()):
            if isinstance(child, torch.nn.Linear) and child.in_features % 128 == 0:
                setattr(mod, name, TernaryLinear(child, trainable=True))
            else:
                swap(child)
    for layer in model.model.layers:
        swap(layer)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=REPO / "out" / "exp-058" / "masked_corpus_4096.pt")
    ap.add_argument("--windows", type=int, default=24)
    ap.add_argument("--out", type=Path, default=REPO / "out" / "exp-058" / "layer_importance.txt")
    args = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    wrap_all(model)
    model.train()

    blob = torch.load(args.corpus, weights_only=False)
    ids_all, lbl_all = blob["ids"], blob["labels"]
    n_win = ids_all.shape[0]
    g = torch.Generator().manual_seed(7)
    order = torch.randperm(n_win, generator=g)[:args.windows]

    n_layers = len(model.model.layers)
    grad_norm = [0.0] * n_layers
    print(f"[imp] accumulating grad norms over {args.windows} masked windows, {n_layers} layers", flush=True)

    for i, w in enumerate(order.tolist()):
        model.zero_grad(set_to_none=True)
        ids = ids_all[w:w+1].to(dev); lbl = lbl_all[w:w+1].to(dev)
        out = model(input_ids=ids, labels=lbl)
        if not torch.isfinite(out.loss):
            continue
        out.loss.backward()
        for li, layer in enumerate(model.model.layers):
            s = 0.0
            for p in layer.parameters():
                if p.grad is not None:
                    s += float(p.grad.detach().norm().item()) ** 2
            grad_norm[li] += s ** 0.5
        if (i + 1) % 4 == 0:
            print(f"[imp]   {i+1}/{args.windows} windows", flush=True)

    # normalize + rank
    ranked = sorted(range(n_layers), key=lambda l: grad_norm[l], reverse=True)
    mx = max(grad_norm) or 1.0
    lines = ["layer  grad_norm  rel   rank"]
    rank_of = {l: r for r, l in enumerate(ranked)}
    for l in range(n_layers):
        lines.append(f"blk.{l:<2d}  {grad_norm[l]:8.2f}  {grad_norm[l]/mx:.2f}  #{rank_of[l]+1}")
    top = ranked[: n_layers // 2]
    lines.append("")
    lines.append(f"TOP-{n_layers//2} by grad signal (train these): {sorted(top)}")
    lines.append(f"(vs the naive 'last-{n_layers//2}': {list(range(n_layers - n_layers//2, n_layers))})")
    text = "\n".join(lines)
    args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
