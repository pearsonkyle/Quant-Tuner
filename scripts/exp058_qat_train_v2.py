"""iter-2 QAT trainer: masked-loss, big-window, LR-scheduled.

Improvements over exp-057 v1 (see the audit in the conversation / iter-2 plan):
  * Consumes a pre-tokenized, assistant-MASKED corpus (build_qat_masked_corpus.py):
    loss is computed only on tool/assistant tokens (labels=-100 elsewhere), so the
    gradient targets tool-call generation instead of boilerplate.
  * Bigger window (whatever the corpus was packed at) + real per-epoch SHUFFLE +
    larger effective batch via --grad-accum.
  * LR warmup + cosine decay (was constant 2e-5).
  * Same working MPS config: fp32 latents, last-N layers, foreach=False,
    checkpoint every --ckpt-every, save-on-signal.

    PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python \
        scripts/exp058_qat_train_v2.py --corpus out/exp-058/masked_corpus.pt \
        --train-layers 18 --epochs 3 --grad-accum 8 --lr 5e-5 --out out/exp-058/trained
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch

from quant_tuner.qat.ternary import TernaryLinear

MODEL = REPO / "out" / "exp-057" / "model"


def parse_layers(spec: str, n_layers: int) -> set[int]:
    """Parse '0-14,32,34,35' -> {0..14,32,34,35}. Empty -> all."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-"); out |= set(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return {l for l in out if 0 <= l < n_layers}


def wrap_model(model, n_train: int, layer_spec: str | None = None) -> int:
    layers = model.model.layers
    if layer_spec:
        trainable_idx = parse_layers(layer_spec, len(layers))
        label = f"explicit layers {sorted(trainable_idx)}"
    else:
        trainable_idx = set(range(max(0, len(layers) - n_train), len(layers)))
        label = f"last {n_train} layers"

    def swap(mod, trainable):
        c = 0
        for name, child in list(mod.named_children()):
            if isinstance(child, torch.nn.Linear) and child.in_features % 128 == 0:
                setattr(mod, name, TernaryLinear(child, trainable=trainable)); c += 1
            else:
                c += swap(child, trainable)
        return c

    nw = sum(swap(layer, i in trainable_idx) for i, layer in enumerate(layers))
    for name, p in model.named_parameters():
        ok = (".linear.weight" in name and "layers." in name
              and int(name.split("layers.")[1].split(".")[0]) in trainable_idx)
        p.requires_grad_(ok)
    nt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[qat] training {label} ({len(trainable_idx)} layers); wrapped {nw}; "
          f"trainable {nt/1e9:.2f}B", flush=True)
    return nw


def lr_at(step, total, base, warmup_frac=0.05):
    warm = max(1, int(total * warmup_frac))
    if step < warm:
        return base * step / warm
    prog = (step - warm) / max(1, total - warm)
    return 0.1 * base + 0.9 * base * 0.5 * (1 + math.cos(math.pi * prog))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--train-layers", type=int, default=18)
    ap.add_argument("--layers", type=str, default=None,
                    help="explicit layer indices to train, e.g. '0-14,32,34,35' (from the "
                         "grad-importance probe); overrides --train-layers")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--optim", choices=["adamw", "adafactor"], default="adamw",
                    help="adafactor = factored state, fits ALL 36 layers in fp32 (adamw swaps)")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--ckpt-every", type=int, default=40)
    ap.add_argument("--out", type=Path, default=REPO / "out" / "exp-058" / "trained")
    args = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    from transformers import AutoModelForCausalLM

    blob = torch.load(args.corpus, weights_only=False)
    ids_all, lbl_all = blob["ids"], blob["labels"]
    n_win, window = ids_all.shape
    total_steps = int(args.epochs * n_win / args.grad_accum)
    print(f"[qat] corpus {n_win} windows x {window} ({blob.get('assistant_frac',0)*100:.0f}% masked); "
          f"{args.epochs} epochs -> {total_steps} steps @ accum {args.grad_accum}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(dev)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    wrap_model(model, args.train_layers, layer_spec=args.layers)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    if args.optim == "adafactor":
        # Factored second moments -> tiny optimizer state, so ALL 36 layers fit in
        # fp32 without swapping (AdamW's two fp32 states would be ~56GB). Fixed-lr
        # config (no internal relative-step schedule); our cosine schedule drives lr.
        from transformers.optimization import Adafactor
        opt = Adafactor(trainable, lr=args.lr, scale_parameter=False,
                        relative_step=False, warmup_init=False)
    else:
        opt = torch.optim.AdamW(trainable, lr=args.lr, foreach=False)

    args.out.mkdir(parents=True, exist_ok=True)
    stop = {"f": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("f", True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("f", True))

    def save_ckpt(at):
        trained = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save({"latents": trained, "args": vars(args), "step": at}, args.out / "trained_latents.pt")
        print(f"[qat] checkpoint @ step {at}: {len(trained)} tensors", flush=True)

    # deterministic per-epoch order (no Math.random): fixed torch.randperm(seed)
    g = torch.Generator().manual_seed(1234)
    order = torch.randperm(n_win, generator=g)
    step = 0; mi = 0; t0 = time.time(); opt.zero_grad(); losses = []
    while step < total_steps and not stop["f"]:
        w = order[mi % n_win].item(); mi += 1
        if mi % n_win == 0:  # reshuffle each epoch
            order = torch.randperm(n_win, generator=g)
        ids = ids_all[w:w+1].to(dev); lbl = lbl_all[w:w+1].to(dev)
        out = model(input_ids=ids, labels=lbl)
        if not torch.isfinite(out.loss):
            print(f"[qat] non-finite loss — skip", flush=True); opt.zero_grad(); continue
        (out.loss / args.grad_accum).backward()
        losses.append(out.loss.item())
        if mi % args.grad_accum == 0:
            for pg in opt.param_groups:
                pg["lr"] = lr_at(step, total_steps, args.lr)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0, foreach=False)
            opt.step(); opt.zero_grad(); step += 1
            if step == 1 or step % 5 == 0:
                mem = torch.mps.current_allocated_memory()/1024**3 if dev == "mps" else 0
                avg = sum(losses[-args.grad_accum*5:]) / len(losses[-args.grad_accum*5:])
                print(f"[qat] step {step}/{total_steps} loss={avg:.4f} lr={opt.param_groups[0]['lr']:.2e} "
                      f"mem={mem:.1f}GiB {(time.time()-t0)/step:.1f}s/step", flush=True)
            if args.ckpt_every and step % args.ckpt_every == 0:
                save_ckpt(step)
    save_ckpt(step)
    print(f"[qat] done at step {step}: loss {losses[0]:.3f} -> {sum(losses[-8:])/8:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
