"""exp-057: continued ternary QAT of Ternary-Bonsai-8B on the tool-call logs.

Straight-through ternary fine-tune (see quant_tuner.qat.ternary) of the unpacked
Qwen3-8B. Memory-frugal for Metal: the whole model is ternarized in the forward,
but only the last ``--train-layers`` transformer blocks carry trainable fp32
latents + optimizer state; everything else is frozen (ternarized, no grad).
Gradient checkpointing keeps activation memory flat. Loss is causal-LM CE on the
calibration corpus (the real tool-call logs). Saves trained latents for export.

    PYTHONPATH=src .venv/bin/python scripts/exp057_qat_train.py \
        --train-layers 8 --steps 200 --seq 1024 --grad-accum 8

Start SMALL to scope memory/speed: --train-layers 4 --steps 20 --seq 512.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch

from quant_tuner.qat.ternary import TernaryLinear

MODEL = REPO / "out" / "exp-057" / "model"
CORPUS = REPO / "out" / "exp-056" / "corpus.cal.txt"  # symlinked/copied from exp-054 below
FALLBACK_CORPUS = REPO / "out" / "exp-054" / "corpora" / "corpus.cal.txt"


def wrap_model(model: torch.nn.Module, n_train: int) -> tuple[int, int]:
    """Ternarize every qualifying linear; make only the last ``n_train`` decoder
    layers trainable. Returns (n_wrapped, n_trainable_params)."""
    layers = model.model.layers
    n_layers = len(layers)
    train_from = max(0, n_layers - n_train)

    def swap(mod: torch.nn.Module, trainable: bool) -> int:
        c = 0
        for name, child in list(mod.named_children()):
            if isinstance(child, torch.nn.Linear):
                if child.in_features % 128 == 0:
                    setattr(mod, name, TernaryLinear(child, trainable=trainable))
                    c += 1
            else:
                c += swap(child, trainable)
        return c

    n_wrapped = 0
    for i, layer in enumerate(layers):
        n_wrapped += swap(layer, trainable=(i >= train_from))
    # freeze everything not a trainable ternary latent (norms, embeddings, lm_head)
    for name, p in model.named_parameters():
        if ".linear.weight" not in name or "layers." not in name:
            p.requires_grad_(False)
        else:
            # only the last n_train layers' linears stay trainable
            li = int(name.split("layers.")[1].split(".")[0])
            p.requires_grad_(li >= train_from)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[qat] {n_layers} layers, training last {n_train} (from idx {train_from}); "
          f"wrapped {n_wrapped} linears; trainable params = {n_trainable/1e9:.2f}B", flush=True)
    return n_wrapped, n_trainable


def make_batches(tok, text: str, seq: int, device: str, limit_tokens: int):
    ids = tok(text, return_tensors="pt").input_ids[0]
    if limit_tokens:
        ids = ids[:limit_tokens]
    n = (ids.numel() // seq) * seq
    ids = ids[:n].view(-1, seq)
    print(f"[qat] corpus -> {ids.shape[0]} blocks of {seq} tokens", flush=True)
    return ids.to(device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-layers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--limit-tokens", type=int, default=400_000)
    ap.add_argument("--corpus", type=Path, default=None)
    ap.add_argument("--optim", choices=["adamw", "sgd"], default="adamw")
    ap.add_argument("--amp", action="store_true",
                    help="bf16 autocast compute with fp32 master latents (faster on MPS, "
                         "keeps latent precision needed for ternary code-flipping)")
    ap.add_argument("--out", type=Path, default=REPO / "out" / "exp-057" / "trained")
    args = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    from transformers import AutoModelForCausalLM, AutoTokenizer

    corpus = args.corpus or (CORPUS if CORPUS.exists() else FALLBACK_CORPUS)
    print(f"[qat] device={dev} dtype={args.dtype} corpus={corpus}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype)
    model.to(dev)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    print(f"[qat] model loaded in {time.time()-t0:.0f}s", flush=True)

    wrap_model(model, args.train_layers)
    model.train()

    batches = make_batches(tok, corpus.read_text(), args.seq, dev, args.limit_tokens)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if args.optim == "sgd":
        opt = torch.optim.SGD(trainable, lr=args.lr, momentum=0.9)  # 1 state (frugal)
    else:
        opt = torch.optim.AdamW(trainable, lr=args.lr)              # 2 fp32 states

    n_blocks = batches.shape[0]
    step = 0
    bi = 0
    t_start = time.time()
    opt.zero_grad()
    losses: list[float] = []
    import contextlib
    amp_ctx = (torch.autocast(device_type=dev, dtype=torch.bfloat16)
               if args.amp else contextlib.nullcontext())
    while step < args.steps:
        ids = batches[bi % n_blocks : (bi % n_blocks) + 1]
        bi += 1
        with amp_ctx:
            out = model(input_ids=ids, labels=ids)
        loss = out.loss / args.grad_accum
        loss.backward()
        losses.append(out.loss.item())
        if bi % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
            step += 1
            if step == 1 or step % 5 == 0:
                mem = torch.mps.current_allocated_memory()/1024**3 if dev == "mps" else 0
                avg = sum(losses[-args.grad_accum*5:]) / len(losses[-args.grad_accum*5:])
                dt = (time.time()-t_start)/step
                print(f"[qat] step {step}/{args.steps} loss={avg:.4f} "
                      f"mem={mem:.1f}GiB {dt:.1f}s/step", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    # save only the trained latents (ternary linears in the trained layers)
    trained = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save({"latents": trained, "args": vars(args),
                "loss_first": losses[0], "loss_last": sum(losses[-8:])/8},
               args.out / "trained_latents.pt")
    print(f"[qat] saved {len(trained)} trained tensors -> {args.out/'trained_latents.pt'}")
    print(f"[qat] loss {losses[0]:.4f} -> {sum(losses[-8:])/8:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
