"""QAT trainer: masked-loss, big-window, LR-scheduled, memory-frugal.

The core of the continued-QAT pipeline for native-ternary models (see
``docs/qat_optimization_audit.md`` for the full rationale). Consumes a pre-tokenized,
assistant-MASKED corpus (:mod:`quant_tuner.qat.corpus`) and continues training the ternary
weights with a straight-through estimator. Key properties:

  * MASKED-CE forward: the lm_head runs only at labeled positions instead of the full
    ``[1, seq, vocab]`` logits tensor (-4-5 GB peak). Parity with HF's ForCausalLMLoss is
    unit-tested.
  * Adafactor (factored 2nd moment, ~MBs vs AdamW's two fp32 states = 55.6 GB at all-36)
    -> full-36-layer fp32 training fits in ~70 GB. Pure per-tensor loop, no MPS deadlock.
    ``weight_decay`` defaults to 0 for BOTH optimizers (decay erodes ternary codes to 0).
  * ``compute_dtype='bf16'``: fp32-master trick (:mod:`quant_tuner.qat.master_opt`) —
    masters+clip+step in fp32, forward/backward in bf16.
  * ``kd_teacher``: online distillation from a same-vocab dense teacher (KL on labeled
    positions only, via ``logits_to_keep``).
  * ``resume``: continue from ``trained_latents.pt`` (data order, step, Adafactor state;
    corpus fingerprint must match). Atomic checkpoints (tmp + rename).
  * Code-flip telemetry every checkpoint — the instrument for the LR probe (at lr 5e-5 the
    expected flip count is ~zero; see the audit).

The CLI shim is ``scripts/exp058_qat_train_v2.py``.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from quant_tuner.qat.corpus import corpus_fingerprint
from quant_tuner.qat.master_opt import MasterOptimizer
from quant_tuner.qat.ternary import TernaryLinear, ternarize_group

REPO = Path(__file__).resolve().parents[3]
MODEL = REPO / "out" / "exp-057" / "model"
MPS_MAX_WINDOW = 4096  # 8192 -> MPSGraph "tensor dims larger than INT_MAX" (32*8192^2 = 2^31)


@dataclass
class QATConfig:
    corpus: Path
    out: Path = REPO / "out" / "exp-058" / "trained"
    model_dir: Path = MODEL
    train_layers: int = 18
    layers: str | None = None
    epochs: float = 3.0
    grad_accum: int = 8
    lr: float = 5e-5
    optim: str = "adamw"
    weight_decay: float = 0.0
    beta1: float | None = None
    dtype: str = "fp32"
    compute_dtype: str = "fp32"
    kd_teacher: Path | None = None
    kd_alpha: float = 0.5
    kd_temp: float = 1.0
    val_corpus: Path | None = None
    val_every: int = 20
    val_windows: int = 16
    train_norms: bool = False
    resume: Path | None = None
    flip_sample: int = 8
    ckpt_every: int = 40


def parse_layers(spec: str, n_layers: int) -> set[int]:
    """Parse '0-14,32,34,35' -> {0..14,32,34,35}. Empty -> all."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out |= set(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return {layer for layer in out if 0 <= layer < n_layers}


def wrap_model(model, n_train: int, layer_spec: str | None = None,
               train_norms: bool = False) -> int:
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
                if not trainable:
                    # Frozen layer: shipped weights are already exactly on the ternary
                    # grid, so TernaryLinear would be a bit-exact no-op costing ~5 W-sized
                    # transient allocs per forward (x2 under checkpoint recompute). Skip
                    # the wrap when we can PROVE exactness; wrap otherwise (e.g. a layer
                    # trained in an earlier run that drifted off-grid).
                    with torch.no_grad():
                        _, _, w_hat = ternarize_group(child.weight)
                        exact = torch.equal(w_hat, child.weight)
                    if exact:
                        child.weight.requires_grad_(False)
                        continue
                    print(f"[qat]   frozen linear off-grid -> wrapping: {name}", flush=True)
                setattr(mod, name, TernaryLinear(child, trainable=trainable))
                c += 1
            else:
                c += swap(child, trainable)
        return c

    nw = sum(swap(layer, i in trainable_idx) for i, layer in enumerate(layers))
    for name, p in model.named_parameters():
        li = int(name.split("layers.")[1].split(".")[0]) if "layers." in name else -1
        is_latent = ".linear.weight" in name and li in trainable_idx
        is_norm = train_norms and li in trainable_idx and "norm" in name
        p.requires_grad_(is_latent or is_norm)
    nt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[qat] training {label} ({len(trainable_idx)} layers"
          f"{', +norms' if train_norms else ''}); wrapped {nw}; "
          f"trainable {nt/1e9:.2f}B", flush=True)
    return nw


def lr_at(step, total, base, warmup_frac=0.05):
    warm = max(1, int(total * warmup_frac))
    if step < warm:
        return base * step / warm
    prog = (step - warm) / max(1, total - warm)
    return 0.1 * base + 0.9 * base * 0.5 * (1 + math.cos(math.pi * prog))


def masked_forward(model, ids: torch.Tensor, lbl: torch.Tensor):
    """Masked-CE forward: lm_head only at labeled positions.

    Selects positions t with lbl[t+1] != -100 (HF shift semantics), runs the
    decoder trunk on the full window, then the lm_head on the K selected hidden
    states only. Returns (ce_loss, logits [1,K,V] fp32, keep_idx) — the mean CE
    over exactly the same target set as transformers' ForCausalLMLoss.
    """
    tgt = lbl[:, 1:]
    keep_idx = (tgt[0] != -100).nonzero(as_tuple=True)[0]
    hidden = model.model(input_ids=ids).last_hidden_state    # [1, S, H]
    h = hidden[:, keep_idx, :]                               # [1, K, H]
    logits = model.lm_head(h).float()                        # [1, K, V]
    ce = F.cross_entropy(logits[0], tgt[0, keep_idx])
    return ce, logits, keep_idx


def kd_kl(teacher, ids: torch.Tensor, keep_idx: torch.Tensor,
          student_logits: torch.Tensor, temp: float) -> torch.Tensor:
    """KL(teacher || student) at the labeled positions, temperature-scaled.

    The teacher gets logits_to_keep=keep_idx (transformers >= 5 accepts an index
    tensor), so it never materializes full-vocab logits at unlabeled positions.
    """
    with torch.no_grad():
        t_logits = teacher(input_ids=ids, logits_to_keep=keep_idx).logits.float()
    t_logp = torch.log_softmax(t_logits[0] / temp, dim=-1)
    s_logp = torch.log_softmax(student_logits[0] / temp, dim=-1)
    return F.kl_div(s_logp, t_logp, log_target=True, reduction="none").sum(-1).mean()


def snapshot_codes(model, k: int = 8) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Snapshot (codes int8, scale fp16) of k trainable linears spread across layers."""
    mods = [(n, m) for n, m in model.named_modules()
            if isinstance(m, TernaryLinear) and m.linear.weight.requires_grad]
    if not mods or k <= 0:
        return {}
    picks = sorted({round(i * (len(mods) - 1) / max(1, min(k, len(mods)) - 1))
                    for i in range(min(k, len(mods)))})
    snaps = {}
    with torch.no_grad():
        for i in picks:
            n, m = mods[i]
            codes, scale, _ = ternarize_group(m.linear.weight.detach().float())
            snaps[n] = (codes.to(torch.int8).cpu(), scale.to(torch.float16).cpu())
    return snaps


def flip_report(model, snaps) -> tuple[dict, str]:
    """Codes flipped / scale drift vs the start-of-run snapshot."""
    mods = dict(model.named_modules())
    stats, lines = {}, []
    with torch.no_grad():
        for name, (codes0, scale0) in snaps.items():
            w = mods[name].linear.weight.detach().float()
            codes, scale, _ = ternarize_group(w)
            c = codes.to(torch.int8).cpu()
            flip_pct = 100.0 * (c != codes0).float().mean().item()
            z2nz = int(((codes0 == 0) & (c != 0)).sum())
            nz2z = int(((codes0 != 0) & (c == 0)).sum())
            s0 = scale0.float()
            drift = ((scale.to(torch.float16).cpu().float() - s0).abs()
                     / s0.clamp_min(1e-8)).mean().item()
            stats[name] = {"flip_pct": flip_pct, "zero_to_nonzero": z2nz,
                           "nonzero_to_zero": nz2z, "scale_drift": drift}
            lines.append(f"  {name}: flips {flip_pct:.4f}% (0->±:{z2nz} ±->0:{nz2z}) "
                         f"scale-drift {drift*100:.2f}%")
    return stats, "\n".join(lines)


def run_validation(model, ids_all, lbl_all, dev, max_windows: int) -> float:
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(min(max_windows, ids_all.shape[0])):
            lbl = lbl_all[i:i + 1]
            if not bool((lbl[0, 1:] != -100).any()):
                continue
            ce, _, _ = masked_forward(model, ids_all[i:i + 1].to(dev), lbl.to(dev))
            tot += float(ce)
            n += 1
    model.train()
    return tot / max(1, n)


def train_qat(cfg: QATConfig) -> int:
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    if cfg.dtype == "bf16":
        print("[qat] WARNING: bf16 latents underflow the ternary threshold — no codes "
              "will flip at stable LRs. Use compute_dtype=bf16 (fp32 masters) instead.",
              flush=True)
    if cfg.compute_dtype == "bf16" and cfg.dtype != "fp32":
        sys.exit("[qat] compute_dtype bf16 requires dtype fp32 (fp32 masters)")
    dtype = torch.float32 if cfg.dtype == "fp32" else torch.bfloat16
    from transformers import AutoModelForCausalLM

    blob = torch.load(cfg.corpus, weights_only=False)
    ids_all, lbl_all = blob["ids"], blob["labels"]
    n_win, window = ids_all.shape
    if dev == "mps" and window > MPS_MAX_WINDOW:
        sys.exit(f"[qat] window {window} > {MPS_MAX_WINDOW}: MPS attention hits the "
                 f"MPSGraph INT_MAX limit (32 heads x 8192^2 = 2^31). Rebuild the "
                 f"corpus with window {MPS_MAX_WINDOW}.")
    fp = blob.get("fingerprint") or corpus_fingerprint(ids_all, lbl_all)
    total_steps = int(cfg.epochs * n_win / cfg.grad_accum)
    print(f"[qat] corpus {n_win} windows x {window} ({blob.get('assistant_frac',0)*100:.0f}% masked, "
          f"fingerprint {fp}); {cfg.epochs} epochs -> {total_steps} steps @ accum {cfg.grad_accum}",
          flush=True)

    val_ids = val_lbl = None
    if cfg.val_corpus:
        vblob = torch.load(cfg.val_corpus, weights_only=False)
        val_ids, val_lbl = vblob["ids"], vblob["labels"]
        print(f"[qat] val corpus {val_ids.shape[0]} windows "
              f"(using {min(cfg.val_windows, val_ids.shape[0])})", flush=True)

    model = AutoModelForCausalLM.from_pretrained(cfg.model_dir, dtype=dtype).to(dev)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()  # transformers>=5 defaults use_reentrant=False
    wrap_model(model, cfg.train_layers, layer_spec=cfg.layers, train_norms=cfg.train_norms)
    model.train()

    teacher = None
    if cfg.kd_teacher:
        tdtype = torch.float16 if dev == "mps" else torch.float32
        teacher = AutoModelForCausalLM.from_pretrained(cfg.kd_teacher, dtype=tdtype).to(dev)
        teacher.config.use_cache = False
        teacher.eval().requires_grad_(False)
        assert teacher.config.vocab_size == model.config.vocab_size, (
            f"teacher vocab {teacher.config.vocab_size} != student "
            f"{model.config.vocab_size} — KD needs a shared tokenizer")
        print(f"[qat] KD teacher {cfg.kd_teacher} ({tdtype}), "
              f"alpha={cfg.kd_alpha} T={cfg.kd_temp}", flush=True)

    trainable_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    t_names = [n for n, _ in trainable_named]
    trainable = [p for _, p in trainable_named]

    def make_inner(params):
        if cfg.optim == "adafactor":
            from transformers import Adafactor
            return Adafactor(params, lr=cfg.lr, scale_parameter=False,
                             relative_step=False, warmup_init=False,
                             beta1=cfg.beta1, weight_decay=cfg.weight_decay)
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay,
                                 foreach=False)

    if cfg.compute_dtype == "bf16":
        # masters are cloned fp32 BEFORE the bf16 cast; the cast keeps Parameter
        # identity, so the wrapper's param references stay live
        opt = MasterOptimizer(trainable, make_inner)
        model.to(torch.bfloat16)
        print("[qat] bf16 compute + fp32 masters "
              f"({sum(m.numel() for m in opt.masters)/1e9:.2f}B master params)", flush=True)
    else:
        opt = make_inner(trainable)
    print(f"[qat] optimizer {cfg.optim} (wd={cfg.weight_decay}"
          f"{f', beta1={cfg.beta1}' if cfg.beta1 else ''})", flush=True)

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    stop = {"f": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("f", True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("f", True))

    g = torch.Generator().manual_seed(1234)
    order = torch.randperm(n_win, generator=g)
    step = 0
    mi = 0
    loss_first = None
    recent: list[float] = []

    if cfg.resume:
        ck = torch.load(cfg.resume, map_location="cpu", weights_only=False)
        ck_fp = ck.get("corpus_fingerprint")
        if ck_fp != fp:
            sys.exit(f"[qat] resume corpus mismatch: ckpt fingerprint {ck_fp} != "
                     f"corpus {fp}. Resuming across a rebuilt corpus would silently "
                     f"misalign the data order — rebuild or drop resume.")
        latents = ck["latents"]
        missing = [n for n in t_names if n not in latents]
        if missing:
            sys.exit(f"[qat] resume layer-set mismatch: ckpt lacks {missing[:3]}... "
                     f"({len(missing)} params). Use the same layers/train_norms.")
        if isinstance(opt, MasterOptimizer):
            opt.load_masters([latents[n] for n in t_names])
        else:
            named = dict(model.named_parameters())
            with torch.no_grad():
                for n in t_names:
                    named[n].copy_(latents[n].to(named[n].device, named[n].dtype))
        step, mi = int(ck.get("step", 0)), int(ck.get("mi", 0))
        for _ in range(mi // n_win):  # replay epoch reshuffles -> deterministic order
            order = torch.randperm(n_win, generator=g)
        if cfg.optim == "adafactor" and ck.get("optim") is not None:
            (opt.inner if isinstance(opt, MasterOptimizer) else opt).load_state_dict(ck["optim"])
            print(f"[qat] resumed at step {step} (mi={mi}) with adafactor state", flush=True)
        else:
            print(f"[qat] resumed at step {step} (mi={mi}); OPTIMIZER STATE RESET "
                  f"({'adamw state is not checkpointed (56 GB at all-36)' if cfg.optim == 'adamw' else 'no state in ckpt'})",
                  flush=True)
        loss_first = ck.get("loss_first")

    snaps = snapshot_codes(model, cfg.flip_sample)
    print(f"[qat] flip telemetry on {len(snaps)} linears", flush=True)
    flip_stats: dict = {}

    def save_ckpt(at):
        nonlocal flip_stats
        if snaps:
            flip_stats, lines = flip_report(model, snaps)
            print(f"[qat] code flips vs run start:\n{lines}", flush=True)
        if isinstance(opt, MasterOptimizer):
            latents = {n: m.detach().cpu() for n, m in zip(t_names, opt.masters, strict=True)}
        else:
            latents = {n: p.detach().cpu() for n, p in trainable_named}
        # for MasterOptimizer save only the INNER state — the masters ARE the
        # latents payload; duplicating them would double the ckpt size (28 GB)
        optim_state = None
        if cfg.optim == "adafactor":
            optim_state = (opt.inner.state_dict() if isinstance(opt, MasterOptimizer)
                           else opt.state_dict())
        payload = {"latents": latents, "args": {k: str(v) for k, v in vars(cfg).items()},
                   "step": at, "mi": mi, "corpus_fingerprint": fp,
                   "loss_first": loss_first,
                   "loss_last": sum(recent[-8:]) / len(recent[-8:]) if recent else None,
                   "flip_stats": flip_stats,
                   "optim": optim_state}
        tmp = out / ".tmp-trained_latents.pt"
        torch.save(payload, tmp)
        os.replace(tmp, out / "trained_latents.pt")
        del latents, payload
        gc.collect()
        if dev == "mps":
            torch.mps.empty_cache()
        print(f"[qat] checkpoint @ step {at}: {len(t_names)} tensors", flush=True)

    def opt_step():
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step, total_steps, cfg.lr)
        if isinstance(opt, MasterOptimizer):
            opt.clip_and_step(1.0)  # clips in fp32 masters, foreach=False inside
        else:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0, foreach=False)
            opt.step()
        opt.zero_grad()

    t0 = time.time()
    n_acc = 0
    opt.zero_grad()
    while step < total_steps and not stop["f"]:
        w = order[mi % n_win].item()
        mi += 1
        if mi % n_win == 0:  # reshuffle each epoch
            order = torch.randperm(n_win, generator=g)
        lbl_cpu = lbl_all[w:w + 1]
        if not bool((lbl_cpu[0, 1:] != -100).any()):
            continue  # no valid shifted target; builder should have dropped it
        ids = ids_all[w:w + 1].to(dev)
        lbl = lbl_cpu.to(dev)
        ce, s_logits, keep_idx = masked_forward(model, ids, lbl)
        if teacher is not None:
            kl = kd_kl(teacher, ids, keep_idx, s_logits, cfg.kd_temp)
            loss = (1 - cfg.kd_alpha) * ce + cfg.kd_alpha * (cfg.kd_temp ** 2) * kl
        else:
            loss = ce
        lv = float(loss.detach())
        if not math.isfinite(lv):
            # skip BEFORE backward: the accumulated group stays valid, n_acc unchanged
            print("[qat] non-finite loss — skip window", flush=True)
            continue
        (loss / cfg.grad_accum).backward()
        n_acc += 1
        if loss_first is None:
            loss_first = lv
        recent.append(lv)
        recent[:] = recent[-cfg.grad_accum * 5:]
        if n_acc == cfg.grad_accum:
            opt_step()
            n_acc = 0
            step += 1
            if step == 1 or step % 5 == 0:
                mem = torch.mps.current_allocated_memory() / 1024**3 if dev == "mps" else 0
                avg = sum(recent) / len(recent)
                print(f"[qat] step {step}/{total_steps} loss={avg:.4f} "
                      f"lr={opt.param_groups[0]['lr']:.2e} "
                      f"mem={mem:.1f}GiB {(time.time()-t0)/step:.1f}s/step", flush=True)
            if val_ids is not None and cfg.val_every and step % cfg.val_every == 0:
                vl = run_validation(model, val_ids, val_lbl, dev, cfg.val_windows)
                print(f"[qat] step {step} VAL masked-CE {vl:.4f}", flush=True)
            if cfg.ckpt_every and step % cfg.ckpt_every == 0:
                save_ckpt(step)
    # drop any partial accum group before the final save
    opt.zero_grad()
    save_ckpt(step)
    if recent and loss_first is not None:
        print(f"[qat] done at step {step}: loss {loss_first:.3f} -> "
              f"{sum(recent[-8:]) / len(recent[-8:]):.3f}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--train-layers", type=int, default=18)
    ap.add_argument("--layers", type=str, default=None,
                    help="explicit layer indices to train, e.g. '0-14,32,34,35'; overrides --train-layers")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--optim", choices=["adamw", "adafactor"], default="adamw",
                    help="adafactor: factored 2nd moment (~MBs vs AdamW's 56 GB at "
                         "all-36) -> full-36 fp32 fits. Per-tensor loop, MPS-safe.")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="default 0: decay on ternary latents erodes codes toward 0")
    ap.add_argument("--beta1", type=float, default=None,
                    help="adafactor momentum (costs a full fp32 state, +27.8 GB at all-36); default off")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32",
                    help="latent dtype. bf16 latents UNSUPPORTED for training — use --compute-dtype bf16")
    ap.add_argument("--compute-dtype", choices=["fp32", "bf16"], default="fp32",
                    help="bf16: fp32-master trick — masters own the latents, fwd/bwd run in bf16")
    ap.add_argument("--kd-teacher", type=Path, default=None,
                    help="HF path of a same-vocab dense teacher (e.g. a SWE-tuned Qwen3-8B); enables KD")
    ap.add_argument("--kd-alpha", type=float, default=0.5,
                    help="loss = (1-a)*CE + a*T^2*KL(teacher||student)")
    ap.add_argument("--kd-temp", type=float, default=1.0)
    ap.add_argument("--val-corpus", type=Path, default=None,
                    help="masked corpus built with --split test; masked-CE validation")
    ap.add_argument("--val-every", type=int, default=20)
    ap.add_argument("--val-windows", type=int, default=16)
    ap.add_argument("--train-norms", action="store_true",
                    help="also train RMSNorm/q_norm/k_norm weights in the trainable layers")
    ap.add_argument("--resume", type=Path, default=None,
                    help="trained_latents.pt to continue from (same corpus required)")
    ap.add_argument("--flip-sample", type=int, default=8,
                    help="trainable linears to track for code-flip telemetry")
    ap.add_argument("--ckpt-every", type=int, default=40)
    ap.add_argument("--out", type=Path, default=REPO / "out" / "exp-058" / "trained")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = QATConfig(
        corpus=args.corpus, out=args.out, model_dir=MODEL,
        train_layers=args.train_layers, layers=args.layers, epochs=args.epochs,
        grad_accum=args.grad_accum, lr=args.lr, optim=args.optim,
        weight_decay=args.weight_decay, beta1=args.beta1, dtype=args.dtype,
        compute_dtype=args.compute_dtype, kd_teacher=args.kd_teacher,
        kd_alpha=args.kd_alpha, kd_temp=args.kd_temp, val_corpus=args.val_corpus,
        val_every=args.val_every, val_windows=args.val_windows,
        train_norms=args.train_norms, resume=args.resume,
        flip_sample=args.flip_sample, ckpt_every=args.ckpt_every,
    )
    return train_qat(cfg)


if __name__ == "__main__":
    sys.exit(main())
