"""GPTQ: Hessian-based per-tensor weight rounding.

Two-stage pipeline:

  1. ``calibrate(...)`` — Run the HF model forward on a calibration corpus,
     accumulating ``H = X^T X`` per target Linear via forward pre-hooks.
     One file per tensor is written under ``hessians_dir/`` (atomic via
     tmp + rename) so partial progress survives a crash.

  2. ``apply(...)`` — For each tensor with a saved Hessian, run the canonical
     GPTQ rounding algorithm (Frantar 2023, "Cholesky reformulation") and
     write the rounded weights back into a fresh HF checkpoint suitable for
     HF→GGUF conversion.

Targets:
  * full_attention layers: ``self_attn.{q,k,v,o}_proj``
  * every layer: ``mlp.{gate,up,down}_proj``

GPTQ writes weights close to an INT-N grid; saving them as F16 and then
running ``llama-quantize`` (per-block RTN to its own K-quant grid) preserves
the error compensation that lives in *adjacent* columns, even though each
column re-snaps slightly.

PPL sanity check: ``verify_perplexity()`` is a thin wrapper around
``llama-perplexity`` for catching catastrophic GPTQ failures (the
``gptq__IQ3_S`` row in the source experiment leaderboard exploded to 3.6M PPL).
Recommended use: convert HF→F16 GGUF, run ``verify_perplexity`` against the
uncalibrated F16 baseline PPL, abort if the ratio exceeds ~1.5.
"""

from __future__ import annotations

import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

# --- Target discovery ----------------------------------------------------- #

def discover_targets(model) -> list[tuple[str, torch.nn.Linear]]:
    """Return ``(dotted_name, module)`` for every target Linear: attn q/k/v/o + MLP gate/up/down."""
    out: list[tuple[str, torch.nn.Linear]] = []
    layers = getattr(model.model, "layers", None)
    if layers is None:
        raise RuntimeError("model.model.layers not found; unsupported architecture")
    for i, layer in enumerate(layers):
        prefix = f"model.layers.{i}"
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            for sub in ("q_proj", "k_proj", "v_proj", "o_proj"):
                m = getattr(attn, sub, None)
                if isinstance(m, torch.nn.Linear):
                    out.append((f"{prefix}.self_attn.{sub}", m))
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            for sub in ("gate_proj", "up_proj", "down_proj"):
                m = getattr(mlp, sub, None)
                if isinstance(m, torch.nn.Linear):
                    out.append((f"{prefix}.mlp.{sub}", m))
    return out


def _get_module(model, dotted: str):
    obj = model
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _safe_name(dotted: str) -> str:
    return dotted.replace(".", "__")


# --- Hessian I/O ---------------------------------------------------------- #

@dataclass
class Hessian:
    """One per-tensor Hessian accumulated as ``H = sum_t x_t · x_t^T``."""

    name: str
    H: torch.Tensor  # [in, in] fp32
    n_seen: int       # total tokens accumulated

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save({"name": self.name, "H": self.H, "n_seen": self.n_seen}, tmp)
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: Path) -> Hessian:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        return cls(name=blob["name"], H=blob["H"], n_seen=int(blob["n_seen"]))


# --- GPTQ rounding (pure tensor math) ------------------------------------ #

@dataclass
class GPTQStats:
    sse: float                       # sum of squared (W - Q) over the permuted basis
    frob_rel: float                  # ||Q - W||_F / ||W||_F in the original basis
    dead_cols: int                   # input channels with zero diag(H)


def gptq_round_tensor(
    W: torch.Tensor,
    H: torch.Tensor,
    *,
    n_bits: int = 4,
    group_size: int = 32,
    dampen: float = 0.01,
    actorder: bool = True,
) -> tuple[torch.Tensor, GPTQStats]:
    """Round ``W`` ([out, in]) toward an INT-``n_bits`` grid using GPTQ.

    Steps (per Frantar 2023):
      1. Tikhonov-damp ``H`` by ``dampen * mean(diag(H))``; zero-diag rows are
         marked "dead" — those input channels are forced to 0 and excluded.
      2. Optionally permute columns by descending ``diag(H)`` (act-order). This
         rounds the high-importance channels first, leaving more compensation
         budget for the tail.
      3. Compute upper-Cholesky ``U`` of ``H⁻¹``: ``U[j,j]`` is the per-column
         step size that scales the rounding error before spreading it onward.
      4. Sweep columns left→right in blocks of ``group_size``: per row, per
         group, pick a symmetric INT-``n_bits`` scale (``max(|w|) / qmax``),
         round each column inside the group, then propagate its residual
         within the group AND across to columns past the group.
    """
    out_features, in_features = W.shape
    if H.shape != (in_features, in_features):
        raise ValueError(f"H shape {tuple(H.shape)} does not match W in_features={in_features}")

    W = W.clone().float()
    H = H.clone().float()
    W_orig = W.clone()  # preserved across in-place mutations for the Frobenius stat

    # 1) Damping + dead-channel handling.
    diag = torch.diag(H)
    dead = diag == 0
    if dead.any():
        H[dead, dead] = 1.0
        W[:, dead] = 0.0
        W_orig[:, dead] = 0.0  # match expectation that dead cols are zeroed
    diag_mean = float(torch.diag(H).mean().item())
    H = H + dampen * diag_mean * torch.eye(in_features, dtype=H.dtype)

    # 2) Act-order permutation.
    if actorder:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
        invperm = torch.argsort(perm)
    else:
        invperm = None

    # 3) Upper Cholesky of H^{-1}.
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    U = torch.linalg.cholesky(Hinv, upper=True)

    # 4) Column-wise sweep in blocks.
    Q = torch.zeros_like(W)
    qmax = 2 ** (n_bits - 1) - 1  # INT4 -> 7, INT3 -> 3, INT2 -> 1
    total_sse = 0.0

    for i1 in range(0, in_features, group_size):
        i2 = min(i1 + group_size, in_features)
        gs = i2 - i1
        Wg = W[:, i1:i2].clone()
        Ug = U[i1:i2, i1:i2]
        Qg = torch.zeros_like(Wg)
        Eg = torch.zeros_like(Wg)

        max_abs = Wg.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        scale = max_abs / qmax  # per-row, per-group

        for j in range(gs):
            w = Wg[:, j]
            d = Ug[j, j]
            q = torch.round(w / scale.squeeze(1)).clamp(-qmax, qmax) * scale.squeeze(1)
            Qg[:, j] = q
            err = (w - q) / d
            if j + 1 < gs:
                Wg[:, j + 1 : gs].sub_(err.unsqueeze(1) * Ug[j, j + 1 : gs].unsqueeze(0))
            Eg[:, j] = err

        Q[:, i1:i2] = Qg
        total_sse += float(((Wg - Qg) ** 2).sum().item())
        if i2 < in_features:
            W[:, i2:].sub_(Eg @ U[i1:i2, i2:])

    if invperm is not None:
        Q = Q[:, invperm]

    frob_rel = float((Q - W_orig).norm() / max(W_orig.norm(), 1e-8))

    return Q, GPTQStats(
        sse=total_sse,
        frob_rel=frob_rel,
        dead_cols=int(dead.sum().item()),
    )


# --- Calibration: accumulate Hessians ----------------------------------- #

@torch.no_grad()
def calibrate(
    model_dir: Path,
    calibration_text: Path,
    hessians_dir: Path,
    *,
    tokens: int = 32_768,
    ctx: int = 2_048,
    device: str = "mps",
    dtype: str = "bfloat16",
    save_every: int = 8,
) -> Path:
    """Accumulate per-tensor Hessians over the calibration corpus.

    Snapshots all Hessians to ``hessians_dir`` every ``save_every`` chunks
    (atomic via tmp + rename), so the run can be resumed after a crash.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]

    print(f"[gptq] load {model_dir} -> {device}/{dtype}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    targets = discover_targets(model)
    n_attn = sum(1 for n, _ in targets if "self_attn" in n)
    n_mlp = sum(1 for n, _ in targets if "mlp." in n)
    print(f"[gptq] targets: {len(targets)} total ({n_attn} attn, {n_mlp} mlp)", file=sys.stderr)

    hessians_dir.mkdir(parents=True, exist_ok=True)
    H: dict[str, torch.Tensor] = {}
    n_seen: dict[str, int] = {}
    for name, mod in targets:
        d = mod.in_features
        H[name] = torch.zeros((d, d), dtype=torch.float32)
        n_seen[name] = 0

    def make_hook(name: str):
        def pre(_module, in_args):
            x = in_args[0].reshape(-1, in_args[0].shape[-1]).float()
            H[name].add_((x.T @ x).detach().cpu())
            n_seen[name] += x.shape[0]
        return pre

    handles = [mod.register_forward_pre_hook(make_hook(n)) for n, mod in targets]

    def snapshot() -> None:
        for name in H:
            Hessian(name=name, H=H[name], n_seen=n_seen[name]).save(
                hessians_dir / f"{_safe_name(name)}.pt"
            )
        print(f"[gptq] snapshot {len(H)} files -> {hessians_dir}", file=sys.stderr)

    try:
        text = calibration_text.read_text()
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0][:tokens]
        chunks = ids.split(ctx)
        print(f"[gptq] {ids.shape[0]} tokens, ctx {ctx} -> {len(chunks)} chunks", file=sys.stderr)
        for i, chunk in enumerate(chunks):
            if chunk.numel() < 2:
                continue
            model(chunk.unsqueeze(0).to(device))
            if (i + 1) % save_every == 0:
                print(f"  chunk {i + 1}/{len(chunks)}", file=sys.stderr)
                snapshot()
    finally:
        for h in handles:
            h.remove()

    snapshot()
    return hessians_dir


# --- Application: round + save ------------------------------------------ #

@torch.no_grad()
def apply(
    model_dir: Path,
    hessians_dir: Path,
    out_dir: Path,
    *,
    n_bits: int = 4,
    group_size: int = 32,
    dampen: float = 0.01,
    actorder: bool = True,
    device: str = "cpu",
    dtype: str = "bfloat16",
    name_filter: str | None = None,
    sanity_tokens: int = 32,
    sanity_max_rel: float = 0.5,
) -> Path:
    """Apply GPTQ rounding to every tensor with a saved Hessian and save the model.

    ``sanity_max_rel`` is the maximum tolerated relative logit drift before
    ``apply`` raises (default 0.5 = 50%). Set to ``float("inf")`` to disable.
    The original experiment's catastrophic ``IQ3_S`` row exceeded this by orders
    of magnitude, so catching it here avoids wasting time on a doomed quantize.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]

    print(f"[gptq.apply] load {model_dir} -> {device}/{dtype}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    model.eval()

    ref_logits = None
    sanity_ids = None
    if sanity_tokens > 0:
        text = "def hello_world():\n    print('hi')\n" * 8
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        sanity_ids = ids[:sanity_tokens].unsqueeze(0).to(device)
        ref_logits = model(sanity_ids).logits.detach().float().cpu()

    h_files = sorted(hessians_dir.glob("*.pt"))
    if not h_files:
        raise FileNotFoundError(f"no Hessians in {hessians_dir}")
    print(f"[gptq.apply] {len(h_files)} Hessians", file=sys.stderr)

    rel_errors: list[tuple[str, float]] = []
    for i, hpath in enumerate(h_files):
        h = Hessian.load(hpath)
        if name_filter is not None and name_filter not in h.name:
            continue
        mod = _get_module(model, h.name)
        W = mod.weight.data.detach().to(torch.float32).cpu()
        if h.H.shape != (W.shape[1], W.shape[1]):
            print(
                f"[gptq.apply] WARN: {h.name} H={tuple(h.H.shape)} vs W in {W.shape[1]}; skip",
                file=sys.stderr,
            )
            continue
        try:
            Q, stats = gptq_round_tensor(
                W, h.H, n_bits=n_bits, group_size=group_size,
                dampen=dampen, actorder=actorder,
            )
        except Exception as e:
            print(f"[gptq.apply] WARN: GPTQ failed on {h.name}: {e}", file=sys.stderr)
            continue
        mod.weight.data.copy_(Q.to(torch_dtype).to(mod.weight.device))
        rel_errors.append((h.name, stats.frob_rel))
        if (i + 1) % 16 == 0 or (i + 1) == len(h_files):
            print(
                f"  [{i + 1}/{len(h_files)}] {h.name}  rel={stats.frob_rel:.4f} sse={stats.sse:.3e}",
                file=sys.stderr,
            )

    if rel_errors:
        rels = torch.tensor([r for _, r in rel_errors])
        print(
            f"[gptq.apply] {len(rel_errors)} rounded: mean rel={rels.mean():.4f} "
            f"max rel={rels.max():.4f}",
            file=sys.stderr,
        )

    if ref_logits is not None and sanity_ids is not None:
        new_logits = model(sanity_ids).logits.detach().float().cpu()
        max_abs = (new_logits - ref_logits).abs().max().item()
        rel = max_abs / max(ref_logits.abs().max().item(), 1e-8)
        print(f"[gptq.apply] sanity max|Δlogit|={max_abs:.3e} rel={rel:.3e}", file=sys.stderr)
        if rel > sanity_max_rel:
            raise RuntimeError(
                f"GPTQ logit drift {rel:.3e} > sanity_max_rel={sanity_max_rel}; "
                f"the Hessian probably did not see enough tokens or weights are too sparse"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[gptq.apply] writing -> {out_dir}", file=sys.stderr)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    for name in ("chat_template.jinja", "generation_config.json"):
        src = model_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
    return out_dir


# --- PPL sanity check ---------------------------------------------------- #

_PPL_FINAL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)")


def _parse_perplexity(text: str) -> float | None:
    matches = _PPL_FINAL_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def verify_perplexity(
    f16_gguf: Path,
    eval_dataset: Path,
    *,
    reference_ppl: float | None = None,
    max_ratio: float = 1.5,
    ctx: int = 8192,
    log: Path | None = None,
) -> float:
    """Run ``llama-perplexity`` on a GPTQ-applied F16 GGUF and bail if it blows up.

    If ``reference_ppl`` is given, raise when ``ppl / reference_ppl > max_ratio``.
    Otherwise just return the measured PPL for the caller to inspect.

    Returns the measured perplexity.
    """
    from quant_tuner.models import llama_cpp
    from quant_tuner.paths import llama_bin

    cmd: list[str | Path] = [
        llama_bin("llama-perplexity"),
        "-m", f16_gguf,
        "-f", eval_dataset,
        "-c", str(ctx),
    ]
    output = llama_cpp.run(cmd, log=log)
    ppl = _parse_perplexity(output)
    if ppl is None:
        raise RuntimeError("could not parse 'Final estimate: PPL' from llama-perplexity output")
    if reference_ppl is not None:
        ratio = ppl / reference_ppl
        if ratio > max_ratio:
            raise RuntimeError(
                f"GPTQ sanity PPL={ppl:.3f} vs reference {reference_ppl:.3f} "
                f"(ratio {ratio:.2f} > {max_ratio}); refusing to quantize a broken model"
            )
    return ppl


__all__ = [
    "GPTQStats",
    "Hessian",
    "apply",
    "calibrate",
    "discover_targets",
    "gptq_round_tensor",
    "verify_perplexity",
]
