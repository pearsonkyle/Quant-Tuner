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

import hashlib
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from quant_tuner.calibrate._device import resolve_device
from quant_tuner.calibrate._hf import forward_no_logits
from quant_tuner.calibrate._ingest import load_chunks
from quant_tuner.calibrate._quant_mix import (
    gqa_or_moe_ge4,
    layer_index,
    target_type_for_member,
)

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
    dampen_used: float               # final damping after any Cholesky-retry escalation


def gptq_round_tensor(
    W: torch.Tensor,
    H: torch.Tensor,
    *,
    n_bits: int = 4,
    group_size: int = 32,
    dampen: float = 0.01,
    actorder: bool = True,
    sym: bool | None = None,
) -> tuple[torch.Tensor, GPTQStats]:
    """Round ``W`` ([out, in]) toward an INT-``n_bits`` grid using GPTQ.

    Steps (per Frantar 2023):
      1. Tikhonov-damp ``H`` by ``dampen * mean(diag(H))``; zero-diag rows are
         marked "dead" — those input channels get H-diag 1.0 (keeps H positive
         definite with zero coupling, so no compensation flows through them)
         and plain RTN on the group grid. If the Cholesky still fails (2-3-bit
         Hessians are often near-singular), the damping escalates ×2.5 up to
         four times before giving up; ``stats.dampen_used`` records the final
         value. At 2-3 bits the damp/Cholesky/inverse chain runs in float64
         on CPU; 4-bit+ keeps the proven (cheaper) fp32 path.
      2. Optionally permute columns by descending ``diag(H)`` (act-order). This
         rounds the high-importance channels first, leaving more compensation
         budget for the tail.
      3. Compute upper-Cholesky ``U`` of ``H⁻¹``: ``U[j,j]`` is the per-column
         step size that scales the rounding error before spreading it onward.
      4. Sweep columns left→right in blocks of ``group_size``: per row, per
         group, pick an INT-``n_bits`` grid, round each column inside the
         group, then propagate its residual within the group AND across to
         columns past the group.

    ``sym`` selects the grid; ``None`` (default) auto-selects:
      * ``True`` (>= 4 bits) — symmetric: ``scale = max|w| / qmax``, codes in
        ``[-qmax, qmax]``. Matches the original implementation.
      * ``False`` (2-3 bits) — asymmetric min+scale, all ``2^n`` codes used.
        A symmetric 2-bit grid has only three usable levels {-s, 0, +s} pinned
        to the group max and destroys the weight distribution; K-quants at
        2 bits (Q2_K, IQ2_*) are asymmetric for the same reason. The range is
        widened to include 0 so dead/zero weights stay exactly representable.
    """
    out_features, in_features = W.shape
    if H.shape != (in_features, in_features):
        raise ValueError(f"H shape {tuple(H.shape)} does not match W in_features={in_features}")

    W = W.clone().float()
    H = H.clone().float()
    W_orig = W.clone()  # preserved across in-place mutations for the Frobenius stat

    # 1) Dead-channel handling: channels the corpus never exercised get
    #    H-diag 1.0 — H stays positive definite and their zero off-diagonals
    #    mean no compensation flows through them, so the sweep plain-RTNs
    #    them on the group grid. (Zeroing the weights instead would destroy
    #    rare-token paths the calibration corpus simply never touched.)
    diag = torch.diag(H)
    dead = diag == 0
    if dead.any():
        H[dead, dead] = 1.0
    diag_mean = float(torch.diag(H).mean().item())

    # 2) Act-order permutation (the damping added below is diagonal, so the
    #    permutation commutes with it).
    if actorder:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
        invperm = torch.argsort(perm)
    else:
        invperm = None

    # 3) Upper Cholesky of H^{-1} under escalating Tikhonov damping. Low-bit
    #    Hessians are often near-singular; rather than dying on the first
    #    factorization failure, retry at 2.5x the damping (an accuracy hit,
    #    but far better than skipping the tensor). At 2-3 bits the chain runs
    #    in float64 on CPU — rounding errors there are ~an order of magnitude
    #    larger, so U's precision matters most exactly where H is worst —
    #    while 4-bit+ keeps the proven fp32 path (half the LAPACK cost and
    #    fp64 peak memory on multi-GB ffn_down Hessians). apply() rounds on
    #    CPU by default; MPS (which lacks fp64) never selects it.
    linalg_dtype = (
        torch.float64 if H.device.type == "cpu" and n_bits < 4 else torch.float32
    )
    damp = dampen
    U: torch.Tensor | None = None
    for attempt in range(5):
        # One working copy per attempt; the explicit dels below cap the number
        # of simultaneously-live [in, in] matrices at two (multi-GB in fp64
        # for large ffn_down tensors) — don't "clean them up".
        Hd = H.to(linalg_dtype) if linalg_dtype != H.dtype else H.clone()
        Hd.diagonal().add_(damp * diag_mean)
        L, info = torch.linalg.cholesky_ex(Hd)
        del Hd
        if int(info) == 0:
            Hinv = torch.cholesky_inverse(L)
            del L
            Uc, info2 = torch.linalg.cholesky_ex(Hinv, upper=True)
            del Hinv
            if int(info2) == 0:
                U = Uc.to(W.dtype)
                break
        if attempt < 4:
            # The 1e-4 floor bootstraps escalation when dampen=0 (0 x 2.5
            # would never escalate); it is far below any damping that matters.
            damp = max(damp * 2.5, 1e-4)
            print(
                f"[gptq] cholesky failed (attempt {attempt + 1}); "
                f"escalating dampen -> {damp:.4g}",
                file=sys.stderr,
            )
    if U is None:
        raise RuntimeError(
            f"Cholesky failed after damping escalation to {damp:.4g}; "
            f"Hessian too ill-conditioned"
        )

    # 4) Column-wise sweep in blocks.
    if sym is None:
        sym = n_bits >= 4
    Q = torch.zeros_like(W)
    qmax = 2 ** (n_bits - 1) - 1   # symmetric:  INT4 -> 7, INT3 -> 3, INT2 -> 1
    q_levels = 2**n_bits - 1       # asymmetric: INT4 -> 15, INT3 -> 7, INT2 -> 3
    total_sse = 0.0

    for i1 in range(0, in_features, group_size):
        i2 = min(i1 + group_size, in_features)
        gs = i2 - i1
        Wg = W[:, i1:i2].clone()
        Ug = U[i1:i2, i1:i2]
        Qg = torch.zeros_like(Wg)
        Eg = torch.zeros_like(Wg)

        if sym:
            max_abs = Wg.abs().amax(dim=1).clamp_min(1e-8)
            scale = max_abs / qmax  # per-row, per-group
            zero = torch.zeros_like(scale)
        else:
            wmin = Wg.amin(dim=1).clamp_max(0.0)  # widen to include 0 so dead
            wmax = Wg.amax(dim=1).clamp_min(0.0)  # weights stay representable
            scale = ((wmax - wmin) / q_levels).clamp_min(1e-8)
            zero = torch.round(-wmin / scale)

        for j in range(gs):
            w = Wg[:, j]
            d = Ug[j, j]
            if sym:
                q = torch.round(w / scale).clamp(-qmax, qmax) * scale
            else:
                q = (torch.round(w / scale + zero).clamp(0, q_levels) - zero) * scale
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
        dampen_used=damp,
    )


def grid_for_quant_type(quant_type: str) -> dict[str, int | bool]:
    """Map a llama-quantize type tag to the GPTQ grid that matches its shape.

    Low-bit K-quants use 16-element inner blocks with asymmetric (min+scale)
    grids; rounding toward the same shape keeps GPTQ's error compensation
    aligned with what llama-quantize will re-snap to. Q6_K is symmetric per-16
    in llama.cpp; Q5_K and 4-bit keep the pragmatic symmetric g32 default.
    """
    qt = quant_type.upper()
    if qt.startswith(("IQ1", "IQ2", "Q2")):
        return {"n_bits": 2, "group_size": 16, "sym": False}
    if qt.startswith(("IQ3", "Q3")):
        return {"n_bits": 3, "group_size": 16, "sym": False}
    if qt.startswith("Q6"):
        return {"n_bits": 6, "group_size": 16, "sym": True}
    if qt.startswith("Q5"):
        return {"n_bits": 5, "group_size": 32, "sym": True}
    return {"n_bits": 4, "group_size": 32, "sym": True}


def grid_for_member(
    grid_mix: str,
    member: str,
    *,
    gqa_ge4: bool,
    n_layers: int | None,
    base_grid: dict[str, int | bool],
) -> dict[str, int | bool]:
    """Rounding grid for one tensor under a mixed ftype.

    Mixed ftypes (Q2_K, IQ2_M, IQ3_M, Q4_K_M, …) store some tensors above the
    base grid — llama-quantize bumps e.g. attn_v to Q4_K at 2 bits under
    GQA ≥ 4 (see ``_quant_mix.target_type_for_member``). Rounding those
    tensors on the base grid injects error the real quantization never makes
    and burns compensation budget, so round each on the grid matching its
    *real* target type instead. Members without an override keep
    ``base_grid``. Pure ftypes (Q3_K_S, IQ3_S without GQA) resolve to zero
    overrides.
    """
    target = target_type_for_member(
        grid_mix, member, gqa_ge4=gqa_ge4, n_layers=n_layers)
    return base_grid if target is None else grid_for_quant_type(target)


# --- Calibration: accumulate Hessians ----------------------------------- #

def _resume_key(tokens: int, ctx: int, calibration_text: Path) -> str:
    """Fingerprint of the calibration inputs a saved Hessian depends on.

    Keys the ``_complete`` slice ledger so a resumed multi-slice run never
    mixes Hessians accumulated over different corpora or token budgets.
    """
    h = hashlib.sha256(f"{tokens}|{ctx}|".encode())
    h.update(calibration_text.read_bytes())
    return h.hexdigest()[:16]


def _layer_slices(
    targets: list[tuple[str, torch.nn.Linear]], layers_per_pass: int,
) -> list[list[tuple[str, torch.nn.Linear]]]:
    """Partition targets into slices of ``layers_per_pass`` consecutive layers.

    ``layers_per_pass <= 0`` returns everything as one slice (single-pass
    behavior). Targets without a layer index in their name ride with the
    first slice.
    """
    if layers_per_pass <= 0:
        return [targets] if targets else []
    by_layer: dict[int, list[tuple[str, torch.nn.Linear]]] = {}
    stray: list[tuple[str, torch.nn.Linear]] = []
    for name, mod in targets:
        idx = layer_index(name)
        if idx is None:
            stray.append((name, mod))
        else:
            by_layer.setdefault(idx, []).append((name, mod))
    slices: list[list[tuple[str, torch.nn.Linear]]] = []
    layers = sorted(by_layer)
    for i in range(0, len(layers), layers_per_pass):
        block: list[tuple[str, torch.nn.Linear]] = []
        for li in layers[i:i + layers_per_pass]:
            block.extend(by_layer[li])
        slices.append(block)
    if stray:
        if slices:
            slices[0].extend(stray)
        else:
            slices.append(stray)
    return slices


@torch.no_grad()
def calibrate(
    model_dir: Path,
    calibration_text: Path,
    hessians_dir: Path,
    *,
    tokens: int = 32_768,
    ctx: int = 2_048,
    device: str = "auto",
    dtype: str = "bfloat16",
    save_every: int = 8,
    hessian_device: str = "auto",
    layers_per_pass: int = 0,
) -> Path:
    """Accumulate per-tensor Hessians over the calibration corpus.

    Snapshots the in-flight Hessians to ``hessians_dir`` every ``save_every``
    chunks (atomic via tmp + rename), so the run can be resumed after a crash.

    ``hessian_device`` controls where the ``[in, in]`` accumulators live:
    ``"auto"`` keeps them on the model device under MPS (unified memory —
    same pool either way, and it avoids an H-sized device→host copy inside
    every hook call) and on CPU otherwise (CUDA VRAM safety). Snapshots are
    always persisted as CPU tensors.

    ``layers_per_pass > 0`` bounds peak Hessian memory: targets are split
    into slices of N layers and the corpus is forwarded once per slice, with
    only that slice's Hessians allocated. ffn_down Hessians are
    ``[ffn_dim, ffn_dim]`` fp32 — holding every layer's at once is GBs on
    large models. Completed slices are recorded in ``_complete`` and skipped
    on re-run (mid-slice snapshots alone don't mark completion). The ledger
    is keyed to a fingerprint of (tokens, ctx, corpus bytes): change any of
    them and every slice re-accumulates instead of silently mixing Hessians
    from two configurations.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(device)
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    h_dev = (
        (device if device == "mps" else "cpu")
        if hessian_device == "auto" else hessian_device
    )

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

    # Strided sample across the WHOLE corpus (see calibrate._ingest) — a
    # head-truncated Hessian never sees the tool-call windows. Tokenized once,
    # shared by every slice pass.
    chunks = load_chunks(tok, calibration_text, ctx=ctx, budget_tokens=tokens,
                         log_tag="gptq")

    # Slice-resume ledger: only meaningful for multi-slice runs (a single
    # pass always re-accumulates, as before). The header keys the ledger to
    # the calibration inputs — a completed slice is only reusable if the
    # Hessians were accumulated over the same corpus and token budget.
    slices = _layer_slices(targets, layers_per_pass)
    multi = len(slices) > 1
    complete_path = hessians_dir / "_complete"
    done_names: set[str] = set()
    if multi:
        header = f"#key={_resume_key(tokens, ctx, calibration_text)}"
        if complete_path.exists():
            lines = complete_path.read_text().split()
            if lines and lines[0] == header:
                done_names = set(lines[1:])
            else:
                print(
                    "[gptq] _complete ledger is from different calibration "
                    "inputs; discarding (all slices re-accumulate)",
                    file=sys.stderr,
                )
                complete_path.unlink()
        if not complete_path.exists():
            complete_path.write_text(header + "\n")

    def run_slice(sl: list[tuple[str, torch.nn.Linear]], tag: str) -> None:
        H: dict[str, torch.Tensor] = {}
        n_seen: dict[str, int] = {}
        for name, mod in sl:
            d = mod.in_features
            H[name] = torch.zeros((d, d), dtype=torch.float32, device=h_dev)
            n_seen[name] = 0

        def make_hook(name: str):
            def pre(_module, in_args):
                x = in_args[0].reshape(-1, in_args[0].shape[-1]).float()
                H[name].add_((x.T @ x).detach().to(H[name].device))
                n_seen[name] += x.shape[0]
            return pre

        handles = [mod.register_forward_pre_hook(make_hook(n)) for n, mod in sl]

        def snapshot() -> None:
            for name in H:
                Hessian(name=name, H=H[name].to("cpu"), n_seen=n_seen[name]).save(
                    hessians_dir / f"{_safe_name(name)}.pt"
                )
            print(f"[gptq] snapshot {len(H)} files -> {hessians_dir}", file=sys.stderr)

        try:
            for i, chunk in enumerate(chunks):
                forward_no_logits(model, chunk.unsqueeze(0).to(device))
                if (i + 1) % save_every == 0:
                    print(f"  {tag} chunk {i + 1}/{len(chunks)}", file=sys.stderr)
                    snapshot()
        finally:
            for hd in handles:
                hd.remove()
        snapshot()

    for si, sl in enumerate(slices):
        tag = f"slice {si + 1}/{len(slices)}"
        if multi and all(_safe_name(n) in done_names for n, _ in sl):
            print(f"[gptq] {tag}: {len(sl)} Hessians already complete; skip",
                  file=sys.stderr)
            continue
        run_slice(sl, tag)
        if multi:
            with complete_path.open("a") as fh:
                for name, _ in sl:
                    fh.write(_safe_name(name) + "\n")
            if device == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()
            elif device.startswith("cuda"):
                torch.cuda.empty_cache()

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
    sym: bool | None = None,
    grid_mix: str | None = None,
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
    Note the default targets 4-bit; at 2-3 bits substantially larger drift is
    expected and legitimate — the pipeline relaxes it per ``n_bits``.

    ``sym=None`` auto-selects per :func:`gptq_round_tensor` (asymmetric below
    4 bits). ``device`` stays ``"cpu"`` by default: the Cholesky path is
    numerically safest there and rounding is not the bottleneck.

    ``grid_mix`` (a llama-quantize ftype, e.g. ``"Q2_K"``) rounds each tensor
    on the grid matching its *real* per-tensor target type under that ftype's
    mix (see :func:`grid_for_member`) instead of the uniform
    ``n_bits``/``group_size``/``sym`` grid. The pipeline defaults it to
    ``quantize.type`` unless the recipe pins the base grid.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(device)
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

    # Per-tensor grid overrides from the target ftype's tensor mix. Tensors
    # without an override keep the uniform n_bits/group_size/sym grid.
    base_grid: dict[str, int | bool] = {
        "n_bits": n_bits, "group_size": group_size, "sym": sym,  # type: ignore[dict-item]
    }
    member_grids: dict[str, dict[str, int | bool]] = {}
    if grid_mix is not None:
        gqa_ge4 = gqa_or_moe_ge4(model.config)
        n_layers = getattr(model.config, "num_hidden_layers", None)
        if not n_layers:
            print(
                f"[gptq.apply] WARN: model config lacks num_hidden_layers; "
                f"grid_mix={grid_mix} layer-scheduled bumps (first-eighth, "
                f"use_more_bits) are disabled — only static overrides apply",
                file=sys.stderr,
            )
        counts: dict[str, int] = {}
        for hpath in h_files:
            member = hpath.stem.replace("__", ".")
            grid = grid_for_member(
                grid_mix, member, gqa_ge4=gqa_ge4, n_layers=n_layers,
                base_grid=base_grid)
            if grid is not base_grid:
                member_grids[member] = grid
                target = target_type_for_member(
                    grid_mix, member, gqa_ge4=gqa_ge4, n_layers=n_layers)
                counts[str(target)] = counts.get(str(target), 0) + 1
        summary = ", ".join(f"{k}×{v}" for k, v in sorted(counts.items())) or "none"
        print(f"[gptq.apply] grid_mix={grid_mix}: member overrides {summary} "
              f"(gqa_or_moe_ge4={gqa_ge4})", file=sys.stderr)

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
        grid = member_grids.get(h.name, base_grid)
        try:
            Q, stats = gptq_round_tensor(
                W, h.H, dampen=dampen, actorder=actorder, **grid,  # type: ignore[arg-type]
            )
        except Exception as e:
            print(f"[gptq.apply] WARN: GPTQ failed on {h.name}: {e}", file=sys.stderr)
            continue
        mod.weight.data.copy_(Q.to(torch_dtype).to(mod.weight.device))
        if stats.dampen_used > dampen:
            print(
                f"[gptq.apply] {h.name}: dampen escalated {dampen:.4g} -> "
                f"{stats.dampen_used:.4g}",
                file=sys.stderr,
            )
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
    "grid_for_member",
    "grid_for_quant_type",
    "verify_perplexity",
]
