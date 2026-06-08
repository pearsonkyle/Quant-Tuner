"""Activation-aware weight scaling (AWQ).

Two-stage pipeline:

  1. ``calibrate(...)`` — Discover per-layer **scale groups** (attention
     ``q/k/v[/gate]`` sharing one ``input_layernorm``; MLP ``gate_proj/up_proj``
     sharing ``post_attention_layernorm``). For each group, capture
     ``mean(|x|)`` per input channel from forward pre-hooks, then grid-search
     ``α`` minimising the AWQ proxy loss

         ``|| Q(W · diag(s^α / geomean)) · diag(geomean / s^α) · X − W · X ||²``

     with ``Q =`` fake symmetric INT4 group-128 RTN. Returns a ``ScaleBundle``.

  2. ``apply(...)`` — For each group, multiply every member weight's
     input-channel column by ``scale`` and pre-divide the preceding RMSNorm
     gain by ``scale``. The F16 forward pass is unchanged; the *quantized*
     forward differs, because scaling input channels redistributes which
     channels carry signal magnitude.

The original AWQ paper fixes ``α = 0.5``; this implementation supports both
fixed-``α`` and grid-searched ``α`` per group. The two variants reported in
the experiment leaderboard are ``a050`` (force 0.5) and ``best`` (grid).

The RMSNorm fold has two forms (controlled by ``rmsnorm_plus_one``):
  * ``False`` — Llama/Mistral/Qwen3 standard: ``γ' = γ / scale``.
  * ``True``  — Qwen3.5: ``(1 + γ') = (1 + γ) / scale``.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

# --- Group discovery ------------------------------------------------------ #

@dataclass(frozen=True)
class ScaleGroup:
    """One AWQ scale group: a set of linears sharing an input + the norm before it.

    ``prev_norm`` is ``None`` for output-projection groups (``o_proj``,
    ``down_proj``) whose input is *not* a normalized tensor — there is no γ to
    fold into and the F16 forward will drift by the scale ratio. Used by
    exp-011 to scale the two largest tensor classes that the naive AWQ port
    leaves untouched.
    """

    group_id: str            # e.g. "L3_attn", "L3_mlp", "L3_attn_out", "L3_mlp_out"
    anchor: str              # dotted module path; first member, used for the hook
    members: tuple[str, ...] # dotted module paths to all sharing linears
    prev_norm: str | None    # dotted module path to the RMSNorm whose γ we fold, or None


def discover_groups(model, *, include_output_proj: bool = False) -> list[ScaleGroup]:
    """Find attention + MLP scale groups in a HuggingFace decoder model.

    With ``include_output_proj=True`` (exp-011), also emit per-layer
    ``L{i}_attn_out`` (``o_proj``) and ``L{i}_mlp_out`` (``down_proj``) groups.
    These have ``prev_norm=None`` since their input is not a normalized
    activation — scales apply to the weight only, no RMSNorm fold cancellation.
    """
    layers = getattr(model.model, "layers", None)
    if layers is None:
        raise RuntimeError("model.model.layers not found; unsupported architecture")

    out: list[ScaleGroup] = []
    for i, layer in enumerate(layers):
        prefix = f"model.layers.{i}"

        attn = getattr(layer, "self_attn", None)
        if attn is not None and isinstance(getattr(attn, "q_proj", None), torch.nn.Linear):
            members = tuple(
                f"{prefix}.self_attn.{sub}"
                for sub in ("q_proj", "k_proj", "v_proj", "gate_proj")
                if isinstance(getattr(attn, sub, None), torch.nn.Linear)
            )
            norm = getattr(layer, "input_layernorm", None)
            if norm is not None and hasattr(norm, "weight"):
                out.append(ScaleGroup(
                    group_id=f"L{i}_attn",
                    anchor=f"{prefix}.self_attn.q_proj",
                    members=members,
                    prev_norm=f"{prefix}.input_layernorm",
                ))

        mlp = getattr(layer, "mlp", None)
        if mlp is not None and isinstance(getattr(mlp, "gate_proj", None), torch.nn.Linear):
            members = [f"{prefix}.mlp.gate_proj"]
            if isinstance(getattr(mlp, "up_proj", None), torch.nn.Linear):
                members.append(f"{prefix}.mlp.up_proj")
            # Gemma-2/3/4 put `pre_feedforward_layernorm` immediately before the MLP
            # and use `post_attention_layernorm` between attention and residual add.
            # Llama/Mistral/Qwen lack `pre_feedforward_layernorm`; their pre-MLP norm
            # is `post_attention_layernorm`. Detect by presence.
            pre_ffn = getattr(layer, "pre_feedforward_layernorm", None)
            if pre_ffn is not None and hasattr(pre_ffn, "weight"):
                norm, norm_name = pre_ffn, "pre_feedforward_layernorm"
            else:
                norm = getattr(layer, "post_attention_layernorm", None)
                norm_name = "post_attention_layernorm"
            if norm is not None and hasattr(norm, "weight"):
                out.append(ScaleGroup(
                    group_id=f"L{i}_mlp",
                    anchor=f"{prefix}.mlp.gate_proj",
                    members=tuple(members),
                    prev_norm=f"{prefix}.{norm_name}",
                ))

        if include_output_proj:
            if attn is not None and isinstance(getattr(attn, "o_proj", None), torch.nn.Linear):
                out.append(ScaleGroup(
                    group_id=f"L{i}_attn_out",
                    anchor=f"{prefix}.self_attn.o_proj",
                    members=(f"{prefix}.self_attn.o_proj",),
                    prev_norm=None,
                ))
            if mlp is not None and isinstance(getattr(mlp, "down_proj", None), torch.nn.Linear):
                out.append(ScaleGroup(
                    group_id=f"L{i}_mlp_out",
                    anchor=f"{prefix}.mlp.down_proj",
                    members=(f"{prefix}.mlp.down_proj",),
                    prev_norm=None,
                ))
    return out


def _get_module(model, dotted: str):
    obj = model
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


# --- Fake quantization for α search -------------------------------------- #

def fake_quant_int4_g128(W: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Symmetric per-row, per-group INT4 round-to-nearest (proxy for Q4_K_M).

    Not bit-exact to llama.cpp's K-quants — but the channel-importance signal
    AWQ relies on is captured well enough for α selection.
    """
    out_f, in_f = W.shape
    pad = (-in_f) % group_size
    if pad:
        W = torch.nn.functional.pad(W, (0, pad))
    Wg = W.view(out_f, -1, group_size)
    max_abs = Wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = max_abs / 7.0  # signed INT4 range [-7, 7]
    q = torch.round(Wg / scale).clamp(-7, 7)
    Wq = (q * scale).view(out_f, -1)
    return Wq[:, :in_f]


def fake_quant_q2k_block16(W: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """Asymmetric per-16-element-block 2-bit RTN (proxy for Q2_K / IQ2_*).

    Mirrors the *shape* of K-quant error rather than being bit-exact: 16-element
    sub-blocks (the K-quant inner block size), per-block ``(min, scale)`` pair,
    2-bit unsigned levels [0, 3]. The per-block scale and min are themselves
    coarsely re-quantized to ~4-bit (relative to the per-row max) to mimic
    K-quant's nested-quantization noise floor.

    Use this proxy for IQ2_XS / IQ2_M / Q2_K_S α search where the default INT4
    proxy systematically over-rewards channels that would actually be lost to
    block-local quantization.
    """
    out_f, in_f = W.shape
    pad = (-in_f) % block_size
    if pad:
        W = torch.nn.functional.pad(W, (0, pad))
    Wb = W.view(out_f, -1, block_size)
    wmin = Wb.amin(dim=-1, keepdim=True)
    wmax = Wb.amax(dim=-1, keepdim=True)
    scale = ((wmax - wmin) / 3.0).clamp(min=1e-8)

    # Nested quantization: per-row, coarsely re-quantize the (scale, min) pair
    # to mimic Q2_K's 4-bit shared scale/min storage.
    s_range = scale.amax(dim=1, keepdim=True).clamp(min=1e-8)
    m_range = wmin.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = (scale / s_range * 15.0).round().clamp(1, 15) * (s_range / 15.0)
    wmin = (wmin / m_range * 7.0).round().clamp(-7, 7) * (m_range / 7.0)

    q = ((Wb - wmin) / scale).round().clamp(0, 3)
    Wq = q * scale + wmin
    return Wq.view(out_f, -1)[:, :in_f]


# Registry of available proxy quantizers. Keys are recipe-facing names.
_PROXIES: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "int4_g128": fake_quant_int4_g128,
    "q2k_b16": fake_quant_q2k_block16,
}


def proxy_loss(
    W: torch.Tensor,
    X: torch.Tensor,
    scale: torch.Tensor,
    *,
    quantizer: Callable[[torch.Tensor], torch.Tensor] = fake_quant_int4_g128,
) -> float:
    """Sum of squared output deltas between fake-quantized scaled W and the original.

    ``X`` is ``[T, in]`` (one chunk of cached activations); the loss is
    ``|| (Q(W · diag(scale)) · diag(1/scale)) X^T − W X^T ||^2`` summed over
    all elements. ``quantizer`` selects the proxy (default INT4 g128; pass
    ``fake_quant_q2k_block16`` for K-quant targets).
    """
    inv = 1.0 / scale
    Wq_back = quantizer(W * scale.unsqueeze(0)) * inv.unsqueeze(0)
    diff = X @ (Wq_back - W).T
    return float((diff.float() ** 2).sum().item())


def scale_from_alpha(s: torch.Tensor, alpha: float) -> torch.Tensor:
    """Convert mean(|x|) per channel into a scale vector at exponent ``α``.

    ``α = 0`` yields the identity (all-ones); for ``α > 0`` the raw ``s^α``
    values are normalized so their geometric mean is 1.0, which keeps the
    overall weight magnitude roughly constant (a critical property: the
    quantizer's per-row scale would otherwise be dominated by a few channels).
    """
    if alpha == 0.0:
        return torch.ones_like(s)
    raw = s.pow(alpha)
    return (raw / raw.log().mean().exp()).clamp_min(1e-8)


# --- Bundle --------------------------------------------------------------- #

@dataclass
class GroupScale:
    group_id: str
    anchor: str
    members: tuple[str, ...]
    prev_norm: str | None  # None for o_proj / down_proj groups (no norm to fold)
    scale: torch.Tensor  # 1-D, length = input-channel count, float32 (group-level)
    alpha: float
    # Per-member overrides written by exp-013's per-tensor α refinement.
    # When non-None, ``apply`` multiplies each member's columns by its own scale
    # instead of the shared ``scale``; the norm fold still uses the group ``scale``.
    member_alphas: dict[str, float] | None = None
    member_scales: dict[str, torch.Tensor] | None = None


@dataclass
class ScaleBundle:
    groups: list[GroupScale] = field(default_factory=list)
    proxy: str = "int4_g128"  # name of the proxy quantizer used during calibration

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "proxy": self.proxy,
                "groups": [
                    {
                        "group_id": g.group_id,
                        "anchor": g.anchor,
                        "members": list(g.members),
                        "prev_norm": g.prev_norm,
                        "scale": g.scale.to(torch.float32),
                        "alpha": float(g.alpha),
                        "member_alphas": (
                            {k: float(v) for k, v in g.member_alphas.items()}
                            if g.member_alphas is not None else None
                        ),
                        "member_scales": (
                            {k: v.to(torch.float32) for k, v in g.member_scales.items()}
                            if g.member_scales is not None else None
                        ),
                    }
                    for g in self.groups
                ]
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> ScaleBundle:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        groups = []
        for g in raw["groups"]:
            ma = g.get("member_alphas")
            ms = g.get("member_scales")
            groups.append(GroupScale(
                group_id=g["group_id"],
                anchor=g["anchor"],
                members=tuple(g["members"]),
                prev_norm=g["prev_norm"],
                scale=g["scale"].float(),
                alpha=float(g["alpha"]),
                member_alphas=({k: float(v) for k, v in ma.items()} if ma else None),
                member_scales=({k: v.float() for k, v in ms.items()} if ms else None),
            ))
        return cls(groups=groups, proxy=raw.get("proxy", "int4_g128"))


# --- Calibration --------------------------------------------------------- #

def calibrate(
    model_dir: Path,
    calibration_text: Path,
    out_path: Path,
    *,
    tokens: int = 4096,
    ctx: int = 1024,
    device: str = "mps",
    dtype: str = "bfloat16",
    alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    force_alpha: float | None = None,
    proxy_tokens: int = 256,
    include_output_proj: bool = False,
    proxy: str = "int4_g128",
    per_tensor_alpha: bool = False,
    holdout_text: Path | None = None,
    cv_strategy: Literal["off", "gate", "mixed"] = "off",
    cv_weight: float = 1.0,
) -> Path:
    """Calibrate per-group AWQ scales and write them to ``out_path``.

    ``force_alpha`` skips the grid search and uses one ``α`` for every group
    (use ``0.5`` to reproduce the original AWQ paper's setting).

    ``include_output_proj`` (exp-011) additionally scales ``o_proj`` and
    ``down_proj``. ``proxy`` (exp-012) selects the proxy quantizer used inside
    the α search — ``"int4_g128"`` (default) or ``"q2k_b16"`` (asymmetric 2-bit
    per-16-block, for IQ2_*/Q2_K targets). ``per_tensor_alpha`` (exp-013) runs
    a second refinement pass that picks each member's own α from a local grid
    around the group α; the per-member scales are stored alongside the shared
    group scale and applied separately at fold time.

    ``holdout_text`` + ``cv_strategy`` (exp-017/018) add a held-out activation
    chunk to defend against the per-tensor α over-fit observed in exp-016
    (PPL collapses below FP16, KLD diverges, tool-call accuracy drops). The
    held-out text MUST come from a distribution disjoint from the calibration
    corpus — same-source-disjoint-sessions catches memorization. Strategies:
      - ``"off"`` (default): no change to α selection.
      - ``"gate"`` (exp-017): after picking the per-member α by calibration
        proxy loss, accept it only if the held-out proxy loss is no worse
        than the group α's held-out loss. Otherwise revert to group α.
      - ``"mixed"`` (exp-018): score each α candidate as
        ``proxy_loss(X_cal, α) + cv_weight · proxy_loss(X_ho, α)``. ``cv_weight``
        > 1 over-weights the held-out signal to push back against over-fit.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]

    if proxy not in _PROXIES:
        raise ValueError(f"unknown proxy {proxy!r}; choose one of {sorted(_PROXIES)}")
    quantizer = _PROXIES[proxy]
    print(f"[awq] proxy={proxy}", file=sys.stderr)

    print(f"[awq] load {model_dir} -> {device}/{dtype}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    groups = discover_groups(model, include_output_proj=include_output_proj)
    n_attn = sum(1 for g in groups if g.group_id.endswith("_attn"))
    n_mlp = sum(1 for g in groups if g.group_id.endswith("_mlp"))
    print(f"[awq] groups: {len(groups)} total ({n_attn} attn, {n_mlp} mlp)", file=sys.stderr)
    if not groups:
        raise RuntimeError("no AWQ scale groups discovered")

    sum_abs: dict[str, torch.Tensor | None] = {g.group_id: None for g in groups}
    tok_count: dict[str, int] = {g.group_id: 0 for g in groups}
    cached_X: dict[str, torch.Tensor | None] = {g.group_id: None for g in groups}
    cached_X_ho: dict[str, torch.Tensor | None] = {g.group_id: None for g in groups}

    def make_hook(gid: str, target_X: dict[str, torch.Tensor | None],
                  update_sum: bool):
        def pre(_module, in_args):
            x_flat = in_args[0].reshape(-1, in_args[0].shape[-1])
            if update_sum:
                abs_sum = x_flat.float().abs().sum(dim=0).detach().cpu()
                sum_abs[gid] = abs_sum if sum_abs[gid] is None else sum_abs[gid] + abs_sum
                tok_count[gid] += x_flat.shape[0]
            if target_X[gid] is None:
                n = min(proxy_tokens, x_flat.shape[0])
                target_X[gid] = x_flat[:n].detach().float().cpu()
        return pre

    # Calibration-corpus pass: collect mean(|x|) and cache X_cal per group.
    handles = [
        _get_module(model, g.anchor).register_forward_pre_hook(
            make_hook(g.group_id, cached_X, update_sum=True)
        )
        for g in groups
    ]
    try:
        text = calibration_text.read_text()
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0][:tokens]
        chunks = ids.split(ctx)
        print(f"[awq] {ids.shape[0]} tokens, ctx {ctx} -> {len(chunks)} chunks", file=sys.stderr)
        with torch.no_grad():
            for i, chunk in enumerate(chunks):
                if chunk.numel() < 2:
                    continue
                model(chunk.unsqueeze(0).to(device))
                print(f"  chunk {i + 1}/{len(chunks)}", file=sys.stderr)
    finally:
        for h in handles:
            h.remove()

    # Held-out pass (exp-017/018): cache X_ho only. mean(|x|) stays from
    # calibration so the *scale vector itself* is calibration-driven; we use
    # held-out X only to evaluate which α generalizes.
    if cv_strategy != "off" and holdout_text is not None:
        if not per_tensor_alpha:
            print(
                "[awq] WARN: cv_strategy set but per_tensor_alpha=False; cv has no effect",
                file=sys.stderr,
            )
        print(f"[awq] capturing held-out activations for cv_strategy={cv_strategy!r} "
              f"(cv_weight={cv_weight})", file=sys.stderr)
        ho_handles = [
            _get_module(model, g.anchor).register_forward_pre_hook(
                make_hook(g.group_id, cached_X_ho, update_sum=False)
            )
            for g in groups
        ]
        try:
            ho_ids = tok(holdout_text.read_text(), return_tensors="pt",
                         add_special_tokens=False).input_ids[0][:tokens]
            ho_chunks = ho_ids.split(ctx)
            print(f"[awq] held-out {ho_ids.shape[0]} tokens -> {len(ho_chunks)} chunks",
                  file=sys.stderr)
            with torch.no_grad():
                for i, chunk in enumerate(ho_chunks):
                    if chunk.numel() < 2:
                        continue
                    model(chunk.unsqueeze(0).to(device))
                    print(f"  ho chunk {i + 1}/{len(ho_chunks)}", file=sys.stderr)
        finally:
            for h in ho_handles:
                h.remove()
    elif cv_strategy != "off" and holdout_text is None:
        raise ValueError(
            f"cv_strategy={cv_strategy!r} requires holdout_text to be set"
        )

    bundle = ScaleBundle(proxy=proxy)
    for g in groups:
        if sum_abs[g.group_id] is None or tok_count[g.group_id] == 0:
            print(f"[awq] WARN: no activations for {g.group_id}; skipping", file=sys.stderr)
            continue
        s = (sum_abs[g.group_id] / tok_count[g.group_id]).clamp_min(1e-8)
        X = cached_X[g.group_id]
        Ws = [_get_module(model, m).weight.detach().float().cpu() for m in g.members]

        if force_alpha is not None:
            best_alpha = force_alpha
        else:
            best_alpha, best_loss = 0.0, float("inf")
            for a in alphas:
                loss = sum(
                    proxy_loss(W, X, scale_from_alpha(s, a), quantizer=quantizer)
                    for W in Ws
                )
                if loss < best_loss:
                    best_loss, best_alpha = loss, a

        member_alphas: dict[str, float] | None = None
        member_scales: dict[str, torch.Tensor] | None = None
        if per_tensor_alpha and len(g.members) > 1:
            local = sorted({
                max(0.0, min(1.0, best_alpha + d)) for d in (-0.25, 0.0, 0.25)
            })
            X_ho = cached_X_ho.get(g.group_id) if cv_strategy != "off" else None
            member_alphas = {}
            member_scales = {}
            for m, W in zip(g.members, Ws, strict=True):
                # Score every candidate on X_cal (and X_ho if cv_strategy="mixed").
                cal_losses: dict[float, float] = {}
                ho_losses: dict[float, float] = {}
                for a in local:
                    sa = scale_from_alpha(s, a)
                    cal_losses[a] = proxy_loss(W, X, sa, quantizer=quantizer)
                    if X_ho is not None:
                        ho_losses[a] = proxy_loss(W, X_ho, sa, quantizer=quantizer)

                if cv_strategy == "mixed" and X_ho is not None:
                    scores = {
                        a: cal_losses[a] + cv_weight * ho_losses[a] for a in local
                    }
                    m_best_a = min(scores, key=scores.get)
                else:
                    m_best_a = min(cal_losses, key=cal_losses.get)

                # Gate (exp-017): accept the per-member α only if it doesn't
                # worsen the held-out proxy loss vs the group α baseline.
                # Reverts to group_α when over-fit is detected.
                if (cv_strategy == "gate" and X_ho is not None
                        and ho_losses[m_best_a] > ho_losses.get(
                            best_alpha, ho_losses[m_best_a])):
                    m_best_a = best_alpha

                member_alphas[m] = float(m_best_a)
                member_scales[m] = scale_from_alpha(s, m_best_a).to(torch.float32)

        bundle.groups.append(GroupScale(
            group_id=g.group_id,
            anchor=g.anchor,
            members=g.members,
            prev_norm=g.prev_norm,
            scale=scale_from_alpha(s, best_alpha).to(torch.float32),
            alpha=float(best_alpha),
            member_alphas=member_alphas,
            member_scales=member_scales,
        ))
        if member_alphas is None:
            print(f"  {g.group_id:>14s}: α={best_alpha:.2f}", file=sys.stderr)
        else:
            per = " ".join(f"{m.split('.')[-1]}={a:.2f}" for m, a in member_alphas.items())
            print(
                f"  {g.group_id:>14s}: group α={best_alpha:.2f} | {per}",
                file=sys.stderr,
            )

    bundle.save(out_path)

    hist: dict[float, int] = {}
    for gs in bundle.groups:
        hist[gs.alpha] = hist.get(gs.alpha, 0) + 1
    print("[awq] alpha histogram:", file=sys.stderr)
    for a in sorted(hist):
        print(f"   α={a:.2f}  {hist[a]:3d} groups", file=sys.stderr)
    return out_path


# --- Application: fold scales into HF weights ---------------------------- #

def fold_rmsnorm_gain(
    norm_weight: torch.Tensor,
    inv_scale: torch.Tensor,
    *,
    plus_one: bool = True,
) -> torch.Tensor:
    """Compute the new RMSNorm gain that pre-divides its output by ``scale``.

    ``inv_scale`` is ``1 / scale`` (float32). With ``plus_one=True`` (Qwen3.5),
    the norm applies ``(1 + γ)`` post-normalize, so the new gain that achieves
    output ``(norm_out / scale)`` is ``(1 + γ) / scale - 1``. With
    ``plus_one=False`` (Llama et al.), the norm is just ``γ`` and the
    transform is ``γ' = γ / scale``.
    """
    gain_f = norm_weight.float()
    if plus_one:
        return (1.0 + gain_f) * inv_scale - 1.0
    return gain_f * inv_scale


def _apply_out_row_counter_fold(
    model,
    group_id: str,
    scale_f: torch.Tensor,
    torch_dtype: torch.dtype,
) -> None:
    """Counter-fold an exp-014 output-projection scale into the previous linear.

    For ``L{i}_attn_out``: o_proj's input channels (length n_q_heads · head_dim)
    correspond to v_proj's output rows. Under GQA (n_q_heads > n_kv_heads), tie
    the scale across each KV group (geomean over the query heads sharing one
    KV head) so the per-channel factor can be expressed on v_proj's reduced
    row count, then divide v_proj's rows by the tied scale.

    For ``L{i}_mlp_out``: down_proj's input is silu(gate_proj(x)) * up_proj(x).
    Dividing up_proj's output rows by ``s`` makes that input divide cleanly
    (elementwise multiplication commutes with the scale). gate_proj is *not*
    touched because silu is nonlinear and scaling its input would not pass
    through.

    Mutates the previous linear's weight (and bias if present) in-place.
    """
    assert "_attn_out" in group_id or "_mlp_out" in group_id, group_id
    layer_i = int(group_id.split("_")[0][1:])

    if group_id.endswith("_attn_out"):
        cfg = model.config
        n_q = cfg.num_attention_heads
        n_kv = getattr(cfg, "num_key_value_heads", n_q)
        head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_q)
        expected_in = n_q * head_dim
        if scale_f.shape[0] != expected_in:
            raise RuntimeError(
                f"attn_out scale len {scale_f.shape[0]} != n_q·head_dim {expected_in}"
            )
        if n_q == n_kv:
            scale_v = scale_f  # MHA: identity tying
        else:
            if n_q % n_kv != 0:
                raise RuntimeError(f"GQA mismatch: n_q={n_q} not divisible by n_kv={n_kv}")
            group_size = n_q // n_kv
            # (n_q, head_dim) -> (n_kv, group_size, head_dim) -> geomean over query heads
            s_grouped = scale_f.view(n_q, head_dim).view(n_kv, group_size, head_dim)
            scale_v = s_grouped.log().mean(dim=1).exp().reshape(n_kv * head_dim)
            # Replace the original scale_f channels with the tied values (so
            # o_proj's column scaling and v_proj's row scaling cancel exactly).
            tied = (scale_v.view(n_kv, 1, head_dim)
                    .expand(n_kv, group_size, head_dim)
                    .reshape(n_q * head_dim))
            # Edit-in-place via copy_ so the o_proj weight (already multiplied
            # by the *untied* scale_f above) is corrected.
            o_proj_path = f"model.layers.{layer_i}.self_attn.o_proj"
            W_o = _get_module(model, o_proj_path).weight
            correction = (tied / scale_f).to(torch_dtype)
            W_o.data.mul_(correction.unsqueeze(0))

        v_proj_path = f"model.layers.{layer_i}.self_attn.v_proj"
        W_v = _get_module(model, v_proj_path).weight
        if W_v.shape[0] != scale_v.shape[0]:
            raise RuntimeError(
                f"v_proj rows {W_v.shape[0]} != scale_v {scale_v.shape[0]}"
            )
        inv = (1.0 / scale_v).to(torch_dtype)
        W_v.data.mul_(inv.unsqueeze(1))
        bias = getattr(_get_module(model, v_proj_path), "bias", None)
        if bias is not None:
            bias.data.mul_(inv)
        return

    # L{i}_mlp_out: scale down_proj input channels, divide up_proj output rows.
    up_path = f"model.layers.{layer_i}.mlp.up_proj"
    W_up = _get_module(model, up_path).weight
    if W_up.shape[0] != scale_f.shape[0]:
        raise RuntimeError(
            f"up_proj rows {W_up.shape[0]} != scale {scale_f.shape[0]}"
        )
    inv = (1.0 / scale_f).to(torch_dtype)
    W_up.data.mul_(inv.unsqueeze(1))
    bias = getattr(_get_module(model, up_path), "bias", None)
    if bias is not None:
        bias.data.mul_(inv)


@torch.no_grad()
def apply(
    model_dir: Path,
    scales: Path | ScaleBundle,
    out_dir: Path,
    *,
    device: str = "mps",
    dtype: str = "bfloat16",
    rmsnorm_plus_one: bool = True,
    sanity_tokens: int = 32,
    sanity_max_rel: float = 0.03,
) -> Path:
    """Fold scales into an HF model and save it to ``out_dir`` for HF→GGUF conversion.

    The transform is mathematically exact in F32: weight columns multiplied by
    ``s`` cancel against the preceding RMSNorm gain divided by ``s``. In bf16
    the round-trip injects ~0.8% relative drift per layer (mantissa noise);
    ``sanity_max_rel`` rejects anything well above that floor.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    bundle = scales if isinstance(scales, ScaleBundle) else ScaleBundle.load(scales)

    print(f"[awq.apply] load {model_dir} -> {device}/{dtype}", file=sys.stderr)
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

    n_applied = 0
    for g in bundle.groups:
        scale_f = g.scale.to(device).float()
        scale = scale_f.to(torch_dtype)
        inv_f = 1.0 / scale_f

        # Per-member scales (exp-013): each member multiplied by its own scale.
        # The shared ``scale`` is still what gets folded into the RMSNorm gain;
        # the residual ``member_scale / scale`` factor stays in the weight and
        # intentionally perturbs the F16 forward (sanity check bounds drift).
        member_mul: dict[str, torch.Tensor] = {}
        if g.member_scales is not None:
            for m, ms in g.member_scales.items():
                member_mul[m] = ms.to(device).to(torch_dtype)

        scaled_members: list[tuple[str, torch.Tensor]] = []
        skip_group = False
        for m in g.members:
            W = _get_module(model, m).weight
            mul = member_mul.get(m, scale)
            if W.shape[1] != mul.shape[0]:
                print(
                    f"[awq.apply] WARN: {m} in={W.shape[1]} vs scale={mul.shape[0]}; "
                    f"skipping group {g.group_id}",
                    file=sys.stderr,
                )
                skip_group = True
                break
            W.data.mul_(mul.unsqueeze(0))
            scaled_members.append((m, mul))
        if skip_group:
            for m, mul in scaled_members:  # revert any partial mutations
                _get_module(model, m).weight.data.mul_((1.0 / mul).unsqueeze(0))
            continue

        if g.prev_norm is None:
            # Output-projection group (exp-014): counter-fold by dividing the
            # *previous linear's* output rows by the same scale. AWQ paper
            # approach — F16 identity holds because attention(softmax(QK^T)V)
            # is linear in V, and down_proj's input (silu(gate) * up) is
            # elementwise so scaling up's rows by 1/s passes through cleanly.
            try:
                _apply_out_row_counter_fold(
                    model, g.group_id, scale_f, torch_dtype,
                )
                n_applied += 1
            except Exception as e:  # noqa: BLE001 — revert + re-raise as warning
                print(
                    f"[awq.apply] WARN: counter-fold failed for {g.group_id}: {e!r}; "
                    f"reverting member scales",
                    file=sys.stderr,
                )
                for m, mul in scaled_members:
                    _get_module(model, m).weight.data.mul_((1.0 / mul).unsqueeze(0))
            continue

        norm_w = _get_module(model, g.prev_norm).weight
        if norm_w.shape[0] != scale.shape[0]:
            print(
                f"[awq.apply] WARN: norm {g.prev_norm} shape mismatch; reverting {g.group_id}",
                file=sys.stderr,
            )
            for m, mul in scaled_members:
                _get_module(model, m).weight.data.mul_((1.0 / mul).unsqueeze(0))
            continue

        new_gain = fold_rmsnorm_gain(norm_w.data, inv_f, plus_one=rmsnorm_plus_one)
        norm_w.data.copy_(new_gain.to(torch_dtype))
        n_applied += 1

    print(f"[awq.apply] {n_applied}/{len(bundle.groups)} groups folded", file=sys.stderr)

    if ref_logits is not None and sanity_ids is not None:
        new_logits = model(sanity_ids).logits.detach().float().cpu()
        max_abs = (new_logits - ref_logits).abs().max().item()
        rel = max_abs / max(ref_logits.abs().max().item(), 1e-8)
        print(f"[awq.apply] sanity max|Δlogit|={max_abs:.3e} rel={rel:.3e}", file=sys.stderr)
        if rel > sanity_max_rel:
            raise RuntimeError(
                f"AWQ sanity check failed: F16 logit drift {rel:.3e} > {sanity_max_rel:.3e} "
                f"(bf16 noise floor is ~0.02)"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[awq.apply] writing -> {out_dir}", file=sys.stderr)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    for name in ("chat_template.jinja", "generation_config.json"):
        src = model_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
    _preserve_auxiliary_tensors(model_dir, out_dir)
    return out_dir


def _preserve_auxiliary_tensors(src_dir: Path, dst_dir: Path) -> None:
    """Copy tensors present in ``src_dir`` but missing from ``dst_dir``.

    ``AutoModelForCausalLM.from_pretrained`` only loads parameters known to the
    causal-LM class, so auxiliary heads (e.g. MTP layers on Qwen3.6) are dropped
    on ``save_pretrained``. Without this rescue, the post-fold HF→GGUF
    conversion would miss those tensors and the resulting GGUF would fail to
    load (e.g. ``missing tensor 'blk.64.attn_norm.weight'``).
    """
    import json

    src_index = src_dir / "model.safetensors.index.json"
    dst_index = dst_dir / "model.safetensors.index.json"
    if not src_index.exists() or not dst_index.exists():
        return

    from safetensors import safe_open
    from safetensors.torch import save_file

    src_map = json.loads(src_index.read_text())["weight_map"]
    dst_obj = json.loads(dst_index.read_text())
    dst_map = dst_obj["weight_map"]
    missing = sorted(set(src_map) - set(dst_map))
    if not missing:
        return

    print(
        f"[awq.apply] preserving {len(missing)} auxiliary tensors not in causal-LM class",
        file=sys.stderr,
    )
    by_shard: dict[str, list[str]] = {}
    for k in missing:
        by_shard.setdefault(src_map[k], []).append(k)

    aux_shard = "model-aux.safetensors"
    aux_tensors: dict[str, torch.Tensor] = {}
    total_bytes = 0
    for shard, keys in by_shard.items():
        with safe_open(src_dir / shard, framework="pt") as f:
            for k in keys:
                t = f.get_tensor(k)
                aux_tensors[k] = t
                total_bytes += t.numel() * t.element_size()

    save_file(aux_tensors, dst_dir / aux_shard, metadata={"format": "pt"})

    dst_map.update({k: aux_shard for k in aux_tensors})
    dst_obj["weight_map"] = dst_map
    md = dst_obj.setdefault("metadata", {})
    md["total_size"] = int(md.get("total_size", 0)) + total_bytes
    dst_index.write_text(json.dumps(dst_obj, indent=2))


__all__ = [
    "GroupScale",
    "ScaleBundle",
    "ScaleGroup",
    "apply",
    "calibrate",
    "discover_groups",
    "fake_quant_int4_g128",
    "fake_quant_q2k_block16",
    "fold_rmsnorm_gain",
    "proxy_loss",
    "scale_from_alpha",
]
