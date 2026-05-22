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
from dataclasses import dataclass, field
from pathlib import Path

import torch

# --- Group discovery ------------------------------------------------------ #

@dataclass(frozen=True)
class ScaleGroup:
    """One AWQ scale group: a set of linears sharing an input + the norm before it."""

    group_id: str            # e.g. "L3_attn", "L3_mlp"
    anchor: str              # dotted module path; first member, used for the hook
    members: tuple[str, ...] # dotted module paths to all sharing linears
    prev_norm: str           # dotted module path to the RMSNorm whose γ we fold


def discover_groups(model) -> list[ScaleGroup]:
    """Find attention + MLP scale groups in a HuggingFace decoder model."""
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
            norm = getattr(layer, "post_attention_layernorm", None)
            if norm is not None and hasattr(norm, "weight"):
                out.append(ScaleGroup(
                    group_id=f"L{i}_mlp",
                    anchor=f"{prefix}.mlp.gate_proj",
                    members=tuple(members),
                    prev_norm=f"{prefix}.post_attention_layernorm",
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


def proxy_loss(W: torch.Tensor, X: torch.Tensor, scale: torch.Tensor) -> float:
    """Sum of squared output deltas between fake-quantized scaled W and the original.

    ``X`` is ``[T, in]`` (one chunk of cached activations); the loss is
    ``|| (Q(W · diag(scale)) · diag(1/scale)) X^T − W X^T ||^2`` summed over
    all elements.
    """
    inv = 1.0 / scale
    Wq_back = fake_quant_int4_g128(W * scale.unsqueeze(0)) * inv.unsqueeze(0)
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
    prev_norm: str
    scale: torch.Tensor  # 1-D, length = input-channel count, float32
    alpha: float


@dataclass
class ScaleBundle:
    groups: list[GroupScale] = field(default_factory=list)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "groups": [
                    {
                        "group_id": g.group_id,
                        "anchor": g.anchor,
                        "members": list(g.members),
                        "prev_norm": g.prev_norm,
                        "scale": g.scale.to(torch.float32),
                        "alpha": float(g.alpha),
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
        return cls(
            groups=[
                GroupScale(
                    group_id=g["group_id"],
                    anchor=g["anchor"],
                    members=tuple(g["members"]),
                    prev_norm=g["prev_norm"],
                    scale=g["scale"].float(),
                    alpha=float(g["alpha"]),
                )
                for g in raw["groups"]
            ]
        )


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
) -> Path:
    """Calibrate per-group AWQ scales and write them to ``out_path``.

    ``force_alpha`` skips the grid search and uses one ``α`` for every group
    (use ``0.5`` to reproduce the original AWQ paper's setting).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]

    print(f"[awq] load {model_dir} -> {device}/{dtype}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    groups = discover_groups(model)
    n_attn = sum(1 for g in groups if g.group_id.endswith("_attn"))
    n_mlp = sum(1 for g in groups if g.group_id.endswith("_mlp"))
    print(f"[awq] groups: {len(groups)} total ({n_attn} attn, {n_mlp} mlp)", file=sys.stderr)
    if not groups:
        raise RuntimeError("no AWQ scale groups discovered")

    sum_abs: dict[str, torch.Tensor | None] = {g.group_id: None for g in groups}
    tok_count: dict[str, int] = {g.group_id: 0 for g in groups}
    cached_X: dict[str, torch.Tensor | None] = {g.group_id: None for g in groups}

    def make_hook(gid: str):
        def pre(_module, in_args):
            x_flat = in_args[0].reshape(-1, in_args[0].shape[-1])
            abs_sum = x_flat.float().abs().sum(dim=0).detach().cpu()
            sum_abs[gid] = abs_sum if sum_abs[gid] is None else sum_abs[gid] + abs_sum
            tok_count[gid] += x_flat.shape[0]
            if cached_X[gid] is None:
                n = min(proxy_tokens, x_flat.shape[0])
                cached_X[gid] = x_flat[:n].detach().float().cpu()
        return pre

    handles = [
        _get_module(model, g.anchor).register_forward_pre_hook(make_hook(g.group_id))
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

    bundle = ScaleBundle()
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
                loss = sum(proxy_loss(W, X, scale_from_alpha(s, a)) for W in Ws)
                if loss < best_loss:
                    best_loss, best_alpha = loss, a
        bundle.groups.append(GroupScale(
            group_id=g.group_id,
            anchor=g.anchor,
            members=g.members,
            prev_norm=g.prev_norm,
            scale=scale_from_alpha(s, best_alpha).to(torch.float32),
            alpha=float(best_alpha),
        ))
        print(f"  {g.group_id:>10s}: α={best_alpha:.2f}", file=sys.stderr)

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

        scaled_members: list[str] = []
        skip_group = False
        for m in g.members:
            W = _get_module(model, m).weight
            if W.shape[1] != scale.shape[0]:
                print(
                    f"[awq.apply] WARN: {m} in={W.shape[1]} vs scale={scale.shape[0]}; "
                    f"skipping group {g.group_id}",
                    file=sys.stderr,
                )
                skip_group = True
                break
            W.data.mul_(scale.unsqueeze(0))
            scaled_members.append(m)
        if skip_group:
            for m in scaled_members:  # revert any partial mutations
                _get_module(model, m).weight.data.mul_((1.0 / scale).unsqueeze(0))
            continue

        norm_w = _get_module(model, g.prev_norm).weight
        if norm_w.shape[0] != scale.shape[0]:
            print(
                f"[awq.apply] WARN: norm {g.prev_norm} shape mismatch; reverting {g.group_id}",
                file=sys.stderr,
            )
            for m in g.members:
                _get_module(model, m).weight.data.mul_((1.0 / scale).unsqueeze(0))
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
    "fold_rmsnorm_gain",
    "proxy_loss",
    "scale_from_alpha",
]
