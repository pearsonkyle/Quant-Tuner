"""Straight-through-estimator ternary quantization for continued QAT.

Ternary-Bonsai-8B is a natively-ternary (1.58-bit) Qwen3-8B: every linear weight
is ``w = s_g * c`` with ``c in {-1, 0, +1}`` and one fp16 scale ``s_g`` per group of
128 input channels (llama.cpp Q2_0 g128). Post-hoc calibration can't improve it
(the model is a lossless quant of itself — see ``ternary_calibration_experiments.md``);
the only lever is continued training with the ternarization *in the loop*.

This module provides that: a per-group **TWN** (Ternary Weight Networks, Li & Liu
2016) ternarizer with a straight-through estimator, so a fine-tune can move the
fp16 *latent* weights while the forward always sees the ternary projection.

Why TWN and not BitNet-b1.58 absmean: we require the quantizer to reproduce the
SHIPPED weights EXACTLY at step 0 (initialise latents from the fp16 checkpoint ->
forward must equal the original model). For a group whose nonzeros all share one
magnitude ``s`` (which is exactly what the shipped ternary weights are):

    mean(|w|) = s * frac         (frac = nonzero fraction)
    Delta     = 0.7 * mean(|w|)  = 0.7 * frac * s   < s   -> keeps every nonzero
    scale     = mean(|w| over kept) = s             -> exact
    w_hat     = scale * sign(w) * mask = s * c       = w   (bit-exact)

BitNet's ``round(w/mean|w|)`` would instead rescale by ``frac`` and NOT reproduce
the model. TWN's threshold+retained-mean form is the one that round-trips.
"""

from __future__ import annotations

import torch

DEFAULT_GROUP_SIZE = 128
# TWN threshold factor: Delta = TWN_THRESH * mean(|W|) per group. 0.7 is the Li&Liu
# value; it comfortably keeps a single-magnitude group's nonzeros (frac<1 => Delta<s).
TWN_THRESH = 0.7


def ternarize_group(
    W: torch.Tensor,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    thresh: float = TWN_THRESH,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-group TWN ternarization of ``W`` ([out, in]).

    Returns ``(codes, scale, w_hat)`` where ``codes`` in {-1,0,+1} ([out,in]),
    ``scale`` is per-group ([out, n_groups]), and ``w_hat = scale_g * codes`` is
    the dequantized ternary weight ([out, in]). ``in`` must be divisible by
    ``group_size`` (true for every Qwen3-8B linear: 4096/12288 are multiples of 128).

    The group scale is quantized to **fp16** before use: the deployed artifact
    stores one fp16 scale per group (F16 GGUF -> Q2_0), so the training forward
    must see the same numerics or a tiny train/serve gap opens after training.
    This is exact at step 0 — the shipped scales are fp16-native, and the fp32
    mean of N identical fp16 values round-trips through fp16 unchanged. Under a
    bf16 ``W`` the fp16 scale is re-rounded to bf16 (the closest bf16
    approximation of the deployed value); the bf16-latents path remains
    unsupported for training regardless (threshold underflow).
    """
    out, inn = W.shape
    if inn % group_size != 0:
        raise ValueError(f"in_features {inn} not divisible by group_size {group_size}")
    ng = inn // group_size
    Wg = W.reshape(out, ng, group_size)
    absW = Wg.abs()
    delta = thresh * absW.mean(dim=2, keepdim=True)          # [out, ng, 1]
    mask = (absW > delta).to(W.dtype)                        # kept positions
    cnt = mask.sum(dim=2, keepdim=True).clamp_min(1.0)
    scale = (absW * mask).sum(dim=2, keepdim=True) / cnt     # mean of kept magnitudes
    scale = scale.to(torch.float16).to(W.dtype)              # deployed Q2_0 numerics
    codes = torch.sign(Wg) * mask                            # {-1,0,+1}
    w_hat = (scale * codes).reshape(out, inn)
    return codes.reshape(out, inn), scale.squeeze(-1), w_hat


class _TernarizeSTE(torch.autograd.Function):
    """Forward = per-group TWN dequantized weight; backward = identity (STE)."""

    @staticmethod
    def forward(ctx, W: torch.Tensor, group_size: int, thresh: float) -> torch.Tensor:  # type: ignore[override]
        _, _, w_hat = ternarize_group(W, group_size=group_size, thresh=thresh)
        return w_hat

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        # Straight-through: pass the gradient to the latent weight unchanged.
        return grad_out, None, None


def ternarize_ste(
    W: torch.Tensor,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    thresh: float = TWN_THRESH,
) -> torch.Tensor:
    """Differentiable per-group ternarization (STE). Use in a Linear's forward."""
    return _TernarizeSTE.apply(W, group_size, thresh)


class TernaryLinear(torch.nn.Module):
    """Wraps an ``nn.Linear`` so its forward uses the STE-ternarized weight.

    The original ``nn.Linear`` (its ``weight`` = the fp16 *latent*, ``bias`` if any)
    is held as a submodule and remains the trainable parameter; the ternary
    projection is recomputed each forward. Setting ``trainable=False`` freezes the
    latent (still ternarized in the forward, but no grad / no optimizer state) —
    the memory-frugal knob for partial-layer QAT on Metal.
    """

    def __init__(
        self,
        linear: torch.nn.Linear,
        *,
        group_size: int = DEFAULT_GROUP_SIZE,
        thresh: float = TWN_THRESH,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.group_size = group_size
        self.thresh = thresh
        self.linear.weight.requires_grad_(trainable)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_hat = ternarize_ste(self.linear.weight, group_size=self.group_size, thresh=self.thresh)
        return torch.nn.functional.linear(x, w_hat, self.linear.bias)

    @property
    def weight(self) -> torch.nn.Parameter:  # convenience for export
        return self.linear.weight
