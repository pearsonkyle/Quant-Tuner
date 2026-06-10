"""Shared helpers for HF-model calibration forward passes."""

from __future__ import annotations


def forward_no_logits(model, input_ids) -> None:
    """Run a forward pass for hook-based stat collection, skipping the LM head.

    Calibration passes only need activations at the hooked Linears — the
    logits are discarded, yet ``model(ids)`` materializes a
    ``[1, ctx, vocab]`` float tensor (several GB at ctx 4096 on a 150k-vocab
    model) and pays the lm_head matmul for every chunk. Calling the decoder
    trunk (``model.model``) directly avoids both. Falls back to the full
    model for architectures without the standard ``.model.layers`` shape.
    """
    base = getattr(model, "model", None)
    if base is not None and hasattr(base, "layers"):
        base(input_ids=input_ids)
    else:
        model(input_ids)
