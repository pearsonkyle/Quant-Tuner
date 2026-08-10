"""Query-chunked SDPA — the lever that lifts the MPS training-window ceiling.

torch has **no MPS training kernel for SDPA** (the fused paths are inference-only), so a
training forward materializes the full ``[B, n_heads, S, S]`` score tensor, and MPSGraph
refuses any tensor with more than ``INT_MAX`` elements. At 32 heads that caps the window
at ``S <= 8191`` (32·8192² == 2³¹ exactly — 8192 fails by *one element*).

Chunking the **query** dimension removes the cap: block ``i`` computes scores of shape
``[B, n_heads, chunk, kv_len]``, which stays far under INT_MAX for any window we care
about. Causality is preserved exactly — block ``i`` attends to keys ``[0 : i+chunk]`` under
a ``tril(diagonal=i)`` mask — and the result is **bit-identical** to
``is_causal=True`` (unit-tested: max abs err 0.0), because softmax is computed per query
row and query rows are independent.

Cost is a Python loop of ``ceil(S/chunk)`` SDPA calls per attention layer; slicing K/V to
``[0 : i+chunk]`` also skips the strictly-masked upper triangle, so the arithmetic is the
same work the unchunked call would have done. Measured fwd+bwd for one 32-head layer on an
M4 Max: 8064 → 0.5 s, 12288 → 1.0 s, 16128 → 1.6 s, 20480 → 3.2 s.

Usage is a single opt-in call before the model is built::

    from quant_tuner.qat.attention import enable_chunked_sdpa
    enable_chunked_sdpa()

This **replaces the registered ``"sdpa"`` implementation in place** rather than registering
a new name on purpose: transformers' mask-creation fast path keys off the string ``"sdpa"``
to return ``attention_mask=None`` for a plain causal decoder. Under a custom name it would
instead build an eager float mask — ``[1, 1, 16128, 16128]`` fp32 is 1 GB of mask, which
costs more than the problem being solved.
"""

from __future__ import annotations

import torch

#: Largest score tensor MPSGraph will accept, minus headroom for the mask.
_INT_MAX = 2**31 - 1
#: Cap on the query block. Smaller = less peak memory, more Python-loop overhead.
DEFAULT_CHUNK = 2048
#: Below this window length the unchunked kernel is used verbatim (no behavior change).
CHUNK_ABOVE = 4096

_original_sdpa = None


def safe_chunk(n_heads: int, kv_len: int, hint: int = DEFAULT_CHUNK) -> int:
    """Largest query block with ``n_heads * chunk * kv_len < INT_MAX``, capped at ``hint``."""
    limit = _INT_MAX // max(1, n_heads * kv_len)
    return max(128, min(hint, limit))


def chunked_causal_sdpa(query, key, value, *, dropout: float = 0.0,
                        scaling: float | None = None, attention_mask=None,
                        chunk_hint: int = DEFAULT_CHUNK, **sdpa_kwargs):
    """Causal SDPA computed in query blocks. Bit-identical to the unchunked call."""
    S = query.shape[2]
    n_heads = query.shape[1]
    chunk = safe_chunk(n_heads, S, chunk_hint)
    outs = []
    for i in range(0, S, chunk):
        j = min(i + chunk, S)
        if attention_mask is None:
            # Causal: block i only ever attends to keys < j, so slice K/V and mask the
            # block-diagonal. tril(diagonal=i) is the causal pattern for absolute rows i..j.
            mask = torch.ones(j - i, j, dtype=torch.bool, device=query.device).tril(diagonal=i)
            k_c, v_c = key[:, :, :j], value[:, :, :j]
        else:
            # A caller-supplied mask may be non-causal (padding, packing); keep all of K/V
            # and slice only the query rows out of it.
            mask = attention_mask[..., i:j, :]
            k_c, v_c = key, value
        outs.append(torch.nn.functional.scaled_dot_product_attention(
            query[:, :, i:j], k_c, v_c, attn_mask=mask, dropout_p=dropout,
            scale=scaling, is_causal=False, **sdpa_kwargs))
    return torch.cat(outs, dim=2)


def enable_chunked_sdpa(chunk_hint: int = DEFAULT_CHUNK,
                        chunk_above: int = CHUNK_ABOVE) -> None:
    """Patch transformers' ``sdpa`` attention to chunk long windows. Idempotent."""
    global _original_sdpa
    from transformers.integrations import sdpa_attention as _sdpa_mod
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if _original_sdpa is not None:
        return
    _original_sdpa = _sdpa_mod.sdpa_attention_forward

    def forward(module, query, key, value, attention_mask=None, dropout: float = 0.0,
                scaling: float | None = None, is_causal: bool | None = None, **kwargs):
        q_len = query.shape[2]
        # Decode steps, short windows and non-causal modules keep the stock kernel, so
        # nothing outside long-window training changes behavior.
        if q_len <= chunk_above or not getattr(module, "is_causal", True):
            return _original_sdpa(module, query, key, value, attention_mask,
                                  dropout=dropout, scaling=scaling,
                                  is_causal=is_causal, **kwargs)
        sdpa_kwargs = {}
        if hasattr(module, "num_key_value_groups"):
            if _sdpa_mod.use_gqa_in_sdpa(attention_mask, key):
                sdpa_kwargs = {"enable_gqa": True}
            else:
                key = _sdpa_mod.repeat_kv(key, module.num_key_value_groups)
                value = _sdpa_mod.repeat_kv(value, module.num_key_value_groups)
        out = chunked_causal_sdpa(query, key, value, dropout=dropout, scaling=scaling,
                                  attention_mask=attention_mask, chunk_hint=chunk_hint,
                                  **sdpa_kwargs)
        return out.transpose(1, 2).contiguous(), None

    _sdpa_mod.sdpa_attention_forward = forward
    ALL_ATTENTION_FUNCTIONS["sdpa"] = forward
    # Keep the "sdpa" mask fast path (returns None for a plain causal decoder). Without
    # this the model would build a [1, 1, S, S] float mask — 1 GB at S=16128.
    assert "sdpa" in ALL_MASK_ATTENTION_FUNCTIONS, (
        "transformers no longer registers an 'sdpa' mask function; re-check that a causal "
        "decoder still gets attention_mask=None before trusting the long-window path")


def disable_chunked_sdpa() -> None:
    """Restore the stock kernel (tests)."""
    global _original_sdpa
    if _original_sdpa is None:
        return
    from transformers.integrations import sdpa_attention as _sdpa_mod
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    _sdpa_mod.sdpa_attention_forward = _original_sdpa
    ALL_ATTENTION_FUNCTIONS["sdpa"] = _original_sdpa
    _original_sdpa = None
