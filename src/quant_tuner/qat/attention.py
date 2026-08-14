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

import contextlib

import torch

# --- prefix context ---------------------------------------------------------------------
#
# Long-window training needs the first N tokens of a window encoded once, under no_grad, and
# then *attended to* by a gradient-carrying tail. transformers' own `past_key_values` cannot
# do this here: `GradientCheckpointingLayer.__call__` sets `past_key_values = None` whenever
# `gradient_checkpointing and training`, so a checkpointed tail silently attends to nothing.
# Since this module already replaces the attention function, the prefix rides here instead —
# captured per attention module on the no_grad pass, concatenated onto K/V on the tail pass.
# RoPE is already applied by the time we see K, so the stored keys keep their true absolute
# positions; the tail only needs matching `position_ids`.

_PREFIX: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
_MODE: str | None = None


@contextlib.contextmanager
def capture_prefix():
    """Record each attention module's K/V. Wrap the no_grad prefix forward in this."""
    global _MODE
    _PREFIX.clear()
    _MODE = "capture"
    try:
        yield _PREFIX
    finally:
        _MODE = None


@contextlib.contextmanager
def use_prefix():
    """Prepend the captured K/V to every attention call. Wrap the tail forward in this."""
    global _MODE
    if not _PREFIX:
        raise RuntimeError("use_prefix() with nothing captured — the tail would train "
                           "without its context")
    _MODE = "use"
    try:
        yield
    finally:
        _MODE = None


def clear_prefix() -> None:
    """Drop the captured K/V (they are per-window and hold real memory)."""
    _PREFIX.clear()


def prefix_len() -> int:
    return next(iter(_PREFIX.values()))[0].shape[2] if _PREFIX else 0

#: Largest score tensor MPSGraph will accept, minus headroom for the mask.
_INT_MAX = 2**31 - 1
#: Cap on the query block. Smaller = less peak memory, more Python-loop overhead.
DEFAULT_CHUNK = 2048
#: Peak bytes allowed for one block's ``[heads, chunk, kv_len]`` score tensor.
#:
#: A FIXED chunk is an element-count cap, not a memory cap: the score tensor grows with
#: ``kv_len``, so the 2048 block that costs 1.97 GiB at kv 8064 costs **8 GiB at kv 32768**
#: — and roughly double that with the softmax alive. Measured: a 32768-window run with a
#: fixed 2048 block drove swap up 26 GB and never completed a step, while the same run at
#: 8064 sat flat. 2 GiB reproduces today's 8064 behavior exactly and shrinks the block
#: automatically as the context grows.
DEFAULT_SCORE_BYTES = 2 * 1024**3
#: Below this window length the unchunked kernel is used verbatim (no behavior change).
CHUNK_ABOVE = 4096

_original_sdpa = None


def safe_chunk(n_heads: int, kv_len: int, hint: int = DEFAULT_CHUNK,
               itemsize: int = 4, score_bytes: int = DEFAULT_SCORE_BYTES) -> int:
    """Largest query block satisfying BOTH the MPSGraph element cap and a memory budget.

    The element cap (``n_heads * chunk * kv_len < INT_MAX``) is what makes long windows
    run at all; the byte budget is what makes them run without swapping.
    """
    limit = _INT_MAX // max(1, n_heads * kv_len)
    by_bytes = score_bytes // max(1, n_heads * kv_len * itemsize)
    return max(64, min(hint, limit, by_bytes))


def chunked_causal_sdpa(query, key, value, *, dropout: float = 0.0,
                        scaling: float | None = None, attention_mask=None,
                        chunk_hint: int = DEFAULT_CHUNK,
                        score_bytes: int = DEFAULT_SCORE_BYTES, **sdpa_kwargs):
    """Causal SDPA computed in query blocks. Bit-identical to the unchunked call.

    Handles ``kv_len > q_len`` (a cached prefix): the queries are then the *last*
    ``q_len`` positions, so query row ``r`` of the whole call sits at absolute position
    ``offset + r`` and may attend to keys ``0 .. offset + r``. Getting this wrong is
    silent — the shapes still broadcast and the loss still falls, it just trains on a
    truncated context — so the offset is derived from the tensors, never assumed 0.
    """
    S = query.shape[2]
    kv_len = key.shape[2]
    offset = kv_len - S
    if offset < 0:
        raise ValueError(f"kv_len {kv_len} < q_len {S}: not a causal decoder call")
    n_heads = query.shape[1]
    chunk = safe_chunk(n_heads, kv_len, chunk_hint, query.element_size(), score_bytes)
    outs = []
    for i in range(0, S, chunk):
        j = min(i + chunk, S)
        if attention_mask is None:
            # Causal: block i only ever attends to keys < offset+j, so slice K/V and mask
            # the block-diagonal. tril(diagonal=offset+i) is the causal pattern for the
            # absolute query rows offset+i .. offset+j.
            mask = torch.ones(j - i, offset + j, dtype=torch.bool,
                              device=query.device).tril(diagonal=offset + i)
            k_c, v_c = key[:, :, :offset + j], value[:, :, :offset + j]
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
                        chunk_above: int = CHUNK_ABOVE,
                        score_bytes: int = DEFAULT_SCORE_BYTES) -> None:
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
        if _MODE == "capture":
            # detach: the prefix is a constant for the tail's backward, and keeping the
            # no_grad tensors alive as graph leaves would defeat the point
            _PREFIX[id(module)] = (key.detach(), value.detach())
        elif _MODE == "use":
            pk = _PREFIX.get(id(module))
            if pk is None:
                raise RuntimeError(f"no captured prefix for {type(module).__name__} — "
                                   "the capture pass did not cover every attention module")
            key = torch.cat([pk[0], key], dim=2)
            value = torch.cat([pk[1], value], dim=2)

        q_len = query.shape[2]
        cached = key.shape[2] != q_len and attention_mask is None
        # Decode steps, short windows and non-causal modules keep the stock kernel, so
        # nothing outside long-window training changes behavior. A maskless call with a
        # cached prefix is the exception: torch's `is_causal=True` aligns its mask
        # top-LEFT, which for kv_len > q_len lets every query see only the prefix head
        # and hides its own recent context. Route that through the offset-aware path
        # whatever the window length.
        if not getattr(module, "is_causal", True) or (q_len <= chunk_above and not cached):
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
                                  score_bytes=score_bytes, **sdpa_kwargs)
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
