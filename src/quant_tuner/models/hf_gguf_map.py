"""HuggingFace → GGUF tensor-name mapping for Linear layers.

Covers attention projections (q/k/v/o + attn_output_gate + fused qkv), MLP
projections (gate/up/down), and the LM head. SSM tensors are intentionally
absent — they don't share the `y = W a` linear-projection structure, so
calibrators that derive activation-aware importance pass them through using
the standard E[a^2] signal instead.
"""

from __future__ import annotations

import re

_HF_TO_GGUF_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_proj$"),    "blk.{bid}.attn_q.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.k_proj$"),    "blk.{bid}.attn_k.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.v_proj$"),    "blk.{bid}.attn_v.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.o_proj$"),    "blk.{bid}.attn_output.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.out_proj$"),  "blk.{bid}.attn_output.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.gate_proj$"), "blk.{bid}.attn_gate.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.gate_proj$"),       "blk.{bid}.ffn_gate.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.up_proj$"),         "blk.{bid}.ffn_up.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.down_proj$"),       "blk.{bid}.ffn_down.weight"),
    (re.compile(r"^lm_head$"),                                    "output.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.qkv_proj$"),  "blk.{bid}.attn_qkv.weight"),
]


def map_hf_to_gguf(hf_name: str) -> str | None:
    """Map an HF module name (e.g. ``model.layers.3.mlp.up_proj``) to a GGUF tensor name."""
    for pat, tmpl in _HF_TO_GGUF_RULES:
        m = pat.match(hf_name)
        if m:
            if m.groups():
                return tmpl.format(bid=m.group(1))
            return tmpl
    return None


def is_ssm(gguf_name: str) -> bool:
    """SSM tensors lack the y = W a structure; calibrators must not rerank them
    with an output-aware prior (E[a^2] passthrough is correct)."""
    return ".ssm_" in gguf_name
