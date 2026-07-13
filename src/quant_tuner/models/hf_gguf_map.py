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
    # Qwen3-Next-style hybrid linear-attention blocks (Mamba-like mixers).
    # The first two land on non-SSM tensors (`attn_*`) and will be hooked by
    # outlier-stat collection. The last three land on `ssm_*` tensors that
    # `is_ssm` filters out of output-aware reranking; mapping them anyway is
    # harmless and makes the HF/GGUF coverage 1:1.
    (re.compile(r"^model\.layers\.(\d+)\.linear_attn\.in_proj_qkv$"), "blk.{bid}.attn_qkv.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.linear_attn\.in_proj_z$"),   "blk.{bid}.attn_gate.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.linear_attn\.in_proj_a$"),   "blk.{bid}.ssm_alpha.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.linear_attn\.in_proj_b$"),   "blk.{bid}.ssm_beta.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.linear_attn\.out_proj$"),    "blk.{bid}.ssm_out.weight"),
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


# Inverse mapping (GGUF -> candidate HF names). Built once from the forward
# rules: a GGUF tensor may correspond to more than one HF module name (e.g.
# attn_output maps from both self_attn.o_proj and self_attn.out_proj), so
# callers get every candidate and disambiguate by state-dict membership.
def _build_inverse() -> dict[str, list[str]]:
    inv: dict[str, list[str]] = {}
    for pat, tmpl in _HF_TO_GGUF_RULES:
        # reconstruct the HF template from the compiled pattern's source
        hf_tmpl = (
            pat.pattern
            .lstrip("^")
            .rstrip("$")
            .replace(r"\.", ".")
            .replace(r"(\d+)", "{bid}")
        )
        gguf_tmpl = tmpl  # may contain "{bid}"
        inv.setdefault(gguf_tmpl, [])
        if hf_tmpl not in inv[gguf_tmpl]:
            inv[gguf_tmpl].append(hf_tmpl)
    return inv


_GGUF_TO_HF_TEMPLATES = _build_inverse()


def gguf_to_hf_names(gguf_name: str) -> list[str]:
    """Inverse of :func:`map_hf_to_gguf`: all candidate HF module names.

    ``gguf_name`` is a tensor name without the ``.weight`` suffix's variance —
    both ``blk.3.ffn_up`` and ``blk.3.ffn_up.weight`` are accepted. Callers
    append ``.weight`` (or check state-dict membership) to pick the real one.
    """
    name = gguf_name[:-len(".weight")] if gguf_name.endswith(".weight") else gguf_name
    m = re.match(r"^blk\.(\d+)\.(.+)$", name)
    bid = m.group(1) if m else None
    suffix = m.group(2) if m else name

    out: list[str] = []
    for gguf_tmpl, hf_tmpls in _GGUF_TO_HF_TEMPLATES.items():
        tmpl_name = gguf_tmpl[:-len(".weight")] if gguf_tmpl.endswith(".weight") else gguf_tmpl
        tm = re.match(r"^blk\.\{bid\}\.(.+)$", tmpl_name)
        if tm:
            if bid is not None and tm.group(1) == suffix:
                out.extend(t.format(bid=bid) for t in hf_tmpls)
        elif tmpl_name == name:  # non-layer tensor (e.g. output)
            out.extend(hf_tmpls)
    return out


def gguf_to_hf_name(gguf_name: str) -> str | None:
    """First candidate HF name for ``gguf_name`` (see :func:`gguf_to_hf_names`)."""
    names = gguf_to_hf_names(gguf_name)
    return names[0] if names else None
