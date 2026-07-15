"""llama-quantize per-tensor mix table, shared by the AWQ and GPTQ calibrators.

Low-bit ftypes are tensor *mixes* — ``llama_tensor_get_type`` (pinned
llama.cpp f3e18281, ``src/llama-quant.cpp``) bumps some tensors above the
ftype's base grid. Both calibrators need that mapping: AWQ scores each group
member with the proxy quantizer matching its real target type
(:func:`quant_tuner.calibrate.awq.proxy_for_member`), and GPTQ rounds each
tensor on the grid matching its real target type
(:func:`quant_tuner.calibrate.gptq.grid_for_member`).

The table is keyed by the leaf module name of the HF Linear
(``v_proj``/``o_proj``/``down_proj``…) — the same dotted-name convention both
calibrators use (``model.layers.N.self_attn.v_proj``).
"""

from __future__ import annotations

import re

LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def layer_index(member: str) -> int | None:
    """Layer index parsed from a dotted HF module name, or ``None``."""
    m = LAYER_RE.search(member)
    return int(m.group(1)) if m else None


def gqa_or_moe_ge4(config) -> bool:
    """Mirror llama.cpp's ``n_gqa() >= 4 || n_expert >= 4`` tensor-mix predicate."""
    heads = getattr(config, "num_attention_heads", None) or 0
    kv = getattr(config, "num_key_value_heads", None) or 0
    if kv and heads // kv >= 4:
        return True
    experts = max(
        getattr(config, "num_local_experts", None) or 0,
        getattr(config, "num_experts", None) or 0,
    )
    return experts >= 4


def use_more_bits(i_layer: int, n_layers: int) -> bool:
    """llama.cpp's layer schedule for Q4_K_M/Q5_K_M bumps: first eighth, last
    eighth, and every third layer in between."""
    return (
        i_layer < n_layers // 8
        or i_layer >= 7 * n_layers // 8
        or (i_layer - n_layers // 8) % 3 == 2
    )


def target_type_for_member(
    quant_type: str,
    member: str,
    *,
    gqa_ge4: bool,
    n_layers: int | None = None,
) -> str | None:
    """llama-quantize tensor type this member really lands on, or ``None``
    when it keeps the ftype's base grid.

    Mirrors ``llama_tensor_get_type`` (pinned llama.cpp f3e18281) for the
    tensors the calibrators touch (attn q/k/v/o + MLP gate/up/down), on the
    architectures the pipeline targets (dense non-Falcon; layer index ==
    attention-tensor index):

    - IQ1/IQ2 branch:
        ``attn_v``     → Q4_K when GQA or MoE ≥ 4, else IQ3_S (S/M) / Q2_K
        ``attn_output``→ IQ3_S for IQ2_S/IQ2_M, IQ2_XXS for IQ1_*
        ``ffn_down``   → one tier up for the first eighth of layers
    - Q2_K:   ``attn_v`` → Q4_K (GQA/MoE ≥ 4) else Q3_K; ``ffn_down`` → Q3_K on
      *every* layer; ``attn_output`` → Q3_K
    - Q2_K_S: ``attn_v`` → Q4_K only under GQA/MoE ≥ 4; ``ffn_down`` → Q4_K for
      the first eighth; ``attn_output`` keeps the base grid
    - Q3_K_M/Q3_K_L: ``attn_v``, ``attn_output`` and ``ffn_down`` (every layer,
      non-Falcon) all land on Q4_K–Q5_K → collapsed to ``"Q4_K"``
    - IQ3_*:  ``attn_v`` → Q4_K for IQ3_M always, and for IQ3_XXS/XS/S under
      GQA/MoE ≥ 4; IQ3_M additionally bumps ``attn_output`` → Q4_K and the
      first eighth of ``ffn_down`` → Q4_K
    - Q4_K_M: ``attn_v`` and ``ffn_down`` → Q6_K on the :func:`use_more_bits`
      layer schedule
    - IQ4_NL/IQ4_XS: ``attn_v`` → Q5_K under GQA/MoE ≥ 4. (The first-eighth
      ``ffn_down`` → Q5_K bump applies only when llama-quantize has NO
      imatrix — every quant-tuner path supplies one, so it is not modeled.)
    - Q3_K_S and IQ3_S (without GQA) have *no* overrides

    Not modeled: the 70B attn_v → Q5_K special case, the 8-expert-MoE
    variations, IQ3_XS/IQ3_XXS attn_q/attn_k *down*-bumps (no shipped recipe
    targets them), and the Falcon arch branches.
    """
    qt = quant_type.upper()
    leaf = member.rsplit(".", 1)[-1]
    idx = layer_index(member)

    def first_eighth() -> bool:
        if not n_layers or idx is None:
            return False
        return idx < max(1, n_layers // 8)

    def more_bits() -> bool:
        if not n_layers or idx is None:
            return False
        return use_more_bits(idx, n_layers)

    # --- 2-bit IQ families (IQ1_*, IQ2_*) ---------------------------------- #
    if qt.startswith(("IQ1", "IQ2")):
        s_or_m = qt.startswith(("IQ2_S", "IQ2_M"))
        if leaf == "v_proj":
            if gqa_ge4:
                return "Q4_K"
            return "IQ3_S" if s_or_m else "Q2_K"
        if leaf == "o_proj":
            if s_or_m:
                return "IQ3_S"
            if qt.startswith("IQ1"):
                return "IQ2_XXS"
            return None
        if leaf == "down_proj" and first_eighth():
            return "IQ3_S" if s_or_m else "Q2_K"
        return None

    # --- Q2_K family -------------------------------------------------------- #
    if qt.startswith("Q2"):
        if leaf == "v_proj":
            if gqa_ge4:
                return "Q4_K"
            return None if qt == "Q2_K_S" else "Q3_K"
        if leaf == "o_proj":
            return "Q3_K" if qt == "Q2_K" else None
        if leaf == "down_proj":
            if qt == "Q2_K":
                return "Q3_K"  # every layer
            if qt == "Q2_K_S" and first_eighth():
                return "Q4_K"
        return None

    # --- 3-bit families ------------------------------------------------------ #
    if qt.startswith("Q3"):
        if qt in ("Q3_K_M", "Q3_K_L") and leaf in ("v_proj", "o_proj", "down_proj"):
            return "Q4_K"  # Q4_K/Q5_K
        return None  # Q3_K_S: base grid everywhere
    if qt.startswith("IQ3"):
        if leaf == "v_proj" and (qt == "IQ3_M" or gqa_ge4):
            return "Q4_K"  # IQ3_M always; IQ3_XXS/XS/S under GQA/MoE >= 4
        if qt == "IQ3_M" and (leaf == "o_proj" or (leaf == "down_proj" and first_eighth())):
            return "Q4_K"
        return None

    # --- 4-bit families ------------------------------------------------------ #
    if qt == "Q4_K_M":
        if leaf in ("v_proj", "down_proj") and more_bits():
            return "Q6_K"
        return None
    if qt.startswith("IQ4"):
        if leaf == "v_proj" and gqa_ge4:
            return "Q5_K"
        return None  # ffn_down bump needs no-imatrix; never the case here

    return None


__all__ = [
    "LAYER_RE",
    "gqa_or_moe_ge4",
    "layer_index",
    "target_type_for_member",
    "use_more_bits",
]
