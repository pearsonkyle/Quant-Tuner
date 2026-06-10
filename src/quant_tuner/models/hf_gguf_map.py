"""HuggingFace → GGUF tensor-name mapping for Linear layers.

Covers attention projections (q/k/v/o + attn_output_gate + fused qkv), MLP
projections (gate/up/down), MoE expert/router tensors (Gemma-4 style fused 3D
experts), and the LM head. SSM tensors are intentionally absent — they don't
share the `y = W a` linear-projection structure, so calibrators that derive
activation-aware importance pass them through using the standard E[a^2]
signal instead.

DiffusionGemma note: the checkpoint's text backbone lives under
``model.decoder.*`` (encoder weights are tied; llama.cpp's converter rewrites
``model.decoder.<x>`` → ``model.<x>``). :func:`map_hf_to_gguf` applies the same
prefix normalization so HF module names from either the decoder or the tied
encoder stack resolve to the canonical GGUF tensor.
"""

from __future__ import annotations

import re

# Trunk prefixes that alias `model.` (DiffusionGemma encoder/decoder stacks).
_TRUNK_PREFIXES: tuple[tuple[str, str], ...] = (
    ("model.decoder.", "model."),
    ("model.encoder.language_model.", "model."),
)

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
    # Gemma-4 / DiffusionGemma MoE. Experts are a single fused 3D parameter per
    # projection ([n_expert, n_out, n_in]); the router is a plain Linear whose
    # input is NOT the experts' pre-norm output (it has its own internal norm).
    (re.compile(r"^model\.layers\.(\d+)\.experts\.gate_up_proj$"), "blk.{bid}.ffn_gate_up_exps.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.experts\.down_proj$"),    "blk.{bid}.ffn_down_exps.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj$"), "blk.{bid}.ffn_gate_up_exps.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.down_proj$"),    "blk.{bid}.ffn_down_exps.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.router\.proj$"),          "blk.{bid}.ffn_gate_inp.weight"),
    # DiffusionGemma decoder-only self-conditioning gated MLP (2D Linears).
    (re.compile(r"^model\.self_conditioning\.gate_proj$"), "self_cond_gate.weight"),
    (re.compile(r"^model\.self_conditioning\.up_proj$"),   "self_cond_up.weight"),
    (re.compile(r"^model\.self_conditioning\.down_proj$"), "self_cond_down.weight"),
]


def map_hf_to_gguf(hf_name: str) -> str | None:
    """Map an HF module name (e.g. ``model.layers.3.mlp.up_proj``) to a GGUF tensor name."""
    for prefix, repl in _TRUNK_PREFIXES:
        if hf_name.startswith(prefix):
            hf_name = repl + hf_name[len(prefix):]
            break
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


_MOE_EXPERT_MARKERS = (".ffn_gate_up_exps.", ".ffn_gate_exps.", ".ffn_up_exps.", ".ffn_down_exps.")


def is_moe_expert(gguf_name: str) -> bool:
    """Stacked per-expert tensors (3D ``[n_expert, n_out, n_in]`` in GGUF).

    Their imatrix entries carry ``n_expert`` independent E[a^2] rows (one per
    expert, expert-major); calibrators must rerank each expert's row against
    that expert's weight slice, never across experts."""
    return any(m in gguf_name for m in _MOE_EXPERT_MARKERS)
