# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
# Adapted from igorbarshteyn/jlens-gguf (Apache-2.0) for quant-tuner; see NOTICE.
"""Read the readout weights the lens needs straight from a model GGUF.

The lens decodes transported residuals with the model's own head:

    logits = softcap( W_U @ final_norm(h_hat) )

so we need, from the model file: the unembedding matrix (``output.weight``,
falling back to the tied ``token_embd.weight``), the final norm
(``output_norm.weight`` [+ ``.bias``]), the norm's epsilon, and Gemma-style
logit softcapping if present. Quantized tensors are dequantized to fp32 with
gguf-py's numpy kernels — no torch anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _kv(reader, key: str, default=None):
    f = reader.fields.get(key)
    if f is None:
        return default
    return f.contents()


@dataclass
class ReadoutWeights:
    """Final-norm + unembedding weights extracted from a model GGUF."""

    arch: str
    n_layers: int
    d_model: int
    n_vocab: int
    w_unembed: np.ndarray                 # [n_vocab, d_model] fp32
    norm_weight: np.ndarray | None        # [d_model] fp32
    norm_bias: np.ndarray | None          # [d_model] fp32 (layernorm models)
    norm_type: str                        # "rms" | "layer"
    eps: float
    logit_softcap: float | None = None
    model_name: str = ""
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_gguf(cls, path: str) -> ReadoutWeights:
        from quant_tuner.paths import ensure_gguf_py

        ensure_gguf_py()
        import gguf
        from gguf.quants import dequantize

        reader = gguf.GGUFReader(path)
        arch = str(_kv(reader, "general.architecture"))
        n_layers = int(_kv(reader, f"{arch}.block_count"))
        d_model = int(_kv(reader, f"{arch}.embedding_length"))
        model_name = str(_kv(reader, "general.name", "") or "")

        tensors = {t.name: t for t in reader.tensors}

        def load_tensor(name: str) -> np.ndarray | None:
            t = tensors.get(name)
            if t is None:
                return None
            if t.tensor_type in (gguf.GGMLQuantizationType.F32,):
                return np.asarray(t.data, dtype=np.float32)
            if t.tensor_type in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16):
                return np.asarray(t.data).astype(np.float32)
            return dequantize(t.data, t.tensor_type).astype(np.float32, copy=False)

        w_unembed = load_tensor("output.weight")
        if w_unembed is None:
            w_unembed = load_tensor("token_embd.weight")  # tied embeddings
            if w_unembed is None:
                raise ValueError(f"{path}: neither output.weight nor token_embd.weight found")
        w_unembed = w_unembed.reshape(-1, d_model)
        n_vocab = w_unembed.shape[0]

        norm_weight = load_tensor("output_norm.weight")
        norm_bias = load_tensor("output_norm.bias")
        # llama.cpp picks LLM_NORM vs LLM_NORM_RMS per architecture; the
        # presence of a bias is a reliable proxy for the LayerNorm family
        # (gpt2, gptneox, bloom, falcon, ...). Modern decoder LLMs are RMS.
        norm_type = "layer" if norm_bias is not None else "rms"
        eps = _kv(reader, f"{arch}.attention.layer_norm_rms_epsilon")
        if eps is None:
            eps = _kv(reader, f"{arch}.attention.layer_norm_epsilon", 1e-5)
        softcap = _kv(reader, f"{arch}.final_logit_softcapping")

        return cls(
            arch=arch,
            n_layers=n_layers,
            d_model=d_model,
            n_vocab=n_vocab,
            w_unembed=w_unembed,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            norm_type=norm_type,
            eps=float(eps),
            logit_softcap=float(softcap) if softcap is not None else None,
            model_name=model_name,
        )

    # ------------------------------------------------------------------ #

    def normalize(self, h: np.ndarray) -> np.ndarray:
        """Apply the model's final norm to ``h`` of shape ``[..., d_model]``."""
        h = h.astype(np.float32, copy=False)
        if self.norm_type == "rms":
            scale = 1.0 / np.sqrt(np.mean(h * h, axis=-1, keepdims=True) + self.eps)
            out = h * scale
        else:  # layernorm
            mu = h.mean(axis=-1, keepdims=True)
            var = h.var(axis=-1, keepdims=True)
            out = (h - mu) / np.sqrt(var + self.eps)
        if self.norm_weight is not None:
            out = out * self.norm_weight
        if self.norm_bias is not None:
            out = out + self.norm_bias
        return out

    def unembed(self, h: np.ndarray) -> np.ndarray:
        """``[..., d_model] -> [..., n_vocab]`` logits (final norm + head)."""
        logits = self.normalize(h) @ self.w_unembed.T
        if self.logit_softcap:
            logits = self.logit_softcap * np.tanh(logits / self.logit_softcap)
        return logits
