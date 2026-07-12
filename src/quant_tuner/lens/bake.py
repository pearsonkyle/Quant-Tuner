"""Bake a runtime ablation into GGUF weights, permanently.

When :mod:`quant_tuner.lens.loops` finds a residual direction whose live
ablation breaks a repetition loop (or an abliteration-style behavior edit),
this module makes it permanent — not by editing the quantized blocks in place
(lossy), but by:

1. **orthogonalizing the FP16 weights**: for each chosen layer, project the
   direction out of the *output* rows of the layers that write to the residual
   stream (``attn_output.weight`` and ``ffn_down.weight``): ``W <- W - v̂(v̂ᵀW)``.
   The residual/skip path is untouched, so this only removes the model's
   ability to *write* that direction — the standard weight-orthogonalization
   ("abliteration") edit, here targeted by an interpretability-derived vector;
2. **requantizing** the edited FP16 through the normal pipeline
   (``llama-quantize`` with the *existing* imatrix), so the baked quant is a
   real quant, not a hand-patched one;
3. **verifying**: KLD + tool-call eval on baked vs original, a loop re-check,
   and (optionally) runtime-intervention-vs-baked parity.

Only ablation/orthogonalization is bakeable this way — additive steering has no
per-layer bias tensor to absorb it in most architectures, so steering stays a
runtime-only capability of the OpenAI-compatible jlens-server.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# GGUF tensors that write into the residual stream (output projection of each
# sublayer). Orthogonalizing their OUTPUT rows removes the direction from what
# the block can add to the stream.
DEFAULT_TARGET_SUFFIXES = ("attn_output.weight", "ffn_down.weight")
MOE_DOWN_SUFFIX = "ffn_down_exps.weight"


@dataclass
class Direction:
    """A residual-space direction plus provenance, saved as ``direction.npz``."""

    v: np.ndarray                # [d] unit vector
    layers: list[int]            # layers this direction was derived at / for
    method: str
    source: dict = field(default_factory=dict)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            v=np.ascontiguousarray(self.v, dtype=np.float32),
            layers=np.asarray(self.layers, dtype=np.int64),
            meta=np.frombuffer(
                json.dumps({"method": self.method, "source": self.source}).encode(),
                dtype=np.uint8,
            ),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> Direction:
        data = np.load(path)
        meta = json.loads(bytes(data["meta"]).decode())
        return cls(
            v=np.asarray(data["v"], dtype=np.float32),
            layers=list(int(x) for x in data["layers"]),
            method=meta.get("method", "unknown"),
            source=meta.get("source", {}),
        )


def _orthogonalize_output_rows(W: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Remove direction ``v`` from the output space of ``W`` (GGUF ``[n_out, n_in]``).

    Each output row w_o gets ``w_o - (v·w_o_proj)``... more precisely we project
    the *columns* of the output-space representation: with W mapping input->output
    and v a vector in output space, the write of direction v is
    ``W <- W - v̂ (v̂ᵀ W)`` so no input can produce a component along v̂.
    """
    v = v.astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-12)
    # W: [n_out, n_in]; v: [n_out]. v̂ᵀW: [n_in]; outer(v̂, v̂ᵀW): [n_out, n_in]
    proj = np.outer(v, v @ W)
    return (W - proj).astype(np.float32)


def orthogonalize_layers(
    f16_gguf: Path,
    direction: Direction,
    out_f16: Path,
    *,
    layers: list[int] | None = None,
    target_suffixes: tuple[str, ...] = DEFAULT_TARGET_SUFFIXES,
    include_moe: bool = True,
) -> Path:
    """Write ``out_f16`` = ``f16_gguf`` with ``direction`` projected out of the
    residual-writing tensors of ``layers`` (default: the direction's own layers)."""
    from quant_tuner.lens.gguf_edit import (
        copy_gguf_with_tensor_edits,
        list_gguf_tensor_names,
        load_gguf_tensor,
    )

    layers = layers if layers is not None else list(direction.layers)
    v = np.asarray(direction.v, dtype=np.float32)
    d = v.shape[0]
    names = set(list_gguf_tensor_names(f16_gguf))
    edits: dict[str, np.ndarray] = {}

    for layer in layers:
        for suffix in target_suffixes:
            name = f"blk.{layer}.{suffix}"
            if name in names:
                W = load_gguf_tensor(f16_gguf, name)     # [n_out, n_in]
                if W.shape[0] != d:
                    logger.warning(
                        "skip %s: output dim %d != direction dim %d",
                        name, W.shape[0], d,
                    )
                    continue
                edits[name] = _orthogonalize_output_rows(W, v)
        if include_moe:
            moe = f"blk.{layer}.{MOE_DOWN_SUFFIX}"
            if moe in names:
                W = load_gguf_tensor(f16_gguf, moe)      # [n_expert, n_out, n_in]
                if W.ndim == 3 and W.shape[1] == d:
                    edits[moe] = np.stack(
                        [_orthogonalize_output_rows(W[e], v) for e in range(W.shape[0])]
                    )

    if not edits:
        raise ValueError(
            f"no residual-writing tensors matched for layers {layers} "
            f"(suffixes {target_suffixes}); direction dim={d}"
        )
    logger.info("orthogonalizing %d tensors across layers %s", len(edits), layers)
    return copy_gguf_with_tensor_edits(f16_gguf, out_f16, edits, edit_dtype=np.float16)


def bake_and_requantize(
    f16_gguf: Path,
    direction: Direction,
    *,
    quant_type: str,
    imatrix: Path | None,
    out_dir: Path,
    layers: list[int] | None = None,
    tensor_types: dict[str, str] | None = None,
    label: str = "baked",
) -> Path:
    """Orthogonalize the FP16, then requantize to a real GGUF via llama-quantize."""
    from quant_tuner.quantize import gguf as gguf_quant

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    edited_f16 = out_dir / f"{label}-f16.gguf"
    orthogonalize_layers(f16_gguf, direction, edited_f16, layers=layers)

    baked = out_dir / f"{quant_type}-{label}.gguf"
    gguf_quant.quantize(
        edited_f16, baked, quant_type, imatrix=imatrix,
        log=out_dir / f"{label}-quantize.log", tensor_types=tensor_types,
    )
    logger.info("baked quant: %s", baked)
    return baked


def verify_bake(
    baked: Path,
    original: Path,
    *,
    reference: Path,
    eval_dataset: Path,
    out_dir: Path,
    holdout: Path | None = None,
    lens_path: Path | None = None,
    loop_prompt_tokens: list[int] | None = None,
    eval_ctx: int = 8192,
) -> dict:
    """Compare a baked quant against the original: KLD, tool-call, loop re-check.

    Returns a dict of before/after metrics and writes ``verify.json``.
    """
    from quant_tuner.bench import bpw as bpw_mod
    from quant_tuner.bench import kld as kld_mod
    from quant_tuner.bench import runner

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = out_dir / "baseline.kld"
    if not baseline.exists():
        kld_mod.build_baseline(reference, eval_dataset, baseline, ctx=eval_ctx)

    result: dict = {"baked": str(baked), "original": str(original)}
    ref_n = bpw_mod.n_params(reference)
    for label, path in (("original", original), ("baked", baked)):
        row = runner.bench_one(
            path, f"{label}", reference_n_params=ref_n,
            eval_dataset=eval_dataset, eval_baseline=baseline,
            tools_dataset=None, tools_baseline=None,
            eval_ctx=eval_ctx, log_dir=out_dir / "logs", suite="kld",
        )
        result[f"{label}_kld"] = {
            "mean_kld": row.mean_kld, "ppl": row.ppl, "same_top_p": row.same_top_p,
        }

    if holdout is not None:
        from quant_tuner.eval.toolcall import Sampling, run_toolcall_eval

        for label, path in (("original", original), ("baked", baked)):
            summary = run_toolcall_eval(Path(holdout), model_path=path, sampling=Sampling())
            result[f"{label}_toolcall"] = {
                "tool_selection_acc": getattr(summary, "tool_selection_acc", None),
                "param_acc": getattr(summary, "param_acc", None),
            }

    if loop_prompt_tokens is not None and lens_path is not None:
        from quant_tuner.lens.loops import loop_autopsy

        report = loop_autopsy(
            baked, lens_path, loop_prompt_tokens,
            runs_dir=out_dir / "loop_runs", n_predict=512,
        )
        result["loop_after_bake"] = {"looped": report.looped, "notes": report.notes}

    (out_dir / "verify.json").write_text(json.dumps(result, indent=2))
    return result
