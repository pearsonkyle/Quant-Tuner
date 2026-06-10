"""Output-aware imatrix calibration.

This module produces a calibrated importance matrix (an "imatrix" GGUF) that
llama-quantize consumes when quantizing the F16 model. Three classes of variant:

  * **analytic** — closed-form ``s_c = ||W[:,c]||^2 * E[a_c^2]``. The contribution
    of input channel ``c`` to ``||y||^2`` for ``y = W a``. Cheap; no model load.
  * **mix_50** / **hybrid_custom** — per-tensor combinations of ``E[a^2]`` with
    the analytic signal, intended to recover channels that look important under
    either prior alone.
  * **outlier_l4** / **outlier_max** — heavy-tail emphasis using ``E[a^4]`` or
    ``max|a|``, captured by an HF forward pass over the calibration corpus.

For every variant, SSM tensors pass through using raw ``E[a^2]`` — they don't
have a ``y = W a`` linear-projection structure, so output-aware re-ranking is
invalid for them.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from quant_tuner.calibrate._device import resolve_device
from quant_tuner.models.hf_gguf_map import is_ssm, map_hf_to_gguf
from quant_tuner.paths import LLAMA_CPP_DIR

Variant = Literal["analytic", "mix_50", "hybrid_custom", "outlier_l4", "outlier_max"]
ANALYTIC_VARIANTS: tuple[Variant, ...] = ("analytic", "mix_50", "hybrid_custom")
OUTLIER_VARIANTS: tuple[Variant, ...] = ("outlier_l4", "outlier_max")


def _ensure_gguf_py() -> None:
    p = str(LLAMA_CPP_DIR / "gguf-py")
    if p not in sys.path:
        sys.path.insert(0, p)


# --- GGUF helpers --------------------------------------------------------- #

def _col_l2_sq(weight: np.ndarray) -> np.ndarray:
    """Sum of squares along the OUTPUT axis -> one value per INPUT channel.

    GGUF stores linear weights as ``[n_out, n_in]``, so summing axis 0 of ``W^2``
    gives ``||W[:, c]||^2`` for every input channel ``c``.
    """
    w = weight.astype(np.float32, copy=False)
    return (w * w).sum(axis=0)


def _load_base_imatrix(path: Path) -> dict[str, np.ndarray]:
    """Read an existing imatrix GGUF as ``{tensor: E[a^2]}`` (sum_sq divided by count)."""
    _ensure_gguf_py()
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    raw = {t.name: np.array(t.data) for t in reader.tensors}
    out: dict[str, np.ndarray] = {}
    for name, arr in raw.items():
        if not name.endswith(".in_sum2"):
            continue
        base = name[: -len(".in_sum2")]
        cnt = raw.get(f"{base}.counts")
        if cnt is None:
            continue
        c = max(int(cnt.flat[0]), 1)
        out[base] = arr.astype(np.float32) / c
    return out


def _load_weights(path: Path) -> dict[str, np.ndarray]:
    _ensure_gguf_py()
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    return {t.name: np.array(t.data) for t in reader.tensors if len(t.shape) >= 2}


def _l1_normalize(v: np.ndarray) -> np.ndarray:
    """Scale ``v`` so its mean magnitude is 1.

    Without this, combining vectors from different signal classes (E[a^2] vs
    ``||W[:,c]||^2 * E[a^2]``) collapses to whichever has the larger raw scale.
    """
    s = float(np.abs(v).sum())
    if s <= 0.0 or not np.isfinite(s):
        return v.astype(np.float32, copy=False)
    return (v.astype(np.float32) * (v.size / s)).astype(np.float32)


# --- Forward-pass stats (E[a^4], max|a|) --------------------------------- #

@dataclass
class ForwardStats:
    """Per-input-channel outlier stats keyed by GGUF tensor name.

    Both arrays are 1-D, length = input-channel count. ``e_a4`` is the streaming
    mean of ``a^4`` over all tokens seen; ``max_abs`` is the per-channel running
    maximum of ``|a|``.
    """

    e_a4: dict[str, np.ndarray]
    max_abs: dict[str, np.ndarray]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        names = list(self.e_a4.keys())
        payload: dict[str, np.ndarray] = {"__tensor_names__": np.array(names)}
        for t in names:
            payload[f"{t}__e_a4"] = self.e_a4[t].astype(np.float32)
            payload[f"{t}__max_abs"] = self.max_abs[t].astype(np.float32)
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: Path) -> ForwardStats:
        z = np.load(path, allow_pickle=True)
        names = list(z["__tensor_names__"])
        return cls(
            e_a4={t: z[f"{t}__e_a4"].astype(np.float32) for t in names},
            max_abs={t: z[f"{t}__max_abs"].astype(np.float32) for t in names},
        )


def collect_forward_stats(
    model_dir: Path,
    calibration_text: Path,
    base_imatrix: Path,
    *,
    tokens: int = 4096,
    ctx: int = 1024,
    device: str = "auto",
    dtype: str = "bfloat16",
) -> ForwardStats:
    """Forward-only pass to record streaming E[a^4] and max|a| per input channel.

    Only tensors present in ``base_imatrix`` (as ``.in_sum2``) are tracked, so
    coverage matches what llama-quantize will actually consume.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _ensure_gguf_py()
    from gguf import GGUFReader

    device = resolve_device(device)
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]

    print(f"[forward_stats] load {model_dir} -> {device}/{dtype}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    reader = GGUFReader(str(base_imatrix))
    gguf_dims = {
        t.name[: -len(".in_sum2")]: int(t.shape[0])
        for t in reader.tensors
        if t.name.endswith(".in_sum2")
    }

    hf_modules: dict[str, torch.nn.Linear] = {}
    mapping: dict[str, str] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            hf_modules[name] = mod
            g = map_hf_to_gguf(name)
            if g is not None and g in gguf_dims:
                mapping[name] = g

    print(
        f"[forward_stats] hooked {len(mapping)}/{len(gguf_dims)} GGUF tensors "
        f"({100 * len(mapping) / max(len(gguf_dims), 1):.0f}%)",
        file=sys.stderr,
    )

    sum_sq_sq: dict[str, torch.Tensor] = {}
    max_abs: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = defaultdict(int)

    def make_hook(gguf_name: str):
        def pre(_module, in_args):
            x = in_args[0].detach()
            flat = x.reshape(-1, x.shape[-1]).float()
            ss = (flat * flat * flat * flat).sum(dim=0)
            mx = flat.abs().amax(dim=0)
            prev_ss = sum_sq_sq.get(gguf_name)
            prev_mx = max_abs.get(gguf_name)
            sum_sq_sq[gguf_name] = ss if prev_ss is None else prev_ss + ss
            max_abs[gguf_name] = mx if prev_mx is None else torch.maximum(prev_mx, mx)
            counts[gguf_name] += flat.shape[0]
        return pre

    handles = [hf_modules[h].register_forward_pre_hook(make_hook(g)) for h, g in mapping.items()]
    try:
        text = calibration_text.read_text()
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0][:tokens]
        chunks = ids.split(ctx)
        print(f"[forward_stats] {ids.shape[0]} tokens, ctx {ctx} -> {len(chunks)} chunks", file=sys.stderr)
        with torch.no_grad():
            for i, chunk in enumerate(chunks):
                if chunk.numel() < 2:
                    continue
                model(chunk.unsqueeze(0).to(device))
                print(f"  chunk {i + 1}/{len(chunks)}", file=sys.stderr)
    finally:
        for h in handles:
            h.remove()

    e_a4 = {
        name: (ss / max(counts[name], 1)).cpu().numpy().astype(np.float32)
        for name, ss in sum_sq_sq.items()
    }
    max_abs_np = {name: t.cpu().numpy().astype(np.float32) for name, t in max_abs.items()}
    return ForwardStats(e_a4=e_a4, max_abs=max_abs_np)


# --- Variant builders ----------------------------------------------------- #
#
# All builders return ``{gguf_tensor: importance_vector}``. SSM tensors always
# pass through the raw E[a^2] signal (memory note: don't rerank ``blk.*.ssm_*``).


def _build_simple(
    f16: Path,
    base_imatrix: Path,
    combiner,
    label: str,
) -> dict[str, np.ndarray]:
    base = _load_base_imatrix(base_imatrix)
    weights = _load_weights(f16)
    out: dict[str, np.ndarray] = {}
    ssm = skipped = 0
    for tname, ea2 in base.items():
        if tname not in weights:
            skipped += 1
            continue
        if is_ssm(tname):
            out[tname] = ea2.astype(np.float32)
            ssm += 1
            continue
        wcol = _col_l2_sq(weights[tname])
        if wcol.shape != ea2.shape:
            skipped += 1
            continue
        out[tname] = combiner(wcol, ea2).astype(np.float32)
    print(
        f"[{label}] {len(out)} tensors ({ssm} SSM passthrough, {skipped} skipped)",
        file=sys.stderr,
    )
    return out


def build_analytic(f16: Path, base_imatrix: Path) -> dict[str, np.ndarray]:
    """``s_c = ||W[:,c]||^2 * E[a_c^2]`` (attn + FFN); SSM tensors passthrough E[a^2]."""
    return _build_simple(f16, base_imatrix, lambda w, e: w * e, "analytic")


def build_mix_50(f16: Path, base_imatrix: Path) -> dict[str, np.ndarray]:
    """Per-tensor geometric mean of L1-normalized ``E[a^2]`` and analytic signals."""
    def combine(w: np.ndarray, e: np.ndarray) -> np.ndarray:
        a = _l1_normalize(e)
        b = _l1_normalize(w * e)
        return np.sqrt(np.maximum(a, 0.0) * np.maximum(b, 0.0))

    return _build_simple(f16, base_imatrix, combine, "mix_50")


def build_hybrid_custom(f16: Path, base_imatrix: Path) -> dict[str, np.ndarray]:
    """Per-tensor L1-normalized ``max(E[a^2], analytic)`` — channels important under either signal."""
    def combine(w: np.ndarray, e: np.ndarray) -> np.ndarray:
        return np.maximum(_l1_normalize(e), _l1_normalize(w * e))

    return _build_simple(f16, base_imatrix, combine, "hybrid_custom")


def build_outlier_l4(
    base_imatrix: Path,
    forward_stats: ForwardStats,
) -> dict[str, np.ndarray]:
    """``s_c = sqrt(E[a_c^4])`` for non-SSM tensors; passthrough ``E[a^2]`` for SSM.

    The L4 norm of a channel's activation distribution emphasizes heavy tails
    more than ``E[a^2]`` does, so rare-but-large activations get more budget.
    """
    base = _load_base_imatrix(base_imatrix)
    out: dict[str, np.ndarray] = {}
    ssm = missing = mismatch = 0
    for tname, ea2 in base.items():
        if is_ssm(tname):
            out[tname] = ea2.astype(np.float32)
            ssm += 1
            continue
        e_a4 = forward_stats.e_a4.get(tname)
        if e_a4 is None:
            out[tname] = ea2.astype(np.float32)
            missing += 1
            continue
        if e_a4.shape != ea2.shape:
            out[tname] = ea2.astype(np.float32)
            mismatch += 1
            continue
        out[tname] = np.sqrt(np.maximum(e_a4, 0.0)).astype(np.float32)
    print(
        f"[outlier_l4] {len(out)} tensors ({ssm} SSM, {missing} no-stats, "
        f"{mismatch} shape-mismatch fallback)",
        file=sys.stderr,
    )
    return out


def build_outlier_max(
    base_imatrix: Path,
    forward_stats: ForwardStats,
) -> dict[str, np.ndarray]:
    """``s_c = E[a_c^2] * max|a_c|^2`` for non-SSM tensors; passthrough ``E[a^2]`` for SSM."""
    base = _load_base_imatrix(base_imatrix)
    out: dict[str, np.ndarray] = {}
    ssm = missing = mismatch = 0
    for tname, ea2 in base.items():
        if is_ssm(tname):
            out[tname] = ea2.astype(np.float32)
            ssm += 1
            continue
        mx = forward_stats.max_abs.get(tname)
        if mx is None:
            out[tname] = ea2.astype(np.float32)
            missing += 1
            continue
        if mx.shape != ea2.shape:
            out[tname] = ea2.astype(np.float32)
            mismatch += 1
            continue
        out[tname] = (ea2 * (mx**2)).astype(np.float32)
    print(
        f"[outlier_max] {len(out)} tensors ({ssm} SSM, {missing} no-stats, "
        f"{mismatch} shape-mismatch fallback)",
        file=sys.stderr,
    )
    return out


# --- GGUF writer ---------------------------------------------------------- #

def write_imatrix(
    out_path: Path,
    importance: dict[str, np.ndarray],
    schema_source: Path,
    dataset_label: str = "<quant-tuner-imatrix>",
) -> Path:
    """Write a new imatrix GGUF mirroring the KV schema of ``schema_source``."""
    _ensure_gguf_py()
    from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter

    src = GGUFReader(str(schema_source))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(out_path), "imatrix")

    for key, field in src.fields.items():
        if not key.startswith("imatrix."):
            continue
        if key == "imatrix.chunk_count":
            writer.add_uint32(key, 1)
        elif key == "imatrix.chunk_size":
            writer.add_uint32(key, int(field.parts[field.data[0]][0]))
        elif key == "imatrix.datasets":
            writer.add_array(key, [dataset_label])

    for tname, vec in importance.items():
        if not np.all(np.isfinite(vec)):
            print(f"  WARN: non-finite values in {tname}; clamping", file=sys.stderr)
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        if (vec < 0).any():
            print(f"  WARN: negative values in {tname}; clamping to 0", file=sys.stderr)
            vec = np.maximum(vec, 0.0)
        writer.add_tensor(
            f"{tname}.in_sum2", vec.astype(np.float32), raw_dtype=GGMLQuantizationType.F32
        )
        writer.add_tensor(
            f"{tname}.counts",
            np.array([1.0], dtype=np.float32),
            raw_dtype=GGMLQuantizationType.F32,
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return out_path


# --- Top-level orchestration --------------------------------------------- #

def calibrate(
    *,
    variant: Variant,
    f16_gguf: Path,
    base_imatrix: Path,
    out_path: Path,
    model_dir: Path | None = None,
    calibration_text: Path | None = None,
    forward_stats_path: Path | None = None,
    forward_tokens: int = 4096,
    forward_ctx: int = 1024,
    device: str = "auto",
    dtype: str = "bfloat16",
) -> Path:
    """Build an output-aware imatrix GGUF.

    Analytic variants (``analytic``, ``mix_50``, ``hybrid_custom``) need only
    ``f16_gguf`` + ``base_imatrix``. Outlier variants additionally need either
    a precomputed ``forward_stats_path`` or (``model_dir`` + ``calibration_text``)
    to compute fresh stats.
    """
    if variant in ANALYTIC_VARIANTS:
        builders = {
            "analytic": build_analytic,
            "mix_50": build_mix_50,
            "hybrid_custom": build_hybrid_custom,
        }
        importance = builders[variant](f16_gguf, base_imatrix)
    elif variant in OUTLIER_VARIANTS:
        if forward_stats_path and forward_stats_path.exists():
            stats = ForwardStats.load(forward_stats_path)
        else:
            if model_dir is None or calibration_text is None:
                raise ValueError(
                    f"variant={variant} requires either an existing --forward-stats "
                    f"or (model_dir + calibration_text) to compute them"
                )
            stats = collect_forward_stats(
                model_dir,
                calibration_text,
                base_imatrix,
                tokens=forward_tokens,
                ctx=forward_ctx,
                device=device,
                dtype=dtype,
            )
            if forward_stats_path is not None:
                stats.save(forward_stats_path)
        importance = (
            build_outlier_l4(base_imatrix, stats)
            if variant == "outlier_l4"
            else build_outlier_max(base_imatrix, stats)
        )
    else:
        raise ValueError(f"unknown variant: {variant}")

    return write_imatrix(
        out_path, importance, base_imatrix, dataset_label=f"<quant-tuner-{variant}>"
    )


__all__ = ["ForwardStats", "Variant", "calibrate", "collect_forward_stats"]
