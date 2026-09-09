"""W4A16 (compressed-tensors) post-training quantization for vLLM serving.

Produces the same checkpoint *format* as ``google/gemma-4-E4B-it-qat-w4a16-ct``
but calibrated on **our** distribution (the usage-log + wiki corpora from
``scripts/build_corpora.py``) and at a **longer calibration context** than the
GGUF pipeline's 4096 default — the serving target is 100k+ context, so the
Hessian statistics should see long sequences.

Design notes
------------
- Calibration reuses :func:`quant_tuner.calibrate._ingest.sample_chunks`
  (deterministic, evenly strided over the whole corpus — never just the head).
- ``llmcompressor`` is imported lazily inside :func:`run_ptq`; everything else
  in this module (config validation, corpus sampling, recipe construction) is
  importable and unit-testable without it. Install with the ``vllm-ptq`` extra.
- Multimodal models (gemma-4 E4B: vision + audio towers, per-layer embeddings)
  are handled by *ignoring* everything outside the language-model decoder
  linears. The default ignore list matches what Google's official QAT W4A16
  checkpoint leaves unquantized. Text-only models simply never match those
  patterns, so the defaults are safe across architectures.
- The exported dir gains ``quant_tuner_ptq.json`` provenance (corpus files +
  SHA-256 fingerprints, ctx, token budget, resolved recipe) so a checkpoint can
  always be traced back to what calibrated it.

Two axes beyond the preset scheme
---------------------------------
1. **Weight grid.** ``scheme="W4A16"`` is a *preset*: int4, group-128,
   symmetric, minmax observer. Deviating from any of that (group 32, asymmetric
   zero-points, an MSE observer, activation reordering) means handing
   llmcompressor an explicit ``config_groups`` spec instead — see
   :func:`build_config_groups`. Both paths are mutually exclusive at the
   modifier: pass a scheme *or* config groups, never both.
2. **KV cache.** ``kv_cache_dtype="fp8_e4m3"`` makes the same oneshot pass
   observe k/v activations and bake **static per-tensor scales** into the
   checkpoint, so ``vllm serve --kv-cache-dtype fp8_e4m3`` picks them up
   automatically and the cache halves. This is the lever for long context and
   concurrency: weights shrink once, the cache shrinks per token per sequence.

   fp8 KV is the failure mode this module guards hardest, because it fails
   *quietly*: a checkpoint whose k/v scales never got written still loads, still
   serves, and vLLM just computes scales on the fly (or refuses, depending on
   version) — the quality and memory difference shows up nowhere in the logs.
   :func:`verify_export` therefore counts the ``k_scale``/``v_scale`` tensors
   actually present and raises when a requested KV scheme produced none.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# What Google's official gemma-4 QAT W4A16 checkpoint leaves in bf16. Regexes
# use llmcompressor's "re:" convention. Text-only models match none of these
# (except lm_head, which we always keep full-precision: a quantized or pruned
# head is exactly the rare-token failure mode we saw in the osoi5 checkpoint).
DEFAULT_IGNORE: tuple[str, ...] = (
    "lm_head",
    "re:.*vision_tower.*",
    "re:.*audio_tower.*",
    "re:.*embed_vision.*",
    "re:.*embed_audio.*",
    "re:.*embed_tokens.*",
    "re:.*per_layer.*",
    "re:.*multi_modal_projector.*",
)

# Schemes whose *weights only* are quantized. A custom weight grid (group size,
# symmetry, observer, act-ordering) is expressible for these as a single
# config_groups entry with no activation args. W8A8 / FP8_DYNAMIC also quantize
# activations, so a hand-built group for them would have to specify
# input_activations too — rather than get that subtly wrong, they are supported
# at their preset settings only.
WEIGHT_ONLY_SCHEMES: dict[str, int] = {"W4A16": 4, "W8A16": 8}
SUPPORTED_SCHEMES: tuple[str, ...] = ("W4A16", "W8A8", "W8A16", "FP8_DYNAMIC")

# llmcompressor's `Observer` registry, read off llmcompressor 0.13.0
# (`Observer.registered_names()`). The observer decides how a per-group scale is
# chosen from the calibration statistics:
#   minmax       — the range endpoints. llmcompressor's default; fastest.
#   mse          — searches the clipping range minimising reconstruction error.
#   imatrix-mse  — mse weighted by per-input-channel activation importance
#                  collected during the calibration forwards. This is the direct
#                  analogue of the GGUF ladder's imatrix, and the observer named
#                  by the published INT4 cards. Costs a forward pass' worth of
#                  statistics, which GPTQ is already paying for.
# Hyphen and underscore spellings both resolve (the registry normalises), and
# the string is stored verbatim in the exported config — VERIFIED on 0.13.0.
KNOWN_OBSERVERS: tuple[str, ...] = (
    "minmax",
    "mse",
    "imatrix-mse",
    "static-minmax",
    "memoryless-minmax",
    "memoryless-mse",
)


def normalize_observer(name: str) -> str:
    """Fold the hyphen/underscore spellings of an observer name together."""
    return name.replace("_", "-").strip().lower()


# Group sizes vLLM's packed-int4 kernels accept. -1 is per-output-channel.
# A checkpoint quantized at any other grouping exports cleanly and then fails
# at `vllm serve`, which is the worst possible place to find out.
VLLM_GROUP_SIZES: tuple[int, ...] = (-1, 32, 64, 128)

# compressed-tensors' `ActivationOrdering`. "static" reorders columns by
# activation magnitude once, from the calibration Hessian, and permutes the
# saved weights — no runtime cost, and the g_idx is folded away. "group" keeps
# a runtime g_idx (slower to serve). "weight" orders by weight magnitude.
# VERIFIED against compressed_tensors.quantization.ActivationOrdering on 0.18.0.
KNOWN_ACTORDER: tuple[str, ...] = ("group", "weight", "dynamic", "static")

# KV-cache quantization schemes, in compressed-tensors `QuantizationArgs` form.
#
# Static per-tensor is what vLLM's `--kv-cache-dtype fp8_e4m3` consumes: it
# reads a scalar `k_scale`/`v_scale` off each attention module and applies it
# without any runtime reduction. `dynamic: False` is the whole point — it is
# what makes the oneshot pass *calibrate* the scales instead of deferring them.
KV_CACHE_SCHEMES: dict[str, dict[str, Any]] = {
    "fp8_e4m3": {
        "num_bits": 8,
        "type": "float",
        "strategy": "tensor",
        "dynamic": False,
        "symmetric": True,
    },
}

# compressed-tensors writes calibrated KV scales under these parameter names,
# on the modules matched by its own `KV_CACHE_TARGETS` —
# `['re:.*(self_attn|attention)$']` as of 0.18.0.
#
# That regex is why a hybrid model yields fewer scales than it has layers, and
# why that is *correct*: Qwen3.8's 48 DeltaNet layers are named `linear_attn`,
# match nothing, and have no KV cache to quantize — only the 16 `self_attn`
# layers do. Check the count against the model's real attention layers, never
# against its layer count.
KV_SCALE_SUFFIXES: tuple[str, ...] = ("k_scale", "v_scale")


@dataclass
class PTQConfig:
    """Configuration for a compressed-tensors PTQ run."""

    model_id: str
    """HF repo id or local path of the bf16 source checkpoint."""

    out_dir: Path
    """Directory the quantized checkpoint is written to."""

    corpus_files: list[Path]
    """Calibration text files (typically ``corpus.cal.txt`` from build_corpora.py)."""

    ctx: int = 8192
    """Calibration sequence length. Deliberately above the GGUF pipeline's 4096
    default (``pipeline.DEFAULT_IMATRIX_CTX``) — the serving target is long
    context. Bounded by what fits during calibration forward passes.

    This is a **packing** parameter as much as a runtime flag: a corpus packed
    for one ctx and read at another straddles window boundaries. Match it to the
    ``ctx`` recorded in the corpus's ``corpora_audit.json``."""

    budget_tokens: int = 524_288
    """Total calibration token budget, strided evenly across all corpus files.

    Read this as *sequences*, not tokens: at ctx 32768 the default is only 16
    sequences, far too few for a stable Hessian. Scale it so the run sees ~128
    distinct sequences (4M tokens at 32K)."""

    scheme: str = "W4A16"
    """llmcompressor quantization scheme (W4A16 = int4 weights / bf16 activations)."""

    group_size: int = 128
    """Weight quantization group size along the input dim. ``-1`` means
    per-output-channel. Anything other than 128 leaves the preset behind and
    builds an explicit ``config_groups`` (see :func:`build_config_groups`)."""

    symmetric: bool = True
    """Symmetric weight grid (no zero-point). ``False`` stores an int8 zero
    point per group — strictly more expressive, ~3% larger, and what a card
    describing "INT4, asymmetric (zero-point), group_size=32" is using."""

    observer: str | None = None
    """How the per-group scale is chosen from the calibration statistics.
    ``None`` keeps llmcompressor's default (minmax). ``"mse"`` searches the
    clipping range minimising reconstruction error — slower, usually better at
    4 bits and below."""

    actorder: str | None = None
    """GPTQ activation reordering. ``"static"`` is the serving-friendly one:
    columns are permuted once at quantization time, so there is no runtime
    ``g_idx`` indirection."""

    dampening_frac: float = 0.01
    """Hessian diagonal dampening, as a fraction of the mean diagonal. Raise it
    if the Cholesky fails on a near-singular layer."""

    block_size: int = 128
    """GPTQ column block size — how many columns are quantized before the error
    is propagated to the rest of the layer."""

    kv_cache_dtype: str | None = None
    """Calibrate and bake static KV-cache scales into the checkpoint.
    ``"fp8_e4m3"`` halves the KV cache at serve time (``vllm serve
    --kv-cache-dtype fp8_e4m3`` picks the scales up automatically), which is
    what buys longer context and more concurrent sequences on the same card.
    ``None`` leaves the cache at the serving dtype."""

    bypass_divisibility_checks: bool = False
    """Skip llmcompressor's check that every quantized weight's input dim is
    divisible by ``group_size``. It is on by default for a reason — a
    non-divisible layer is what most runtimes choke on — but a small group size
    on an unusual intermediate dim can trip it where vLLM would in fact serve
    the result. Set this only after checking the offending layer by hand."""

    ignore: tuple[str, ...] = field(default_factory=lambda: DEFAULT_IGNORE)
    """Module patterns excluded from quantization (llmcompressor syntax)."""

    device_map: str = "auto"
    """Passed to model loading; "auto" lets accelerate shard/offload. GPTQ runs
    layer-sequentially, so a model larger than VRAM still calibrates."""

    trust_remote_code: bool = False

    model_class: str | None = None
    """transformers class used to load the checkpoint. ``None`` uses
    ``AutoModelForCausalLM``, which resolves via ``MODEL_FOR_CAUSAL_LM_MAPPING``
    and on multimodal checkpoints picks the **text-only** class — silently
    dropping every tower tensor from the export. Qwen3.5 is exactly that case:
    ``qwen3_5`` maps to ``Qwen3_5ForCausalLM`` (no ``model.visual.*``), while the
    checkpoint declares ``Qwen3_5ForConditionalGeneration``. Name the class here
    to keep the tower (and then ignore it — see :func:`audit_ignore`)."""

    pipeline: str = "sequential"
    """llmcompressor calibration pipeline. "sequential" (layer-sliced, lowest
    memory) breaks on architectures with cross-layer state — gemma-4's shared
    KV (`shared_kv_states` flows from share-source layers to later sliding
    layers) raises KeyError under the tracer. Use "basic" there: full-model
    forwards with hooks; slower and hungrier (all Hessians accumulate at once)
    but architecture-agnostic."""

    def validate(self) -> None:
        if not self.corpus_files:
            raise ValueError("at least one corpus file is required")
        missing = [str(p) for p in self.corpus_files if not Path(p).is_file()]
        if missing:
            raise ValueError(f"corpus files not found: {missing}")
        if self.ctx < 128:
            raise ValueError(f"ctx must be >= 128, got {self.ctx}")
        if self.budget_tokens < self.ctx:
            raise ValueError(
                f"budget_tokens ({self.budget_tokens}) must cover at least one "
                f"ctx-sized sample ({self.ctx})"
            )
        if self.scheme not in SUPPORTED_SCHEMES:
            raise ValueError(f"unsupported scheme: {self.scheme}")
        if self.pipeline not in ("sequential", "basic", "independent"):
            raise ValueError(f"unsupported pipeline: {self.pipeline}")

        # compressed-tensors would accept any group size; vLLM's packed-int4
        # (marlin) kernels would not, and the checkpoint only fails at serve
        # time — long after the calibration is spent. Reject here instead.
        if self.group_size not in VLLM_GROUP_SIZES:
            raise ValueError(
                f"group_size {self.group_size} is not servable by vLLM's int4 "
                f"kernels; use one of {VLLM_GROUP_SIZES} (-1 = per-channel)"
            )
        if self.observer is not None and normalize_observer(self.observer) not in {
            normalize_observer(o) for o in KNOWN_OBSERVERS
        }:
            raise ValueError(
                f"unknown observer {self.observer!r}; known: {KNOWN_OBSERVERS}"
            )
        if self.actorder is not None and self.actorder not in KNOWN_ACTORDER:
            raise ValueError(
                f"unknown actorder {self.actorder!r}; known: {KNOWN_ACTORDER}"
            )
        if not 0 < self.dampening_frac < 1:
            raise ValueError(f"dampening_frac must be in (0, 1), got {self.dampening_frac}")
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")

        if self.kv_cache_dtype is not None and self.kv_cache_dtype not in KV_CACHE_SCHEMES:
            raise ValueError(
                f"unknown kv_cache_dtype {self.kv_cache_dtype!r}; "
                f"known: {tuple(KV_CACHE_SCHEMES)}"
            )

        # A custom weight grid has to be expressed as config_groups, and this
        # module only knows how to write a weight-only group. Refuse rather than
        # silently drop the customization on an activation-quantized scheme.
        if self.custom_weight_grid() and self.scheme not in WEIGHT_ONLY_SCHEMES:
            raise ValueError(
                f"scheme {self.scheme} quantizes activations as well as weights, so "
                "it is supported at its preset settings only; group_size / symmetric "
                "/ observer / actorder overrides need a weight-only scheme "
                f"({tuple(WEIGHT_ONLY_SCHEMES)})"
            )

    def custom_weight_grid(self) -> bool:
        """True when the weight grid deviates from the preset scheme, and an
        explicit ``config_groups`` therefore has to be built."""
        return (
            self.group_size != 128
            or not self.symmetric
            or self.observer is not None
            or self.actorder is not None
        )


def build_config_groups(cfg: PTQConfig) -> dict[str, Any] | None:
    """Explicit llmcompressor ``config_groups`` for a custom weight grid.

    Returns ``None`` when the preset ``scheme`` already describes the run — the
    two are mutually exclusive at the modifier, and passing a scheme is the
    better-tested path, so it stays the default.

    Emitted as plain dicts, not ``compressed_tensors`` pydantic models, so this
    stays importable (and testable) without the ``vllm-ptq`` extra;
    llmcompressor validates them on the way in.
    """
    if not cfg.custom_weight_grid():
        return None

    weights: dict[str, Any] = {
        "num_bits": WEIGHT_ONLY_SCHEMES[cfg.scheme],
        "type": "int",
        "symmetric": cfg.symmetric,
        # -1 is the sentinel for per-output-channel; anything else is a group
        # along the input dim.
        "strategy": "channel" if cfg.group_size == -1 else "group",
    }
    if cfg.group_size != -1:
        weights["group_size"] = cfg.group_size
    if cfg.observer is not None:
        weights["observer"] = cfg.observer
    if cfg.actorder is not None:
        weights["actorder"] = cfg.actorder

    return {
        "group_0": {
            "targets": ["Linear"],
            "weights": weights,
            "input_activations": None,
            "output_activations": None,
        }
    }


def build_kv_cache_scheme(cfg: PTQConfig) -> dict[str, Any] | None:
    """KV-cache ``QuantizationArgs`` for ``cfg.kv_cache_dtype``, or ``None``."""
    if cfg.kv_cache_dtype is None:
        return None
    return dict(KV_CACHE_SCHEMES[cfg.kv_cache_dtype])


def build_calibration_samples(cfg: PTQConfig, tokenizer: Any) -> list[list[int]]:
    """Tokenize the corpus files and return ctx-sized id lists, evenly strided
    to ~``budget_tokens`` total. Deterministic; budget is split across files
    proportionally to their token counts (every file contributes)."""
    import torch

    from quant_tuner.calibrate._ingest import sample_chunks

    per_file_ids: list[torch.Tensor] = []
    for path in cfg.corpus_files:
        text = Path(path).read_text(encoding="utf-8")
        ids = tokenizer.encode(text, add_special_tokens=False)
        per_file_ids.append(torch.tensor(ids, dtype=torch.long))

    total = sum(t.numel() for t in per_file_ids)
    if total == 0:
        raise ValueError("corpus files tokenized to zero tokens")

    samples: list[list[int]] = []
    for ids in per_file_ids:
        share = max(cfg.ctx, int(cfg.budget_tokens * ids.numel() / total))
        for chunk in sample_chunks(ids, cfg.ctx, share):
            samples.append(chunk.tolist())
    if not samples:
        raise ValueError("no calibration samples produced — corpus too small for ctx")
    return samples


def resolve_model_class(model_class: str | None) -> Any:
    """Resolve a transformers class by name; ``None`` → ``AutoModelForCausalLM``."""
    import transformers

    if model_class is None:
        return transformers.AutoModelForCausalLM
    cls = getattr(transformers, model_class, None)
    if cls is None:
        raise ValueError(f"transformers has no class named {model_class!r}")
    return cls


def model_module_names(model_id: str | Path, model_class: str | None = None) -> list[str]:
    """Module names of the *instantiated* model, built on the meta device.

    This — not the weight map — is what llmcompressor's ``ignore`` patterns are
    actually matched against, and the two can disagree profoundly: a tensor
    present in the checkpoint but absent from the module tree is dropped from
    the export entirely. See :func:`dropped_tensors`.
    """
    import torch
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id)
    cls = resolve_model_class(model_class)
    with torch.device("meta"):
        model = cls.from_config(cfg) if model_class is None else cls._from_config(cfg)
    return [n for n, _ in model.named_modules() if n]


def checkpoint_tensor_names(model_id: str | Path) -> list[str]:
    """Every tensor name in a local checkpoint, from the index or the shards."""
    root = Path(model_id)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        return list(json.loads(index.read_text())["weight_map"])

    from safetensors import safe_open

    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors found under {root}")
    names: list[str] = []
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            names.extend(f.keys())
    return names


def checkpoint_module_names(model_id: str | Path) -> list[str]:
    """Module names implied by a local checkpoint's safetensors weight map.

    Derived by stripping the parameter leaf (``.weight``/``.bias``/``.scale``…)
    off each tensor name. This describes what is *on disk*, which is not
    necessarily what gets loaded — pair it with :func:`model_module_names`.
    """
    names = checkpoint_tensor_names(model_id)
    return sorted({n.rsplit(".", 1)[0] if "." in n else n for n in names})


def _match_counts(names: list[str], ignore: tuple[str, ...] | list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pattern in ignore:
        if pattern.startswith("re:"):
            rx = re.compile(pattern[3:])
            counts[pattern] = sum(1 for n in names if rx.match(n))
        else:
            counts[pattern] = sum(1 for n in names if n == pattern)
    return counts


def audit_ignore(
    model_id: str | Path,
    ignore: tuple[str, ...] | list[str],
    model_class: str | None = None,
) -> dict[str, int]:
    """Map each ignore pattern to the number of *live modules* it matches.

    A pattern matching **zero** modules is the silent failure this exists to
    catch: ``DEFAULT_IGNORE``'s ``re:.*vision_tower.*`` matches nothing on a
    checkpoint whose tower is ``model.visual.*``, so the tower gets quantized to
    int4 against a text-only calibration corpus without any error.
    """
    return _match_counts(model_module_names(model_id, model_class), ignore)


def dropped_tensors(model_id: str | Path, model_class: str | None = None) -> list[str]:
    """Checkpoint tensors with no corresponding module in the loaded model.

    These are silently discarded by ``from_pretrained`` and will be **absent
    from the export** — a strictly worse outcome than being quantized, and one
    no ``ignore`` entry can prevent. On Qwen3.5 this is how the ``mtp.*`` draft
    head disappears (transformers lists it in
    ``_keys_to_ignore_on_load_unexpected``), and, under the default text-only
    class, the entire ``model.visual.*`` tower with it.
    """
    live = set(model_module_names(model_id, model_class))
    return [n for n in checkpoint_module_names(model_id) if n not in live]


def count_kv_scales(out_dir: str | Path) -> dict[str, int]:
    """Count calibrated ``k_scale``/``v_scale`` tensors in an exported checkpoint.

    Read this per *suffix*, not as a total: a hybrid-attention model only has a
    KV cache on its softmax-attention layers (Qwen3.8: 16 of 64 — the other 48
    are linear/DeltaNet layers carrying recurrent state), so the right count is
    "one per real attention layer", not "one per layer".
    """
    # The leaf matters here, so read raw tensor names — checkpoint_module_names
    # would strip `.k_scale` off and report the attention module instead.
    raw = checkpoint_tensor_names(out_dir)
    return {
        suffix: sum(1 for n in raw if n.endswith("." + suffix) or n == suffix)
        for suffix in KV_SCALE_SUFFIXES
    }


def verify_export(out_dir: str | Path, cfg: PTQConfig) -> dict[str, Any]:
    """Check that the export actually carries what was asked for.

    Two silent failures this catches, both of which produce a checkpoint that
    loads and serves without complaint:

    - **no ``quantization_config``** — vLLM serves it as bf16, at full size,
      and the only symptom is that it did not get smaller.
    - **a requested KV scheme that wrote no scales** — vLLM falls back to an
      uncalibrated cache. Nothing in the run log says so.

    Returns the observed summary (also stored in ``quant_tuner_ptq.json``).
    """
    out = Path(out_dir)
    config = json.loads((out / "config.json").read_text())
    if "quantization_config" not in config:
        raise RuntimeError(
            "exported config.json has no quantization_config — PTQ did not apply"
        )
    qcfg = config["quantization_config"]

    kv_scales = count_kv_scales(out)
    observed = {
        "format": qcfg.get("format"),
        "config_groups": qcfg.get("config_groups"),
        "kv_cache_scheme": qcfg.get("kv_cache_scheme"),
        "kv_scale_tensors": kv_scales,
    }

    if cfg.kv_cache_dtype is not None:
        if not qcfg.get("kv_cache_scheme"):
            raise RuntimeError(
                f"kv_cache_dtype={cfg.kv_cache_dtype!r} was requested but the exported "
                "quantization_config has no kv_cache_scheme — the KV observers never "
                "attached. vLLM would serve this with an uncalibrated cache."
            )
        if not any(kv_scales.values()):
            raise RuntimeError(
                f"kv_cache_scheme is present but the checkpoint holds no "
                f"{'/'.join(KV_SCALE_SUFFIXES)} tensors — calibration produced no "
                "scales. Check that the calibration forwards actually ran through "
                "attention (a traced sequential pipeline that skipped attention "
                "yields exactly this)."
            )
    return observed


def _fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_ptq(cfg: PTQConfig) -> Path:
    """Quantize ``cfg.model_id`` to a compressed-tensors checkpoint at
    ``cfg.out_dir``. Returns the output directory.

    Requires the ``vllm-ptq`` extra (``uv sync --extra vllm-ptq``).
    """
    cfg.validate()

    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import GPTQModifier
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "llmcompressor is required for vLLM PTQ — install the 'vllm-ptq' extra"
        ) from exc

    from transformers import AutoTokenizer

    from datasets import Dataset

    kv_cache_scheme = build_kv_cache_scheme(cfg)
    if kv_cache_scheme is not None and "kv_cache_scheme" not in GPTQModifier.model_fields:
        # Older llmcompressor releases have no KV-cache support on the mixin.
        # Silently dropping the kwarg would spend the whole calibration and
        # produce a checkpoint with no scales, so refuse before the model load.
        raise RuntimeError(
            "the installed llmcompressor GPTQModifier has no 'kv_cache_scheme' field — "
            "KV-cache calibration needs a newer release (>= 0.5). Upgrade the "
            "'vllm-ptq' extra or drop --kv-cache-dtype."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id, trust_remote_code=cfg.trust_remote_code
    )
    samples = build_calibration_samples(cfg, tokenizer)

    # llmcompressor consumes a tokenized dataset: input_ids + attention_mask.
    dataset = Dataset.from_list(
        [{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in samples]
    )

    requested = resolve_model_class(cfg.model_class)
    model = requested.from_pretrained(
        cfg.model_id,
        torch_dtype="bfloat16",
        device_map=cfg.device_map,
        trust_remote_code=cfg.trust_remote_code,
    )
    # A named class that does not survive the load means the export will be
    # missing whatever that class was chosen to keep — hours of calibration
    # spent on the wrong module tree, discoverable only afterwards from
    # provenance. Fail before any of it is spent.
    if cfg.model_class is not None and type(model).__name__ != cfg.model_class:
        raise RuntimeError(
            f"requested model_class={cfg.model_class!r} but from_pretrained "
            f"returned {type(model).__name__!r} — the export would not contain "
            "the modules that class was selected for"
        )

    # Preset scheme *or* explicit config_groups — never both; llmcompressor
    # rejects the pair. build_config_groups() returns None whenever the preset
    # already describes the requested grid, which keeps the common path on the
    # better-tested scheme= branch.
    config_groups = build_config_groups(cfg)
    modifier_kwargs: dict[str, Any] = {
        "ignore": list(cfg.ignore),
        "dampening_frac": cfg.dampening_frac,
        "block_size": cfg.block_size,
        # Under the basic pipeline every module's Hessian accumulates at once —
        # offload them to CPU RAM or the largest layers OOM the GPU.
        "offload_hessians": (cfg.pipeline == "basic"),
        "bypass_divisibility_checks": cfg.bypass_divisibility_checks,
    }
    if config_groups is not None:
        modifier_kwargs["config_groups"] = config_groups
    else:
        modifier_kwargs["targets"] = "Linear"
        modifier_kwargs["scheme"] = cfg.scheme
    if kv_cache_scheme is not None:
        modifier_kwargs["kv_cache_scheme"] = kv_cache_scheme

    recipe = GPTQModifier(**modifier_kwargs)

    # `processor` must be passed explicitly: given a dataset, llmcompressor
    # otherwise auto-initializes one via AutoProcessor, which raises on a
    # multimodal checkpoint whose processor needs image/video config it cannot
    # resolve. Our dataset is already tokenized, so the tokenizer is the correct
    # processor here — it is used for saving, not for preprocessing. The
    # annotation says `str | ProcessorMixin | None`, but llmcompressor accepts
    # and expects a tokenizer on the text path; the type is narrower than the
    # contract.
    oneshot(
        model=model,
        processor=tokenizer,  # type: ignore[arg-type]
        dataset=dataset,
        recipe=recipe,
        max_seq_length=cfg.ctx,
        num_calibration_samples=len(samples),
        output_dir=str(cfg.out_dir),
        pipeline=cfg.pipeline,
    )
    tokenizer.save_pretrained(str(cfg.out_dir))

    out = Path(cfg.out_dir)
    observed = verify_export(out, cfg)

    provenance = {
        "tool": "quant_tuner.vllm_export.w4a16",
        "model_id": str(cfg.model_id),
        "scheme": cfg.scheme,
        "group_size": cfg.group_size,
        "symmetric": cfg.symmetric,
        "observer": cfg.observer,
        "actorder": cfg.actorder,
        "dampening_frac": cfg.dampening_frac,
        "block_size": cfg.block_size,
        "bypass_divisibility_checks": cfg.bypass_divisibility_checks,
        "config_groups": config_groups,
        "kv_cache_dtype": cfg.kv_cache_dtype,
        "kv_cache_scheme": kv_cache_scheme,
        "ctx": cfg.ctx,
        "budget_tokens": cfg.budget_tokens,
        "num_samples": len(samples),
        "model_class": type(model).__name__,
        "pipeline": cfg.pipeline,
        "ignore": list(cfg.ignore),
        "ignore_match_counts": _match_counts(
            [n for n, _ in model.named_modules() if n], cfg.ignore
        ),
        "dropped_checkpoint_tensors": dropped_tensors(cfg.model_id, cfg.model_class),
        "exported": observed,
        "corpus": [
            {"path": str(p), "sha256": _fingerprint(Path(p))} for p in cfg.corpus_files
        ],
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "quant_tuner_ptq.json").write_text(json.dumps(provenance, indent=2))
    return out
