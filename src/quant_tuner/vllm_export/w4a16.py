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
  in this module (config validation, corpus sampling) is importable and
  unit-testable without it. Install with the ``vllm-ptq`` extra.
- Multimodal models (gemma-4 E4B: vision + audio towers, per-layer embeddings)
  are handled by *ignoring* everything outside the language-model decoder
  linears. The default ignore list matches what Google's official QAT W4A16
  checkpoint leaves unquantized. Text-only models simply never match those
  patterns, so the defaults are safe across architectures.
- The exported dir gains ``quant_tuner_ptq.json`` provenance (corpus files +
  SHA-256 fingerprints, ctx, token budget, scheme) so a checkpoint can always
  be traced back to what calibrated it.
"""

from __future__ import annotations

import hashlib
import json
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


@dataclass
class PTQConfig:
    """Configuration for a W4A16 compressed-tensors PTQ run."""

    model_id: str
    """HF repo id or local path of the bf16 source checkpoint."""

    out_dir: Path
    """Directory the quantized checkpoint is written to."""

    corpus_files: list[Path]
    """Calibration text files (typically ``corpus.cal.txt`` from build_corpora.py)."""

    ctx: int = 8192
    """Calibration sequence length. Deliberately above the GGUF pipeline's 4096
    default (``pipeline.DEFAULT_IMATRIX_CTX``) — the serving target is long
    context. Bounded by what fits during calibration forward passes."""

    budget_tokens: int = 524_288
    """Total calibration token budget, strided evenly across all corpus files."""

    scheme: str = "W4A16"
    """llmcompressor quantization scheme (W4A16 = int4 weights / bf16 activations)."""

    kv_cache_scheme: str | None = None
    """Quantize the KV cache alongside the weights. ``"fp8_e4m3"`` emits calibrated
    per-tensor ``k_scale``/``v_scale`` into the checkpoint, which vLLM consumes
    under ``--kv-cache-dtype fp8_e4m3``; ``None`` leaves the cache at bf16.

    This is ORTHOGONAL to ``scheme`` -- KV quantization is applied to the k/v
    projections' outputs, not to weights, so W4A16 + fp8 KV is a normal pairing
    and does not require quantized activations.

    Worth sizing before reaching for it. On gemma-4-E4B it saves ~1.77 GiB at
    131k context and ~0.46 GiB at 32k, because 35 of 42 layers are sliding
    attention with a 512-token window (they hold 512 positions regardless of
    context) and only 7 are full attention, over 2 KV heads. That is small
    against the weights for a single stream -- but it scales linearly with
    concurrency, so it earns its place on a server and not on a workstation."""

    group_size: int = 128

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
        if self.scheme not in ("W4A16", "W8A8", "W8A16", "FP8_DYNAMIC"):
            raise ValueError(f"unsupported scheme: {self.scheme}")
        if self.kv_cache_scheme not in (None, "fp8_e4m3"):
            raise ValueError(
                f"unsupported kv_cache_scheme: {self.kv_cache_scheme} "
                "(supported: fp8_e4m3, or None to leave the cache at bf16)"
            )
        if self.pipeline not in ("sequential", "basic", "independent"):
            raise ValueError(f"unsupported pipeline: {self.pipeline}")


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


def checkpoint_module_names(model_id: str | Path) -> list[str]:
    """Module names implied by a local checkpoint's safetensors weight map.

    Derived by stripping the parameter leaf (``.weight``/``.bias``/``.scale``…)
    off each tensor name. This describes what is *on disk*, which is not
    necessarily what gets loaded — pair it with :func:`model_module_names`.
    """
    root = Path(model_id)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        names = list(json.loads(index.read_text())["weight_map"])
    else:
        from safetensors import safe_open

        shards = sorted(root.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no safetensors found under {root}")
        names = []
        for shard in shards:
            with safe_open(str(shard), framework="pt") as f:
                names.extend(f.keys())
    return sorted({n.rsplit(".", 1)[0] if "." in n else n for n in names})


def _match_counts(names: list[str], ignore: tuple[str, ...] | list[str]) -> dict[str, int]:
    import re

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

    # Preset schemes carry their own group size (W4A16 = group-128); llmcompressor
    # >= 0.12 rejects a top-level group_size kwarg. Non-default groupings need a
    # custom config_groups spec — reject loudly rather than silently ignoring.
    if cfg.group_size != 128:
        raise ValueError(
            "only group_size=128 is supported (defined by the preset scheme); "
            "use a custom config_groups recipe for other groupings"
        )
    # Under the basic pipeline every module's Hessian accumulates at once —
    # offload them to CPU RAM or the largest layers OOM the GPU.
    recipe: Any = GPTQModifier(
        targets="Linear",
        scheme=cfg.scheme,
        ignore=list(cfg.ignore),
        offload_hessians=(cfg.pipeline == "basic"),
    )

    if cfg.kv_cache_scheme is not None:
        # compressed-tensors spells an fp8-e4m3 cache as a static, symmetric,
        # per-tensor float8 scheme on the attention output. Calibration is what
        # makes it worth doing: without scales vLLM falls back to a dynamic
        # guess, which is exactly the accuracy that calibration buys back.
        kv_spec = {
            "num_bits": 8,
            "type": "float",
            "strategy": "tensor",
            "dynamic": False,
            "symmetric": True,
        }
        # Verified on llmcompressor 0.13.0: GPTQModifier takes kv_cache_scheme
        # directly and resolves it to zp_dtype=torch.float8_e4m3fn, so W4A16 and
        # the fp8 cache come out of one modifier. The extra runs because this
        # package only floors llmcompressor at >=0.8 and the field lives on
        # QuantizationMixin, which older GPTQModifiers do not inherit -- and a
        # silently dropped kwarg would yield a checkpoint with no KV scales that
        # still looks like a success.
        import inspect

        if "kv_cache_scheme" in inspect.signature(GPTQModifier).parameters:
            recipe = GPTQModifier(
                targets="Linear",
                scheme=cfg.scheme,
                ignore=list(cfg.ignore),
                offload_hessians=(cfg.pipeline == "basic"),
                kv_cache_scheme=kv_spec,
            )
        else:
            from llmcompressor.modifiers.quantization import QuantizationModifier

            recipe = [
                recipe,
                QuantizationModifier(
                    targets="Linear",
                    ignore=list(cfg.ignore),
                    kv_cache_scheme=kv_spec,
                ),
            ]

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

    provenance = {
        "tool": "quant_tuner.vllm_export.w4a16",
        "model_id": str(cfg.model_id),
        "scheme": cfg.scheme,
        "kv_cache_scheme": cfg.kv_cache_scheme,
        "group_size": cfg.group_size,
        "ctx": cfg.ctx,
        "budget_tokens": cfg.budget_tokens,
        "num_samples": len(samples),
        "model_class": type(model).__name__,
        "ignore": list(cfg.ignore),
        "ignore_match_counts": _match_counts(
            [n for n, _ in model.named_modules() if n], cfg.ignore
        ),
        "dropped_checkpoint_tensors": dropped_tensors(cfg.model_id, cfg.model_class),
        "corpus": [
            {"path": str(p), "sha256": _fingerprint(Path(p))} for p in cfg.corpus_files
        ],
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out = Path(cfg.out_dir)
    (out / "quant_tuner_ptq.json").write_text(json.dumps(provenance, indent=2))

    # A checkpoint without a quantization_config would load as bf16 and silently
    # not be quantized — fail loudly instead.
    config = json.loads((out / "config.json").read_text())
    if "quantization_config" not in config:
        raise RuntimeError(
            "exported config.json has no quantization_config — PTQ did not apply"
        )
    # Same reasoning one level down: a KV scheme that was requested but did not
    # survive into the config leaves a checkpoint vLLM serves with a bf16 cache,
    # indistinguishable from success except for the memory it does not save.
    if cfg.kv_cache_scheme is not None and not config["quantization_config"].get(
        "kv_cache_scheme"
    ):
        raise RuntimeError(
            f"kv_cache_scheme={cfg.kv_cache_scheme!r} was requested but the "
            "exported quantization_config has none — the calibrated k/v "
            "scales were not written"
        )
    return out
