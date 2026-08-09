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

    group_size: int = 128

    ignore: tuple[str, ...] = field(default_factory=lambda: DEFAULT_IGNORE)
    """Module patterns excluded from quantization (llmcompressor syntax)."""

    device_map: str = "auto"
    """Passed to model loading; "auto" lets accelerate shard/offload. GPTQ runs
    layer-sequentially, so a model larger than VRAM still calibrates."""

    trust_remote_code: bool = False

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

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from datasets import Dataset

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id, trust_remote_code=cfg.trust_remote_code
    )
    samples = build_calibration_samples(cfg, tokenizer)

    # llmcompressor consumes a tokenized dataset: input_ids + attention_mask.
    dataset = Dataset.from_list(
        [{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in samples]
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype="bfloat16",
        device_map=cfg.device_map,
        trust_remote_code=cfg.trust_remote_code,
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
    recipe = GPTQModifier(
        targets="Linear",
        scheme=cfg.scheme,
        ignore=list(cfg.ignore),
        offload_hessians=(cfg.pipeline == "basic"),
    )

    oneshot(
        model=model,
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
        "group_size": cfg.group_size,
        "ctx": cfg.ctx,
        "budget_tokens": cfg.budget_tokens,
        "num_samples": len(samples),
        "ignore": list(cfg.ignore),
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
    return out
