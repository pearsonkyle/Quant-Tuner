"""Torch-side KL divergence between a bf16 reference and a quantized checkpoint.

Why this exists
---------------
``bench/kld.py`` shells out to ``llama-perplexity --kl-divergence-base`` and so
only speaks **GGUF**. A compressed-tensors safetensors checkpoint (the
``vllm_export`` path) cannot be fed to it, and there is no other HF-side KLD in
the tree. This module is the equivalent for that path.

Comparability with the GGUF ladder
----------------------------------
Deliberately mirrors the *shape* of the GGUF ladder's measurement so the two can
be read side by side — same six eval corpora, each its own distribution and
**never concatenated**, each chunked at ``eval_ctx`` 8192, reporting **median**
KLD (robust to per-token tails) and top-token agreement.

It is **not** numerically comparable, for two reasons that must be stated
wherever these numbers appear:

1. The reference here is the **bf16 HF model**, whereas the GGUF ladder's
   reference is the F16 GGUF. Same weights, different runtime.
2. Unlike ``llama-perplexity`` — which has no ``--parse-special`` — this
   tokenizes chat control tokens to their real single ids. That makes the
   ``tools``/``agentic``/``broad``/``cal8k`` numbers *more* correct here, and
   therefore not interchangeable with the GGUF card's.

Method
------
For each position, with ``p`` the reference distribution and ``q`` the
quantized one::

    KLD = sum_v p(v) * (log p(v) - log q(v))

accumulated in fp32 and **chunked over the vocab dimension** — at 248,320 vocab
by 8192 positions a single fp32 logits tensor is 8 GB, so materializing
``softmax`` over the whole thing is not an option. Two passes over the vocab:
one for a numerically-stable ``logsumexp`` (online rescaling) plus argmax, one
to accumulate the divergence.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

# The six exp-060-32k eval distributions, in the order the ladder reports them.
# cal8k is a FIT probe (drawn from the calibration distribution), NOT a holdout —
# it is expected to score better and must never be read as generalization.
DEFAULT_EVAL_CORPORA: tuple[tuple[str, str], ...] = (
    ("external", "corpus.eval.txt"),
    ("general", "corpus.eval.general.txt"),
    ("tools", "corpus.eval.tools.txt"),
    ("agentic", "corpus.eval.agentic.txt"),
    ("broad", "corpus.eval.broad.txt"),
    ("cal8k", "corpus.eval.cal8k.txt"),
)

DEFAULT_EVAL_CTX = 8192
"""Matches the GGUF ladder's eval chunking so the comparison has the same shape."""


@dataclass
class HFKLDMetrics:
    """Per-corpus result. Field order is the CSV column order."""

    corpus: str
    n_tokens: int
    n_chunks: int
    n_positions: int
    median_kld: float
    mean_kld: float
    p90_kld: float
    p99_kld: float
    max_kld: float
    top1_agree: float
    top5_agree: float
    ref_ppl: float
    quant_ppl: float
    ppl_ratio: float


CSV_COLUMNS: tuple[str, ...] = tuple(HFKLDMetrics.__dataclass_fields__)


def chunk_corpus(
    path: Path,
    tokenizer: Any,
    ctx: int = DEFAULT_EVAL_CTX,
    drop_last: bool = True,
) -> list[list[int]]:
    """Tokenize ``path`` and split into non-overlapping ``ctx``-sized chunks.

    ``add_special_tokens=False`` because the corpora are already chat-templated
    text; in-text control tokens encode to their single special ids by default
    (never pass ``split_special_tokens=True`` — see CLAUDE.md).
    """
    ids = tokenizer.encode(Path(path).read_text(encoding="utf-8"), add_special_tokens=False)
    chunks = [ids[i : i + ctx] for i in range(0, len(ids), ctx)]
    if drop_last and len(chunks) > 1 and len(chunks[-1]) < ctx:
        chunks.pop()
    return [c for c in chunks if len(c) >= 2]


def _logsumexp_and_topk(
    logits: torch.Tensor, vocab_chunk: int, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable per-row logsumexp and top-k indices, chunked over the vocab dim.

    Uses online rescaling so only one pass over the vocab is needed: when a
    later chunk raises the running max, the accumulated sum is rescaled rather
    than recomputed.
    """
    import torch

    n_pos, vocab = logits.shape
    running_max = torch.full((n_pos,), float("-inf"), device=logits.device, dtype=torch.float32)
    running_sum = torch.zeros(n_pos, device=logits.device, dtype=torch.float32)
    top_vals = torch.full((n_pos, k), float("-inf"), device=logits.device, dtype=torch.float32)
    top_idx = torch.zeros((n_pos, k), device=logits.device, dtype=torch.long)

    for start in range(0, vocab, vocab_chunk):
        block = logits[:, start : start + vocab_chunk].to(torch.float32)
        block_max = block.max(dim=1).values
        new_max = torch.maximum(running_max, block_max)
        # Rescale the running sum onto the new max, then add this block.
        running_sum = running_sum * torch.exp(running_max - new_max) + torch.exp(
            block - new_max.unsqueeze(1)
        ).sum(dim=1)
        running_max = new_max

        width = min(k, block.shape[1])
        b_vals, b_idx = block.topk(width, dim=1)
        cat_vals = torch.cat([top_vals, b_vals], dim=1)
        cat_idx = torch.cat([top_idx, b_idx + start], dim=1)
        sel = cat_vals.topk(k, dim=1)
        top_vals = sel.values
        top_idx = cat_idx.gather(1, sel.indices)

    return running_max + torch.log(running_sum), top_idx


def compare_logits(
    ref_logits: torch.Tensor,
    quant_logits: torch.Tensor,
    targets: torch.Tensor,
    vocab_chunk: int = 16384,
    top_k: int = 5,
) -> dict[str, torch.Tensor]:
    """Per-position KLD, top-1/top-5 agreement and both NLLs for one chunk.

    ``ref_logits``/``quant_logits`` are ``[n_pos, vocab]``; ``targets`` is the
    ``[n_pos]`` next-token id for each position. All reductions run in fp32 and
    never materialize a full fp32 ``[n_pos, vocab]`` tensor.
    """
    import torch

    ref_logz, ref_top = _logsumexp_and_topk(ref_logits, vocab_chunk, top_k)
    q_logz, q_top = _logsumexp_and_topk(quant_logits, vocab_chunk, top_k)

    n_pos, vocab = ref_logits.shape
    kld = torch.zeros(n_pos, device=ref_logits.device, dtype=torch.float32)
    for start in range(0, vocab, vocab_chunk):
        end = start + vocab_chunk
        r = ref_logits[:, start:end].to(torch.float32) - ref_logz.unsqueeze(1)
        q = quant_logits[:, start:end].to(torch.float32) - q_logz.unsqueeze(1)
        # sum_v p * (log p - log q); p = exp(r) is formed only per vocab block.
        kld += (torch.exp(r) * (r - q)).sum(dim=1)

    # Numerical floor: KLD is non-negative by construction, but fp32 accumulation
    # over 248k terms can land a hair below zero when the two are near-identical.
    kld = kld.clamp_min(0.0)

    tgt = targets.unsqueeze(1)
    ref_nll = ref_logz - ref_logits.gather(1, tgt).squeeze(1).to(torch.float32)
    q_nll = q_logz - quant_logits.gather(1, tgt).squeeze(1).to(torch.float32)

    return {
        "kld": kld,
        "top1": (ref_top[:, 0] == q_top[:, 0]).to(torch.float32),
        "top5": (ref_top[:, 0].unsqueeze(1) == q_top).any(dim=1).to(torch.float32),
        "ref_nll": ref_nll,
        "quant_nll": q_nll,
    }


@dataclass
class _Accum:
    kld: list = field(default_factory=list)
    top1: list = field(default_factory=list)
    top5: list = field(default_factory=list)
    ref_nll: list = field(default_factory=list)
    quant_nll: list = field(default_factory=list)


def _forward_logits(model: Any, ids: list[int], device: str) -> torch.Tensor:
    import torch

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    # no_grad, NOT inference_mode. compressed-tensors decompresses the int4 weights
    # inside its quantized_forward; anything materialized under inference_mode becomes
    # an *inference tensor*, which cannot be used again once the module has been moved
    # between devices ("Inference tensors do not track version counter"). Two-pass
    # evaluation moves each model on and off the GPU per corpus, so it hits this on the
    # SECOND corpus — the first one always passes, which makes it look like a memory
    # problem rather than a mode problem. no_grad produces identical numerics.
    with torch.no_grad():
        out = model(input_ids=input_ids)
    return out.logits[0]


def _reduce(acc: _Accum, corpus: str, chunks: list[list[int]]) -> HFKLDMetrics:
    import torch

    kld = torch.cat(acc.kld)
    ref_nll = torch.cat(acc.ref_nll)
    quant_nll = torch.cat(acc.quant_nll)
    ref_ppl = float(torch.exp(ref_nll.mean()))
    quant_ppl = float(torch.exp(quant_nll.mean()))
    return HFKLDMetrics(
        corpus=corpus,
        n_tokens=sum(len(c) for c in chunks),
        n_chunks=len(chunks),
        n_positions=int(kld.numel()),
        median_kld=float(kld.median()),
        mean_kld=float(kld.mean()),
        p90_kld=float(kld.quantile(0.90)),
        p99_kld=float(kld.quantile(0.99)),
        max_kld=float(kld.max()),
        top1_agree=float(torch.cat(acc.top1).mean()) * 100.0,
        top5_agree=float(torch.cat(acc.top5).mean()) * 100.0,
        ref_ppl=ref_ppl,
        quant_ppl=quant_ppl,
        ppl_ratio=quant_ppl / ref_ppl if ref_ppl else float("nan"),
    )


def evaluate_corpus(
    ref_model: Any,
    quant_model: Any,
    chunks: list[list[int]],
    corpus: str,
    device: str = "cuda",
    vocab_chunk: int = 16384,
    skip_first: int = 1,
    progress: bool = True,
) -> HFKLDMetrics:
    """Both models resident: run each chunk through both, reduce, move on.

    ``skip_first`` drops the leading positions of every chunk, whose predictions
    are conditioned on almost no context and would otherwise dominate the tail
    statistics. Both models see identical inputs, so this does not bias the
    comparison — it only removes a high-variance head.
    """
    import torch

    acc = _Accum()
    for i, ids in enumerate(chunks):
        ref_logits = _forward_logits(ref_model, ids, device)
        quant_logits = _forward_logits(quant_model, ids, device)

        # Position t predicts token t+1, so the last position has no target.
        lo, hi = skip_first, len(ids) - 1
        targets = torch.tensor(ids[lo + 1 : hi + 1], dtype=torch.long, device=device)
        stats = compare_logits(
            ref_logits[lo:hi], quant_logits[lo:hi], targets, vocab_chunk=vocab_chunk
        )
        for key in ("kld", "top1", "top5", "ref_nll", "quant_nll"):
            getattr(acc, key).append(stats[key].to("cpu"))
        del ref_logits, quant_logits, stats
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        if progress:
            print(f"  [{corpus}] chunk {i + 1}/{len(chunks)}", flush=True)

    return _reduce(acc, corpus, chunks)


def evaluate_corpus_two_pass(
    ref_model: Any,
    quant_model: Any,
    chunks: list[list[int]],
    corpus: str,
    device: str = "cuda",
    vocab_chunk: int = 16384,
    skip_first: int = 1,
    progress: bool = True,
) -> HFKLDMetrics:
    """One model on the GPU at a time, reference logits cached in CPU RAM.

    Insurance for the case where both models will not co-reside: a bf16 27B
    reference is ~52 GB, and if transformers *decompresses* the W4A16 checkpoint
    on load rather than keeping it packed, the pair needs ~104 GB — more than
    this box's 97.9 GB card.

    The cost is CPU RAM: one fp16 ``[positions, vocab]`` tensor per chunk, so
    ~4 GB per 8192-token chunk at a 248k vocab. Bounded per corpus (the largest
    here is 18 chunks ≈ 73 GB), never across corpora, and freed before the next
    one.

    Caching in fp16 is **lossless** for bf16 logits: fp16 carries 10 mantissa
    bits to bf16's 7, and logits sit far inside fp16's ±65504 range, so the cast
    cannot lose information. Results are therefore bit-identical to
    :func:`evaluate_corpus` — only the schedule differs (asserted by
    ``test_kld_hf.py::test_two_pass_cache_roundtrip_is_lossless``).
    """
    import torch

    # Evict BEFORE admitting. On the first corpus the quantized model is still on
    # the CPU so the order looks harmless, but this function leaves it resident on
    # the GPU when it returns — so on every subsequent corpus, admitting the
    # reference first means both models are on the card at once and the second
    # allocation OOMs. Symmetric with the swap below, which already evicts first.
    quant_model.to("cpu")
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    ref_model.to(device)

    cached: list[torch.Tensor] = []
    for i, ids in enumerate(chunks):
        logits = _forward_logits(ref_model, ids, device)
        cached.append(logits[skip_first : len(ids) - 1].to("cpu", torch.float16))
        del logits
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        if progress:
            print(f"  [{corpus}] ref {i + 1}/{len(chunks)}", flush=True)

    ref_model.to("cpu")
    quant_model.to(device)
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    acc = _Accum()
    for i, ids in enumerate(chunks):
        quant_logits = _forward_logits(quant_model, ids, device)
        lo, hi = skip_first, len(ids) - 1
        targets = torch.tensor(ids[lo + 1 : hi + 1], dtype=torch.long, device=device)
        stats = compare_logits(
            cached[i].to(device), quant_logits[lo:hi], targets, vocab_chunk=vocab_chunk
        )
        for key in ("kld", "top1", "top5", "ref_nll", "quant_nll"):
            getattr(acc, key).append(stats[key].to("cpu"))
        cached[i] = torch.empty(0)  # release as we go
        del quant_logits, stats
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        if progress:
            print(f"  [{corpus}] quant {i + 1}/{len(chunks)}", flush=True)

    return _reduce(acc, corpus, chunks)


def write_csv(rows: list[HFKLDMetrics], out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return out


def iter_corpora(
    corpora_dir: Path,
    names: tuple[tuple[str, str], ...] = DEFAULT_EVAL_CORPORA,
) -> Iterator[tuple[str, Path]]:
    for label, filename in names:
        path = Path(corpora_dir) / filename
        if not path.is_file():
            raise FileNotFoundError(f"eval corpus missing: {path}")
        yield label, path
