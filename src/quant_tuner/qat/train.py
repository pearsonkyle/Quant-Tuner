"""QAT trainer: masked-loss, big-window, LR-scheduled, memory-frugal.

The core of the continued-QAT pipeline for native-ternary models (see
``docs/qat_optimization_audit.md`` for the full rationale). Consumes a pre-tokenized,
assistant-MASKED corpus (:mod:`quant_tuner.qat.corpus`) and continues training the ternary
weights with a straight-through estimator. Key properties:

  * MASKED-CE forward: the lm_head runs only at labeled positions instead of the full
    ``[1, seq, vocab]`` logits tensor (-4-5 GB peak). Parity with HF's ForCausalLMLoss is
    unit-tested.
  * Adafactor (factored 2nd moment, ~MBs vs AdamW's two fp32 states = 55.6 GB at all-36)
    -> full-36-layer fp32 training fits in ~70 GB. Pure per-tensor loop, no MPS deadlock.
    ``weight_decay`` defaults to 0 for BOTH optimizers (decay erodes ternary codes to 0).
  * ``compute_dtype='bf16'``: fp32-master trick (:mod:`quant_tuner.qat.master_opt`) —
    masters+clip+step in fp32, forward/backward in bf16.
  * ``kd_teacher``: online distillation from a same-vocab dense teacher (KL on labeled
    positions only, via ``logits_to_keep``).
  * ``resume``: continue from ``trained_latents.pt`` (data order, step, Adafactor state;
    corpus fingerprint must match). Atomic checkpoints (tmp + rename).
  * Code-flip telemetry every checkpoint — the instrument for the LR probe (at lr 5e-5 the
    expected flip count is ~zero; see the audit).

The CLI shim is ``scripts/exp058_qat_train_v2.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from quant_tuner.qat._device import MPS_MAX_WINDOW, resolve_backend
from quant_tuner.qat.attention import (
    DEFAULT_CHUNK,
    capture_prefix,
    clear_prefix,
    enable_chunked_sdpa,
    enable_fp32_gqa_repeat,
    use_prefix,
)
from quant_tuner.qat.corpus import corpus_fingerprint
from quant_tuner.qat.kd_precompute import kd_loss_from_topk
from quant_tuner.qat.kd_table import KDTable, stop_logp_of
from quant_tuner.qat.master_opt import MasterOptimizer
from quant_tuner.qat.stop_probe import StopProbe
from quant_tuner.qat.stop_probe import format_line as stop_probe_fmt
from quant_tuner.qat.ternary import TernaryLinear, ternarize_group

REPO = Path(__file__).resolve().parents[3]
MODEL = REPO / "out" / "exp-057" / "model"

__all__ = ["MPS_MAX_WINDOW", "QATConfig", "main", "train_qat"]


@dataclass
class QATConfig:
    corpus: Path
    out: Path = REPO / "out" / "exp-058" / "trained"
    model_dir: Path = MODEL
    train_layers: int = 18
    layers: str | None = None
    epochs: float = 3.0
    grad_accum: int = 8
    lr: float = 5e-5
    optim: str = "adamw"
    weight_decay: float = 0.0
    beta1: float | None = None
    dtype: str = "fp32"
    compute_dtype: str = "fp32"
    kd_teacher: Path | None = None
    #: precomputed top-K teacher table (kd_precompute). Offline KD: no teacher
    #: in memory, so it composes with an all-36 student where --kd-teacher
    #: (which loads a dense teacher alongside) does not.
    kd_table: Path | None = None
    kd_alpha: float = 0.5
    kd_temp: float = 1.0
    #: β for the stop-token log-ratio hinge (0 = off; needs a forced-stop --kd-table).
    #: The KL's restoring force on the stop logit is P_s−P_t — proportional to the drift,
    #: so α-insensitive at small P; this hinge is O(1) in log space at any magnitude.
    stop_anchor: float = 0.0
    #: free band (nats) around the teacher's log P(stop) before the hinge engages.
    stop_anchor_margin: float = 1.0
    #: abort when the probe's CONTROL falls below this (0 disables). The diagnostic-only
    #: abort missed the a0.75 run's real failure: sentence_period retreated under the
    #: threshold at step 150 while after_tool_call fell 0.9998 -> 0.8876 — the sft32k
    #: loop failure, unguarded.
    probe_abort_control: float = 0.0
    val_corpus: Path | None = None
    val_every: int = 20
    # Termination telemetry cadence. 0 disables. Defaults to the validation cadence
    # so the two series line up on the same steps in the report.
    probe_every: int = 25
    #: abort the run (exit 3, checkpoint saved) when the probe's diagnostic exceeds this.
    #: 0 disables. A collapsing run is visible by ~step 50 and monotone from there; this
    #: converts an 11-hour post-mortem into a 40-minute one.
    probe_abort: float = 0.0
    #: "group-scale" gives each latent tensor lr * median(s)/median-of-medians (clamped
    #: to [0.5, 2.0]) so large-scale tensors (v/up/gate) are not flip-starved. fp32 only.
    lr_scale: str = "none"
    val_windows: int = 16
    train_norms: bool = False
    resume: Path | None = None
    flip_sample: int = 8
    ckpt_every: int = 40
    ckpt_keep: int = 2
    warmup_frac: float = 0.05
    grad_spike_factor: float = 4.0
    #: "auto" (patch only where the backend needs it, or for --trained-tail),
    #: "on" (always patch — for benchmarking it against a fused kernel), "off"
    chunked_attention: str = "auto"
    #: None -> the backend's default (5 on MPS, off on CUDA); see qat._device
    empty_cache_every: int | None = None
    metrics_jsonl: bool = True
    trained_tail: int = 0
    stop_weight: float = 1.0
    device: str = "auto"
    #: torch.set_float32_matmul_precision; see the --matmul-precision help
    matmul_precision: str = "highest"
    #: Which layers are ternarized AT ALL (None = every layer). Distinct from ``layers``/
    #: ``train_layers``, which choose what gets GRADIENTS. On a natively-ternary model the
    #: two coincide — everything is already on the grid, so a frozen layer is ternary for
    #: free. On a DENSE model they must not: a progressive schedule needs a third state,
    #: "still bf16, not on the grid, not being moved there yet", and without this every
    #: layer would be ternarized from step 0 — the all-at-once approach the schedule exists
    #: to avoid.
    ternary_layers: str | None = None
    #: Tensor-name substrings kept DENSE wherever they appear, e.g. "down_proj". Measured
    #: on gemma-4-E4B (docs/gemma4_ternary_feasibility.md): ternarizing mlp.down_proj alone
    #: takes perplexity from ~19 to 149 while every other kind stays in band, so it is the
    #: one tensor worth spending bits on. Applied on top of ``ternary_layers``.
    dense_kinds: tuple[str, ...] = ()


def parse_layers(spec: str, n_layers: int) -> set[int]:
    """Parse '0-14,32,34,35' -> {0..14,32,34,35}. Empty -> all."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out |= set(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return {layer for layer in out if 0 <= layer < n_layers}


#: Where a decoder's layer list lives, most-specific first. A plain CausalLM keeps it at
#: ``model.model.layers``; a multimodal wrapper like ``Gemma4ForConditionalGeneration``
#: nests it under a language_model, and reading the wrong one raises rather than silently
#: training a vision tower — but only because we look it up instead of hard-coding.
_LAYER_PATHS = ("model.language_model.layers", "model.layers",
                "model.model.layers", "model.decoder.layers")


def decoder_layers(model):
    """The transformer block list, wherever this architecture puts it."""
    for path in _LAYER_PATHS:
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__len__") and len(obj):
            return obj
    raise AttributeError(
        f"no decoder layer list found on {type(model).__name__} (tried {_LAYER_PATHS}); "
        f"add its path to _LAYER_PATHS")


def wrap_model(model, n_train: int, layer_spec: str | None = None,
               train_norms: bool = False, ternary_spec: str | None = None,
               dense_kinds: tuple[str, ...] = ()) -> int:
    layers = decoder_layers(model)
    if layer_spec:
        trainable_idx = parse_layers(layer_spec, len(layers))
        label = f"explicit layers {sorted(trainable_idx)}"
    else:
        trainable_idx = set(range(max(0, len(layers) - n_train), len(layers)))
        label = f"last {n_train} layers"

    ternary_idx = (parse_layers(ternary_spec, len(layers)) if ternary_spec
                   else set(range(len(layers))))
    if not trainable_idx <= ternary_idx:
        raise ValueError(
            f"layers {sorted(trainable_idx - ternary_idx)} are set to train but not to "
            f"ternarize — their gradients would move a latent nothing quantizes, which "
            f"is plain fine-tuning wearing a QAT label")
    n_dense = 0
    #: Parameter ids of weights deliberately kept DENSE inside a trainable layer. They are
    #: ordinary ``Linear.weight``s, not ``.linear.weight`` latents, so the requires_grad
    #: pass below cannot recognise them by name — and they DO train: letting the dense
    #: tensors adapt to their ternarized neighbours is most of why a partial schedule beats
    #: all-at-once.
    dense_trainable: set[int] = set()
    #: Parameter ids of the ternary LATENTS in trainable layers, collected as they are
    #: wrapped. Identity, not name: the old name match (".linear.weight" plus a layer index
    #: parsed from "layers.") also hits a multimodal model's TOWERS, whose encoders have
    #: their own layers.N and submodules literally called `linear`. On gemma-4-E4B that
    #: marked 85 tensors / 167.8 M params of vision+audio tower trainable — they never
    #: receive a gradient from a text forward, so nothing failed loudly; they just consumed
    #: optimizer state and were exposed to weight decay. Bonsai has no towers, which is why
    #: this survived until a multimodal model was tried.
    latent_trainable: set[int] = set()

    def swap(mod, trainable, ternary=True, prefix=""):
        nonlocal n_dense
        c = 0
        for name, child in list(mod.named_children()):
            full = f"{prefix}{name}"
            if isinstance(child, torch.nn.Linear) and child.in_features % 128 == 0:
                if not ternary or any(k in full for k in dense_kinds):
                    n_dense += 1
                    if trainable:
                        dense_trainable.add(id(child.weight))
                    continue
                if not trainable:
                    # Frozen layer: shipped weights are already exactly on the ternary
                    # grid, so TernaryLinear would be a bit-exact no-op costing ~5 W-sized
                    # transient allocs per forward (x2 under checkpoint recompute). Skip
                    # the wrap when we can PROVE exactness; wrap otherwise (e.g. a layer
                    # trained in an earlier run that drifted off-grid).
                    with torch.no_grad():
                        _, _, w_hat = ternarize_group(child.weight)
                        exact = torch.equal(w_hat, child.weight)
                    if exact:
                        child.weight.requires_grad_(False)
                        continue
                    print(f"[qat]   frozen linear off-grid -> wrapping: {name}", flush=True)
                setattr(mod, name, TernaryLinear(child, trainable=trainable))
                if trainable:
                    latent_trainable.add(id(child.weight))
                c += 1
            else:
                c += swap(child, trainable, ternary, f"{full}.")
        return c

    nw = sum(swap(layer, i in trainable_idx, i in ternary_idx)
             for i, layer in enumerate(layers))
    trainable_ids = latent_trainable | dense_trainable
    if train_norms:
        for i in sorted(trainable_idx):
            trainable_ids |= {id(q) for n_, q in layers[i].named_parameters() if "norm" in n_}
    for p in model.parameters():
        p.requires_grad_(id(p) in trainable_ids)
    nt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dense_note = (f"; {n_dense} linears left DENSE "
                  f"({len(dense_trainable)} of them trainable)" if n_dense else "")
    print(f"[qat] training {label} ({len(trainable_idx)} layers"
          f"{', +norms' if train_norms else ''}); ternary layers "
          f"{len(ternary_idx)}/{len(layers)}; wrapped {nw}{dense_note}; "
          f"trainable {nt/1e9:.2f}B", flush=True)
    return nw


def latent_lr_mults(
    named_params, *, clamp: tuple[float, float] = (0.5, 2.0),
) -> dict[str, float]:
    """Per-tensor lr multipliers proportional to the median TWN group scale.

    A flip needs the latent to cross its group's threshold, and that distance is
    proportional to the group scale ``s`` — while adafactor's normalized step is ~lr
    for every coordinate. So at uniform lr, large-scale tensors are structurally
    starved of code flips. Measured on the shipped model: v/up/gate carry the largest
    scales (0.035-0.048, growing with depth) and flip least (0.0000-0.0002% in the
    59-step arms); q/down carry the smallest (~0.022-0.026) and flip most (0.0145%,
    0.0057%); sorting the tracked tensors by ``s`` almost exactly anti-orders them by
    observed flip rate. ``lr_t = lr * s_t / median(s)`` (clamped) equalizes flip
    opportunity; non-2D params (norms, biases) keep 1.0.
    """
    from quant_tuner.qat.ternary import DEFAULT_GROUP_SIZE, ternarize_group
    scales: dict[str, float] = {}
    with torch.no_grad():
        for n, p in named_params:
            if p.ndim == 2 and p.shape[1] % DEFAULT_GROUP_SIZE == 0:
                _, s, _ = ternarize_group(p.detach())
                scales[n] = float(s.float().median())
    if not scales:
        return {n: 1.0 for n, _ in named_params}
    ref = sorted(scales.values())[len(scales) // 2]
    return {n: (min(max(scales[n] / ref, clamp[0]), clamp[1]) if n in scales else 1.0)
            for n, _ in named_params}


def lr_at(step, total, base, warmup_frac=0.05):
    warm = max(1, int(total * warmup_frac))
    if step < warm:
        return base * step / warm
    prog = (step - warm) / max(1, total - warm)
    return 0.1 * base + 0.9 * base * 0.5 * (1 + math.cos(math.pi * prog))


class GradSpikeGuard:
    """Skip an optimizer step whose PRE-clip grad norm dwarfs the recent median.

    On the sft8k-full run the loss went 1.06 -> 9.80 within five steps of the LR reaching
    its peak, and took ~90 steps (~9 GPU-hours) of cosine decay to unwind. Gradient
    clipping does not prevent that: clipping rescales the direction but still takes a
    full-size step along it, and it hides the excursion from every logged number.

    The guard compares each step's pre-clip norm against the median of a trailing window.
    A step above `factor` x median is dropped (grads zeroed, LR schedule untouched), so a
    handful of pathological batches cannot move the weights.

    **It is NOT warmup-aware, and that is a real hazard — read this before enabling it.**
    The only thing delaying it is `min_history`, so with 20 norms accumulated it goes live
    at step 21 regardless of where warmup ends. On the sft8k-full run the LR peaked at
    step 30 and the loss rose 1.06 -> 9.80 over the next five steps while validation
    improved *monotonically* through it — a healthy post-warmup reorganization, not a
    divergence. A factor of 4.0 would have skipped exactly those steps and suppressed it
    silently, since a skipped step is invisible in the loss curve.

    So: leave it OFF (`factor=0`) for a run whose warmup transient is expected, and use it
    only to protect a run that has already been shown to diverge. Skipping is recorded in
    `n_skipped` so a run that skips constantly is visible as a too-low `factor` rather
    than as a mysteriously slow run.
    """

    def __init__(self, factor: float = 4.0, window: int = 25, min_history: int = 20):
        self.factor = factor
        self.window = window
        self.min_history = min_history
        self.history: list[float] = []
        self.n_skipped = 0
        self.last_median = 0.0

    def check(self, norm: float) -> bool:
        """True if this step should be SKIPPED. Feeds the history either way."""
        if not math.isfinite(norm):
            self.n_skipped += 1
            return True
        skip = False
        if self.factor > 0 and len(self.history) >= self.min_history:
            h = sorted(self.history)
            self.last_median = h[len(h) // 2]
            if self.last_median > 0 and norm > self.factor * self.last_median:
                skip = True
                self.n_skipped += 1
        # a skipped (outlier) norm must not enter the history, or a run of spikes drags
        # the median up until the guard stops firing
        if not skip:
            self.history.append(norm)
            self.history[:] = self.history[-self.window:]
        return skip


#: Labeled positions per lm_head call when logits are not needed. `[K, vocab]` fp32 at
#: K=8064/V=151669 is 4.6 GB for the logits alone (~14 GB with softmax + backward), and K
#: swings from ~400 to the full window depending on a window's trainable density — an
#: intermittent multi-GB spike that OOM-kills a long run at an unpredictable step. 1024
#: caps it at ~0.6 GB.
LOGIT_CHUNK = 1024


@contextlib.contextmanager
def prefix_window(model, ids: torch.Tensor, n_prefix: int):
    """Encode the first ``n_prefix`` tokens under no_grad into the attention prefix store.

    This is what makes a 32K window trainable on a 128 GB box. Activation memory under
    gradient checkpointing is ``n_layers x S x hidden`` — 19.3 GB fp32 at S=32768 for this
    model, on top of 32.8 GB of params and 27.8 GB of grads, which is what pushed the
    16128 attempt into terminal swap. Prefix K/V costs ``n_layers x 2 x n_kv x head_dim``
    per token instead: 288 KB/token fp32 here, i.e. **7.2 GB for a 24576-token prefix**,
    and carries no autograd graph at all.

    The K/V ride in `qat.attention`'s prefix store rather than a transformers `Cache`,
    because `GradientCheckpointingLayer.__call__` nulls `past_key_values` whenever
    ``gradient_checkpointing and training`` — a checkpointed tail handed a real cache
    attends to nothing, and the loss still falls.

    The trade is real and worth stating: gradients do not reach the prefix, so the model
    learns to *use* long context, not to *build* it. For the failure this addresses —
    termination, whose signal is the trajectory's final `<|im_end|>` — the whole gradient
    lives in the tail anyway.
    """
    if not n_prefix:
        yield
        return
    with torch.no_grad(), capture_prefix() as store:
        model.model(input_ids=ids[:, :n_prefix])
    n_attn = sum(1 for m in model.modules() if type(m).__name__.endswith("Attention"))
    if len(store) != n_attn:
        raise RuntimeError(f"captured {len(store)} of {n_attn} attention modules — the "
                           "tail would train with a partial context")
    try:
        # The block must span the BACKWARD too, not just the forward: gradient
        # checkpointing re-runs each layer during backward, and a recompute that no longer
        # sees the prefix produces a differently-shaped attention and torch raises
        # CheckpointError ("Recomputed values ... have different metadata").
        with use_prefix():
            yield
    finally:
        clear_prefix()


LOG_HALF = -0.6931471805599453   # log(0.5): teacher-side split between
                                 # continue-positions and stop-positions


def masked_forward(model, ids: torch.Tensor, lbl: torch.Tensor, *,
                   need_logits: bool = True, logit_chunk: int = LOGIT_CHUNK,
                   n_prefix: int = 0, weights: torch.Tensor | None = None,
                   kd=None, kd_temp: float = 1.0,
                   stop_anchor=None):
    """Masked-CE forward: lm_head only at labeled positions.

    Selects positions t with lbl[t+1] != -100 (HF shift semantics), runs the
    decoder trunk on the full window, then the lm_head on the K selected hidden
    states only. Returns (ce_loss, logits [1,K,V] fp32 or None, keep_idx) — the mean CE
    over exactly the same target set as transformers' ForCausalLMLoss.

    ``n_prefix > 0`` assumes the caller is inside `prefix_window`: the first ``n_prefix``
    tokens have already been encoded under no_grad and only the remaining tokens carry
    gradient. Targets falling inside the prefix are dropped from the loss — they have no
    graph — so the reported CE is over the tail's targets only and is NOT comparable with
    a full-window CE on the same data.

    ``kd`` is an optional :class:`~quant_tuner.qat.kd_table.KDWindow` of precomputed
    teacher top-K logprobs for THIS window, already aligned to ``keep_idx``. When given the
    return is ``(ce, logits, keep_idx, kl, anchor)`` — two elements longer — and the KD
    term is computed inside the SAME logit chunks as CE, because a separate pass would
    materialize the ``[K, V]`` logits a second time (5.8 GiB at 29% density on a 32768
    window).

    ``stop_anchor`` (KD path only) is ``(stop_id, t_stop_logp [K] fp32, margin)``: adds a
    per-position hinge on the STOP token's log-prob gap to the teacher,
    ``relu(|log P_s(stop) − log P_t(stop)| − margin)``, returned (mean over K) as the
    5th element (else None). This exists because the KL's restoring force on the stop
    logit is ``P_s − P_t`` — proportional to the drift itself, ~0.01 when the student
    drifts through 1e-2 against a 1e-6 teacher, which is why tripling α (0.5 → 0.75)
    barely changed the collapse trajectory. The hinge's gradient is O(1) in LOG space
    at any drift magnitude; the margin leaves an e^margin band around the teacher free
    so it does not fight ordinary calibration differences.

    ``weights`` is an optional per-vocab-id CE weight vector, used to upweight the
    terminating `<|im_end|>` target: it is 0.57% of labels but carries the entire stop
    decision, so at uniform weight the run optimizes ~176 "keep going" tokens for every
    "stop" one.

    With ``need_logits=False`` (the plain masked-CE path — only KD needs the logits
    themselves) the lm_head + CE run in ``logit_chunk``-sized blocks of labeled
    positions, each recomputed in the backward pass, so peak logits memory is
    ``logit_chunk × vocab`` instead of ``K × vocab``. The loss is identical: chunk
    losses are re-weighted by chunk size, so this is a mean over all K, not a mean of
    means (they differ whenever K is not a multiple of the chunk).
    """
    tgt = lbl[:, 1:]
    keep_idx = (tgt[0] != -100).nonzero(as_tuple=True)[0]

    if n_prefix > 0:
        # position_ids must be the ABSOLUTE positions: RoPE is applied before the attention
        # function sees K, so the stored prefix keys carry positions 0..n_prefix-1 and the
        # tail has to continue the sequence, not restart it.
        pos = torch.arange(n_prefix, ids.shape[1], device=ids.device).unsqueeze(0)
        hidden = model.model(input_ids=ids[:, n_prefix:],
                             position_ids=pos).last_hidden_state      # [1, S-n_prefix, H]
        # keep_idx indexes the SHIFTED targets, i.e. hidden position t predicts tgt[t];
        # only t >= n_prefix has a graph. Re-base onto the tail's own coordinates.
        n_drop = int((keep_idx < n_prefix).sum())
        keep_idx = keep_idx[keep_idx >= n_prefix]
        h = hidden[:, keep_idx - n_prefix, :]
        if kd is not None and n_drop:
            # The KD window was validated against the FULL keep_idx (rows in position
            # order); targets inside the prefix carry no gradient and were just dropped,
            # so drop their teacher rows too or every chunk pairs a student position
            # with an earlier token's distribution. Same for the anchor's teacher rows.
            kd = kd.slice(n_drop, len(kd))
            if stop_anchor is not None:
                stop_anchor = (stop_anchor[0], stop_anchor[1][n_drop:], stop_anchor[2])
    else:
        hidden = model.model(input_ids=ids).last_hidden_state         # [1, S, H]
        h = hidden[:, keep_idx, :]                                    # [1, K, H]
    targets = tgt[0, keep_idx]
    K = keep_idx.numel()
    if K == 0:
        raise ValueError("no labeled target carries a gradient (prefix covers the window)")
    if kd is not None and len(kd) != K:
        raise ValueError(
            f"KD window has {len(kd)} rows for {K} gradient-carrying targets — the "
            f"teacher rows would be paired with the wrong positions.")
    if stop_anchor is not None:
        if kd is None:
            raise ValueError("stop_anchor requires the KD path (teacher rows)")
        if int(stop_anchor[1].numel()) != K:
            raise ValueError(
                f"stop anchor has {int(stop_anchor[1].numel())} teacher rows for {K} "
                f"targets — would pair positions wrongly.")

    def anchor_pen(lg, t_stop):
        """One-sided hinge on the stop token's log-prob gap to the teacher; summed.

        One-sided PER POSITION TYPE, not L1: continue-positions (teacher P(stop) < 0.5)
        outnumber stop-positions 176:1 in this corpus, so a symmetric |gap| hinge exerts
        a massive net-DOWNWARD trunk-level pressure on P(stop) — measured: it crushed
        the diagnostic to 0.0000 while collapsing the control 0.9987 -> 0.6974 by step
        175. Here each position only penalizes drift in its harmful direction — stopping
        MORE than the teacher where the teacher continues, stopping LESS where the
        teacher stops — and exerts zero force once the student is on the safe side, so
        the 176:1 imbalance has nothing to push with.
        """
        sid, _, margin = stop_anchor
        s_stop = lg[:, sid] - torch.logsumexp(lg, dim=-1)
        direction = torch.where(t_stop > LOG_HALF, -1.0, 1.0)
        return (direction * (s_stop - t_stop) - margin).clamp_min(0.0).sum()

    if need_logits or (logit_chunk >= K and kd is None):
        logits = model.lm_head(h).float()                    # [1, K, V]
        ce = F.cross_entropy(logits[0], targets, weight=weights)
        if kd is not None:
            kl = kd_loss_from_topk(logits[0], kd.idx, kd.logp, kd.tail, temp=kd_temp)
            anchor = (anchor_pen(logits[0], stop_anchor[1]) / K
                      if stop_anchor is not None else None)
            return ce, (logits if need_logits else None), keep_idx, kl, anchor
        return ce, (logits if need_logits else None), keep_idx

    def block_sum(hb, tb):
        return F.cross_entropy(model.lm_head(hb).float(), tb, weight=weights,
                               reduction="sum")

    def block_sum_kd(hb, tb, kidx, klogp, ktail, tstop):
        """CE sum, KD sum AND stop-anchor sum for one chunk, from ONE lm_head call.

        KD has to run inside the same chunk as CE: computing it separately would
        materialize [K, V] logits a second time, which is the 5.8 GiB spike (at 29%
        density on a 32768 window) that the chunking exists to avoid. All are summed,
        not averaged, so the caller can divide by the true totals — a mean of per-chunk
        means is wrong whenever K is not a multiple of the chunk.
        """
        lg = model.lm_head(hb).float()
        ce_s = F.cross_entropy(lg, tb, weight=weights, reduction="sum")
        kl_s = kd_loss_from_topk(lg, kidx, klogp, ktail, temp=kd_temp) * hb.shape[0]
        an_s = (anchor_pen(lg, tstop) if tstop is not None
                else lg.new_zeros(()))
        return ce_s, kl_s, an_s

    if kd is not None:
        total = h.new_zeros((), dtype=torch.float32)
        kl_total = h.new_zeros((), dtype=torch.float32)
        an_total = h.new_zeros((), dtype=torch.float32)
        denom = (weights[targets].sum() if weights is not None
                 else torch.as_tensor(float(K), device=h.device))
        for i in range(0, K, logit_chunk):
            hb, tb = h[0, i:i + logit_chunk], targets[i:i + logit_chunk]
            kb = kd.slice(i, i + logit_chunk)
            ts = (stop_anchor[1][i:i + logit_chunk].to(hb.device)
                  if stop_anchor is not None else None)
            if torch.is_grad_enabled() and hb.requires_grad:
                ce_s, kl_s, an_s = torch.utils.checkpoint.checkpoint(
                    block_sum_kd, hb, tb, kb.idx, kb.logp, kb.tail, ts,
                    use_reentrant=False)
            else:
                ce_s, kl_s, an_s = block_sum_kd(hb, tb, kb.idx, kb.logp, kb.tail, ts)
            total = total + ce_s
            kl_total = kl_total + kl_s
            an_total = an_total + an_s
        anchor = an_total / float(K) if stop_anchor is not None else None
        return total / denom, None, keep_idx, kl_total / float(K), anchor

    total = h.new_zeros((), dtype=torch.float32)
    # With a `weight` vector the denominator is sum(w[target]), not K — otherwise
    # upweighting a rare token silently rescales the whole loss (and with it the
    # effective LR) by however often that token happened to appear in the window.
    denom = (weights[targets].sum() if weights is not None
             else torch.as_tensor(float(K), device=h.device))
    for i in range(0, K, logit_chunk):
        hb, tb = h[0, i:i + logit_chunk], targets[i:i + logit_chunk]
        if torch.is_grad_enabled() and hb.requires_grad:
            total = total + torch.utils.checkpoint.checkpoint(
                block_sum, hb, tb, use_reentrant=False)
        else:
            total = total + block_sum(hb, tb)
    return total / denom, None, keep_idx


def kd_kl(teacher, ids: torch.Tensor, keep_idx: torch.Tensor,
          student_logits: torch.Tensor, temp: float) -> torch.Tensor:
    """KL(teacher || student) at the labeled positions, temperature-scaled.

    The teacher gets logits_to_keep=keep_idx (transformers >= 5 accepts an index
    tensor), so it never materializes full-vocab logits at unlabeled positions.
    """
    with torch.no_grad():
        t_logits = teacher(input_ids=ids, logits_to_keep=keep_idx).logits.float()
    t_logp = torch.log_softmax(t_logits[0] / temp, dim=-1)
    s_logp = torch.log_softmax(student_logits[0] / temp, dim=-1)
    return F.kl_div(s_logp, t_logp, log_target=True, reduction="none").sum(-1).mean()


def latent_weights(model, opt) -> dict[str, torch.Tensor]:
    """Map module name -> the tensor that IS the trained latent.

    Under ``--compute-dtype bf16`` the live ``linear.weight`` is a bf16 *copy* of an fp32
    master owned by the optimizer, and :func:`export_qat` ternarizes the **masters**. Read
    the live copy instead and the flip telemetry describes a model that is never exported:
    bf16 carries 8 mantissa bits, so a latent sitting within ~0.2% of the TWN threshold
    ternarizes differently in the two, and flips get recorded at the wrong step. Since flip
    velocity — not loss — is how this project decides whether a ternary run is learning at
    all, that instrument has to read the same tensor the artifact will.

    Returns ``{}`` in the fp32 case, where the live weight already is the latent.
    """
    if not isinstance(opt, MasterOptimizer):
        return {}
    by_param = {id(p): m for p, m in zip(opt.params, opt.masters, strict=True)}
    return {name: by_param[id(mod.linear.weight)]
            for name, mod in model.named_modules()
            if isinstance(mod, TernaryLinear) and id(mod.linear.weight) in by_param}


def snapshot_codes(model, k: int = 8, latents: dict[str, torch.Tensor] | None = None,
                   ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Snapshot (codes int8, scale fp16) of k trainable linears spread across layers.

    ``latents`` (see :func:`latent_weights`) overrides where each module's latent is read
    from — required under bf16 compute, a no-op otherwise.
    """
    mods = [(n, m) for n, m in model.named_modules()
            if isinstance(m, TernaryLinear) and m.linear.weight.requires_grad]
    if not mods or k <= 0:
        return {}
    picks = sorted({round(i * (len(mods) - 1) / max(1, min(k, len(mods)) - 1))
                    for i in range(min(k, len(mods)))})
    snaps = {}
    with torch.no_grad():
        for i in picks:
            n, m = mods[i]
            w = (latents or {}).get(n, m.linear.weight)
            codes, scale, _ = ternarize_group(w.detach().float())
            snaps[n] = (codes.to(torch.int8).cpu(), scale.to(torch.float16).cpu())
    return snaps


def flip_report(model, snaps, prev: dict | None = None,
                latents: dict[str, torch.Tensor] | None = None) -> tuple[dict, str]:
    """Codes flipped / scale drift vs the start-of-run snapshot.

    Beyond the cumulative flip count this records three things the raw percentage
    conflates, each of which answered a real question about a live run:

    * ``sign_flip`` vs ``zero_to_nonzero``/``nonzero_to_zero`` — a tensor that
      reorganizes signs at constant density is doing something different from one
      recruiting weights that shipped as zero. Measured split: q/k and down_proj
      reorganize (ratio ~1), v_proj and gate_proj densify (ratio 3-8).
    * ``density`` — absolute nonzero fraction, so the direction of travel is readable
      without integrating the deltas.
    * ``flip_pct_delta`` (needs ``prev``) — cumulative flips cannot distinguish a
      tensor that settled early from one still oscillating; the per-interval velocity
      can. A run whose velocity has peaked on every tensor is converging.

    ``scale_drift`` stays the mean absolute relative move (comparable with older runs);
    ``scale_drift_signed`` is added because the absolute value hides whether scales are
    systematically growing or shrinking.

    ``latents`` must be passed whatever was passed to :func:`snapshot_codes`, or the
    comparison is against a differently-rounded baseline.
    """
    mods = dict(model.named_modules())
    stats, lines = {}, []
    with torch.no_grad():
        for name, (codes0, scale0) in snaps.items():
            w = (latents or {}).get(name, mods[name].linear.weight).detach().float()
            codes, scale, _ = ternarize_group(w)
            c = codes.to(torch.int8).cpu()
            flip_pct = 100.0 * (c != codes0).float().mean().item()
            z2nz = int(((codes0 == 0) & (c != 0)).sum())
            nz2z = int(((codes0 != 0) & (c == 0)).sum())
            sign = int(((codes0 != 0) & (c != 0) & (c != codes0)).sum())
            s0 = scale0.float()
            rel = (scale.to(torch.float16).cpu().float() - s0) / s0.clamp_min(1e-8)
            drift = rel.abs().mean().item()
            st = {"flip_pct": flip_pct, "zero_to_nonzero": z2nz, "nonzero_to_zero": nz2z,
                  "sign_flip": sign,
                  # >1 recruiting dead weights, <1 pruning, ~1 pure sign reorganization
                  "densify_ratio": (z2nz / nz2z) if nz2z else None,
                  "density": float((c != 0).float().mean()),
                  "density_start": float((codes0 != 0).float().mean()),
                  "scale_drift": drift, "scale_drift_signed": rel.mean().item(),
                  "numel": int(c.numel())}
            if prev and name in prev:
                st["flip_pct_delta"] = flip_pct - prev[name]["flip_pct"]
                st["z2nz_delta"] = z2nz - prev[name]["zero_to_nonzero"]
            stats[name] = st
            vel = f" Δ{st['flip_pct_delta']:+.4f}" if "flip_pct_delta" in st else ""
            lines.append(f"  {name}: flips {flip_pct:.4f}%{vel} "
                         f"(0->±:{z2nz} ±->0:{nz2z} ±->∓:{sign}) "
                         f"density {st['density_start']*100:.1f}->{st['density']*100:.1f}% "
                         f"scale-drift {drift*100:.2f}% ({st['scale_drift_signed']*100:+.2f}%)")
    return stats, "\n".join(lines)


def run_validation(model, ids_all, lbl_all, dev, max_windows: int,
                   n_prefix: int = 0) -> float:
    """Masked CE on held-out windows.

    ``n_prefix`` must match training: it changes which targets are scored (prefix targets
    are dropped), so a val number taken at a different split is not on the same scale.
    """
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(min(max_windows, ids_all.shape[0])):
            lbl = lbl_all[i:i + 1]
            if not bool((lbl[0, 1:] != -100).any()):
                continue
            if n_prefix and not bool((lbl[0, 1 + n_prefix:] != -100).any()):
                continue  # every target sits in the prefix; nothing to score
            ids = ids_all[i:i + 1].to(dev)
            with prefix_window(model, ids, n_prefix):
                ce, _, _ = masked_forward(model, ids, lbl.to(dev),
                                          need_logits=False, n_prefix=n_prefix)
            tot += float(ce)
            n += 1
    model.train()
    return tot / max(1, n)


def write_run_config(out: Path, cfg: QATConfig, **facts) -> Path:
    """Record everything needed to reproduce and compare this run, at step 0.

    Comparing two QAT runs after the fact needs the hyper-parameters, and those used to
    exist only in the launch command — so a run directory could not answer "what lr and
    window produced this?" once the shell was gone. Written before the first step so a
    killed run still explains itself, and never overwritten on ``--resume`` (the resumed
    leg gets its own numbered file) because the two legs can differ in lr or corpus and
    collapsing them would misattribute whichever one is read.
    """
    import dataclasses
    import subprocess

    rec: dict = {"kind": "run_config"}
    for f in dataclasses.fields(cfg):
        v = getattr(cfg, f.name)
        rec[f.name] = str(v) if isinstance(v, Path) else v
    rec.update(facts)
    rec["argv"] = sys.argv
    rec["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec["torch"] = torch.__version__
    try:  # provenance is best-effort — a dirty tree or no git must not kill a 10 h run
        rec["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        rec["git_commit"] = None

    path = out / "run_config.json"
    n = 1
    while path.exists():           # a resume leg is a new record, not a replacement
        path = out / f"run_config.{n}.json"
        n += 1
    path.write_text(json.dumps(rec, indent=2, default=str) + "\n")
    print(f"[qat] run config -> {path}", flush=True)
    return path


def train_qat(cfg: QATConfig) -> int:
    backend = resolve_backend(cfg.device)
    dev = backend.name
    empty_cache_every = (cfg.empty_cache_every if cfg.empty_cache_every is not None
                         else backend.default_empty_cache_every)
    print(f"[qat] device {backend.describe()}", flush=True)
    if cfg.matmul_precision != "highest":
        # TF32/bf16 tensor cores for the fp32 MATMULS only. This is a different knob from
        # --compute-dtype: the latents, the TWN threshold, ternarize_group and its
        # deliberate fp16 scale rounding are all elementwise fp32 and stay bit-exact, so
        # the codes AND the scales a step produces are unchanged. Only the matmul's
        # internal accumulation is reduced (TF32 keeps 10 mantissa bits, bf16 8).
        #
        # --compute-dtype bf16 rounds the latent itself to 8 mantissa bits before
        # ternarizing. Measured, that does NOT move the codes: on the shipped weights and
        # on real trained latents, 0 of 117M codes differ, because a ternary latent sits
        # at 0 or +-s while delta = 0.7*mean|W| sits between them — nothing is within even
        # fp32 precision of the threshold. What it does move is the SCALE (0.05-0.10% off
        # the fp16 value the exported Q2_0 carries) and the gradients, and therefore what
        # GradSpikeGuard sees as a spike.
        torch.set_float32_matmul_precision(cfg.matmul_precision)
        print(f"[qat] fp32 matmul precision '{cfg.matmul_precision}' "
              f"(tensor cores; latents and ternarization stay exact fp32)", flush=True)
    if backend.name == "cpu":
        print("[qat] WARNING: no accelerator found — training on CPU is ~100x slower "
              "and is almost certainly not what you want.", flush=True)
    if cfg.dtype == "bf16":
        print("[qat] WARNING: bf16 latents underflow the ternary threshold — no codes "
              "will flip at stable LRs. Use compute_dtype=bf16 (fp32 masters) instead.",
              flush=True)
    if cfg.compute_dtype == "bf16" and cfg.dtype != "fp32":
        sys.exit("[qat] compute_dtype bf16 requires dtype fp32 (fp32 masters)")
    dtype = torch.float32 if cfg.dtype == "fp32" else torch.bfloat16
    from transformers import AutoModelForCausalLM

    blob = torch.load(cfg.corpus, weights_only=False)
    ids_all, lbl_all = blob["ids"], blob["labels"]
    n_win, window = ids_all.shape
    if cfg.trained_tail and cfg.chunked_attention == "off":
        sys.exit("[qat] --trained-tail needs the patched attention: the prefix K/V ride in "
                 "qat.attention's store, not a transformers Cache. Drop "
                 "--no-chunked-attention.")
    # Query-chunked SDPA is REQUIRED on Metal (it removes the MPSGraph INT_MAX
    # score-tensor cap; bit-identical output — see qat.attention) and is what carries the
    # prefix K/V for --trained-tail on every device. On CUDA neither applies to a plain
    # full-gradient run: FlashAttention never materializes the score matrix, so the
    # chunked path is pure overhead there and stays off unless asked for.
    # fp32 + GQA has no fused SDPA kernel, so transformers' enable_gqa=True drops the
    # whole call to the math backend and materializes [batch, heads, S, S] — 7.75 GiB at
    # a 8064 window, which OOMs a 95 GiB card. Expanding K/V instead reaches the
    # memory-efficient kernel. bf16/fp16 keep the native grouped path.
    if backend.is_cuda and cfg.compute_dtype == "fp32":
        enable_fp32_gqa_repeat()
        print("[qat] fp32 GQA: expanding K/V so SDPA reaches the memory-efficient kernel "
              "(enable_gqa=True would fall back to math and materialize [heads,S,S])",
              flush=True)

    want_chunked = (cfg.chunked_attention == "on" or
                    (cfg.chunked_attention == "auto"
                     and (backend.needs_chunked_sdpa or cfg.trained_tail)))
    if want_chunked:
        enable_chunked_sdpa()
        print(f"[qat] chunked SDPA enabled (query blocks of {DEFAULT_CHUNK}) — the "
              f"{MPS_MAX_WINDOW}-token MPSGraph cap does not apply; the limit is memory",
              flush=True)
    elif backend.max_window and window > backend.max_window:
        sys.exit(f"[qat] window {window} > {backend.max_window}: MPS attention hits the "
                 f"MPSGraph INT_MAX limit (n_heads x S^2 must stay < 2^31; at 32 heads "
                 f"that is S <= {backend.max_window}). Either rebuild the corpus at 8064 or "
                 f"drop --no-chunked-attention.")
    else:
        print(f"[qat] stock SDPA on {backend.name} (fused/flash kernels; no score matrix "
              f"is materialized, so there is no window cap to chunk around)", flush=True)
    # Per-window source label, when the builder recorded one. The corpus mixes sources with
    # very different assistant fractions (0.08 refusals .. 0.79 broad-instruct), so a single
    # loss curve cannot say which data is driving the flips; a per-source breakdown can.
    win_src = blob.get("window_source")
    src_names = blob.get("source_names") or sorted((blob.get("per_source") or {}).keys())
    fp = blob.get("fingerprint") or corpus_fingerprint(ids_all, lbl_all)
    total_steps = int(cfg.epochs * n_win / cfg.grad_accum)
    print(f"[qat] corpus {n_win} windows x {window} ({blob.get('assistant_frac',0)*100:.0f}% masked, "
          f"fingerprint {fp}); {cfg.epochs} epochs -> {total_steps} steps @ accum {cfg.grad_accum}",
          flush=True)

    val_ids = val_lbl = None
    if cfg.val_corpus:
        vblob = torch.load(cfg.val_corpus, weights_only=False)
        val_ids, val_lbl = vblob["ids"], vblob["labels"]
        print(f"[qat] val corpus {val_ids.shape[0]} windows "
              f"(using {min(cfg.val_windows, val_ids.shape[0])})", flush=True)

    model = AutoModelForCausalLM.from_pretrained(cfg.model_dir, dtype=dtype).to(dev)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()  # transformers>=5 defaults use_reentrant=False
    wrap_model(model, cfg.train_layers, layer_spec=cfg.layers,
               train_norms=cfg.train_norms, ternary_spec=cfg.ternary_layers,
               dense_kinds=cfg.dense_kinds)
    model.train()

    teacher = None
    if cfg.kd_teacher:
        tdtype = backend.teacher_dtype
        teacher = AutoModelForCausalLM.from_pretrained(cfg.kd_teacher, dtype=tdtype).to(dev)
        teacher.config.use_cache = False
        teacher.eval().requires_grad_(False)
        assert teacher.config.vocab_size == model.config.vocab_size, (
            f"teacher vocab {teacher.config.vocab_size} != student "
            f"{model.config.vocab_size} — KD needs a shared tokenizer")
        print(f"[qat] KD teacher {cfg.kd_teacher} ({tdtype}), "
              f"alpha={cfg.kd_alpha} T={cfg.kd_temp}", flush=True)

    trainable_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    t_names = [n for n, _ in trainable_named]
    trainable = [p for _, p in trainable_named]

    def make_inner(params):
        if cfg.optim == "adafactor":
            # scale_parameter=False + relative_step=False makes this "Adam with a rank-1
            # second moment", and beta1 defaults to None — i.e. NO MOMENTUM. That is the
            # variable the 8-bit options below exist to test: a ternary latent only
            # changes anything when it crosses the ternarization threshold, and crossing
            # needs pressure accumulated over many steps. Without momentum a latent near
            # the threshold jitters on instantaneous gradients, while a coarse signal
            # present in nearly every batch is reinforced every step regardless.
            from transformers import Adafactor
            return Adafactor(params, lr=cfg.lr, scale_parameter=False,
                             relative_step=False, warmup_init=False,
                             beta1=cfg.beta1, weight_decay=cfg.weight_decay)
        if cfg.optim in ("adamw8bit", "lion8bit", "ademamix8bit"):
            # 8-bit state is what makes real per-parameter moments affordable here.
            # Measured against Adafactor's 70.6 GiB peak on a 95 GiB card at all-36:
            #   AdamW      +55.6 GiB -> ~126 GiB   OOM
            #   Adafactor + beta1  +27.8 GiB -> ~98 GiB   OOM
            #   AdamW8bit  +13.9 GiB -> ~84 GiB    fits
            #   Lion8bit    +7.0 GiB -> ~78 GiB    fits
            # NOTE the CLAUDE.md line "an 8-bit optimizer is a no-op here" is about
            # 8-bit ADAFACTOR (whose state is already ~9 MB). Against AdamW it is the
            # difference between fitting and not.
            import bitsandbytes as bnb
            cls = {"adamw8bit": bnb.optim.AdamW8bit,
                   "lion8bit": bnb.optim.Lion8bit,
                   "ademamix8bit": bnb.optim.AdEMAMix8bit}[cfg.optim]
            kw = {"lr": cfg.lr, "weight_decay": cfg.weight_decay}
            if cfg.beta1 is not None and cfg.optim != "ademamix8bit":
                kw["betas"] = (cfg.beta1, 0.999 if cfg.optim == "adamw8bit" else 0.99)
            return cls(params, **kw)
        # foreach fuses ~250 tiny per-tensor kernels into a handful of multi-tensor ones
        # on CUDA; on MPS the same kernels deadlock at full-model scale (qat._device).
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay,
                                 foreach=backend.foreach)

    if cfg.lr_scale not in ("none", "group-scale"):
        sys.exit(f"[qat] unknown --lr-scale {cfg.lr_scale!r}")
    if cfg.lr_scale != "none" and cfg.compute_dtype == "bf16":
        sys.exit("[qat] --lr-scale is not wired for the bf16 master path (fp32 is the "
                 "supported training dtype for this model anyway)")
    if cfg.compute_dtype == "bf16":
        # masters are cloned fp32 BEFORE the bf16 cast; the cast keeps Parameter
        # identity, so the wrapper's param references stay live
        opt = MasterOptimizer(trainable, make_inner, foreach=backend.foreach)
        model.to(torch.bfloat16)
        print("[qat] bf16 compute + fp32 masters "
              f"({sum(m.numel() for m in opt.masters)/1e9:.2f}B master params)", flush=True)
    else:
        if cfg.lr_scale == "group-scale":
            # One param group per tensor, each carrying its multiplier; the schedule
            # (opt_step) applies `base_lr * lr_mult` every step, so warmup/cosine and
            # the multiplier compose. Extra dict keys survive add_param_group.
            mults = latent_lr_mults(trainable_named)
            opt = make_inner([{"params": [p], "lr_mult": mults[n]}
                              for n, p in trainable_named])
            scaled = [m for m in mults.values() if m != 1.0]
            print(f"[qat] lr-scale group-scale: {len(scaled)} tensors, "
                  f"mult {min(scaled):.2f}..{max(scaled):.2f} "
                  f"(median-scale-normalized, clamp [0.5, 2.0])", flush=True)
        else:
            opt = make_inner(trainable)
    print(f"[qat] optimizer {cfg.optim} (wd={cfg.weight_decay}"
          f"{f', beta1={cfg.beta1}' if cfg.beta1 else ''})", flush=True)

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    stop = {"f": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("f", True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("f", True))

    g = torch.Generator().manual_seed(1234)
    order = torch.randperm(n_win, generator=g)
    step = 0
    mi = 0
    loss_first = None
    recent: list[float] = []

    if cfg.resume:
        # mmap=True keeps the ~28 GB of latents as file-backed pages the kernel can evict,
        # instead of anonymous memory that can only go to swap. Without it, resuming an
        # all-36 run costs model (30 GB) + checkpoint (26 GB) resident simultaneously and
        # the process is OOM-killed during startup — observed twice.
        try:
            ck = torch.load(cfg.resume, map_location="cpu", weights_only=False, mmap=True)
        except (RuntimeError, ValueError) as e:  # legacy (non-zipfile) checkpoint
            print(f"[qat] mmap load unavailable ({e}); falling back to a full read", flush=True)
            ck = torch.load(cfg.resume, map_location="cpu", weights_only=False)
        ck_fp = ck.get("corpus_fingerprint")
        if ck_fp != fp:
            sys.exit(f"[qat] resume corpus mismatch: ckpt fingerprint {ck_fp} != "
                     f"corpus {fp}. Resuming across a rebuilt corpus would silently "
                     f"misalign the data order — rebuild or drop resume.")
        latents = ck["latents"]
        missing = [n for n in t_names if n not in latents]
        if missing:
            sys.exit(f"[qat] resume layer-set mismatch: ckpt lacks {missing[:3]}... "
                     f"({len(missing)} params). Use the same layers/train_norms.")
        # Consume tensor-by-tensor and drop each reference as it lands, so peak overhead is
        # ONE tensor rather than the whole payload. `[latents[n] for n in t_names]` would
        # have pinned all 28 GB at once.
        if isinstance(opt, MasterOptimizer):
            with torch.no_grad():
                for m, n in zip(opt.masters, t_names, strict=True):
                    m.copy_(latents.pop(n).to(m.device, torch.float32))
                for p, m in zip(opt.params, opt.masters, strict=True):
                    p.copy_(m.to(p.dtype))
        else:
            named = dict(model.named_parameters())
            with torch.no_grad():
                for n in t_names:
                    named[n].copy_(latents.pop(n).to(named[n].device, named[n].dtype))
        step, mi = int(ck.get("step", 0)), int(ck.get("mi", 0))
        for _ in range(mi // n_win):  # replay epoch reshuffles -> deterministic order
            order = torch.randperm(n_win, generator=g)
        if cfg.optim == "adafactor" and ck.get("optim") is not None:
            (opt.inner if isinstance(opt, MasterOptimizer) else opt).load_state_dict(ck["optim"])
            print(f"[qat] resumed at step {step} (mi={mi}) with adafactor state", flush=True)
        else:
            print(f"[qat] resumed at step {step} (mi={mi}); OPTIMIZER STATE RESET "
                  f"({'optimizer state is not checkpointed for ' + cfg.optim if cfg.optim != 'adafactor' else 'no state in ckpt'})",
                  flush=True)
        loss_first = ck.get("loss_first")
        # `ck` is function-scoped, so without this it stays alive for the WHOLE run —
        # 28 GB of checkpoint sitting alongside a 30 GB model for 50+ hours.
        latents.clear()
        ck.clear()
        del latents, ck
        gc.collect()
        backend.empty_cache()

    # Under bf16 compute the latents live in the optimizer's fp32 masters, and that is
    # what export ternarizes — so that is what the flip telemetry must read.
    latents_for_flips = latent_weights(model, opt)
    snaps = snapshot_codes(model, cfg.flip_sample, latents=latents_for_flips)
    print(f"[qat] flip telemetry on {len(snaps)} linears"
          f"{' (reading fp32 masters)' if latents_for_flips else ''}", flush=True)

    # Termination telemetry. Built from the model's own tokenizer so the probe prompt is
    # rendered by the same chat template the corpus was packed with — a probe built from a
    # different template measures a prompt the model never sees.
    kd_table = None
    if cfg.kd_table:
        kd_table = KDTable.load(cfg.kd_table, corpus_fingerprint=fp)
        print(f"[qat] KD {kd_table}", flush=True)
        # A PARTIAL table is the quiet failure: windows it does not cover would train on
        # plain CE while the rest train on CE+KL, so the objective silently changes from
        # window to window and the run is neither one experiment nor the other. Refuse it
        # rather than letting a --max-windows smoke table drive a real run.
        missing = [w for w in range(n_win) if not kd_table.has_window(w)]
        if missing:
            sys.exit(f"[qat] KD table covers {n_win - len(missing)}/{n_win} windows "
                     f"(first uncovered: {missing[0]}). Windows without teacher rows would "
                     f"train on plain CE while the others train on CE+KL — the objective "
                     f"would change from window to window. Re-run kd_precompute without "
                     f"--max-windows, or pass a corpus matching the table.")
        print(f"[qat] KD alpha={cfg.kd_alpha} T={cfg.kd_temp}; loss = "
              f"{1 - cfg.kd_alpha:g}*CE + {cfg.kd_alpha:g}*T^2*KL", flush=True)
        if kd_table.coverage() < 0.8:
            print(f"[qat] WARNING top-{kd_table.topk} captures only "
                  f"{kd_table.coverage():.1%} of the teacher's mass — the KL is a weaker "
                  f"constraint than it looks; consider a larger --topk", flush=True)

    stop_anchor_id = None
    if cfg.stop_anchor > 0:
        if kd_table is None:
            sys.exit("[qat] --stop-anchor needs --kd-table (the teacher's per-position "
                     "P(stop) lives in the forced-stop table)")
        from transformers import AutoTokenizer

        from quant_tuner.qat.dialect import detect as _detect
        _t = AutoTokenizer.from_pretrained(str(cfg.model_dir))
        stop_anchor_id = _t.convert_tokens_to_ids(_detect(_t).stop_piece)
        # Fail at startup, not at window 0: the plain top-K table lacks the stop id in
        # ~98% of rows and there is nothing to anchor to.
        stop_logp_of(kd_table.for_window(
            0, kd_table._pos[kd_table._lo[0]:kd_table._hi[0]]), stop_anchor_id)
        print(f"[qat] stop anchor beta={cfg.stop_anchor} margin="
              f"{cfg.stop_anchor_margin} nats on token {stop_anchor_id} — O(1) log-space "
              f"restoring force (the KL's is P_s−P_t, vanishing exactly when needed)",
              flush=True)

    stop_probe = None
    if cfg.probe_every:
        try:
            from transformers import AutoTokenizer
            _tok = AutoTokenizer.from_pretrained(str(cfg.model_dir))
            stop_probe = StopProbe.build(_tok)
            print(f"[qat] stop-probe every {cfg.probe_every} steps "
                  f"({len(stop_probe.prompts)} points, stop id {stop_probe.stop_id})",
                  flush=True)
        except Exception as exc:
            print(f"[qat] stop-probe unavailable ({exc}) — continuing without it",
                  flush=True)
    flip_stats: dict = {}

    # Machine-readable telemetry. The stdout log is human-facing and has to be re-parsed
    # (scripts/parse_qat_log.py) to plot anything; this is the same numbers, already
    # structured, appended so a resume extends rather than truncates the series.
    metrics_path = out / "metrics.jsonl"
    metrics_fh = metrics_path.open("a") if cfg.metrics_jsonl else None

    # The run's own provenance, written BEFORE the first step so it exists even if the run
    # is killed. Until this existed the hyper-parameters lived only in the launch command
    # and died with the shell: a finished run directory could not say what lr, window or
    # stop-weight produced it, which makes two runs uncomparable after the fact. Everything
    # here is either cfg, or a fact the trainer alone knows (corpus fingerprint, resolved
    # device, effective step count).
    write_run_config(out, cfg, fingerprint=fp, n_windows=n_win, window=window,
                     total_steps=total_steps, device=str(dev),
                     assistant_frac=blob.get("assistant_frac"))

    def emit(kind: str, **fields) -> None:
        if metrics_fh is None:
            return
        metrics_fh.write(json.dumps({"kind": kind, **fields}) + "\n")
        metrics_fh.flush()

    def save_ckpt(at):
        nonlocal flip_stats
        if snaps:
            flip_stats, lines = flip_report(model, snaps, prev=flip_stats,
                                            latents=latents_for_flips)
            print(f"[qat] code flips vs run start:\n{lines}", flush=True)
            for tname, st in flip_stats.items():
                emit("flip", step=at, tensor=tname, **st)
        # The whole-dict .cpu() copy below is a ~28 GB transient at all-36. Both observed
        # OOM kills happened exactly at a checkpoint boundary (steps 180 and 20, both
        # multiples of --ckpt-every), i.e. peak-training memory + this spike. Release the
        # cached MPS blocks and the flip-report temporaries FIRST so the copy has headroom.
        gc.collect()
        backend.empty_cache()
        if isinstance(opt, MasterOptimizer):
            latents = {n: m.detach().cpu() for n, m in zip(t_names, opt.masters, strict=True)}
        else:
            latents = {n: p.detach().cpu() for n, p in trainable_named}
        # for MasterOptimizer save only the INNER state — the masters ARE the
        # latents payload; duplicating them would double the ckpt size (28 GB)
        optim_state = None
        if cfg.optim == "adafactor":
            optim_state = (opt.inner.state_dict() if isinstance(opt, MasterOptimizer)
                           else opt.state_dict())
        payload = {"latents": latents, "args": {k: str(v) for k, v in vars(cfg).items()},
                   "step": at, "mi": mi, "corpus_fingerprint": fp,
                   "loss_first": loss_first,
                   "loss_last": sum(recent[-8:]) / len(recent[-8:]) if recent else None,
                   "flip_stats": flip_stats,
                   "optim": optim_state}
        tmp = out / ".tmp-trained_latents.pt"
        torch.save(payload, tmp)
        os.replace(tmp, out / "trained_latents.pt")
        # Rotate a few step-stamped hard links beside it. `trained_latents.pt` is
        # overwritten every save, so a run that degrades (or diverges) has nothing to
        # roll back TO — observed the hard way on sft8k-full, where the last healthy
        # pre-divergence state was gone by the time the divergence was visible. Links
        # cost no extra disk until the next save rewrites the name.
        if cfg.ckpt_keep > 0:
            try:
                stamped = out / f"trained_latents.step{at}.pt"
                stamped.unlink(missing_ok=True)
                os.link(out / "trained_latents.pt", stamped)
                old = sorted(out.glob("trained_latents.step*.pt"),
                             key=lambda p: int(p.stem.split("step")[-1]))
                for p in old[:-cfg.ckpt_keep]:
                    p.unlink(missing_ok=True)
            except OSError as e:  # a filesystem without hard links must not kill the run
                print(f"[qat] checkpoint rotation skipped ({e})", flush=True)
        del latents, payload
        gc.collect()
        backend.empty_cache()
        print(f"[qat] checkpoint @ step {at}: {len(t_names)} tensors", flush=True)

    def opt_step() -> tuple[float, bool]:
        """Step unless the guard rejects it. Returns (pre-clip grad norm, skipped)."""
        base = lr_at(step, total_steps, cfg.lr, cfg.warmup_frac)
        for pg in opt.param_groups:
            pg["lr"] = base * pg.get("lr_mult", 1.0)
        if isinstance(opt, MasterOptimizer):
            # clip_and_step needs the norm BEFORE deciding, so measure on the masters
            gn = opt.stage_grads_and_norm()
            if guard.check(gn):
                opt.zero_grad()
                return gn, True
            opt.step_staged(1.0)
        else:
            gn = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0,
                                                      foreach=backend.foreach))
            if guard.check(gn):
                opt.zero_grad()
                return gn, True
            opt.step()
        opt.zero_grad()
        return gn, False

    # Prefix-context: encode all but the last `trained_tail` tokens under no_grad so a
    # window far longer than the activation budget still conditions the trained tail.
    n_prefix = 0
    if cfg.trained_tail and cfg.trained_tail < window:
        n_prefix = window - cfg.trained_tail
        kv_gib = 36 * 2 * n_prefix * 8 * 128 * (4 if dtype == torch.float32 else 2) / 1024**3
        print(f"[qat] prefix-context: {n_prefix} tokens no_grad (KV cache ~{kv_gib:.1f} GiB) "
              f"+ {cfg.trained_tail} tokens with gradient. Targets inside the prefix are "
              f"dropped, so this loss is NOT comparable with a full-window run.", flush=True)
    elif cfg.trained_tail:
        print(f"[qat] --trained-tail {cfg.trained_tail} >= window {window}: whole window "
              "carries gradient (no prefix)", flush=True)

    ce_weights = None
    if cfg.stop_weight != 1.0:
        # `stop_id` is the dialect-neutral key; `im_end_id` is the Qwen-era name kept so
        # corpora built before qat/dialect.py existed still train.
        im_end_id = blob.get("stop_id", blob.get("im_end_id"))
        if im_end_id is None:
            sys.exit("[qat] --stop-weight needs 'stop_id' in the corpus blob; rebuild it")
        ce_weights = torch.ones(model.config.vocab_size, device=dev, dtype=torch.float32)
        ce_weights[int(im_end_id)] = cfg.stop_weight
        print(f"[qat] stop-token weight {cfg.stop_weight}x on id {im_end_id} "
              f"({blob.get('im_end_targets', '?')} targets in the corpus)", flush=True)

    guard = GradSpikeGuard(cfg.grad_spike_factor)
    t0 = time.time()
    step0 = step  # step we entered the loop at; a resume starts above 0
    n_acc = 0
    grad_norm = 0.0
    tokens_seen = 0
    n_tail_empty = 0
    src_loss: dict[str, list[float]] = {}
    kl_acc: list[float] = []          # KD KL over the current accum group
    an_acc: list[float] = []          # stop-anchor penalty over the current accum group
    opt.zero_grad()
    while step < total_steps and not stop["f"]:
        w = order[mi % n_win].item()
        mi += 1
        if mi % n_win == 0:  # reshuffle each epoch
            order = torch.randperm(n_win, generator=g)
        lbl_cpu = lbl_all[w:w + 1]
        if not bool((lbl_cpu[0, 1:] != -100).any()):
            continue  # no valid shifted target; builder should have dropped it
        if n_prefix and not bool((lbl_cpu[0, 1 + n_prefix:] != -100).any()):
            n_tail_empty += 1
            continue  # every target sits in the frozen prefix — nothing to backprop
        ids = ids_all[w:w + 1].to(dev)
        lbl = lbl_cpu.to(dev)
        # only the KD path consumes the logits; otherwise chunk them (multi-GB spike)
        # The prefix block spans forward AND backward: checkpoint recompute happens inside
        # .backward(), and a recompute that cannot see the prefix raises CheckpointError.
        with prefix_window(model, ids, n_prefix):
            kd_win = None
            if kd_table is not None:
                # Aligned to this window's keep_idx, and verified against it — a table
                # built from a different pack would resolve without erroring and distil
                # every position against another token's distribution.
                kd_win = kd_table.for_window(w, (lbl[:, 1:][0] != -100)
                                             .nonzero(as_tuple=True)[0]).to(dev)
            anchor_arg = None
            if kd_win is not None and cfg.stop_anchor > 0:
                anchor_arg = (stop_anchor_id, stop_logp_of(kd_win, stop_anchor_id),
                              cfg.stop_anchor_margin)
            # NOT `out` — that is the run directory in this scope, and shadowing it
            # made save_ckpt do `tuple / str` four tests later.
            fwd = masked_forward(model, ids, lbl,
                                 need_logits=teacher is not None,
                                 n_prefix=n_prefix, weights=ce_weights,
                                 kd=kd_win, kd_temp=cfg.kd_temp,
                                 stop_anchor=anchor_arg)
            if kd_win is not None:
                ce, s_logits, keep_idx, kl, anchor = fwd
                loss = (1 - cfg.kd_alpha) * ce + cfg.kd_alpha * (cfg.kd_temp ** 2) * kl
                if anchor is not None:
                    loss = loss + cfg.stop_anchor * anchor
                    an_acc.append(float(anchor.detach()))
                kl_v = float(kl.detach())
            else:
                ce, s_logits, keep_idx = fwd
                kl_v = None
                if teacher is not None:
                    kl = kd_kl(teacher, ids, keep_idx, s_logits, cfg.kd_temp)
                    loss = (1 - cfg.kd_alpha) * ce + cfg.kd_alpha * (cfg.kd_temp ** 2) * kl
                else:
                    loss = ce
            lv = float(loss.detach())
            if not math.isfinite(lv):
                # skip BEFORE backward: the accumulated group stays valid, n_acc unchanged
                print("[qat] non-finite loss — skip window", flush=True)
                continue
            (loss / cfg.grad_accum).backward()
        n_acc += 1
        tokens_seen += int(ids.shape[1])
        if win_src is not None:
            sname = src_names[int(win_src[w])] if src_names else str(int(win_src[w]))
            src_loss.setdefault(sname, []).append(lv)
        if kl_v is not None:
            kl_acc.append(kl_v)
        if loss_first is None:
            loss_first = lv
        recent.append(lv)
        recent[:] = recent[-cfg.grad_accum * 5:]
        if n_acc == cfg.grad_accum:
            grad_norm, skipped = opt_step()
            if skipped:
                print(f"[qat] step {step + 1}: grad spike {grad_norm:.1f} > "
                      f"{cfg.grad_spike_factor}x median {guard.last_median:.1f} — step "
                      f"SKIPPED ({guard.n_skipped} so far)", flush=True)
            n_acc = 0
            step += 1
            # Periodic allocator-cache release. On MPS this is load-bearing: over a long
            # all-36 run the allocator fragments and the working set creeps until it swaps
            # (s/step balloons) and macOS OOM-kills the process. Every 25 steps was NOT
            # enough at window 8064/all-36 — a kill landed at ~step 12 with swap at 63 GB,
            # mid-interval, before the release ever fired, so the MPS default is 5. On CUDA
            # the failure mode does not exist and the release costs a device sync plus the
            # blocks the allocator would have reused, so the default there is off.
            if empty_cache_every and step % empty_cache_every == 0:
                backend.empty_cache()
            if step == 1 or step % 5 == 0:
                mem, mem_peak = backend.allocated_gib(), backend.peak_gib()
                avg = sum(recent) / len(recent)
                # base schedule lr; with --lr-scale each group runs base * its lr_mult
                cur_lr = lr_at(step, total_steps, cfg.lr, cfg.warmup_frac)
                emit("step", step=step, total_steps=total_steps, loss=avg,
                     kd_kl=(sum(kl_acc) / len(kl_acc)) if kl_acc else None,
                     stop_anchor=(sum(an_acc) / len(an_acc)) if an_acc else None,
                     lr=cur_lr, grad_norm=grad_norm,
                     grad_median=guard.last_median, n_skipped=guard.n_skipped, mem_gib=mem,
                     mem_peak_gib=mem_peak, device=backend.name,
                     n_tail_empty=n_tail_empty,
                     tokens_seen=tokens_seen, elapsed_s=time.time() - t0,
                     s_per_step=(time.time() - t0) / max(1, step - step0),
                     loss_by_source={k: sum(v) / len(v) for k, v in src_loss.items()})
                kl_str = f" kl={sum(kl_acc)/len(kl_acc):.4f}" if kl_acc else ""
                if an_acc:
                    kl_str += f" an={sum(an_acc)/len(an_acc):.4f}"
                src_loss.clear()
                kl_acc.clear()
                an_acc.clear()
                print(f"[qat] step {step}/{total_steps} loss={avg:.4f}{kl_str} "
                      f"lr={cur_lr:.2e} "
                      # pre-clip; a divergence shows up here BEFORE the loss reacts
                      f"gnorm={grad_norm:.2f} "
                      # live bytes, then the high-water mark — the peak is what decides
                      # whether the next checkpoint save or validation OOMs
                      f"mem={mem:.1f}/{mem_peak:.1f}GiB "
                      # steps run in THIS process, not the absolute step — after a resume
                      # the latter divides by a step count this process never spent time on
                      # and under-reports by (step / steps_here), which is exactly the
                      # number a run's wall-clock gets sized from.
                      f"{(time.time()-t0)/max(1, step-step0):.1f}s/step", flush=True)
            if val_ids is not None and cfg.val_every and step % cfg.val_every == 0:
                # Release first: validation allocates a fresh set of long-window
                # activations on top of a training cache that is already at the working-set
                # ceiling. Measured at a 32768 window — the val interval was the only place
                # swap moved (+11 GiB), and it dragged the surrounding steps with it.
                backend.empty_cache()
                t_val = time.time()
                vl = run_validation(model, val_ids, val_lbl, dev, cfg.val_windows, n_prefix)
                val_s = time.time() - t_val
                backend.empty_cache()
                emit("val", step=step, val_masked_ce=vl, val_windows=cfg.val_windows,
                     val_seconds=val_s)
                # Report the cost: at a long window validation is not free next to a step,
                # and `--val-windows` is the knob that pays for itself.
                print(f"[qat] step {step} VAL masked-CE {vl:.4f} "
                      f"({cfg.val_windows} windows in {val_s:.0f}s)", flush=True)
            if stop_probe is not None and cfg.probe_every and step % cfg.probe_every == 0:
                # Termination telemetry. The masked-CE validation cannot see this: it
                # scores the model on the corpus's own continuations, and a model that has
                # collapsed the stop decision into "sentence end -> <|im_end|>" still
                # scores well there — sft32k's val went flat for 225 steps while its
                # P(stop | sentence end) went to 0.97. Five short forwards, no gradients.
                t_pr = time.time()
                probs = None
                try:
                    probs = stop_probe.measure(model, dev)
                    emit("stopprobe", step=step, seconds=time.time() - t_pr, **probs)
                    print(f"[qat] step {step} STOPPROBE "
                          f"{stop_probe_fmt(probs, stop_probe.dialect)}",
                          flush=True)
                except Exception as exc:            # never let telemetry kill a long run
                    print(f"[qat] step {step} STOPPROBE failed: {exc}", flush=True)
                backend.empty_cache()
                # The diagnostic point is per chat family (gemma-4's control point means
                # the opposite of Qwen's), so read its name off the probe, not a constant.
                diag = stop_probe.diagnostic
                ctrl = stop_probe.control
                if (probs is not None and cfg.probe_abort_control
                        and probs.get(ctrl, 1.0) < cfg.probe_abort_control):
                    # The OTHER failure direction: losing the ability to stop where
                    # stopping is right — the sft32k loop failure. The a0.75 run showed
                    # the diagnostic can retreat under its threshold while this one
                    # collapses monotonically (0.9998 -> 0.8876 by step 150), unguarded.
                    print(f"[qat] step {step} PROBE-ABORT: {ctrl}="
                          f"{probs[ctrl]:.4f} < --probe-abort-control "
                          f"{cfg.probe_abort_control} — the model is losing the ability "
                          f"to STOP where stopping is right (the loop failure); saving "
                          f"and stopping.", flush=True)
                    save_ckpt(step)
                    if metrics_fh is not None:
                        metrics_fh.close()
                    sys.exit(3)
                if (probs is not None and cfg.probe_abort
                        and probs.get(diag, 0.0) > cfg.probe_abort):
                    # Every collapse so far was visible by ~step 50 and monotone from
                    # there; a full run spends 8+ hours past that point learning nothing
                    # we will ship. Save what exists and stop while the checkpoint is
                    # still analyzable.
                    print(f"[qat] step {step} PROBE-ABORT: {diag}="
                          f"{probs[diag]:.4f} > --probe-abort "
                          f"{cfg.probe_abort} — termination is collapsing; saving and "
                          f"stopping.", flush=True)
                    save_ckpt(step)
                    if metrics_fh is not None:
                        metrics_fh.close()
                    sys.exit(3)
            if cfg.ckpt_every and step % cfg.ckpt_every == 0:
                save_ckpt(step)
    # drop any partial accum group before the final save
    opt.zero_grad()
    save_ckpt(step)
    if metrics_fh is not None:
        metrics_fh.close()
        print(f"[qat] metrics -> {metrics_path}", flush=True)
    if recent and loss_first is not None:
        print(f"[qat] done at step {step}: loss {loss_first:.3f} -> "
              f"{sum(recent[-8:]) / len(recent[-8:]):.3f}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--train-layers", type=int, default=18)
    ap.add_argument("--layers", type=str, default=None,
                    help="explicit layer indices to train, e.g. '0-14,32,34,35'; overrides --train-layers")
    ap.add_argument("--ternary-layers", type=str, default=None,
                    help="layer indices to TERNARIZE, e.g. '24-41'; default = all. "
                         "Separate from --layers, which picks what gets gradients: on a "
                         "dense model (gemma-4) a layer left out of this stays bf16 and "
                         "off the grid, which is what makes a progressive schedule "
                         "expressible at all. Every trained layer must also be ternarized.")
    ap.add_argument("--dense-kind", action="append", default=None, metavar="SUBSTR",
                    help="keep linears whose name contains SUBSTR dense, in every layer "
                         "(repeatable). Measured on gemma-4-E4B: 'down_proj' alone takes "
                         "PPL ~19 -> 149 when ternarized, against <1.0 KLD for every other "
                         "kind — see docs/gemma4_ternary_feasibility.md.")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--optim",
                    choices=["adamw", "adafactor", "adamw8bit", "lion8bit",
                             "ademamix8bit"], default="adamw",
                    help="adafactor: factored 2nd moment (~MBs vs AdamW's 56 GB at "
                         "all-36) -> full-36 fp32 fits. Per-tensor loop, MPS-safe.")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="default 0: decay on ternary latents erodes codes toward 0")
    ap.add_argument("--beta1", type=float, default=None,
                    help="adafactor momentum (costs a full fp32 state, +27.8 GB at all-36); default off")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32",
                    help="latent dtype. bf16 latents UNSUPPORTED for training — use --compute-dtype bf16")
    ap.add_argument("--compute-dtype", choices=["fp32", "bf16"], default="fp32",
                    help="bf16: fp32-master trick — masters own the latents, fwd/bwd run in bf16")
    ap.add_argument("--kd-teacher", type=Path, default=None,
                    help="HF path of a same-vocab dense teacher (e.g. a SWE-tuned Qwen3-8B); enables KD")
    ap.add_argument("--kd-table", type=Path,
                    help="precomputed top-K teacher table from scripts/kd_precompute.py. "
                         "Offline KD — no teacher in memory, unlike --kd-teacher.")
    ap.add_argument("--kd-alpha", type=float, default=0.5,
                    help="loss = (1-a)*CE + a*T^2*KL(teacher||student)")
    ap.add_argument("--kd-temp", type=float, default=1.0)
    ap.add_argument("--stop-anchor", type=float, default=0.0,
                    help="beta for the stop-token log-ratio hinge, "
                         "relu(|log P_s(stop) - log P_t(stop)| - margin) per position. "
                         "O(1) restoring force in log space where the KL's vanishes with "
                         "the drift. Needs a forced-stop --kd-table (--include-ids).")
    ap.add_argument("--stop-anchor-margin", type=float, default=1.0,
                    help="free band in nats around the teacher's log P(stop)")
    ap.add_argument("--probe-abort-control", type=float, default=0.0,
                    help="abort (exit 3, checkpoint saved) when the probe CONTROL "
                         "drops below this (0 disables). The other failure direction: "
                         "losing the ability to stop after a tool call = the loop "
                         "failure. Healthy runs sit >= 0.995; collapses pass 0.95 "
                         "decisively.")
    ap.add_argument("--val-corpus", type=Path, default=None,
                    help="masked corpus built with --split test; masked-CE validation")
    ap.add_argument("--val-every", type=int, default=20)
    ap.add_argument("--probe-every", type=int, default=25,
                    help="measure P(<|im_end|>) at fixed positions every N steps "
                         "(0 disables). Masked-CE cannot see a collapsed stop "
                         "decision; this can.")
    ap.add_argument("--probe-abort", type=float, default=0.0,
                    help="abort (exit 3, checkpoint saved) when the probe diagnostic "
                         "exceeds this (0 disables). E.g. 0.09 = 10x the vanilla "
                         "0.0092 — every observed collapse blew far past that.")
    ap.add_argument("--lr-scale", choices=["none", "group-scale"], default="none",
                    help="group-scale: per-tensor lr proportional to the median TWN "
                         "group scale (clamped [0.5, 2.0]) so large-scale tensors "
                         "(v/up/gate) are not flip-starved at uniform lr. fp32 only.")
    ap.add_argument("--val-windows", type=int, default=16)
    ap.add_argument("--train-norms", action="store_true",
                    help="also train RMSNorm/q_norm/k_norm weights in the trainable layers")
    ap.add_argument("--resume", type=Path, default=None,
                    help="trained_latents.pt to continue from (same corpus required)")
    ap.add_argument("--flip-sample", type=int, default=8,
                    help="trainable linears to track for code-flip telemetry")
    ap.add_argument("--ckpt-every", type=int, default=40)
    ap.add_argument("--warmup-frac", type=float, default=0.05,
                    help="fraction of total steps spent warming up (default 0.05). The "
                         "sft8k-full run diverged 4 steps after warmup ended; a longer "
                         "ramp is the cheap half of the fix.")
    ap.add_argument("--grad-spike-factor", type=float, default=4.0,
                    help="skip an optimizer step whose pre-clip grad norm exceeds this "
                         "multiple of the trailing median (0 disables). Clipping alone "
                         "does not prevent a divergence — it still steps full-size along "
                         "the clipped direction.")
    ap.add_argument("--ckpt-keep", type=int, default=2,
                    help="step-stamped hard links to keep beside trained_latents.pt "
                         "(default 2, 0 disables). Without these a run that diverges has "
                         "nothing to roll back to — the live file is already overwritten.")
    ap.add_argument("--no-metrics-jsonl", dest="metrics_jsonl", action="store_false",
                    help="skip the structured metrics.jsonl sidecar")
    ap.add_argument("--empty-cache-every", type=int, default=None,
                    help="release the allocator cache every N steps (default: 5 on MPS, "
                         "off on CUDA). On MPS at all-36/window 8064 a cadence of 25 let "
                         "the working set creep into swap and OOM-kill the process "
                         "mid-interval, and the release costs ~1 s against a ~390 s step. "
                         "CUDA has no such failure mode and the release costs a device "
                         "sync plus reusable blocks; 0 disables.")
    ap.add_argument("--matmul-precision", choices=["highest", "high", "medium"],
                    default="highest",
                    help="torch.set_float32_matmul_precision for the fp32 path. 'highest' "
                         "(default) is true fp32 — what every published run used. 'high' "
                         "uses TF32 tensor cores (10 mantissa bits) and 'medium' bf16 "
                         "ones (8). Unlike --compute-dtype this leaves the LATENTS in "
                         "exact fp32, so the TWN threshold and every code flip are "
                         "unperturbed; only the matmul accumulation is reduced. No effect "
                         "under --compute-dtype bf16, which is already not doing fp32 "
                         "matmuls.")
    ap.add_argument("--device", default="auto",
                    help="cuda / mps / cpu / cuda:N (default auto: cuda > mps > cpu)")
    ap.add_argument("--chunked-attention", choices=["auto", "on", "off"], default="auto",
                    help="query-chunked SDPA (qat.attention). 'auto' patches it only where "
                         "the backend needs it — on MPS, where the stock kernel caps the "
                         "window at 8191 tokens (n_heads*S^2 < 2^31), and for "
                         "--trained-tail on any device, which carries its prefix K/V. On "
                         "CUDA the fused/flash kernel never materializes the score matrix, "
                         "so auto leaves it off. 'on' forces it (benchmarking); 'off' "
                         "refuses it everywhere.")
    ap.add_argument("--no-chunked-attention", dest="chunked_attention",
                    action="store_const", const="off",
                    help="alias for --chunked-attention off")
    ap.add_argument("--trained-tail", type=int, default=0,
                    help="prefix-context mode: encode all but the last N tokens of each "
                         "window under no_grad into a KV cache and backprop only through "
                         "the tail. Activation memory becomes O(N) instead of O(window) at "
                         "a cache cost of ~288 KB/prefix-token (fp32), which is what makes "
                         "a 32768 window fit where a full-gradient 16128 could not. "
                         "Gradients do not reach the prefix. 0 = off.")
    ap.add_argument("--stop-weight", type=float, default=1.0,
                    help="CE weight on the terminating <|im_end|> target. It is ~0.57%% of "
                         "labels yet carries the entire stop decision, so at 1.0 the run "
                         "sees ~176 'keep going' targets per 'stop' one — the measured "
                         "cause of sft8k-full's 97%% loop rate. Try 5-10.")
    ap.add_argument("--model-dir", type=Path, default=MODEL,
                    help=f"HF model to continue training (default {MODEL})")
    ap.add_argument("--out", type=Path, default=REPO / "out" / "exp-058" / "trained")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = QATConfig(
        corpus=args.corpus, out=args.out, model_dir=args.model_dir,
        train_layers=args.train_layers, layers=args.layers,
        ternary_layers=args.ternary_layers,
        dense_kinds=tuple(args.dense_kind or ()), epochs=args.epochs,
        grad_accum=args.grad_accum, lr=args.lr, optim=args.optim,
        weight_decay=args.weight_decay, beta1=args.beta1, dtype=args.dtype,
        compute_dtype=args.compute_dtype, kd_teacher=args.kd_teacher,
        kd_table=args.kd_table,
        kd_alpha=args.kd_alpha, kd_temp=args.kd_temp,
        stop_anchor=args.stop_anchor, stop_anchor_margin=args.stop_anchor_margin,
        val_corpus=args.val_corpus,
        val_every=args.val_every, val_windows=args.val_windows,
        probe_every=args.probe_every, probe_abort=args.probe_abort,
        probe_abort_control=args.probe_abort_control,
        lr_scale=args.lr_scale,
        train_norms=args.train_norms, resume=args.resume,
        flip_sample=args.flip_sample, ckpt_every=args.ckpt_every,
        ckpt_keep=args.ckpt_keep, warmup_frac=args.warmup_frac,
        grad_spike_factor=args.grad_spike_factor,
        chunked_attention=args.chunked_attention,
        empty_cache_every=args.empty_cache_every,
        metrics_jsonl=args.metrics_jsonl,
        trained_tail=args.trained_tail, stop_weight=args.stop_weight,
        device=args.device, matmul_precision=args.matmul_precision,
    )
    return train_qat(cfg)


if __name__ == "__main__":
    sys.exit(main())
