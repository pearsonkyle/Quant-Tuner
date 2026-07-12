"""Infinite-loop diagnosis: why does a low-bit quant repeat, and can we break it?

Heavy quantization frequently sends a model into a repetition loop mid-generation
(the same phrase, tool call, or line forever). This module:

1. detects the loop (:func:`detect_repetition`, k-gram cycle search);
2. autopsies it (:func:`loop_autopsy`): captures the generation region on the
   quant AND an FP16 reference and measures, per position approaching loop
   onset, the *collapse depth* — the earliest layer at which the lens top-1
   already commits to the loop token. A quant that commits earlier than FP16
   has lost the competing continuations low-bit noise should not have erased;
3. proposes a fix (:func:`loop_direction` + :func:`intervention_sweep`):
   builds the residual-space direction that separates loop positions from
   pre-loop positions and ablates it live across late layers, keeping whatever
   α actually breaks the loop. A winning direction is saved as ``direction.npz``
   for :mod:`quant_tuner.lens.bake` to fold into the weights permanently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LoopSpan:
    start: int              # index into the generated tokens where the cycle begins
    period: int             # cycle length in tokens
    n_repeats: int          # how many times it repeats
    tokens: list[int]       # one period

    @property
    def end(self) -> int:
        return self.start + self.period * self.n_repeats


def detect_repetition(
    tokens: list[int],
    *,
    max_period: int = 64,
    min_repeats: int = 3,
    min_span: int = 24,
) -> LoopSpan | None:
    """Find the earliest tail cycle: a period-``p`` block repeated ``>=min_repeats``.

    Scans periods ascending (shortest cycle wins) and, for each, finds the
    longest run of exact repeats ending at the sequence tail. Returns None if no
    cycle covers at least ``min_span`` tokens.
    """
    n = len(tokens)
    best: LoopSpan | None = None
    for p in range(1, min(max_period, n // min_repeats) + 1):
        # walk backwards from the end counting how many period-p blocks match
        reps = 1
        i = n - p
        while i - p >= 0 and tokens[i - p : i] == tokens[i : i + p]:
            reps += 1
            i -= p
        if reps >= min_repeats and p * reps >= min_span:
            start = n - p * reps
            span = LoopSpan(start=start, period=p, n_repeats=reps, tokens=tokens[start : start + p])
            # prefer the earliest onset; among those, the shortest period
            if best is None or span.start < best.start:
                best = span
    return best


@dataclass
class LoopReport:
    looped: bool
    loop: LoopSpan | None
    generated_text: str
    collapse_depth_quant: list[float] = field(default_factory=list)   # per pre-onset position
    collapse_depth_ref: list[float] = field(default_factory=list)
    quant_run_id: str | None = None
    ref_run_id: str | None = None
    notes: str = ""


def load_trajectory_prompt(traj_json: Path, client) -> list[int]:
    """Template + tokenize a swebench ``.traj.json`` conversation into tokens.

    Accepts the mini-swe / openai-agents trajectory shape (a ``messages`` or
    ``trajectory`` list of ``{role, content}``); everything up to (but not
    including) the final assistant turn becomes the prompt.
    """
    import json

    data = json.loads(Path(traj_json).read_text())
    msgs = data.get("messages") or data.get("trajectory") or data.get("history") or []
    conv = [
        {"role": m.get("role", "user"), "content": m.get("content", "") or ""}
        for m in msgs
        if m.get("role") in ("system", "user", "assistant", "tool")
    ]
    while conv and conv[-1]["role"] == "assistant":
        conv.pop()
    prompt = client.apply_template(conv, add_assistant=True)
    return client.tokenize(prompt, add_special=True, parse_special=True)


def _collapse_depth(grid, layers: list[int], target_token: int, position_index: int) -> float:
    """Fraction-of-depth at which the lens top-1 first locks onto ``target_token``.

    0.0 = the very first layer already predicts the loop token; 1.0 = never
    until the final layer. Lower means the model committed to repeating sooner.
    """
    for li in range(len(layers)):
        if int(grid.ids[li, position_index, 0]) == int(target_token):
            return li / max(len(layers) - 1, 1)
    return 1.0


def loop_autopsy(
    model_gguf: Path,
    lens_path: Path,
    prompt_tokens: list[int],
    *,
    runs_dir: Path,
    reference_gguf: Path | None = None,
    n_predict: int = 512,
    sampling: dict | None = None,
    ctx: int = 8192,
    ngl: int = 99,
    window: int = 24,
) -> LoopReport:
    """Generate on the quant, detect a loop, and compare collapse depth vs FP16."""
    from quant_tuner.lens.backend import running_jlens_server
    from quant_tuner.lens.capture import capture_run, load_run
    from quant_tuner.lens.lens import JacobianLensGGUF
    from quant_tuner.lens.model_reader import ReadoutWeights
    from quant_tuner.lens.readout import LensReadout

    lens = JacobianLensGGUF.load(str(lens_path))
    sampling = sampling or {"greedy": True}

    with running_jlens_server(model_gguf, ctx=ctx, ngl=ngl) as client:
        fr = client.forward(prompt_tokens, capture=False, n_predict=n_predict, sampling=sampling)
        gen = [g["token"] for g in fr.generated]
        loop = detect_repetition(gen)
        if loop is None:
            return LoopReport(looped=False, loop=None, generated_text=fr.generated_text,
                              notes="no repetition detected in generation")

        # capture the region approaching + including the first loop period
        n_prompt = len(prompt_tokens)
        onset_abs = n_prompt + loop.start
        keep = list(range(onset_abs - window, onset_abs + loop.period))
        keep = [p for p in keep if 0 <= p < n_prompt + len(gen)]
        run_dir = capture_run(
            client, prompt_tokens, runs_dir=runs_dir,
            keep_positions=keep, n_predict=len(gen), sampling=sampling,
            tags={"model": Path(model_gguf).name, "role": "loop"},
        )
        run = load_run(run_dir)
        weights = ReadoutWeights.from_gguf(str(model_gguf))
        readout = LensReadout(weights, lens)
        grid = run.readout_topk(readout, k=4)
        layers = run.manifest.capture_layers
        loop_tok = loop.tokens[0]
        depths_q = [
            _collapse_depth(grid, layers, loop_tok, pi)
            for pi in range(min(window, len(run.manifest.positions)))
        ]

    depths_ref: list[float] = []
    ref_id = None
    if reference_gguf is not None:
        with running_jlens_server(reference_gguf, ctx=ctx, ngl=ngl) as rclient:
            # replay the quant's exact token sequence (prompt + its generation)
            ref_tokens = run.manifest.tokens
            ref_dir = capture_run(
                rclient, ref_tokens, runs_dir=runs_dir,
                keep_positions=keep, tags={"model": Path(reference_gguf).name, "role": "loop_ref"},
            )
            ref_run = load_run(ref_dir)
            ref_id = ref_run.manifest.run_id
            rweights = ReadoutWeights.from_gguf(str(reference_gguf))
            rreadout = LensReadout(rweights, lens)
            rgrid = ref_run.readout_topk(rreadout, k=4)
            rlayers = ref_run.manifest.capture_layers
            depths_ref = [
                _collapse_depth(rgrid, rlayers, loop_tok, pi)
                for pi in range(min(window, len(ref_run.manifest.positions)))
            ]

    return LoopReport(
        looped=True, loop=loop, generated_text=fr.generated_text,
        collapse_depth_quant=depths_q, collapse_depth_ref=depths_ref,
        quant_run_id=run.manifest.run_id, ref_run_id=ref_id,
        notes=(
            f"loop: period {loop.period}, {loop.n_repeats}x from gen index {loop.start}"
        ),
    )


def loop_direction(run, loop: LoopSpan, layer: int, *, window: int = 24):
    """Residual direction separating loop positions from the pre-loop run-up.

    ``v = mean(h over loop positions) - mean(h over pre-loop positions)``,
    normalized. Returned as a :class:`quant_tuner.lens.bake.Direction`.
    """
    from quant_tuner.lens.bake import Direction

    acts = run.activations(layer)                # [P, d] over kept positions
    P = acts.shape[0]
    split = max(1, min(window, P) // 2)
    pre = acts[:split].mean(axis=0)
    loop_h = acts[split:].mean(axis=0)
    v = loop_h - pre
    n = float(np.linalg.norm(v))
    if n > 0:
        v = v / n
    return Direction(
        v=v.astype(np.float32),
        layers=[layer],
        method="loop_mean_diff",
        source={"run_id": run.manifest.run_id, "loop_period": loop.period, "layer": layer},
    )


@dataclass
class InterventionResult:
    layers: list[int]
    alpha: float
    broke_loop: bool
    generated_text: str
    loop_after: LoopSpan | None


def intervention_sweep(
    client,
    direction,
    *,
    layers: list[int],
    prompt_tokens: list[int],
    alphas: tuple[float, ...] = (1.0,),
    n_predict: int = 256,
    sampling: dict | None = None,
) -> list[InterventionResult]:
    """Ablate ``direction`` across ``layers`` at each α; record loop-break.

    Ablation is a rank-1 low-rank edit ``h -= (v̂·h) v̂`` (A = -v̂, B = v̂ᵀ),
    applied at every layer in ``layers``. Success = generation no longer loops.
    """
    sampling = sampling or {"greedy": True}
    v = np.ascontiguousarray(direction.v, dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-12)
    results: list[InterventionResult] = []
    for alpha in alphas:
        # scale the ablation by alpha (alpha=1 => full projection removed)
        A = (-alpha * v).reshape(-1, 1)      # [d, 1]
        B = v.reshape(1, -1)                 # [1, d]
        ivs = [
            {"layer": int(l), "pos_start": 0, "pos_end": -1, "mode": "lowrank", "a": A, "b": B}
            for l in layers
        ]
        fr = client.forward(
            prompt_tokens, capture=False, interventions=ivs,
            n_predict=n_predict, sampling=sampling,
        )
        gen = [g["token"] for g in fr.generated]
        loop_after = detect_repetition(gen)
        results.append(InterventionResult(
            layers=list(layers), alpha=float(alpha),
            broke_loop=loop_after is None,
            generated_text=fr.generated_text, loop_after=loop_after,
        ))
        logger.info("ablate a=%.2f layers=%s -> %s", alpha, layers,
                    "broke loop" if loop_after is None else "still loops")
    return results
