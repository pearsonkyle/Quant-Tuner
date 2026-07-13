"""Diff two capture runs: where (layer x position) do two models disagree?

The unit of comparison is a :class:`~quant_tuner.lens.capture.Run` pair over
the *same token sequence* — FP16 vs quant, quant vs quant, or clean vs
intervened. Convention throughout: ``diff_runs(run, ref)`` treats ``ref`` as
the reference (typically FP16) and ``run`` as the candidate under test, so
"rank shift" answers *the token the reference wanted — where did the
candidate put it?* and KLD is ``D_KL(ref || run)`` (the same direction
llama-perplexity's ``--kl-divergence`` uses against an FP16 baseline).

Because runs store residual activations, every per-layer readout here is a
full-vocab recomputation through each model's own head under the shared
lens — no top-k truncation. ``final_kld_exact`` distinguishes rows backed by
server-stored logits (bit-exact model output at flagged decision positions)
from rows recomputed through the numpy readout (corr > 0.9999, but not the
server's arithmetic).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from quant_tuner.lens.capture import Run, _log_softmax
from quant_tuner.lens.readout import LensReadout

logger = logging.getLogger(__name__)

DIFF_JSON = "diff.json"
DIFF_ARRAYS = "diff_arrays.npz"


@dataclass
class RunDiff:
    """All (layer x position) comparison surfaces between two runs."""

    run_id: str
    ref_id: str
    run_model: str
    ref_model: str
    layers: list[int]
    positions: list[int]
    tokens: list[int]

    top1_run: np.ndarray               # [L, P] int32 — candidate's lens top-1
    top1_ref: np.ndarray               # [L, P] int32 — reference's lens top-1
    agree: np.ndarray                  # [L, P] bool
    ref_top1_rank_in_run: np.ndarray   # [L, P] int32 (0 = candidate agrees)
    layer_cos: np.ndarray              # [L] mean cosine(h_run, h_ref)
    layer_rel_l2: np.ndarray           # [L] median ||h_run - h_ref|| / ||h_ref||
    layer_kld: np.ndarray              # [L] mean_p D_KL(ref_readout || run_readout)
    final_kld: np.ndarray              # [P] D_KL(ref || run) at the model head
    final_kld_exact: np.ndarray        # [P] bool — server-stored logits on both sides
    pinned: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    # token_id -> {"rank_run": [L, P] int32, "rank_ref": [L, P] int32}

    def summary(self) -> dict[str, float]:
        """Scalar roll-up for CSV rows / leaderboard sidecars."""
        final_li = len(self.layers) - 1
        return {
            "lens_agree_mean": float(self.agree.mean()),
            "lens_agree_final_layer": float(self.agree[final_li].mean()),
            "lens_ref_top1_rank_mean": float(self.ref_top1_rank_in_run.mean()),
            "lens_ref_top1_rank_final": float(self.ref_top1_rank_in_run[final_li].mean()),
            "lens_layer_kld_auc": float(self.layer_kld.mean()),
            "lens_layer_rel_l2_auc": float(self.layer_rel_l2.mean()),
            "lens_final_kld_mean": float(self.final_kld.mean()),
            "lens_first_divergent_layer": float(self._first_divergent_layer()),
        }

    def _first_divergent_layer(self, threshold: float = 0.5) -> int:
        """First layer where top-1 agreement drops below ``threshold``.

        Later is better: a quant that only diverges in the last few layers has
        preserved the computation and merely blurred the readout.
        """
        per_layer = self.agree.mean(axis=1)
        below = np.nonzero(per_layer < threshold)[0]
        return int(self.layers[below[0]]) if below.size else int(self.layers[-1]) + 1

    def to_meta(self) -> dict:
        return {
            "run_id": self.run_id,
            "ref_id": self.ref_id,
            "run_model": self.run_model,
            "ref_model": self.ref_model,
            "layers": self.layers,
            "positions": self.positions,
            "n_tokens": len(self.tokens),
            "pinned_tokens": sorted(self.pinned),
            "summary": self.summary(),
        }

    def save(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / DIFF_JSON).write_text(json.dumps(self.to_meta(), indent=1))
        arrays: dict[str, np.ndarray] = {
            "top1_run": self.top1_run,
            "top1_ref": self.top1_ref,
            "agree": self.agree,
            "ref_top1_rank_in_run": self.ref_top1_rank_in_run,
            "layer_cos": self.layer_cos,
            "layer_rel_l2": self.layer_rel_l2,
            "layer_kld": self.layer_kld,
            "final_kld": self.final_kld,
            "final_kld_exact": self.final_kld_exact,
            "tokens": np.asarray(self.tokens, dtype=np.int64),
        }
        for tid, ranks in self.pinned.items():
            arrays[f"pin_{tid}_run"] = ranks["rank_run"]
            arrays[f"pin_{tid}_ref"] = ranks["rank_ref"]
        np.savez(out_dir / DIFF_ARRAYS, **arrays)
        return out_dir

    @classmethod
    def load(cls, out_dir: Path) -> RunDiff:
        out_dir = Path(out_dir)
        meta = json.loads((out_dir / DIFF_JSON).read_text())
        data = np.load(out_dir / DIFF_ARRAYS)
        pinned: dict[int, dict[str, np.ndarray]] = {}
        for tid in meta.get("pinned_tokens", []):
            pinned[int(tid)] = {
                "rank_run": data[f"pin_{tid}_run"],
                "rank_ref": data[f"pin_{tid}_ref"],
            }
        return cls(
            run_id=meta["run_id"],
            ref_id=meta["ref_id"],
            run_model=meta["run_model"],
            ref_model=meta["ref_model"],
            layers=list(meta["layers"]),
            positions=list(meta["positions"]),
            tokens=list(data["tokens"]),
            top1_run=data["top1_run"],
            top1_ref=data["top1_ref"],
            agree=data["agree"],
            ref_top1_rank_in_run=data["ref_top1_rank_in_run"],
            layer_cos=data["layer_cos"],
            layer_rel_l2=data["layer_rel_l2"],
            layer_kld=data["layer_kld"],
            final_kld=data["final_kld"],
            final_kld_exact=data["final_kld_exact"],
            pinned=pinned,
        )


def _rank_of(logprobs: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Rank (0 = top) of ``ids[p]`` in each row ``logprobs[p]``."""
    target = np.take_along_axis(logprobs, ids[:, None], axis=-1)
    return (logprobs > target).sum(axis=-1).astype(np.int32)


def _kld_rows(ref_lp: np.ndarray, run_lp: np.ndarray) -> np.ndarray:
    """Row-wise ``D_KL(ref || run)`` for log-softmax rows ``[P, vocab]``."""
    return (np.exp(ref_lp) * (ref_lp - run_lp)).sum(axis=-1)


def validate_pair(run: Run, ref: Run) -> tuple[list[int], list[int]]:
    """Common (layers, positions) for a diff, or raise with the mismatch."""
    if run.manifest.tokens != ref.manifest.tokens:
        a, b = run.manifest.tokens, ref.manifest.tokens
        first = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))
        raise ValueError(
            f"runs are over different token sequences (lengths {len(a)} vs {len(b)}, "
            f"first mismatch at position {first}) — capture both models on the same "
            "prompt (greedy, n_predict=0) or replay one run's tokens through the other"
        )
    if run.manifest.positions != ref.manifest.positions:
        raise ValueError(
            f"runs kept different positions ({len(run.manifest.positions)} vs "
            f"{len(ref.manifest.positions)}); re-capture with matching keep_positions"
        )
    layers = sorted(set(run.manifest.capture_layers) & set(ref.manifest.capture_layers))
    if not layers:
        raise ValueError("runs share no captured layers")
    return layers, list(run.manifest.positions)


def diff_runs(
    run: Run,
    ref: Run,
    *,
    readout_run: LensReadout,
    readout_ref: LensReadout,
    use_lens: bool = True,
    pinned_tokens: list[int] | None = None,
) -> RunDiff:
    """Compare ``run`` (candidate) against ``ref`` (reference), layer by layer."""
    layers, positions = validate_pair(run, ref)
    if readout_run.weights.n_vocab != readout_ref.weights.n_vocab:
        raise ValueError(
            f"vocab mismatch: {readout_run.weights.n_vocab} vs "
            f"{readout_ref.weights.n_vocab} — are these quants of the same base model?"
        )
    pins = [int(t) for t in (pinned_tokens or [])]

    L, P = len(layers), len(positions)
    top1_run = np.zeros((L, P), dtype=np.int32)
    top1_ref = np.zeros((L, P), dtype=np.int32)
    rank_shift = np.zeros((L, P), dtype=np.int32)
    layer_cos = np.zeros(L, dtype=np.float32)
    layer_rel_l2 = np.zeros(L, dtype=np.float32)
    layer_kld = np.zeros(L, dtype=np.float32)
    pinned = {
        t: {
            "rank_run": np.zeros((L, P), dtype=np.int32),
            "rank_ref": np.zeros((L, P), dtype=np.int32),
        }
        for t in pins
    }
    final_run_lp: np.ndarray | None = None
    final_ref_lp: np.ndarray | None = None

    for li, layer in enumerate(layers):
        h_run = run.activations(layer)
        h_ref = ref.activations(layer)
        dot = (h_run * h_ref).sum(axis=-1)
        norms = np.linalg.norm(h_run, axis=-1) * np.linalg.norm(h_ref, axis=-1) + 1e-12
        layer_cos[li] = float((dot / norms).mean())
        ref_norm = np.linalg.norm(h_ref, axis=-1) + 1e-12
        layer_rel_l2[li] = float(np.median(np.linalg.norm(h_run - h_ref, axis=-1) / ref_norm))

        run_lp = _log_softmax(readout_run.lens_logits(h_run, layer, use_lens=use_lens))
        ref_lp = _log_softmax(readout_ref.lens_logits(h_ref, layer, use_lens=use_lens))
        top1_run[li] = run_lp.argmax(axis=-1).astype(np.int32)
        top1_ref[li] = ref_lp.argmax(axis=-1).astype(np.int32)
        rank_shift[li] = _rank_of(run_lp, top1_ref[li])
        layer_kld[li] = float(_kld_rows(ref_lp, run_lp).mean())
        for t in pins:
            ids = np.full(P, t, dtype=np.int64)
            pinned[t]["rank_run"][li] = _rank_of(run_lp, ids)
            pinned[t]["rank_ref"][li] = _rank_of(ref_lp, ids)
        if layer == layers[-1]:
            final_run_lp, final_ref_lp = run_lp, ref_lp
        else:
            del run_lp, ref_lp

    # final-position KLD: prefer server-stored exact logits where both sides
    # have them; otherwise the final-layer readout (J = I there == model head)
    assert final_run_lp is not None and final_ref_lp is not None
    final_kld = _kld_rows(final_ref_lp, final_run_lp).astype(np.float32)
    exact = np.zeros(P, dtype=bool)
    stored_run = run.final_logprobs()
    stored_ref = ref.final_logprobs()
    for pi, pos in enumerate(positions):
        if pos in stored_run and pos in stored_ref:
            final_kld[pi] = float(
                (np.exp(stored_ref[pos]) * (stored_ref[pos] - stored_run[pos])).sum()
            )
            exact[pi] = True

    return RunDiff(
        run_id=run.manifest.run_id,
        ref_id=ref.manifest.run_id,
        run_model=Path(run.manifest.model_path).name,
        ref_model=Path(ref.manifest.model_path).name,
        layers=layers,
        positions=positions,
        tokens=run.manifest.tokens,
        top1_run=top1_run,
        top1_ref=top1_ref,
        agree=top1_run == top1_ref,
        ref_top1_rank_in_run=rank_shift,
        layer_cos=layer_cos,
        layer_rel_l2=layer_rel_l2,
        layer_kld=layer_kld,
        final_kld=final_kld,
        final_kld_exact=exact,
        pinned=pinned,
    )
