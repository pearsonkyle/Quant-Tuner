"""Capture runs: activations + metadata persisted to disk — the A/B unit.

A *run* is one forward pass of one model over one token sequence, with the
residual stream captured at chosen layers/positions and (optionally) exact
full-vocab final logprobs at flagged decision positions:

    <runs_dir>/<run_id>/
        run.json            RunManifest (see below)
        activations.npz     layer_<l> -> [n_kept_positions, d] float16
        final_logprobs.npz  optional: positions [m], logprobs [m, vocab] f32

Design decisions:

- **Store activations, not readouts.** Lens readouts are recomputed lazily
  (and cached per (lens, k)); a run stays valid when the lens is refit.
- **Runs are content-addressed.** ``run_id`` hashes the capture request
  (model fingerprint, tokens, layers, kept positions, interventions,
  sampling, llama commit), so re-capturing an identical spec is idempotent —
  the existing directory is returned, ``experiments.step()``-style.
- **Exact vs approximate KLD.** Full-vocab logprobs are persisted only for
  ``logits_positions`` (decision points); elsewhere the diff engine falls
  back to a top-k + tail-mass approximation and says so.
- **h_rms drift alarm.** The manifest records each captured layer's median
  residual norm; readout against a lens fit on a different model (or a badly
  damaged quant) trips a loud warning instead of silently nonsense readouts.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from quant_tuner.lens.client import NativeClient
from quant_tuner.lens.lens import JacobianLensGGUF
from quant_tuner.lens.readout import LensReadout

logger = logging.getLogger(__name__)

MANIFEST_NAME = "run.json"
ACTIVATIONS_NAME = "activations.npz"
FINAL_LOGPROBS_NAME = "final_logprobs.npz"

# relative drift of observed residual norms vs the lens's fitted h_rms that
# triggers a warning (the lens was fit on a different distribution/model)
H_RMS_WARN_RATIO = 2.0


def file_fingerprint(path: Path | str) -> str:
    """Cheap content fingerprint for multi-GiB GGUFs: size + first/last MiB."""
    p = Path(path)
    h = hashlib.sha256()
    size = p.stat().st_size
    h.update(str(size).encode())
    with open(p, "rb") as f:
        h.update(f.read(1 << 20))
        if size > (2 << 20):
            f.seek(-(1 << 20), 2)
            h.update(f.read(1 << 20))
    return h.hexdigest()[:16]


@dataclass
class RunManifest:
    """Everything needed to interpret (and re-produce) a capture run."""

    run_id: str
    model_path: str
    model_fingerprint: str
    tokens: list[int]                    # full sequence incl. generated tokens
    n_prompt: int
    n_gen: int
    capture_layers: list[int]
    positions: list[int]                 # absolute indices retained on disk
    logits_positions: list[int]          # positions with exact final logprobs
    interventions: list[dict]            # UI-level specs (empty = clean run)
    n_predict: int
    sampling: dict
    llama_commit: str
    h_rms_observed: dict[str, float]     # str(layer) -> median ||h|| (JSON-safe)
    generated_text: str = ""
    created_at: str = ""
    tags: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def from_json(cls, text: str) -> RunManifest:
        data = json.loads(text)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def _run_id(
    model_fingerprint: str,
    tokens: list[int],
    layers: list[int],
    positions: list[int] | None,
    interventions: list[dict],
    n_predict: int,
    sampling: dict,
    llama_commit: str,
) -> str:
    payload = json.dumps(
        {
            "model": model_fingerprint,
            "tokens": tokens,
            "layers": layers,
            "positions": positions,
            "interventions": interventions,
            "n_predict": n_predict,
            "sampling": sampling,
            "llama_commit": llama_commit,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _log_softmax(rows: np.ndarray) -> np.ndarray:
    z = rows - rows.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def capture_run(
    client: NativeClient,
    tokens: list[int],
    *,
    runs_dir: Path,
    layers: list[int] | None = None,
    keep_positions: list[int] | None = None,
    logits_positions: list[int] | None = None,
    interventions: list[dict] | None = None,
    native_interventions: list[dict] | None = None,
    n_predict: int = 0,
    sampling: dict | None = None,
    tags: dict | None = None,
    overwrite: bool = False,
) -> Path:
    """Run the model over ``tokens`` and persist a capture run; returns its dir.

    ``layers=None`` captures every layer. ``keep_positions`` bounds what is
    written to disk (absolute indices; negatives resolve against the final
    sequence length) — capture the decision tail, not a 30k-token prompt.
    ``interventions`` are UI-level specs recorded in the manifest;
    ``native_interventions`` are the translated add/lowrank edits actually
    sent to the server (pass both when steering, neither for a clean run).
    """
    runs_dir = Path(runs_dir)
    props = client.props()
    n_layer = int(props["n_layer"])
    layer_list = sorted(set(range(n_layer)) if layers is None else {int(l) for l in layers})
    sampling = dict(sampling or {"greedy": True})
    interventions = list(interventions or [])
    llama_commit = str(props.get("llama_commit", "unknown"))
    model_path = str(props.get("model_path", ""))
    fingerprint = file_fingerprint(model_path) if Path(model_path).exists() else "unknown"

    run_id = _run_id(
        fingerprint, [int(t) for t in tokens], layer_list, keep_positions,
        interventions, n_predict, sampling, llama_commit,
    )
    run_dir = runs_dir / run_id
    if run_dir.exists() and not overwrite and (run_dir / MANIFEST_NAME).exists():
        logger.info("capture run %s already exists — reusing", run_id)
        if tags:  # merge new tags in (tags are not part of the identity)
            run = load_run(run_dir)
            run.manifest.tags.update(tags)
            (run_dir / MANIFEST_NAME).write_text(run.manifest.to_json())
        return run_dir

    fr = client.forward(
        tokens,
        capture_layers=layer_list,
        dtype="f16",
        interventions=native_interventions or None,
        n_predict=n_predict,
        sampling=sampling,
        logits_positions=logits_positions or None,
    )

    n_total = len(fr.tokens)
    if keep_positions is None:
        kept = list(range(n_total))
    else:
        kept = sorted({p % n_total for p in keep_positions})
    kept_arr = np.asarray(kept, dtype=np.int64)

    h_rms: dict[str, float] = {}
    arrays: dict[str, np.ndarray] = {}
    for layer, acts in fr.activations.items():
        h_rms[str(layer)] = float(np.median(np.linalg.norm(acts, axis=-1)))
        arrays[f"layer_{layer}"] = acts[kept_arr].astype(np.float16)

    manifest = RunManifest(
        run_id=run_id,
        model_path=model_path,
        model_fingerprint=fingerprint,
        tokens=[int(t) for t in fr.tokens],
        n_prompt=fr.n_prompt,
        n_gen=fr.n_gen,
        capture_layers=layer_list,
        positions=kept,
        logits_positions=sorted(fr.logits),
        interventions=interventions,
        n_predict=n_predict,
        sampling=sampling,
        llama_commit=llama_commit,
        h_rms_observed=h_rms,
        generated_text=fr.generated_text,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        tags=dict(tags or {}),
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(run_dir / ACTIVATIONS_NAME, **arrays)
    if fr.logits:
        positions = sorted(fr.logits)
        rows = np.stack([fr.logits[p] for p in positions]).astype(np.float32)
        np.savez(
            run_dir / FINAL_LOGPROBS_NAME,
            positions=np.asarray(positions, dtype=np.int64),
            logprobs=_log_softmax(rows),
        )
    (run_dir / MANIFEST_NAME).write_text(manifest.to_json())
    logger.info(
        "captured run %s: %d layers x %d positions (of %d tokens)%s",
        run_id, len(layer_list), len(kept), n_total,
        f", {len(fr.logits)} exact-logit rows" if fr.logits else "",
    )
    return run_dir


@dataclass
class ReadoutGrid:
    """Lens top-k over (layer, kept-position) for one run."""

    layers: list[int]
    positions: list[int]
    ids: np.ndarray        # [L, P, k] int32
    logprobs: np.ndarray   # [L, P, k] float32 (log-softmax over full vocab)
    tail_logmass: np.ndarray  # [L, P] float32: log(1 - sum(exp(topk)))


class Run:
    """A capture run on disk. Activations load lazily, per layer."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.manifest = RunManifest.from_json((self.run_dir / MANIFEST_NAME).read_text())
        self._npz: Any = None  # np.lib.npyio.NpzFile once loaded
        self._acts: dict[int, np.ndarray] = {}

    def __repr__(self) -> str:
        m = self.manifest
        return (
            f"Run({m.run_id}, model={Path(m.model_path).name}, "
            f"L={len(m.capture_layers)}, P={len(m.positions)}, tags={m.tags})"
        )

    def activations(self, layer: int) -> np.ndarray:
        """``[n_kept_positions, d]`` float32 residuals at ``layer``."""
        if layer not in self._acts:
            if self._npz is None:
                self._npz = np.load(self.run_dir / ACTIVATIONS_NAME)
            self._acts[layer] = np.asarray(self._npz[f"layer_{layer}"], dtype=np.float32)
        return self._acts[layer]

    def final_logprobs(self) -> dict[int, np.ndarray]:
        """Exact full-vocab log-softmax rows at the flagged decision positions."""
        path = self.run_dir / FINAL_LOGPROBS_NAME
        if not path.exists():
            return {}
        data = np.load(path)
        return {
            int(p): data["logprobs"][i]
            for i, p in enumerate(data["positions"])
        }

    def check_h_rms(self, lens: JacobianLensGGUF) -> list[str]:
        """Warn-worthy layers where observed ||h|| drifts far from the lens fit."""
        bad = []
        for layer_s, observed in self.manifest.h_rms_observed.items():
            fitted = lens.h_rms.get(int(layer_s))
            if not fitted or not observed:
                continue
            ratio = observed / fitted
            if ratio > H_RMS_WARN_RATIO or ratio < 1.0 / H_RMS_WARN_RATIO:
                bad.append(f"layer {layer_s}: observed {observed:.1f} vs lens {fitted:.1f}")
        if bad:
            logger.warning(
                "run %s: residual norms drift from the lens's fit distribution "
                "(%s) — was this lens fit on a different model?",
                self.manifest.run_id, "; ".join(bad[:4]),
            )
        return bad

    def readout_topk(
        self,
        readout: LensReadout,
        *,
        k: int = 8,
        use_lens: bool = True,
        cache: bool = True,
    ) -> ReadoutGrid:
        """Lens top-k grid, cached on disk per (lens content, k, use_lens)."""
        lens = readout.lens
        lens_key = hashlib.sha256(
            json.dumps(
                {
                    "layers": lens.source_layers,
                    "method": lens.fit_method,
                    "n_prompts": lens.n_prompts,
                    "d": lens.d_model,
                    "k": k,
                    "use_lens": use_lens,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:10]
        cache_path = self.run_dir / f"readout_topk_{lens_key}.npz"
        layers = [l for l in self.manifest.capture_layers]
        if cache and cache_path.exists():
            data = np.load(cache_path)
            return ReadoutGrid(
                layers=list(data["layers"]),
                positions=list(self.manifest.positions),
                ids=data["ids"],
                logprobs=data["logprobs"],
                tail_logmass=data["tail_logmass"],
            )

        self.check_h_rms(lens)
        P = len(self.manifest.positions)
        ids = np.zeros((len(layers), P, k), dtype=np.int32)
        logprobs = np.zeros((len(layers), P, k), dtype=np.float32)
        tail = np.zeros((len(layers), P), dtype=np.float32)
        for li, layer in enumerate(layers):
            logits = readout.lens_logits(self.activations(layer), layer, use_lens=use_lens)
            lp = _log_softmax(logits)
            top_ids, top_vals = LensReadout.topk(lp, k)
            ids[li] = top_ids.astype(np.int32)
            logprobs[li] = top_vals.astype(np.float32)
            covered = np.exp(top_vals).sum(axis=-1)
            tail[li] = np.log(np.clip(1.0 - covered, 1e-12, 1.0)).astype(np.float32)
            del logits, lp
        grid = ReadoutGrid(
            layers=layers, positions=list(self.manifest.positions),
            ids=ids, logprobs=logprobs, tail_logmass=tail,
        )
        if cache:
            np.savez(
                cache_path,
                layers=np.asarray(layers, dtype=np.int64),
                ids=ids, logprobs=logprobs, tail_logmass=tail,
            )
        return grid


def load_run(run_dir: Path) -> Run:
    return Run(run_dir)


def find_runs(runs_dir: Path, **tag_filters: str) -> list[RunManifest]:
    """All run manifests under ``runs_dir`` whose tags match ``tag_filters``."""
    out: list[RunManifest] = []
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return out
    for manifest_path in sorted(runs_dir.glob(f"*/{MANIFEST_NAME}")):
        try:
            manifest = RunManifest.from_json(manifest_path.read_text())
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("skipping malformed manifest %s: %s", manifest_path, e)
            continue
        if all(str(manifest.tags.get(k)) == str(v) for k, v in tag_filters.items()):
            out.append(manifest)
    out.sort(key=lambda m: m.created_at)
    return out
