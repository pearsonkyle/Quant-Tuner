"""Why does calibrated quantization work? A layer-divergence + weight-noise study.

Ties three signals together across calibration variants at the same bit-width:

1. **layer-divergence profile** ``D_l`` — per-layer readout/activation drift of a
   quant vs FP16 over an eval corpus (from cached capture runs; see
   :mod:`quant_tuner.lens.diff`). *Where* in the network does calibration keep
   the model faithful?
2. **per-tensor weight RMSE** — the raw weight-space damage each quant inflicts
   (:mod:`quant_tuner.lens.quant_noise`).
3. **imatrix concentration** — how peaked each tensor's importance vector is
   (gini / normalized entropy / top-1% mass), read the same way
   ``calibrate.imatrix._load_base_imatrix`` reads ``.in_sum2 / .counts``.

The hypothesis the OmniCoder/gemma results suggest: calibration spends its
fidelity budget on mid-to-late attn_output / ffn_down tensors whose importance
is concentrated, and the payoff shows up as reduced late-layer ``D_l`` on the
in-distribution (tools) corpus specifically.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def layer_divergence_profile(run_quant, run_ref, *, readout_quant, readout_ref) -> dict[str, list[float]]:
    """Per-layer {rel_l2, cos, kld} of a quant capture vs an FP16 capture."""
    from quant_tuner.lens.diff import diff_runs

    d = diff_runs(run_quant, run_ref, readout_run=readout_quant, readout_ref=readout_ref)
    return {
        "layers": [int(l) for l in d.layers],
        "rel_l2": [float(x) for x in d.layer_rel_l2],
        "cos": [float(x) for x in d.layer_cos],
        "kld": [float(x) for x in d.layer_kld],
    }


def _gini(x: np.ndarray) -> float:
    x = np.abs(np.asarray(x, dtype=np.float64))
    if x.sum() == 0:
        return 0.0
    xs = np.sort(x)
    n = xs.shape[0]
    cum = np.cumsum(xs)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def _norm_entropy(x: np.ndarray) -> float:
    x = np.abs(np.asarray(x, dtype=np.float64))
    s = x.sum()
    if s == 0:
        return 1.0
    p = x / s
    nz = p[p > 0]
    h = -(nz * np.log(nz)).sum()
    return float(h / np.log(len(x)))  # 1 = uniform, 0 = a single spike


def imatrix_concentration(imatrix_gguf: Path) -> dict[str, dict[str, float]]:
    """Per-tensor concentration stats of the imatrix importance vectors."""
    from quant_tuner.paths import ensure_gguf_py

    ensure_gguf_py()
    from gguf import GGUFReader

    reader = GGUFReader(str(imatrix_gguf))
    raw = {t.name: np.array(t.data) for t in reader.tensors}
    out: dict[str, dict[str, float]] = {}
    for name, arr in raw.items():
        if not name.endswith(".in_sum2"):
            continue
        base = name[: -len(".in_sum2")]
        cnt = raw.get(f"{base}.counts")
        if cnt is None:
            continue
        importance = arr.astype(np.float64) / max(int(cnt.flat[0]), 1)
        srt = np.sort(np.abs(importance))[::-1]
        top1pct = srt[: max(1, len(srt) // 100)].sum() / (srt.sum() + 1e-12)
        out[base] = {
            "gini": _gini(importance),
            "entropy_norm": _norm_entropy(importance),
            "top1pct_mass": float(top1pct),
            "n_channels": int(len(importance)),
        }
    return out


@dataclass
class StudyConfig:
    f16_gguf: str
    quants: dict[str, str]              # label -> quant gguf path
    imatrices: dict[str, str]           # label -> imatrix gguf path (optional per label)
    lens: str
    eval_corpora: dict[str, str]        # corpus label -> path (e.g. tools, general)
    max_tokens: int = 4096

    @classmethod
    def from_yaml(cls, path: Path) -> StudyConfig:
        import yaml

        d = yaml.safe_load(Path(path).read_text())
        return cls(
            f16_gguf=d["f16_gguf"],
            quants=d["quants"],
            imatrices=d.get("imatrices", {}),
            lens=d["lens"],
            eval_corpora=d.get("eval_corpora", {}),
            max_tokens=int(d.get("max_tokens", 4096)),
        )


def calibration_study(cfg: StudyConfig, out_dir: Path) -> Path:
    """Run the full study: divergence profiles + weight RMSE + imatrix stats.

    Writes ``study.json`` (+ ``study.md``) under ``out_dir`` and returns it.
    Layer-divergence needs capture runs; those are produced here per (quant,
    corpus) using the first ``max_tokens`` of each corpus as a single prompt.
    """
    from quant_tuner.lens.backend import running_jlens_server
    from quant_tuner.lens.capture import capture_run, load_run
    from quant_tuner.lens.lens import JacobianLensGGUF
    from quant_tuner.lens.model_reader import ReadoutWeights
    from quant_tuner.lens.quant_noise import per_tensor_rmse
    from quant_tuner.lens.readout import LensReadout

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    tmp_dir = out_dir / "dequant"
    lens = JacobianLensGGUF.load(cfg.lens)

    # 1) reference (FP16) captures per corpus
    ref_weights = ReadoutWeights.from_gguf(cfg.f16_gguf)
    ref_readout = LensReadout(ref_weights, lens)
    corpus_tokens: dict[str, list[int]] = {}
    ref_runs: dict[str, str] = {}
    with running_jlens_server(Path(cfg.f16_gguf)) as client:
        for cname, cpath in cfg.eval_corpora.items():
            text = Path(cpath).read_text()[: cfg.max_tokens * 8]
            tokens = client.tokenize(text, add_special=True, parse_special=True)[: cfg.max_tokens]
            corpus_tokens[cname] = tokens
            rd = capture_run(client, tokens, runs_dir=runs_dir,
                             tags={"model": "f16", "corpus": cname, "role": "reference"})
            ref_runs[cname] = str(rd)

    result: dict = {"quants": {}, "imatrix": {}}
    for label, qpath in cfg.quants.items():
        q_weights = ReadoutWeights.from_gguf(qpath)
        q_readout = LensReadout(q_weights, lens)
        entry: dict = {"path": qpath, "divergence": {}, "weight_rmse": {}}
        with running_jlens_server(Path(qpath)) as client:
            for cname, tokens in corpus_tokens.items():
                qd = capture_run(client, tokens, runs_dir=runs_dir,
                                 tags={"model": label, "corpus": cname, "role": "candidate"})
                prof = layer_divergence_profile(
                    load_run(qd), load_run(Path(ref_runs[cname])),
                    readout_quant=q_readout, readout_ref=ref_readout,
                )
                entry["divergence"][cname] = prof
        try:
            entry["weight_rmse"] = per_tensor_rmse(Path(cfg.f16_gguf), Path(qpath), tmp_dir)
        except Exception as e:
            logger.warning("weight RMSE failed for %s: %s", label, e)
        result["quants"][label] = entry

    for label, ipath in cfg.imatrices.items():
        try:
            result["imatrix"][label] = imatrix_concentration(Path(ipath))
        except Exception as e:
            logger.warning("imatrix concentration failed for %s: %s", label, e)

    (out_dir / "study.json").write_text(json.dumps(result, indent=1))
    _render_study_md(result, out_dir / "study.md")
    logger.info("wrote study to %s", out_dir)
    return out_dir / "study.json"


def _render_study_md(result: dict, out_md: Path) -> None:
    lines = ["# Why calibration works — layer divergence + weight noise", ""]
    for label, entry in result["quants"].items():
        lines.append(f"## {label}")
        lines.append("")
        for cname, prof in entry.get("divergence", {}).items():
            layers = prof["layers"]
            rel = prof["rel_l2"]
            auc = float(np.mean(rel)) if rel else float("nan")
            peak_layer = layers[int(np.argmax(rel))] if rel else -1
            lines.append(
                f"- **{cname}**: mean rel-L2 divergence {auc:.4f}, "
                f"peak at layer {peak_layer}"
            )
        rmse = entry.get("weight_rmse", {})
        if rmse:
            worst = sorted(rmse.items(), key=lambda kv: -kv[1])[:5]
            lines.append("- worst-hit tensors (rel weight RMSE): " +
                         ", ".join(f"{n}={v:.3f}" for n, v in worst))
        lines.append("")
    out_md.write_text("\n".join(lines) + "\n")
