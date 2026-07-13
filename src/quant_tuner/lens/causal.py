"""Exact causal-Jacobian lens fitting via the vendored anthropics/jacobian-lens.

The default lens fit (:func:`quant_tuner.lens.fitting.fit_lens`) is a
forward-only ridge regression that runs on any GGUF. This module is the
optional *exact* path: it loads the FP16 HF snapshot with the reference
PyTorch implementation, fits ``J_l = E[∂h_final/∂h_l]`` by backprop, saves a
``.pt``, and converts it to the same GGUF lens format via
:mod:`quant_tuner.lens.pt_convert` (torch-free).

The submodule is imported lazily and only here, so the default regression path
never depends on it — and its transformers-version constraints (it was written
against 4.x) only bite if you actually run ``fit-causal``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _ensure_jlens_vendor() -> None:
    from quant_tuner.paths import REPO_ROOT

    vendor = REPO_ROOT / "vendor" / "jacobian-lens"
    if not (vendor / "jlens").exists():
        raise RuntimeError(
            "vendor/jacobian-lens is not checked out. Run:\n"
            "  git submodule update --init vendor/jacobian-lens"
        )
    p = str(vendor)
    if p not in sys.path:
        sys.path.insert(0, p)


def fit_causal_lens(
    hf_model: Path,
    model_gguf: Path,
    corpus: Path,
    out_path: Path,
    *,
    n_prompts: int = 200,
    max_seq_len: int = 128,
    checkpoint_path: Path | None = None,
):
    """Fit the exact causal lens on the HF snapshot, then convert to lens GGUF."""
    _ensure_jlens_vendor()
    try:
        import jlens
        import transformers
    except ImportError as e:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "fit-causal needs torch + transformers + the vendored jacobian-lens "
            "installed. If transformers>=5 conflicts with the reference impl "
            "(written against 4.x), fit in a separate venv:\n"
            "  python -m venv .venv-causal && .venv-causal/bin/pip install "
            "-e vendor/jacobian-lens && .venv-causal/bin/python -m jlens ...\n"
            "then `quant-tuner lens convert-pt lens.pt lens.gguf`.\n"
            f"(import error: {e})"
        ) from e

    from quant_tuner.lens.fitting import load_corpus
    from quant_tuner.lens.pt_convert import convert_pt_lens

    hf = transformers.AutoModelForCausalLM.from_pretrained(str(hf_model))
    tok = transformers.AutoTokenizer.from_pretrained(str(hf_model))
    try:
        hf = hf.cuda()
    except Exception:
        logger.info("running causal fit on CPU (no CUDA)")
    model = jlens.from_hf(hf, tok)

    prompts = load_corpus(str(corpus), n_prompts=n_prompts)
    ckpt = str(checkpoint_path) if checkpoint_path else str(Path(out_path).with_suffix(".pt"))
    logger.info("fitting causal Jacobian on %d prompts ...", len(prompts))
    lens = jlens.fit(model, prompts=prompts, checkpoint_path=ckpt)
    lens.save(ckpt)
    logger.info("saved reference lens -> %s; converting to GGUF", ckpt)
    return convert_pt_lens(ckpt, str(out_path), base_model=str(model_gguf))
