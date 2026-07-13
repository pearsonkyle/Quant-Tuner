#!/usr/bin/env python3
"""exp-104: why calibrated quantization works (Jacobian lens divergence study).

Runs ``quant_tuner.lens.study.calibration_study`` over a set of same-bit-width
gemma-4-31B quants that differ only in calibration (e.g. ``Q2_K-plain`` vs
``qat-Q2_K_S-imatrix``; IQ2_M imatrix vs AWQ), plus the complementary OmniCoder
Q4_K_M imatrix-variant sweep. For each variant it computes the per-layer
divergence profile vs FP16 on in-distribution (tools) and general corpora, the
per-tensor weight RMSE, and — where an imatrix is available — its per-tensor
importance concentration.

This is config-driven; write a study YAML (see ``--emit-example``) and pass it
with ``--config``.

Usage::

    PYTHONPATH=src python scripts/lens_exp104_why_calibration.py --emit-example > study.yaml
    PYTHONPATH=src python scripts/lens_exp104_why_calibration.py --config study.yaml --out out/lens/exp104
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from quant_tuner.lens.study import StudyConfig, calibration_study  # noqa: E402

_EXAMPLE = """\
# exp-104 study config: gemma-4-31B calibration contrasts at ~2 bpw.
f16_gguf: out/exp-044/vanilla/model-f16.gguf
lens: out/lens/gemma-4-31B.lens.gguf
max_tokens: 4096
quants:
  q2k_plain:      out/exp-052/gemma-4-31B-it-Q2_K-plain.gguf
  q2k_imatrix:    uploads/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/gemma-4-31B-it-qat-Q2_K_S-imatrix.gguf
  iq2m_imatrix:   uploads/pearsonkyle/gemma-4-31b-it-imatrix-GGUF/gemma-4-31B-it-IQ2_M.gguf
imatrices:
  q2k_imatrix:    out/exp-052/imatrix.gguf
eval_corpora:
  tools:   out/exp-044/corpora/corpus.eval.tools.txt
  general: out/exp-044/corpora/corpus.eval.general.txt
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=_REPO / "out" / "lens" / "exp104")
    ap.add_argument("--emit-example", action="store_true", help="print an example study YAML and exit")
    args = ap.parse_args()

    if args.emit_example:
        print(_EXAMPLE)
        return 0
    if args.config is None:
        raise SystemExit("pass --config study.yaml (or --emit-example to see the shape)")

    cfg = StudyConfig.from_yaml(args.config)
    out_json = calibration_study(cfg, args.out)
    print(f"[exp-104] wrote {out_json} and study.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
