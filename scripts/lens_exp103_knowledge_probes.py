#!/usr/bin/env python3
"""exp-103: knowledge loss at 2 bpw — erased vs suppressed (Jacobian lens).

Runs a probe set through gemma-4-31B FP16 and each 2-bit quant, classifying each
probe as correct / suppressed / absent, and reports the ref->quant transition
matrix. The hypothesis: calibration shifts failures from *absent* (knowledge
erased) toward *suppressed* (knowledge present in the residual stream, readout
degraded) — which is why calibrated quants respond to mild steering.

Usage::

    PYTHONPATH=src python scripts/lens_exp103_knowledge_probes.py \
        --f16 out/exp-044/vanilla/model-f16.gguf \
        --lens out/lens/gemma-4-31B.lens.gguf \
        --probes artifacts/probes.jsonl \
        --quant uploads/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/gemma-4-31B-it-qat-Q2_K_S-imatrix.gguf \
        --out out/lens/exp103
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from quant_tuner.lens.backend import running_jlens_server  # noqa: E402
from quant_tuner.lens.lens import JacobianLensGGUF  # noqa: E402
from quant_tuner.lens.model_reader import ReadoutWeights  # noqa: E402
from quant_tuner.lens.probes import (  # noqa: E402
    compare_probes,
    load_probes,
    render_probe_report,
    run_probes,
    write_probe_results,
)
from quant_tuner.lens.readout import LensReadout  # noqa: E402

_DEFAULT_F16 = _REPO / "out" / "exp-044" / "vanilla" / "model-f16.gguf"
_DEFAULT_QUANTS = [
    _REPO / "uploads" / "pearsonkyle" / "gemma-4-31B-it-awq-2bit-GGUF"
    / "gemma-4-31B-it-qat-Q2_K_S-imatrix.gguf",
    _REPO / "uploads" / "pearsonkyle" / "gemma-4-31b-it-imatrix-GGUF"
    / "gemma-4-31B-it-IQ2_M.gguf",
    _REPO / "out" / "exp-052" / "gemma-4-31B-it-Q2_K-plain.gguf",
]


def _probe_model(model: Path, lens, probes, ngl):
    readout = LensReadout(ReadoutWeights.from_gguf(str(model)), lens)
    with running_jlens_server(model, ngl=ngl) as client:
        return run_probes(client, readout, probes)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--f16", type=Path, default=_DEFAULT_F16)
    ap.add_argument("--quant", type=Path, action="append", dest="quants")
    ap.add_argument("--lens", type=Path, required=True)
    ap.add_argument("--probes", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=_REPO / "out" / "lens" / "exp103")
    ap.add_argument("--ngl", type=int, default=99)
    args = ap.parse_args()

    quants = args.quants or _DEFAULT_QUANTS
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    lens = JacobianLensGGUF.load(str(args.lens))
    probes = load_probes(args.probes)

    print(f"[exp-103] reference: {args.f16.name} ({len(probes)} probes)")
    ref_results = _probe_model(args.f16, lens, probes, args.ngl)
    write_probe_results(ref_results, out / "f16_probes.jsonl")

    summary_rows = []
    for quant in quants:
        if not Path(quant).exists():
            print(f"[exp-103] SKIP missing quant {quant}")
            continue
        print(f"[exp-103] {Path(quant).name}")
        q_results = _probe_model(quant, lens, probes, args.ngl)
        write_probe_results(q_results, out / f"{Path(quant).stem}_probes.jsonl")
        cmp = compare_probes(q_results, ref_results)
        cmp.quant_model = Path(quant).name
        cmp.ref_model = args.f16.name
        render_probe_report(cmp, out / f"{Path(quant).stem}.probe_report.md")
        s = cmp.summary()
        summary_rows.append((Path(quant).name, s))
        print(f"    preserved={s['probe_preserved']} suppressed={s['probe_suppressed']} "
              f"erased={s['probe_erased']} suppressed_frac={s['probe_suppressed_frac_of_lost']:.2f}")

    _render(summary_rows, out / "exp103.md", args.f16.name)
    print(f"[exp-103] wrote reports to {out}")
    return 0


def _render(rows, out_md: Path, ref_name: str) -> None:
    lines = [f"# exp-103: knowledge loss at 2 bpw (vs {ref_name})", "",
             "| quant | preserved | suppressed | erased | suppressed-fraction-of-lost |",
             "| --- | --- | --- | --- | --- |"]
    for name, s in rows:
        lines.append(
            f"| {name} | {s['probe_preserved']} | {s['probe_suppressed']} | "
            f"{s['probe_erased']} | {s['probe_suppressed_frac_of_lost']:.2f} |"
        )
    out_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
