#!/usr/bin/env python3
"""exp-101: tool-call representations under quantization (Jacobian lens).

For a set of gemma-4-31B quants + the FP16 reference, replay the tool-call
holdout through the lens and measure where the tool-call decision forms:
emergence layer of the gold tool token, its final lens rank, and per-decision
divergence vs FP16. Writes a leaderboard sidecar CSV and per-decision JSONL,
plus a short markdown summary.

The story this tests: calibrated 2-bit quants should diverge from FP16 only in
late layers and keep the gold tool token's emergence layer close to FP16, while
the uncalibrated ``Q2_K-plain`` anchor should show earlier emergence-layer
degradation and gold-rank inflation — quantifying exactly where gemma's agentic
weakness lives.

Usage::

    PYTHONPATH=src python scripts/lens_exp101_toolcall_lens.py \
        --f16 out/exp-044/vanilla/model-f16.gguf \
        --lens out/lens/gemma-4-31B.lens.gguf \
        --holdout out/omnicoder_q4_k_m/toolcall_holdout.jsonl \
        --quant uploads/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/gemma-4-31B-it-qat-Q2_K_S-imatrix.gguf \
        --out out/lens/exp101
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from quant_tuner.lens.replay import (  # noqa: E402
    replay_toolcall_lens,
    write_decisions_jsonl,
    write_lens_sidecar,
)

# gemma-4-31B defaults (paths as they exist in the study workspace)
_DEFAULT_F16 = _REPO / "out" / "exp-044" / "vanilla" / "model-f16.gguf"
_DEFAULT_QUANTS = [
    _REPO / "uploads" / "pearsonkyle" / "gemma-4-31B-it-awq-2bit-GGUF"
    / "gemma-4-31B-it-qat-Q2_K_S-imatrix.gguf",
    _REPO / "uploads" / "pearsonkyle" / "gemma-4-31b-it-imatrix-GGUF"
    / "gemma-4-31B-it-IQ2_M.gguf",
    _REPO / "out" / "exp-052" / "gemma-4-31B-it-Q2_K-plain.gguf",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--f16", type=Path, default=_DEFAULT_F16)
    ap.add_argument("--quant", type=Path, action="append", dest="quants")
    ap.add_argument("--lens", type=Path, required=True)
    ap.add_argument("--holdout", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=_REPO / "out" / "lens" / "exp101")
    ap.add_argument("--tail", type=int, default=16)
    ap.add_argument("--max-sessions", type=int, default=None)
    ap.add_argument("--ngl", type=int, default=99)
    args = ap.parse_args()

    quants = args.quants or _DEFAULT_QUANTS
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    runs_dir = out / "runs"
    ref_runs = out / "runs_ref"

    # reference (FP16) capture pass first, so quant replays can diff against it
    print(f"[exp-101] reference pass: {args.f16.name}")
    replay_toolcall_lens(
        args.holdout, args.f16, args.lens, runs_dir=ref_runs,
        tail_positions=args.tail, max_sessions=args.max_sessions,
        ngl=args.ngl, label="f16",
    )

    summaries = []
    for quant in quants:
        if not Path(quant).exists():
            print(f"[exp-101] SKIP missing quant {quant}")
            continue
        print(f"[exp-101] {Path(quant).name}")
        summary = replay_toolcall_lens(
            args.holdout, quant, args.lens, runs_dir=runs_dir,
            reference_runs_dir=ref_runs, tail_positions=args.tail,
            max_sessions=args.max_sessions, ngl=args.ngl,
        )
        summaries.append(summary)
        write_decisions_jsonl(summary, out / f"{Path(quant).stem}.decisions.jsonl")
        print(f"    decisions={summary.n_decisions} "
              f"gold_rank={summary.gold_tool_rank_final_mean:.2f} "
              f"emerge_layer={summary.emergence_layer_mean:.1f}")

    if summaries:
        write_lens_sidecar(summaries, out / "lens_toolcall.csv")
        _render(summaries, out / "exp101.md")
        print(f"[exp-101] wrote sidecar + summary to {out}")
    return 0


def _render(summaries, out_md: Path) -> None:
    lines = ["# exp-101: tool-call representations under quantization", "",
             "| quant | decisions | gold rank (final) | emergence layer | decision KLD vs FP16 |",
             "| --- | --- | --- | --- | --- |"]
    for s in summaries:
        kld = f"{s.decision_kld_vs_ref_mean:.4f}" if s.decision_kld_vs_ref_mean is not None else "-"
        lines.append(
            f"| {s.model} | {s.n_decisions} | {s.gold_tool_rank_final_mean:.2f} | "
            f"{s.emergence_layer_mean:.1f} | {kld} |"
        )
    out_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
