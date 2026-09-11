"""Continued QAT for a natively-ternary model, then export to a Q2_0 GGUF.

For models that ship already ternarized (the Qwen-Bonsai family and the
prism-ml/Ternary-Bonsai-8B reference), post-hoc calibration is a structural
no-op — the "F16" is a lossless container of w = s·c with c in {-1, 0, +1},
so there is no quantization error for imatrix/AWQ/GPTQ to recover. The only
lever is more training with the ternarization in the loop (TWN straight-through
estimator). This script wraps ``qat.train_qat`` + ``qat.export_qat`` for the
common "I have a ternary Qwen checkpoint and a usage-log corpus" case.

    uv run python examples/ternary_qat.py \
        --model /path/to/ternary-qwen-8b \
        --corpus out/corpora/sft.jsonl.gz \
        --out out/ternary-qwen-8b

The corpus must be the SFT export (``sft.jsonl.gz``) from
``build_universal_corpus.py`` — full conversations, real tool schemas, the
``split`` field matching the calibration corpus. See
``docs/ternary_qat.md`` for the full method and the termination-probe
invariants (every trained run so far broke the model's stop decision; the
probe is what catches it during training, not after).

Note: the working recipe in ``docs/ternary_qat.md`` uses a 32B teacher KD
table + termination steering + a bounded repetition hinge. This example wires
the minimal path (masked CE only); add ``--kd-table`` / ``--steer-weight`` to
match the published recipe.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True,
                   help="HF dir of the natively-ternary checkpoint (bf16 container)")
    p.add_argument("--corpus", type=Path, required=True,
                   help="sft.jsonl.gz from build_universal_corpus.py")
    p.add_argument("--out", type=Path, default=Path("./out/ternary"))
    p.add_argument("--epochs", type=float, default=2.2,
                   help="fractional epochs (default 2.2 — the measured sweet spot; "
                        "8 epochs memorizes)")
    p.add_argument("--lr", type=float, default=5e-4,
                   help="learning rate (default 5e-4 — the code-flip threshold; "
                        "3e-4 flips ~0%%, 1e-3 over-flips)")
    p.add_argument("--layers", type=int, default=36,
                   help="how many decoder layers get gradients (default all)")
    p.add_argument("--device", default="auto",
                   help="cuda | mps | cpu (default auto: cuda > mps > cpu)")
    p.add_argument("--compute-dtype", default="fp32",
                   choices=["fp32", "bf16"],
                   help="fp32 on MPS (required: bf16 underflows the ternary "
                        "threshold); bf16 on CUDA is fine")
    p.add_argument("--stop-weight", type=float, default=6.0,
                   help="CE weight on the terminating </s> token (default 6.0)")
    p.add_argument("--probe-every", type=int, default=25,
                   help="run the termination probe every N steps (default 25; 0 disables)")
    p.add_argument("--tag", default="qwen-bonsai",
                   help="export tag for the GGUF filename")
    p.add_argument("--dry-run", action="store_true",
                   help="print the QATConfig and exit (no model load)")
    a = p.parse_args()

    from quant_tuner.qat.export import export_qat
    from quant_tuner.qat.train import QATConfig, train_qat

    cfg = QATConfig(
        corpus=a.corpus,
        out=a.out,
        model_dir=a.model,
        train_layers=a.layers,
        epochs=a.epochs,
        lr=a.lr,
        device=a.device,
        compute_dtype=a.compute_dtype,
        stop_weight=a.stop_weight,
        probe_every=a.probe_every,
    )

    if a.dry_run:
        import dataclasses
        for f in dataclasses.fields(cfg):
            v = getattr(cfg, f.name)
            if not f.name.startswith("_"):
                print(f"  {f.name}: {v}")
        return 0

    train_qat(cfg)
    # Export the trained latents to a Q2_0 GGUF. Requires
    # LLAMA_CPP_DIR=vendor/llama.cpp-prism (the ftype-41 fork).
    export_qat(a.out / "latents.pt", tag=a.tag, model_dir=a.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
