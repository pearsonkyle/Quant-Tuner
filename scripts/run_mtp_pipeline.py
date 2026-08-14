"""Orchestrate per-model MTP head training, GGUF rebuild, and acceptance sweep.

For each model (tesslate → jackrong → qwen):
  1. Train MTP head (293 steps, 2 epochs, lr=2e-5) — IQ3_S-trained.gguf
  2. Install trained weights into model_extracted (replaces donor shard)
  3. Reconvert HF → F16 GGUF (picks up new MTP weights)
  4. Requantize IQ3_S using existing imatrix-hybrid_custom.gguf (no recalibration needed;
     MTP tensors are a tiny fraction of the calibration signal)
  5. Save as IQ3_S-trained.gguf
  6. Train noisy MTP head (same as above + --quant-noise 0.029) — IQ3_S-trained-noisy.gguf
  7. Install noisy weights, reconvert + requantize → IQ3_S-trained-noisy.gguf

After all three: run plot_mtp_acceptance.py --variants vanilla,trained,trained-noisy

Usage::

    # Full pipeline (waits for tesslate training if already running)
    uv run python scripts/run_mtp_pipeline.py --logs logtrain.jsonl

    # Skip already-trained models
    uv run python scripts/run_mtp_pipeline.py --logs logtrain.jsonl --models jackrong,qwen

    # Dry run
    uv run python scripts/run_mtp_pipeline.py --logs logtrain.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.experiments import log, phase
from quant_tuner.quantize import convert, gguf

WORKSPACE = REPO / "out/benchmark_9b_iq3s"
TRAINING_STEPS = 293   # 2 epochs over 585 chunks @ grad_accum=4
TRAINING_LR    = 2e-5


# ---------------------------------------------------------------------------
# Install trained MTP weights into model_extracted
# ---------------------------------------------------------------------------

def _install_trained_weights(slug: str, workspace: Path) -> None:
    """Copy trained safetensors shard into model_extracted and update the index."""
    model_dir   = workspace / slug / "model_extracted"
    trained_src = workspace / slug / "mtp_trained" / "model-mtp-trained.safetensors"

    if not trained_src.exists():
        raise FileNotFoundError(f"Trained weights not found: {trained_src}")

    import shutil
    dest_shard = "model-mtp-trained.safetensors"
    shutil.copy2(str(trained_src), str(model_dir / dest_shard))
    log(f"  copied {trained_src.name} → model_extracted/{dest_shard}")

    # Update index: redirect all mtp.* keys to the new shard
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    wmap: dict[str, str] = index["weight_map"]
    updated = 0
    for key in list(wmap):
        if key.startswith("mtp."):
            wmap[key] = dest_shard
            updated += 1
    index_path.write_text(json.dumps(index, indent=2))
    log(f"  updated {updated} mtp.* entries in safetensors index → {dest_shard}")


# ---------------------------------------------------------------------------
# Train one model's MTP head
# ---------------------------------------------------------------------------

def _train(slug: str, workspace: Path, logs: Path, dry_run: bool) -> None:
    model_dir  = workspace / slug / "model_extracted"
    out_dir    = workspace / slug / "mtp_trained"
    done_flag  = out_dir / "model-mtp-trained.safetensors"

    if done_flag.exists():
        log(f"  [{slug}] trained weights already exist — skipping training")
        return

    cmd = [
        "uv", "run", "python", str(REPO / "scripts/train_mtp_head.py"),
        "--model",      str(model_dir),
        "--logs",       str(logs),
        "--out",        str(out_dir),
        "--steps",      str(TRAINING_STEPS),
        "--lr",         str(TRAINING_LR),
    ]
    if dry_run:
        log(f"  [dry-run] would run: {' '.join(cmd)}")
        return

    log_path = workspace / slug / "logs" / "train_mtp.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"  [{slug}] training MTP head → {log_path}")

    with open(log_path, "a") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf)

    # Stream progress to our log every 60s
    while proc.poll() is None:
        time.sleep(60)
        # Print last meaningful line from training log
        try:
            lines = log_path.read_text().splitlines()
            recent = [l for l in lines[-20:] if "step" in l or "PPL" in l or "top-1" in l or "DONE" in l]
            if recent:
                log(f"    {recent[-1].strip()}")
        except Exception:
            pass

    if proc.returncode != 0:
        raise RuntimeError(f"[{slug}] training failed (exit {proc.returncode}) — see {log_path}")
    log(f"  [{slug}] training complete")


# ---------------------------------------------------------------------------
# Rebuild GGUF with trained MTP weights
# ---------------------------------------------------------------------------

def _rebuild_gguf(slug: str, workspace: Path, dry_run: bool) -> None:
    model_dir   = workspace / slug / "model_extracted"
    f16         = workspace / slug / "model-f16.gguf"
    imatrix     = workspace / slug / "imatrix-hybrid_custom.gguf"
    out_quant   = workspace / slug / "IQ3_S-trained.gguf"
    log_dir     = workspace / slug / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if not imatrix.exists():
        raise FileNotFoundError(f"imatrix not found: {imatrix} — run benchmark_9b_models.py first")

    if dry_run:
        log(f"  [dry-run] would reconvert + quantize IQ3_S-trained.gguf for {slug}")
        return

    # Delete stale F16 so reconversion picks up new MTP shard
    if f16.exists():
        f16.unlink()
        log(f"  [{slug}] deleted stale model-f16.gguf")

    with phase(f"[{slug}] reconvert HF → F16 GGUF"):
        convert.hf_to_f16_gguf(model_dir, f16, log=log_dir / "convert.log")

    with phase(f"[{slug}] quantize IQ3_S-trained"):
        gguf.quantize(f16, out_quant, "IQ3_S",
                      imatrix=imatrix, log=log_dir / "quantize_trained.log")

    log(f"  [{slug}] built {out_quant.name} ({out_quant.stat().st_size / 1e9:.2f} GB)")


# ---------------------------------------------------------------------------
# Train noisy MTP head (same as _train but with quantization-noise injection)
# ---------------------------------------------------------------------------

def _train_noisy(slug: str, workspace: Path, logs: Path, dry_run: bool) -> None:
    model_dir  = workspace / slug / "model_extracted"
    out_dir    = workspace / slug / "mtp_trained_noisy"
    done_flag  = out_dir / "model-mtp-trained.safetensors"

    if done_flag.exists():
        log(f"  [{slug}] noisy trained weights already exist — skipping")
        return

    cmd = [
        "uv", "run", "python", str(REPO / "scripts/train_mtp_head.py"),
        "--model",      str(model_dir),
        "--logs",       str(logs),
        "--out",        str(out_dir),
        "--steps",      str(TRAINING_STEPS),
        "--lr",         str(TRAINING_LR),
        "--quant-noise", "0.029",
        "--quant-type",  "IQ3_S",
    ]
    if dry_run:
        log(f"  [dry-run] would run: {' '.join(cmd)}")
        return

    log_path = workspace / slug / "logs" / "train_mtp_noisy.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"  [{slug}] training noisy MTP head → {log_path}")

    with open(log_path, "a") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf)

    while proc.poll() is None:
        time.sleep(60)
        try:
            lines = log_path.read_text().splitlines()
            recent = [l for l in lines[-20:] if "step" in l or "PPL" in l or "top-1" in l or "DONE" in l]
            if recent:
                log(f"    {recent[-1].strip()}")
        except Exception:
            pass

    if proc.returncode != 0:
        raise RuntimeError(f"[{slug}] noisy training failed (exit {proc.returncode}) — see {log_path}")
    log(f"  [{slug}] noisy training complete")


def _install_noisy_weights(slug: str, workspace: Path) -> None:
    """Copy noisy-trained safetensors shard into model_extracted and update the index."""
    model_dir   = workspace / slug / "model_extracted"
    trained_src = workspace / slug / "mtp_trained_noisy" / "model-mtp-trained.safetensors"

    if not trained_src.exists():
        raise FileNotFoundError(f"Noisy trained weights not found: {trained_src}")

    import shutil
    dest_shard = "model-mtp-trained-noisy.safetensors"
    shutil.copy2(str(trained_src), str(model_dir / dest_shard))
    log(f"  copied {trained_src.name} → model_extracted/{dest_shard}")

    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    wmap: dict[str, str] = index["weight_map"]
    updated = 0
    for key in list(wmap):
        if key.startswith("mtp."):
            wmap[key] = dest_shard
            updated += 1
    index_path.write_text(json.dumps(index, indent=2))
    log(f"  updated {updated} mtp.* entries in safetensors index → {dest_shard}")


def _rebuild_gguf_noisy(slug: str, workspace: Path, dry_run: bool) -> None:
    model_dir   = workspace / slug / "model_extracted"
    f16         = workspace / slug / "model-f16.gguf"
    imatrix     = workspace / slug / "imatrix-hybrid_custom.gguf"
    out_quant   = workspace / slug / "IQ3_S-trained-noisy.gguf"
    log_dir     = workspace / slug / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if not imatrix.exists():
        raise FileNotFoundError(f"imatrix not found: {imatrix} — run benchmark_9b_models.py first")

    if dry_run:
        log(f"  [dry-run] would reconvert + quantize IQ3_S-trained-noisy.gguf for {slug}")
        return

    if f16.exists():
        f16.unlink()
        log(f"  [{slug}] deleted stale model-f16.gguf")

    with phase(f"[{slug}] reconvert HF → F16 GGUF (noisy)"):
        convert.hf_to_f16_gguf(model_dir, f16, log=log_dir / "convert_noisy.log")

    with phase(f"[{slug}] quantize IQ3_S-trained-noisy"):
        gguf.quantize(f16, out_quant, "IQ3_S",
                      imatrix=imatrix, log=log_dir / "quantize_trained_noisy.log")

    log(f"  [{slug}] built {out_quant.name} ({out_quant.stat().st_size / 1e9:.2f} GB)")


# ---------------------------------------------------------------------------
# Train wiki-mixed MTP head (tool-call logs + 30 % general text)
# ---------------------------------------------------------------------------

WIKI_MIX = 0.3


def _train_mixed(slug: str, workspace: Path, logs: Path, dry_run: bool) -> None:
    model_dir  = workspace / slug / "model_extracted"
    out_dir    = workspace / slug / "mtp_trained_mixed"
    done_flag  = out_dir / "model-mtp-trained.safetensors"

    if done_flag.exists():
        log(f"  [{slug}] mixed trained weights already exist — skipping")
        return

    cmd = [
        "uv", "run", "python", str(REPO / "scripts/train_mtp_head.py"),
        "--model",     str(model_dir),
        "--logs",      str(logs),
        "--out",       str(out_dir),
        "--steps",     str(TRAINING_STEPS),
        "--lr",        str(TRAINING_LR),
        "--wiki-mix",  str(WIKI_MIX),
    ]
    if dry_run:
        log(f"  [dry-run] would run: {' '.join(cmd)}")
        return

    log_path = workspace / slug / "logs" / "train_mtp_mixed.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"  [{slug}] training mixed MTP head → {log_path}")

    with open(log_path, "a") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf)

    while proc.poll() is None:
        time.sleep(60)
        try:
            lines = log_path.read_text().splitlines()
            recent = [l for l in lines[-20:] if "step" in l or "PPL" in l or "top-1" in l or "DONE" in l]
            if recent:
                log(f"    {recent[-1].strip()}")
        except Exception:
            pass

    if proc.returncode != 0:
        raise RuntimeError(f"[{slug}] mixed training failed (exit {proc.returncode}) — see {log_path}")
    log(f"  [{slug}] mixed training complete")


def _install_mixed_weights(slug: str, workspace: Path) -> None:
    """Copy mixed-trained safetensors shard into model_extracted and update the index."""
    model_dir   = workspace / slug / "model_extracted"
    trained_src = workspace / slug / "mtp_trained_mixed" / "model-mtp-trained.safetensors"

    if not trained_src.exists():
        raise FileNotFoundError(f"Mixed trained weights not found: {trained_src}")

    import shutil
    dest_shard = "model-mtp-trained-mixed.safetensors"
    shutil.copy2(str(trained_src), str(model_dir / dest_shard))
    log(f"  copied {trained_src.name} → model_extracted/{dest_shard}")

    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    wmap: dict[str, str] = index["weight_map"]
    updated = 0
    for key in list(wmap):
        if key.startswith("mtp."):
            wmap[key] = dest_shard
            updated += 1
    index_path.write_text(json.dumps(index, indent=2))
    log(f"  updated {updated} mtp.* entries in safetensors index → {dest_shard}")


def _rebuild_gguf_mixed(slug: str, workspace: Path, dry_run: bool) -> None:
    model_dir   = workspace / slug / "model_extracted"
    f16         = workspace / slug / "model-f16.gguf"
    imatrix     = workspace / slug / "imatrix-hybrid_custom.gguf"
    out_quant   = workspace / slug / "IQ3_S-trained-mixed.gguf"
    log_dir     = workspace / slug / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if not imatrix.exists():
        raise FileNotFoundError(f"imatrix not found: {imatrix} — run benchmark_9b_models.py first")

    if dry_run:
        log(f"  [dry-run] would reconvert + quantize IQ3_S-trained-mixed.gguf for {slug}")
        return

    if f16.exists():
        f16.unlink()
        log(f"  [{slug}] deleted stale model-f16.gguf")

    with phase(f"[{slug}] reconvert HF → F16 GGUF (mixed)"):
        convert.hf_to_f16_gguf(model_dir, f16, log=log_dir / "convert_mixed.log")

    with phase(f"[{slug}] quantize IQ3_S-trained-mixed"):
        gguf.quantize(f16, out_quant, "IQ3_S",
                      imatrix=imatrix, log=log_dir / "quantize_trained_mixed.log")

    log(f"  [{slug}] built {out_quant.name} ({out_quant.stat().st_size / 1e9:.2f} GB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", type=Path, default=REPO / "logtrain.jsonl")
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--models", type=str, default="tesslate,jackrong,qwen",
                        help="Comma-separated model slugs to process in order")
    parser.add_argument("--skip-sweep", action="store_true",
                        help="Skip the final acceptance rate sweep")
    parser.add_argument("--sweep-reps", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    slugs = [s.strip() for s in args.models.split(",")]

    log("=" * 60)
    log(f"MTP pipeline: {' → '.join(slugs)}")
    log(f"workspace:    {args.workspace}")
    log(f"steps/model:  {TRAINING_STEPS} ({TRAINING_STEPS / (585 / 4):.1f} epochs)")
    log("=" * 60)

    for slug in slugs:
        log(f"\n{'='*60}")
        log(f"  MODEL: {slug}")
        log(f"{'='*60}")

        with phase(f"[{slug}] train MTP head"):
            _train(slug, args.workspace, args.logs, args.dry_run)

        with phase(f"[{slug}] install trained weights"):
            if not args.dry_run:
                _install_trained_weights(slug, args.workspace)
            else:
                log(f"  [dry-run] would install trained weights for {slug}")

        with phase(f"[{slug}] rebuild IQ3_S-trained GGUF"):
            _rebuild_gguf(slug, args.workspace, args.dry_run)

        with phase(f"[{slug}] train noisy MTP head"):
            _train_noisy(slug, args.workspace, args.logs, args.dry_run)

        with phase(f"[{slug}] install noisy weights"):
            if not args.dry_run:
                _install_noisy_weights(slug, args.workspace)
            else:
                log(f"  [dry-run] would install noisy weights for {slug}")

        with phase(f"[{slug}] rebuild IQ3_S-trained-noisy GGUF"):
            _rebuild_gguf_noisy(slug, args.workspace, args.dry_run)

        with phase(f"[{slug}] train mixed MTP head (wiki-mix={WIKI_MIX})"):
            _train_mixed(slug, args.workspace, args.logs, args.dry_run)

        with phase(f"[{slug}] install mixed weights"):
            if not args.dry_run:
                _install_mixed_weights(slug, args.workspace)
            else:
                log(f"  [dry-run] would install mixed weights for {slug}")

        with phase(f"[{slug}] rebuild IQ3_S-trained-mixed GGUF"):
            _rebuild_gguf_mixed(slug, args.workspace, args.dry_run)

    # Final acceptance sweep across all models, all variants
    if not args.skip_sweep:
        log(f"\n{'='*60}")
        log("  ACCEPTANCE SWEEP: vanilla + trained + trained-noisy + trained-mixed, all models")
        log(f"{'='*60}")

        sweep_cmd = [
            "uv", "run", "python",
            str(REPO / "scripts/plot_mtp_acceptance.py"),
            "--workspace", str(args.workspace),
            "--holdout-jsonl", str(args.logs),
            "--models", ",".join(slugs),
            "--variants", "vanilla,trained,trained-noisy,trained-mixed",
            "--reps", str(args.sweep_reps),
        ]
        if args.dry_run:
            log(f"  [dry-run] would run: {' '.join(sweep_cmd)}")
        else:
            sweep_log = args.workspace / "mtp_acceptance_final.log"
            log(f"  sweep log → {sweep_log}")
            with open(sweep_log, "w") as lf:
                proc = subprocess.Popen(sweep_cmd, stdout=lf, stderr=lf)
                while proc.poll() is None:
                    time.sleep(60)
                    try:
                        lines = sweep_log.read_text().splitlines()
                        recent = [l for l in lines[-10:]
                                  if any(x in l for x in ["n_max", "acceptance", "===", "wrote"])]
                        if recent:
                            log(f"    {recent[-1].strip()}")
                    except Exception:
                        pass
                if proc.returncode != 0:
                    log(f"WARNING: sweep exited with code {proc.returncode}")
                else:
                    # Print the summary table from the sweep log
                    try:
                        for line in sweep_log.read_text().splitlines():
                            if any(x in line for x in ["===", "vanilla", "trained", "wrote"]):
                                log(f"  {line.strip()}")
                    except Exception:
                        pass

    log("\n=== MTP PIPELINE COMPLETE ===")
    log(f"  trained GGUFs:       {args.workspace}/<slug>/IQ3_S-trained.gguf")
    log(f"  noisy trained GGUFs: {args.workspace}/<slug>/IQ3_S-trained-noisy.gguf")
    log(f"  acceptance plot:     {args.workspace}/mtp_acceptance.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
