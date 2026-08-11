"""Train MTP heads against real IQ3_S hidden states, with a wiki+toolcall mix.

For each model slug:
  1. Capture IQ3_S `h_pre_norm` activations to a memory-mapped cache,
     using `wiki_mix` of general text alongside the tool-call train split.
  2. Train the MTP head with --iq3s-cache + --wiki-mix (forces quant-noise=0;
     the real IQ3_S noise is baked into the cache).
  3. Copy the trained mtp.* tensors into model_extracted under a dedicated
     shard, reconvert HF → F16 GGUF, and requantize to IQ3_S using the
     existing imatrix-hybrid_custom.gguf, producing IQ3_S-trained-iq3s.gguf.

After this script finishes, run scripts/run_mtp_eval_suite.py with the
new `trained-iq3s` variant to fill in the comparison tables in
EXPERIMENT.md.

Usage::

    uv run python scripts/run_mtp_iq3s_pipeline.py --logs logtrain.jsonl
    uv run python scripts/run_mtp_iq3s_pipeline.py --models tesslate --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.experiments import log, phase  # noqa: E402
from quant_tuner.quantize import convert, gguf  # noqa: E402

WORKSPACE = REPO / "out/benchmark_9b_iq3s"
TRAINING_STEPS = 293     # 2 epochs over the wiki-mixed chunk count @ grad_accum=4
TRAINING_LR    = 2e-5
GRAD_ACCUM     = 4
WIKI_MIX       = 0.3
SEQ_LEN        = 512
MAX_TRAIN_TOKENS = 300_000
SEED           = 42
MAX_CHUNKS     = TRAINING_STEPS * GRAD_ACCUM   # chunks consumed by training
ALL_MODELS = ("tesslate", "jackrong", "qwen")


def _stream_subprocess(cmd: list[str], log_path: Path, label: str) -> None:
    """Run a subprocess, tee its output to log_path, and tail progress lines."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"  [{label}] → {log_path}")
    with open(log_path, "a") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf)
    while proc.poll() is None:
        time.sleep(60)
        try:
            lines = log_path.read_text().splitlines()
            recent = [ln for ln in lines[-20:]
                      if any(k in ln for k in ("step", "PPL", "top-1", "chunk", "DONE"))]
            if recent:
                log(f"    {recent[-1].strip()}")
        except OSError:
            pass
    if proc.returncode != 0:
        raise RuntimeError(f"[{label}] subprocess exited rc={proc.returncode} — see {log_path}")


# ---------------------------------------------------------------------------
# Stage 1: capture
# ---------------------------------------------------------------------------

def _capture(slug: str, logs: Path, dry_run: bool) -> Path:
    model_dir   = WORKSPACE / slug / "model_extracted"
    gguf_path   = WORKSPACE / slug / "IQ3_S-custom.gguf"
    cache_path  = WORKSPACE / slug / "iq3s_hidden_states_wiki.npy"
    done_flag   = cache_path.with_suffix(cache_path.suffix + ".done")

    if done_flag.exists() and cache_path.exists():
        log(f"  [{slug}] capture already done — skipping")
        return cache_path

    if not gguf_path.exists():
        raise FileNotFoundError(f"IQ3_S GGUF not found: {gguf_path}")

    cmd = [
        "uv", "run", "python", str(REPO / "scripts/capture_iq3s_hidden_states.py"),
        "--model-slug",       slug,
        "--gguf",             str(gguf_path),
        "--logs",             str(logs),
        "--workspace",        str(model_dir),
        "--out",              str(cache_path),
        "--seq-len",          str(SEQ_LEN),
        "--max-train-tokens", str(MAX_TRAIN_TOKENS),
        "--seed",             str(SEED),
        "--wiki-mix",         str(WIKI_MIX),
        "--max-chunks",       str(MAX_CHUNKS),
    ]
    if dry_run:
        log(f"  [dry-run] {' '.join(cmd)}")
        return cache_path

    _stream_subprocess(cmd, WORKSPACE / slug / "logs" / "capture_iq3s.log", f"{slug} capture")
    return cache_path


# ---------------------------------------------------------------------------
# Stage 2: train against the cache
# ---------------------------------------------------------------------------

def _train(slug: str, cache_path: Path, logs: Path, dry_run: bool) -> Path:
    model_dir = WORKSPACE / slug / "model_extracted"
    out_dir   = WORKSPACE / slug / "mtp_trained_iq3s"
    weights   = out_dir / "model-mtp-trained.safetensors"

    if weights.exists():
        log(f"  [{slug}] trained-iq3s weights already exist — skipping")
        return weights

    cmd = [
        "uv", "run", "python", str(REPO / "scripts/train_mtp_head.py"),
        "--model",        str(model_dir),
        "--logs",         str(logs),
        "--out",          str(out_dir),
        "--steps",        str(TRAINING_STEPS),
        "--lr",           str(TRAINING_LR),
        "--seq-len",      str(SEQ_LEN),
        "--max-train-tokens", str(MAX_TRAIN_TOKENS),
        "--seed",         str(SEED),
        "--wiki-mix",     str(WIKI_MIX),
        "--iq3s-cache",   str(cache_path),
    ]
    if dry_run:
        log(f"  [dry-run] {' '.join(cmd)}")
        return weights

    _stream_subprocess(cmd, WORKSPACE / slug / "logs" / "train_mtp_iq3s.log", f"{slug} train")
    return weights


# ---------------------------------------------------------------------------
# Stage 3: install + rebuild GGUF
# ---------------------------------------------------------------------------

def _install(slug: str) -> None:
    model_dir   = WORKSPACE / slug / "model_extracted"
    trained_src = WORKSPACE / slug / "mtp_trained_iq3s" / "model-mtp-trained.safetensors"
    if not trained_src.exists():
        raise FileNotFoundError(f"trained-iq3s weights missing: {trained_src}")

    dest_shard = "model-mtp-trained-iq3s.safetensors"
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
    log(f"  pointed {updated} mtp.* entries → {dest_shard}")


def _rebuild_gguf(slug: str, dry_run: bool) -> Path:
    model_dir = WORKSPACE / slug / "model_extracted"
    f16       = WORKSPACE / slug / "model-f16.gguf"
    imatrix   = WORKSPACE / slug / "imatrix-hybrid_custom.gguf"
    out_quant = WORKSPACE / slug / "IQ3_S-trained-iq3s.gguf"
    log_dir   = WORKSPACE / slug / "logs"

    if out_quant.exists():
        log(f"  [{slug}] {out_quant.name} already exists — skipping rebuild")
        return out_quant
    if not imatrix.exists():
        raise FileNotFoundError(f"imatrix not found: {imatrix}")

    if dry_run:
        log(f"  [dry-run] would convert + quantize → {out_quant.name}")
        return out_quant

    log_dir.mkdir(parents=True, exist_ok=True)
    if f16.exists():
        f16.unlink()
        log(f"  [{slug}] deleted stale model-f16.gguf")

    with phase(f"[{slug}] reconvert HF → F16 GGUF (iq3s)"):
        convert.hf_to_f16_gguf(model_dir, f16, log=log_dir / "convert_iq3s.log")
    with phase(f"[{slug}] quantize IQ3_S-trained-iq3s"):
        gguf.quantize(f16, out_quant, "IQ3_S",
                      imatrix=imatrix, log=log_dir / "quantize_iq3s.log")
    log(f"  [{slug}] built {out_quant.name} ({out_quant.stat().st_size / 1e9:.2f} GB)")
    return out_quant


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", type=Path, default=REPO / "logtrain.jsonl")
    p.add_argument("--models", type=str, default=",".join(ALL_MODELS),
                   help="comma-separated subset of " + ",".join(ALL_MODELS))
    p.add_argument("--skip-rebuild", action="store_true",
                   help="capture + train only; skip GGUF reconvert+requant")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in ALL_MODELS:
            log(f"ERROR: unknown model slug '{m}'  (expected one of {ALL_MODELS})")
            return 1

    log("=== MTP IQ3_S-cache pipeline ===")
    log(f"models:          {models}")
    log(f"logs:            {args.logs}")
    log(f"wiki_mix:        {WIKI_MIX}")
    log(f"steps/model:     {TRAINING_STEPS}")
    log(f"workspace:       {WORKSPACE}")

    for slug in models:
        with phase(f"[{slug}] capture IQ3_S hidden states"):
            cache = _capture(slug, args.logs, args.dry_run)
        with phase(f"[{slug}] train MTP head (iq3s-cache, wiki={WIKI_MIX})"):
            _train(slug, cache, args.logs, args.dry_run)
        if args.skip_rebuild:
            continue
        if not args.dry_run:
            _install(slug)
        _rebuild_gguf(slug, args.dry_run)

    log("=== DONE ===")
    log("Next:  uv run python scripts/run_mtp_eval_suite.py --holdout artifacts/toolcall_holdout.jsonl "
        "--mmlu-holdout out/mmlu_pro/holdout.json --variants trained-iq3s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
