"""Capture IQ3_S GGUF h_pre_norm hidden states for MTP-head fine-tuning.

Walks the same training chunks as scripts/train_mtp_head.py through an IQ3_S
GGUF (via the vendored llama.cpp `llama-mtp-capture` binary) and dumps the
post-last-transformer-layer / pre-output-norm hidden state to a
memory-mapped .npy file. The trainer reads from this cache instead of
running an FP16 HF forward pass.

This script is torch-free on purpose: llama.cpp and PyTorch/MPS fight over
the Apple Metal GPU, so capture must run in its own process.

The actual model load + decode happens in the vendored C++ tool
(vendor/llama.cpp/tools/mtp-capture/). This Python script handles
tokenization, parameter parity with training, and stitching the C++
tool's raw output into a numpy .npy file.

Usage::

    uv run python scripts/capture_iq3s_hidden_states.py \\
        --model-slug tesslate \\
        --gguf out/benchmark_9b_iq3s/tesslate/IQ3_S-custom.gguf \\
        --logs logtrain.jsonl \\
        --workspace out/benchmark_9b_iq3s/tesslate/model_extracted \\
        --out out/benchmark_9b_iq3s/tesslate/iq3s_hidden_states.npy
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_tuner.paths import llama_bin  # noqa: E402


def _log(msg: str) -> None:
    print(msg, flush=True)


def _read_gguf_n_embd(gguf_path: Path) -> int:
    """Parse the GGUF header to extract the embedding length without loading the model.

    Supports the subset of GGUF v2/v3 we need: scan key/value metadata until we hit
    `<arch>.embedding_length` (uint32 or uint64).
    """
    with gguf_path.open("rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise RuntimeError(f"not a GGUF file: {gguf_path}")
        version = struct.unpack("<I", f.read(4))[0]
        if version not in (2, 3):
            raise RuntimeError(f"unsupported GGUF version {version}")
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        def _read_str() -> str:
            ln = struct.unpack("<Q", f.read(8))[0]
            return f.read(ln).decode("utf-8")

        def _skip_value(vtype: int) -> object:
            if vtype == 0:  # UINT8
                return f.read(1)
            if vtype == 1:  # INT8
                return f.read(1)
            if vtype == 2:  # UINT16
                return f.read(2)
            if vtype == 3:  # INT16
                return f.read(2)
            if vtype == 4:  # UINT32
                return struct.unpack("<I", f.read(4))[0]
            if vtype == 5:  # INT32
                return struct.unpack("<i", f.read(4))[0]
            if vtype == 6:  # FLOAT32
                f.read(4); return None
            if vtype == 7:  # BOOL
                f.read(1); return None
            if vtype == 8:  # STRING
                _read_str(); return None
            if vtype == 9:  # ARRAY
                inner = struct.unpack("<I", f.read(4))[0]
                count = struct.unpack("<Q", f.read(8))[0]
                for _ in range(count):
                    _skip_value(inner)
                return None
            if vtype == 10:  # UINT64
                return struct.unpack("<Q", f.read(8))[0]
            if vtype == 11:  # INT64
                return struct.unpack("<q", f.read(8))[0]
            if vtype == 12:  # FLOAT64
                f.read(8); return None
            raise RuntimeError(f"unknown GGUF value type {vtype}")

        for _ in range(n_kv):
            key = _read_str()
            vtype = struct.unpack("<I", f.read(4))[0]
            val = _skip_value(vtype)
            if key.endswith(".embedding_length") and isinstance(val, int):
                return val
    raise RuntimeError(f"could not find <arch>.embedding_length in {gguf_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-slug", type=str, required=True,
                   help="Model identifier for logging only (e.g. tesslate)")
    p.add_argument("--gguf", type=Path, required=True,
                   help="Path to the IQ3_S GGUF to evaluate")
    p.add_argument("--logs", type=Path, default=REPO / "logtrain.jsonl",
                   help="Usage-log JSONL (tokenized into training chunks)")
    p.add_argument("--workspace", type=Path, required=True,
                   help="Path to model_extracted (provides the tokenizer)")
    p.add_argument("--out", type=Path, required=True,
                   help="Output .npy path (float32, [n_chunks, seq_len, n_embd])")
    p.add_argument("--seq-len", type=int, default=512,
                   help="Sequence length per chunk (must match training)")
    p.add_argument("--max-train-tokens", type=int, default=300_000,
                   help="Max training tokens (must match training)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for chunk shuffling (must match training)")
    p.add_argument("--wiki-mix", type=float, default=0.0,
                   help="Wiki-mix fraction (must match training)")
    p.add_argument("--wiki-text", type=Path, default=None)
    p.add_argument("--n-gpu-layers", type=int, default=-1,
                   help="Layers to offload to Metal/CUDA. -1 = all.")
    p.add_argument("--max-chunks", type=int, default=0,
                   help="Cap the number of chunks to capture (0 = all). "
                        "Training only consumes steps*grad_accum chunks; passing "
                        "that value here avoids spending GPU time on chunks the "
                        "trainer will never read.")
    p.add_argument("--keep-tokens-bin", action="store_true",
                   help="Keep the intermediate tokens.bin file (default: delete on success).")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    done_flag = args.out.with_suffix(args.out.suffix + ".done")
    ids_path  = args.out.with_name(args.out.stem + "_chunk_ids.npy")
    tokens_bin = args.out.with_name(args.out.stem + "_tokens.bin")

    capture_bin = llama_bin("llama-mtp-capture")

    _log(f"[{args.model_slug}] capture IQ3_S hidden states")
    _log(f"  gguf:        {args.gguf}")
    _log(f"  logs:        {args.logs}")
    _log(f"  workspace:   {args.workspace}")
    _log(f"  out:         {args.out}")
    _log(f"  ids out:     {ids_path}")
    _log(f"  seq_len:     {args.seq_len}")
    _log(f"  max_tokens:  {args.max_train_tokens}")
    _log(f"  seed:        {args.seed}")
    _log(f"  wiki_mix:    {args.wiki_mix}")
    _log(f"  capture:     {capture_bin}")

    if done_flag.exists() and args.out.exists() and ids_path.exists():
        _log(f"  skip-if-exists: {done_flag.name} present, exiting 0")
        return 0

    for path in (args.gguf, args.workspace, args.logs):
        if not path.exists():
            _log(f"ERROR: required input missing: {path}")
            return 1
    if not capture_bin.exists():
        _log(f"ERROR: llama-mtp-capture binary not built: {capture_bin}")
        _log("       run: cmake --build vendor/llama.cpp/build --target llama-mtp-capture -j")
        return 1

    n_embd = _read_gguf_n_embd(args.gguf)
    _log(f"  n_embd:      {n_embd} (read from GGUF metadata)")

    # --- Tokenize chunks ---
    from quant_tuner.mtp.chunks import build_token_batches
    _log("tokenising chunks …")
    chunks = build_token_batches(
        args.workspace,
        args.logs,
        seq_len=args.seq_len,
        max_tokens=args.max_train_tokens,
        seed=args.seed,
        wiki_mix=args.wiki_mix,
        wiki_text=args.wiki_text,
    )
    if not chunks:
        _log("ERROR: no chunks produced — check --logs / --workspace")
        return 1
    if args.max_chunks > 0 and len(chunks) > args.max_chunks:
        _log(f"  truncating chunks: {len(chunks)} → {args.max_chunks} (--max-chunks)")
        chunks = chunks[: args.max_chunks]
    n_chunks = len(chunks)
    expected_chunk_len = args.seq_len + 2
    if any(len(c) != expected_chunk_len for c in chunks):
        _log(f"ERROR: chunk length mismatch (expected {expected_chunk_len})")
        return 1
    bytes_total = n_chunks * args.seq_len * n_embd * 4
    _log(f"  {n_chunks} chunks of length {expected_chunk_len}")
    _log(f"  cache size: ({n_chunks}, {args.seq_len}, {n_embd}) float32 ≈ {bytes_total / 1e9:.2f} GB")

    if args.dry_run:
        _log("dry-run — exiting")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # --- Persist chunk_ids.npy (full seq_len+2 token IDs per chunk) ---
    chunk_ids = np.asarray(chunks, dtype=np.int32)
    np.save(str(ids_path), chunk_ids)
    _log(f"  wrote {ids_path}")

    # --- Pre-create the .npy file with the right header, capture data offset ---
    h_arr = np.lib.format.open_memmap(
        str(args.out), mode="w+", dtype=np.float32,
        shape=(n_chunks, args.seq_len, n_embd),
    )
    out_offset = int(h_arr.offset)
    del h_arr  # close, leaves file with reserved space

    # --- Write tokens.bin (input slices only, length seq_len each) ---
    inputs = chunk_ids[:, : args.seq_len].astype(np.int32, copy=False)
    inputs.tofile(str(tokens_bin))
    _log(f"  wrote {tokens_bin}  ({inputs.size * 4 / 1e6:.1f} MB)")

    # --- Run the C++ capture tool ---
    cmd = [
        str(capture_bin),
        "-m", str(args.gguf),
        "--tokens", str(tokens_bin),
        "--out", str(args.out),
        "--out-offset", str(out_offset),
        "--seq-len", str(args.seq_len),
        "--n-chunks", str(n_chunks),
        "--expected-n-embd", str(n_embd),
        "-ngl", str(args.n_gpu_layers),
    ]
    _log("running: " + " ".join(cmd))
    t0 = time.time()
    rc = subprocess.call(cmd)
    if rc != 0:
        _log(f"ERROR: llama-mtp-capture exited rc={rc}")
        return rc
    elapsed = time.time() - t0
    _log(f"  capture done in {elapsed/60:.1f} min")

    # --- Sanity stats on the first chunk ---
    peek = np.load(str(args.out), mmap_mode="r")
    if peek.shape != (n_chunks, args.seq_len, n_embd):
        _log(f"ERROR: output shape mismatch: {peek.shape} vs ({n_chunks}, {args.seq_len}, {n_embd})")
        return 1
    chunk0 = np.asarray(peek[0], dtype=np.float32)
    _log(f"  chunk[0] mean={chunk0.mean():.4f} std={chunk0.std():.4f} "
         f"min={chunk0.min():.2f} max={chunk0.max():.2f}")
    del peek

    if not args.keep_tokens_bin:
        try:
            tokens_bin.unlink()
        except OSError:
            pass

    done_flag.write_text(json.dumps({
        "model_slug": args.model_slug,
        "gguf": str(args.gguf),
        "n_chunks": n_chunks,
        "seq_len": args.seq_len,
        "n_embd": n_embd,
        "seed": args.seed,
        "max_train_tokens": args.max_train_tokens,
        "wiki_mix": args.wiki_mix,
    }, indent=2))
    _log(f"=== DONE  →  {args.out}  +  {ids_path}  +  {done_flag.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
