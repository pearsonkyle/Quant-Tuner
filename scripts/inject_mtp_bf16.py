#!/usr/bin/env python3
"""Re-inject a checkpoint's MTP draft head (bf16) into a quantized export.

Why this is needed
------------------
transformers 5.12.1 declares ``_keys_to_ignore_on_load_unexpected = ['^mtp.*']``
on **every** ``qwen3_5`` class, so ``from_pretrained`` discards the 15 ``mtp.*``
tensors before llmcompressor ever sees them. They are therefore absent from the
W4A16 export — and no ``ignore`` pattern can prevent that, because ``ignore``
operates on *modules* and the draft head never becomes one. The only way to keep
it is to copy the raw tensors back afterwards.

They are kept at **bf16** (0.85 GB). Quantizing them to int8 would save ~0.4 GB
on a ~14 GB checkpoint while requiring hand-authored compressed-tensors scales
that vLLM's loader may not accept — a bad trade. bf16 is also strictly better
than the GGUF ladder's Q8_0 pin on the same head.

The variant directory **hardlinks** the unchanged shards rather than copying
them, so it costs ~0.85 GB on disk, not another full checkpoint.

    PYTHONPATH=src .venv/bin/python scripts/inject_mtp_bf16.py \
        --source out/exp-060/model_extracted \
        --quant  out/exp-060-w4a16-32k/checkpoint \
        --out    out/exp-060-w4a16-32k/checkpoint-mtp
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from pathlib import Path

MTP_SHARD = "model-mtp-bf16.safetensors"


def _load_index(root: Path) -> dict:
    """Read the shard index, synthesizing one for a single-shard checkpoint.

    llmcompressor writes one ``model.safetensors`` with no index when the
    export fits in a single file. Adding a second shard then requires an index
    to exist, so build it from the lone file's own header.
    """
    path = root / "model.safetensors.index.json"
    if path.is_file():
        return json.loads(path.read_text())

    shards = sorted(p for p in root.glob("*.safetensors"))
    if len(shards) != 1:
        raise FileNotFoundError(
            f"no index at {path} and {len(shards)} shards in {root} — cannot "
            "infer a weight map"
        )
    shard = shards[0]
    with open(shard, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    weight_map = {k: shard.name for k in header if k != "__metadata__"}
    return {
        "metadata": {"total_size": shard.stat().st_size},
        "weight_map": weight_map,
    }


def collect_mtp_tensors(source: Path, prefix: str = "mtp.") -> dict:
    """Load every ``prefix`` tensor from the source checkpoint, preserving dtype."""
    from safetensors.torch import load_file

    weight_map = _load_index(source)["weight_map"]
    wanted = {k: v for k, v in weight_map.items() if k.startswith(prefix)}
    if not wanted:
        raise ValueError(f"no tensors with prefix {prefix!r} in {source}")

    by_shard: dict[str, list[str]] = {}
    for name, shard in wanted.items():
        by_shard.setdefault(shard, []).append(name)

    tensors = {}
    for shard, names in by_shard.items():
        loaded = load_file(str(source / shard))
        for name in names:
            tensors[name] = loaded[name]
        del loaded
    return tensors


def build_variant(source: Path, quant: Path, out: Path, prefix: str = "mtp.") -> Path:
    """Create ``out`` as ``quant`` + the source's ``prefix`` tensors at bf16."""
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    index = _load_index(quant)
    weight_map = dict(index["weight_map"])

    # Hardlink every existing file across; fall back to a copy across devices.
    for item in sorted(quant.iterdir()):
        if item.is_dir():
            continue
        target = out / item.name
        if target.exists():
            target.unlink()
        try:
            os.link(item, target)
        except OSError:
            shutil.copy2(item, target)

    tensors = collect_mtp_tensors(source, prefix)
    added_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    save_file(tensors, str(out / MTP_SHARD), metadata={"format": "pt"})

    for name in tensors:
        weight_map[name] = MTP_SHARD
    index["weight_map"] = weight_map
    index.setdefault("metadata", {})
    index["metadata"]["total_size"] = (
        int(index["metadata"].get("total_size", 0)) + added_bytes
    )
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    provenance = out / "quant_tuner_mtp_injection.json"
    provenance.write_text(
        json.dumps(
            {
                "source": str(source),
                "quant": str(quant),
                "prefix": prefix,
                "n_tensors": len(tensors),
                "added_bytes": added_bytes,
                "dtype": str(next(iter(tensors.values())).dtype),
                "shard": MTP_SHARD,
                "note": (
                    "transformers drops mtp.* on load "
                    "(_keys_to_ignore_on_load_unexpected); these tensors are "
                    "carried over verbatim from the bf16 source, NOT quantized. "
                    "Whether vLLM can use them for speculative decoding is a "
                    "separate question from whether they are present."
                ),
            },
            indent=2,
        )
    )
    print(f"injected {len(tensors)} {prefix}* tensors ({added_bytes / 1e9:.2f} GB) -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, type=Path, help="bf16 source checkpoint")
    ap.add_argument("--quant", required=True, type=Path, help="quantized export")
    ap.add_argument("--out", required=True, type=Path, help="variant dir to create")
    ap.add_argument("--prefix", default="mtp.", help="tensor prefix to carry over")
    args = ap.parse_args()
    build_variant(args.source, args.quant, args.out, args.prefix)


if __name__ == "__main__":
    main()
