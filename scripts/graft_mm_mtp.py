#!/usr/bin/env python3
"""Graft the vision tower and MTP draft head back onto a text-only W4A16 export.

Why this beats re-running PTQ
-----------------------------
Neither grafted component is ever quantized under ANY build path:

* the vision tower is explicitly protected by ``--ignore 're:.*visual.*'`` (audited:
  281 modules), so a multimodal rebuild would keep it bf16 — byte-identical to the
  source weights copied here;
* ``mtp.*`` cannot be quantized at all, because transformers declares
  ``_keys_to_ignore_on_load_unexpected = ['^mtp.*']`` and the tensors never become
  modules.

So the only thing a 3-hour rebuild would change is the text trunk — which is already
quantized, on the same corpus, over the same modules. This assembles the same parts.

What actually failed, and what this fixes
-----------------------------------------
Quantizing through the text-only ``Qwen3_5ForCausalLM`` made transformers write a FLAT
text config (``model_type: "qwen3_5_text"``). Both of vLLM's relevant features are gated
on config identity rather than on weights:

* MTP detection matches ``model_type in ("qwen3_5", "qwen3_5_moe")`` and rewrites it to
  ``qwen3_5_mtp``. ``qwen3_5_text`` matches neither, so an unknown method reaches the
  validator and the server refuses to start.
* the multimodal loader keys off the ``Qwen3_5ForConditionalGeneration`` architecture,
  which maps ``model.language_model.*`` -> ``language_model.model.*`` and
  ``model.visual.*`` -> ``visual.*`` — exactly the naming the HF-native export already
  has, which is why THIS variant needs no tensor rename.

So the config is restored from the source (multimodal shape) and our
``quantization_config`` is re-attached to it.

The ignore-list trap
--------------------
llmcompressor **expands regex ignores into concrete module names and drops the ones that
matched nothing**. The saved config therefore has 97 literal entries, zero ``re:``
patterns, and no mention of visual or mtp. With ``targets: ["Linear"]``, every Linear not
named in ``ignore`` is assumed quantized — so the grafted bf16 tensors would be read as
packed int4 and fail to load. This re-adds the regex patterns.

    PYTHONPATH=src .venv/bin/python scripts/graft_mm_mtp.py \\
        --source out/exp-060/model_extracted \\
        --quant  out/exp-060-w4a16-32k/checkpoint \\
        --out    out/exp-060-w4a16-32k/checkpoint-mm-graft
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from pathlib import Path

VISION_SHARD = "model-visual-bf16.safetensors"
MTP_SHARD = "model-mtp-bf16.safetensors"
EXTRA_IGNORE = ["re:.*visual.*", "re:.*mtp.*"]

# Processor assets the TEXT-ONLY export never had. Restoring the multimodal config makes
# vLLM build an image processor, and it hard-errors without preprocessor_config.json:
#   OSError: Can't load image processor for '<dir>'
# The tokenizer pair is carried too: some processor paths rebuild a slow tokenizer from
# vocab.json + merges.txt rather than tokenizer.json.
PROCESSOR_ASSETS = (
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
    "merges.txt",
)


def load_index(root: Path) -> dict:
    path = root / "model.safetensors.index.json"
    if path.is_file():
        return json.loads(path.read_text())
    shards = sorted(root.glob("*.safetensors"))
    if len(shards) != 1:
        raise FileNotFoundError(f"no index at {path} and {len(shards)} shards in {root}")
    shard = shards[0]
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return {
        "metadata": {"total_size": shard.stat().st_size},
        "weight_map": {k: shard.name for k in header if k != "__metadata__"},
    }


def collect(source: Path, predicate) -> dict:
    """Load matching tensors from the source, preserving dtype."""
    from safetensors.torch import load_file

    weight_map = load_index(source)["weight_map"]
    wanted = {k: v for k, v in weight_map.items() if predicate(k)}
    if not wanted:
        raise ValueError("no tensors matched — refusing to write an empty shard")
    by_shard: dict[str, list[str]] = {}
    for name, shard in wanted.items():
        by_shard.setdefault(shard, []).append(name)
    out = {}
    for shard, names in sorted(by_shard.items()):
        loaded = load_file(str(source / shard))
        for name in names:
            out[name] = loaded[name]
        del loaded
    return out


def build_config(source: Path, quant: Path) -> dict:
    """Source's multimodal config + our quantization_config, with regex ignores restored."""
    cfg = json.loads((source / "config.json").read_text())
    qcfg = json.loads((quant / "config.json").read_text())["quantization_config"]

    ignore = list(qcfg.get("ignore", []))
    for pat in EXTRA_IGNORE:
        if pat not in ignore:
            ignore.append(pat)
    qcfg["ignore"] = ignore

    cfg["quantization_config"] = qcfg
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=Path, help="bf16 multimodal checkpoint")
    ap.add_argument("--quant", required=True, type=Path, help="text-only W4A16 export (HF naming)")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    from safetensors.torch import save_file

    a.out.mkdir(parents=True, exist_ok=True)
    index = load_index(a.quant)
    weight_map = dict(index["weight_map"])

    # The quantized trunk must keep the HF-native model.language_model.* naming: that is
    # precisely what vLLM's multimodal mapper expects, and why this variant needs no rename.
    trunk = [k for k in weight_map if "language_model" in k]
    if not trunk:
        raise SystemExit(
            f"{a.quant} has no model.language_model.* tensors — this looks like the RENAMED "
            "serving variant. Graft from the HF-native export instead."
        )

    for item in sorted(a.quant.iterdir()):
        if item.is_dir():
            continue
        target = a.out / item.name
        if target.exists():
            target.unlink()
        try:
            os.link(item, target)          # hardlink: the 16 GiB trunk is not copied
        except OSError:
            shutil.copy2(item, target)

    # Copy (not hardlink) the processor assets: in the source these are symlinks into the
    # HF blob cache, and a published checkpoint must not depend on that cache existing.
    copied = []
    for name in PROCESSOR_ASSETS:
        src = a.source / name
        if not src.exists():
            continue
        dst = a.out / name
        if dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)          # follows the symlink, copies real bytes
        copied.append(name)
    print(f"  + processor assets: {copied or 'NONE FOUND'}")
    if "preprocessor_config.json" not in copied:
        raise SystemExit(
            "preprocessor_config.json not found in the source — vLLM will refuse to build "
            "an image processor for a multimodal config and the server will not start."
        )

    added = 0
    for shard, pred, label in (
        (VISION_SHARD, lambda k: "visual" in k, "vision tower"),
        (MTP_SHARD, lambda k: k.startswith("mtp."), "mtp head"),
    ):
        tensors = collect(a.source, pred)
        nbytes = sum(t.numel() * t.element_size() for t in tensors.values())
        save_file(tensors, str(a.out / shard), metadata={"format": "pt"})
        for name in tensors:
            weight_map[name] = shard
        added += nbytes
        dtypes = {str(t.dtype) for t in tensors.values()}
        print(f"  + {label:12s} {len(tensors):4d} tensors  {nbytes / 1024**3:.2f} GiB  {dtypes}")
        del tensors

    total = sum((a.out / f).stat().st_size for f in set(weight_map.values()))
    (a.out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2)
    )

    cfg = build_config(a.source, a.quant)
    (a.out / "config.json").write_text(json.dumps(cfg, indent=2))

    (a.out / "quant_tuner_graft.json").write_text(json.dumps({
        "source": str(a.source),
        "quant": str(a.quant),
        "grafted": {"visual": VISION_SHARD, "mtp": MTP_SHARD},
        "added_bytes": added,
        "config_from": "source (multimodal) + quant's quantization_config",
        "extra_ignore": EXTRA_IGNORE,
        "why": (
            "vision and mtp are bf16 under every build path, so grafting them is "
            "equivalent to re-running PTQ through the multimodal class — only the text "
            "trunk would differ, and it is already quantized on the same corpus."
        ),
    }, indent=2))

    print(f"\nwrote {a.out}")
    print(f"  model_type   : {cfg.get('model_type')}  (was qwen3_5_text)")
    print(f"  architectures: {cfg.get('architectures')}")
    print(f"  tensors      : {len(weight_map)}")
    print(f"  total size   : {total / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
