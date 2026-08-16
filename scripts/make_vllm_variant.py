#!/usr/bin/env python3
"""Rename an HF-native Qwen3.5 text export into the layout vLLM expects.

The divergence
--------------
transformers' ``Qwen3_5ForCausalLM`` *saves* its decoder nested under
``model.language_model.*`` (it keeps the multimodal wrapper's naming even in the
text-only class). vLLM's ``Qwen3_5ForCausalLM`` builds ``self.model =
Qwen3_5Model(...)`` and its ``hf_to_vllm_mapper`` has **no** ``language_model.``
prefix rule, so it looks for ``model.layers.*`` and dies with::

    ValueError: There is no module or parameter named 'language_model'
                in Qwen3_5Model

Only the multimodal ``Qwen3_5ForConditionalGeneration`` path knows the nested
form, and that path also demands vision weights a text-only export does not
have. So the fix is a key rename, not a config edit.

``quantization_config.ignore`` must be rewritten too: vLLM matches those
patterns against *its* layer names, so a stale ``model.language_model.…`` entry
silently stops matching and the layer is no longer excluded.

The HF-native export is left untouched — ``bench/kld_hf.py`` loads it through
transformers and needs the original naming. Keep both.

    PYTHONPATH=src .venv/bin/python scripts/make_vllm_variant.py \
        --src out/exp-060-w4a16-32k/checkpoint \
        --out out/exp-060-w4a16-32k/checkpoint-vllm
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

OLD_PREFIX = "model.language_model."
NEW_PREFIX = "model."


def rename_key(key: str) -> str:
    return NEW_PREFIX + key[len(OLD_PREFIX) :] if key.startswith(OLD_PREFIX) else key


def build(src: Path, out: Path) -> Path:
    from safetensors.torch import load_file, save_file

    out.mkdir(parents=True, exist_ok=True)

    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors in {src}")

    renamed_total = 0
    for shard in shards:
        tensors = load_file(str(shard))
        new = {}
        for k, v in tensors.items():
            nk = rename_key(k)
            renamed_total += nk != k
            new[nk] = v
        save_file(new, str(out / shard.name), metadata={"format": "pt"})
        del tensors, new
        print(f"  wrote {shard.name}", flush=True)

    # Non-weight files copy across; config.json needs its ignore list rewritten.
    for item in sorted(src.iterdir()):
        if item.suffix == ".safetensors" or item.is_dir():
            continue
        if item.name == "model.safetensors.index.json":
            idx = json.loads(item.read_text())
            idx["weight_map"] = {rename_key(k): v for k, v in idx["weight_map"].items()}
            (out / item.name).write_text(json.dumps(idx, indent=2))
            continue
        shutil.copy2(item, out / item.name)

    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text())
    q = cfg.get("quantization_config", {})
    if "ignore" in q:
        before = list(q["ignore"])
        q["ignore"] = [rename_key(x) for x in before]
        changed = sum(a != b for a, b in zip(before, q["ignore"], strict=True))
        print(f"  rewrote {changed}/{len(before)} quantization_config.ignore entries")
    cfg_path.write_text(json.dumps(cfg, indent=2))

    (out / "quant_tuner_vllm_rename.json").write_text(
        json.dumps(
            {
                "source": str(src),
                "rule": f"{OLD_PREFIX} -> {NEW_PREFIX}",
                "tensors_renamed": renamed_total,
                "why": (
                    "vLLM's text-only Qwen3_5ForCausalLM expects model.layers.*; "
                    "transformers saves model.language_model.layers.*. The HF "
                    "export is kept as-is for transformers-side eval."
                ),
            },
            indent=2,
        )
    )
    print(f"renamed {renamed_total} tensors -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    build(args.src, args.out)


if __name__ == "__main__":
    main()
