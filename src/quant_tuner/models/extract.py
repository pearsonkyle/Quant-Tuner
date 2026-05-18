"""Fetch a HF model locally and (for VLM source models) extract the text-only LM.

Generalized from the OmniCoder-9B extractor. For ordinary CausalLM models this
function just snapshot_downloads the repo and returns the local path.
"""

from __future__ import annotations

import gc
import json
import shutil
from glob import glob
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer


def _parse_size(s: str) -> int:
    s = s.strip().upper()
    if s.endswith("GB"):
        return int(float(s[:-2]) * 1e9)
    if s.endswith("MB"):
        return int(float(s[:-2]) * 1e6)
    return int(s)


def _build_text_config(vlm_config: dict) -> dict:
    text_cfg = dict(vlm_config.get("text_config", {}))
    text_cfg["architectures"] = vlm_config.get("architectures", text_cfg.get("architectures", []))
    text_cfg.setdefault("torch_dtype", vlm_config.get("torch_dtype", "bfloat16"))
    for field in ("pad_token_id", "bos_token_id", "eos_token_id"):
        if text_cfg.get(field) is None and vlm_config.get(field) is not None:
            text_cfg[field] = vlm_config[field]
    return {k: v for k, v in text_cfg.items() if v is not None}


def fetch_model(source: str, cache_dir: Path | None = None) -> Path:
    """Return a local path to `source` — either an existing dir, or snapshot_download'd."""
    src_path = Path(source)
    if src_path.is_dir():
        return src_path
    return Path(snapshot_download(source, cache_dir=str(cache_dir) if cache_dir else None))


def extract_text_lm(
    source: str,
    output_dir: Path,
    *,
    drop_prefixes: tuple[str, ...] = ("model.visual.", "visual.", "mtp."),
    rename_prefix: tuple[str, str] | None = ("model.language_model.", "model."),
    causal_lm_arch: str | None = None,
    max_shard_size: str = "4GB",
) -> Path:
    """Strip vision/MTP weights from a VLM and write a CausalLM checkpoint.

    For models that are already CausalLM, this is a no-op pass-through —
    the source directory is returned unchanged.
    """
    model_path = fetch_model(source)
    with open(model_path / "config.json") as f:
        config = json.load(f)
    arch = (config.get("architectures") or [""])[0]
    if "ForCausalLM" in arch and not config.get("text_config"):
        return model_path

    output_dir.mkdir(parents=True, exist_ok=True)
    new_config = _build_text_config(config)
    if causal_lm_arch:
        new_config["architectures"] = [causal_lm_arch]
    with open(output_dir / "config.json", "w") as f:
        json.dump(new_config, f, indent=2)

    AutoTokenizer.from_pretrained(model_path, trust_remote_code=True).save_pretrained(output_dir)

    shard_files = sorted(glob(str(model_path / "model*.safetensors"))) or sorted(
        glob(str(model_path / "*.safetensors"))
    )
    max_shard_bytes = _parse_size(max_shard_size)
    current: dict = {}
    cur_size = 0
    shard_idx = 1
    weight_map: dict[str, str] = {}
    total_bytes = 0

    def flush() -> None:
        nonlocal current, cur_size, shard_idx
        if not current:
            return
        name = f"model-{shard_idx:05d}-of-PLACEHOLDER.safetensors"
        save_file(current, str(output_dir / name))
        for k in current:
            weight_map[k] = name
        shard_idx += 1
        current = {}
        cur_size = 0
        gc.collect()

    for sf in shard_files:
        with safe_open(sf, framework="pt", device="cpu") as f:
            for key in f.keys():
                if any(key.startswith(p) for p in drop_prefixes):
                    continue
                tensor = f.get_tensor(key)
                new_key = key
                if rename_prefix and new_key.startswith(rename_prefix[0]):
                    new_key = rename_prefix[1] + new_key[len(rename_prefix[0]) :]
                t_size = tensor.numel() * tensor.element_size()
                if current and cur_size + t_size > max_shard_bytes:
                    flush()
                current[new_key] = tensor
                cur_size += t_size
                total_bytes += t_size
    flush()

    n_shards = shard_idx - 1
    for i in range(1, n_shards + 1):
        old = output_dir / f"model-{i:05d}-of-PLACEHOLDER.safetensors"
        new = output_dir / f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        if old.exists():
            old.rename(new)
        for k in list(weight_map):
            if weight_map[k] == old.name:
                weight_map[k] = new.name

    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, f, indent=2)

    gen_cfg = model_path / "generation_config.json"
    if gen_cfg.exists():
        shutil.copy2(gen_cfg, output_dir / "generation_config.json")

    return output_dir
