"""Finding the MTP / nextn draft layer in a GGUF, and pinning it near-lossless.

Every MTP release so far hardcoded ``{"blk.64.": "q8_0"}`` because Qwen3.6-27B happens to
have 64 trunk layers, so its single nextn layer lands at index 64. That constant is wrong
for any other model, and wrong silently: ``llama-quantize`` accepts a ``--tensor-type``
pattern that matches nothing, so a mistyped pin produces a GGUF whose draft head was
quantized to 2 bits along with the trunk. It still loads, still drafts, and only shows up
as a mediocre acceptance rate that looks like the head just being weak.

So: read the layer index out of the GGUF instead. Two signals, either of which is enough:

* ``blk.<i>.*`` with ``i >= <arch>.block_count`` — the converter appends draft blocks after
  the trunk.
* a tensor named ``nextn`` / ``mtp`` / ``eh_proj``.

**Both are needed in practice.** On the shipped Qwopus3.6-27B-Coder F16 the trunk is 64
layers and the draft head is ``blk.64``, but the GGUF declares ``block_count=65`` — the
converter counts the draft layer as a block. The index test alone finds nothing there; the
``nextn.eh_proj`` name is what identifies it. Conversely a converter that renames the head
generically is caught by the index test. Verified against that file: 15 tensors, ``blk.64``,
pin ``{"blk.64.": "q8_0"}`` — the same constant the hand-written exp-041/045 scripts used.
"""

from __future__ import annotations

import re
from pathlib import Path

_BLK = re.compile(r"(?:^|\.)blk\.(\d+)\.")
_NAME_HINTS = ("nextn", "mtp", "eh_proj")


def mtp_layer_indices(tensor_names: list[str], block_count: int | None) -> list[int]:
    """Indices of ``blk.<i>`` layers that are draft heads, not trunk layers.

    A layer qualifies if its index is past the declared trunk depth, or if any of its
    tensors is named like a draft head (``nextn``/``mtp``/``eh_proj``) — some converters
    emit the fusion projection under a hinted name while keeping the index in range.
    """
    found: set[int] = set()
    for name in tensor_names:
        m = _BLK.search(name)
        if not m:
            continue
        i = int(m.group(1))
        lower = name.lower()
        if (block_count is not None and i >= block_count) or any(
            h in lower for h in _NAME_HINTS
        ):
            found.add(i)
    return sorted(found)


def mtp_tensor_names(tensor_names: list[str], block_count: int | None) -> list[str]:
    """Every tensor belonging to a draft layer, plus un-blocked ``nextn``/``mtp`` tensors."""
    idx = set(mtp_layer_indices(tensor_names, block_count))
    out = []
    for name in tensor_names:
        m = _BLK.search(name)
        in_draft_layer = int(m.group(1)) in idx if m else False
        hinted_loose_tensor = not m and any(h in name.lower() for h in _NAME_HINTS)
        if in_draft_layer or hinted_loose_tensor:
            out.append(name)
    return out


def pin_map(layer_indices: list[int], quant: str = "q8_0") -> dict[str, str]:
    """``tensor_types`` pin for :func:`quant_tuner.quantize.gguf.quantize`.

    ``"blk.64."`` (with the trailing dot) is used rather than ``"blk.64"`` so the pattern
    cannot also swallow ``blk.640`` on a hypothetical deeper model.
    """
    return {f"blk.{i}.": quant for i in layer_indices}


def read_gguf_tensor_names(path: Path) -> tuple[list[str], int | None]:
    """``(tensor_names, block_count)`` for a GGUF, using the vendored gguf-py if needed."""
    try:
        from gguf import GGUFReader
    except ImportError:  # fall back to the llama.cpp submodule's copy
        from quant_tuner import paths

        paths.ensure_gguf_py()
        from gguf import GGUFReader

    reader = GGUFReader(str(path))
    names = [t.name for t in reader.tensors]
    block_count: int | None = None
    for field_name, field in reader.fields.items():
        if field_name.endswith(".block_count"):
            try:
                block_count = int(field.parts[field.data[0]][0])
            except Exception:  # noqa: BLE001 - kv layouts vary; fall back to name hints
                block_count = None
            break
    return names, block_count


def describe(path: Path, quant: str = "q8_0") -> dict:
    """Inspect a GGUF and return the pin plus what it matched. Empty pin ⇒ no MTP head."""
    names, block_count = read_gguf_tensor_names(Path(path))
    idx = mtp_layer_indices(names, block_count)
    tensors = mtp_tensor_names(names, block_count)
    return {
        "gguf": str(path),
        "block_count": block_count,
        "mtp_layer_indices": idx,
        "mtp_tensor_count": len(tensors),
        "mtp_tensors": tensors,
        "pin": pin_map(idx, quant),
        "has_mtp": bool(idx),
    }


def config_declares_mtp(config: dict) -> int:
    """Number of draft layers an HF ``config.json`` declares (0 = none).

    Note the exp-050 trap: Ornith-1.0-9B's config claimed ``mtp_num_hidden_layers=1`` while
    shipping no ``mtp.*`` weights, and extracting with ``keep_mtp=True`` produced a phantom
    block that crashed llama.cpp. A non-zero answer here is a claim to verify against the
    actual weights, not a fact.
    """
    for key in ("mtp_num_hidden_layers", "num_nextn_predict_layers",
                "nextn_predict_layers", "num_mtp_layers"):
        v = config.get(key)
        if isinstance(v, int) and v > 0:
            return v
    text_cfg = config.get("text_config")
    if isinstance(text_cfg, dict):
        return config_declares_mtp(text_cfg)
    return 0


__all__ = [
    "config_declares_mtp",
    "describe",
    "mtp_layer_indices",
    "mtp_tensor_names",
    "pin_map",
    "read_gguf_tensor_names",
]
