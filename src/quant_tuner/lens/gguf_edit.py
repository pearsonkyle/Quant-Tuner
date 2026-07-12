"""Copy a GGUF, replacing named tensors — the write half of "bake it back".

``copy_gguf_with_tensor_edits`` clones a source GGUF verbatim (all kv metadata,
all tensors) except for the tensors named in ``edits``, which are written from
the supplied numpy arrays. Untouched tensors pass through with their original
raw bytes and dtype — no dequantize/requantize round-trip, so an unedited quant
copy is byte-for-byte identical (unit-tested). Edits are only ever applied to an
FP16/F32 source in the bake pipeline; the edited file is then requantized fresh
by ``llama-quantize`` rather than block-edited in place (which would be lossy).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def copy_gguf_with_tensor_edits(
    src: Path,
    dst: Path,
    edits: dict[str, np.ndarray],
    *,
    edit_dtype=np.float32,
) -> Path:
    """Write ``dst`` = ``src`` with ``edits`` substituted for matching tensors."""
    from quant_tuner.paths import ensure_gguf_py

    ensure_gguf_py()
    from gguf import GGUFReader, GGUFWriter

    src, dst = Path(src), Path(dst)
    reader = GGUFReader(str(src))
    arch = _kv(reader, "general.architecture")
    writer = GGUFWriter(str(dst), arch=str(arch))

    _copy_kv(reader, writer)

    edited = set(edits)
    seen = set()
    for tensor in reader.tensors:
        name = tensor.name
        if name in edits:
            arr = np.ascontiguousarray(edits[name], dtype=edit_dtype)
            if tuple(arr.shape) != tuple(int(x) for x in reversed(tensor.shape)) and \
               tuple(arr.shape) != tuple(int(x) for x in tensor.shape):
                logger.warning(
                    "edit for %s has shape %s but source tensor is %s",
                    name, arr.shape, tuple(int(x) for x in tensor.shape),
                )
            writer.add_tensor(name, arr)
            seen.add(name)
        else:
            # pass raw bytes through with the original quant type (no re-encode)
            writer.add_tensor(
                name,
                np.asarray(tensor.data),
                raw_shape=[int(x) for x in tensor.shape][::-1],
                raw_dtype=tensor.tensor_type,
            )
    missing = edited - seen
    if missing:
        raise KeyError(f"edit targets not found in {src}: {sorted(missing)}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    logger.info("wrote %s (%d tensors, %d edited)", dst, len(reader.tensors), len(edits))
    return dst


def load_gguf_tensor(path: Path, name: str) -> np.ndarray:
    """Load a single tensor from a GGUF as float32 (dequantizing if needed)."""
    from quant_tuner.paths import ensure_gguf_py

    ensure_gguf_py()
    import gguf
    from gguf import GGUFReader
    from gguf.quants import dequantize

    reader = GGUFReader(str(path))
    for t in reader.tensors:
        if t.name == name:
            if t.tensor_type == gguf.GGMLQuantizationType.F32:
                return np.asarray(t.data, dtype=np.float32)
            if t.tensor_type in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16):
                return np.asarray(t.data).astype(np.float32)
            return dequantize(t.data, t.tensor_type).astype(np.float32)
    raise KeyError(f"{name} not in {path}")


def list_gguf_tensor_names(path: Path) -> list[str]:
    from quant_tuner.paths import ensure_gguf_py

    ensure_gguf_py()
    from gguf import GGUFReader

    return [t.name for t in GGUFReader(str(path)).tensors]


def _kv(reader, key: str, default=None):
    f = reader.fields.get(key)
    if f is None:
        return default
    return f.contents()


def _copy_kv(reader, writer) -> None:
    """Copy every kv field except the ones GGUFWriter manages itself."""
    from gguf import GGUFValueType

    managed = {
        "general.architecture",
        "GGUF.version",
        "GGUF.tensor_count",
        "GGUF.kv_count",
    }
    for key, field in reader.fields.items():
        if key in managed:
            continue
        try:
            value = field.contents()
        except Exception:
            logger.debug("skipping unreadable kv %s", key)
            continue
        if value is None:
            continue
        vtype = field.types[0] if field.types else None
        try:
            if vtype == GGUFValueType.ARRAY:
                writer.add_array(key, list(value))
            elif vtype == GGUFValueType.STRING:
                writer.add_string(key, str(value))
            elif vtype == GGUFValueType.BOOL:
                writer.add_bool(key, bool(value))
            elif vtype in (GGUFValueType.FLOAT32, GGUFValueType.FLOAT64):
                writer.add_float32(key, float(value))
            elif vtype is not None:
                _add_int(writer, key, int(value))
        except Exception as e:
            logger.debug("could not copy kv %s (%s): %s", key, vtype, e)


def _add_int(writer, key: str, value: int) -> None:
    if value > 2**31:
        writer.add_int64(key, value)
    elif value >= 0:
        writer.add_uint32(key, value)
    else:
        writer.add_int32(key, value)
