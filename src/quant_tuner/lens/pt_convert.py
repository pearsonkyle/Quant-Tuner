# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
# Adapted from igorbarshteyn/jlens-gguf (Apache-2.0) for quant-tuner; see NOTICE.
"""Convert reference PyTorch lenses (.pt) to GGUF — without needing torch.

The reference implementation saves lenses with ``torch.save`` as::

    {"J": {layer: Tensor[d, d]}, "n_prompts": int,
     "source_layers": [...], "d_model": int}

``torch.save`` writes a zip archive: a pickle stream (``data.pkl``) plus one
raw little-endian buffer per storage (``data/{key}``). Loading it back needs
only the pickle protocol and numpy — we stub the handful of torch classes the
stream references. If torch *is* installed we simply use it.
"""

from __future__ import annotations

import io
import pickle
import zipfile
from dataclasses import dataclass

import numpy as np

from quant_tuner.lens.lens import JacobianLensGGUF

_DTYPES = {
    "FloatStorage": np.dtype("<f4"),
    "DoubleStorage": np.dtype("<f8"),
    "HalfStorage": np.dtype("<f2"),
    "BFloat16Storage": np.dtype("<u2"),  # decoded below
    "IntStorage": np.dtype("<i4"),
    "LongStorage": np.dtype("<i8"),
    "ShortStorage": np.dtype("<i2"),
    "CharStorage": np.dtype("<i1"),
    "ByteStorage": np.dtype("<u1"),
    "BoolStorage": np.dtype("?"),
}


@dataclass
class _Storage:
    dtype_name: str
    key: str
    numel: int


class _StubTensor:
    """Stands in for torch.Tensor during unpickling; resolves to numpy."""

    def __init__(self, storage: _Storage, offset: int, size: tuple, stride: tuple):
        self.storage = storage
        self.offset = offset
        self.size = size
        self.stride = stride

    def to_numpy(self, read_storage) -> np.ndarray:
        raw = read_storage(self.storage)
        dt = _DTYPES[self.storage.dtype_name]
        flat = np.frombuffer(raw, dtype=dt)
        if self.storage.dtype_name == "BFloat16Storage":
            flat = (flat.astype(np.uint32) << 16).view(np.float32)
        if not self.size:
            return flat[self.offset].copy()
        arr = np.lib.stride_tricks.as_strided(
            flat[self.offset :],
            shape=tuple(self.size),
            strides=tuple(s * flat.dtype.itemsize for s in self.stride),
        )
        return np.ascontiguousarray(arr)


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
    return _StubTensor(storage, storage_offset, size, stride)


class _TorchUnpickler(pickle.Unpickler):
    def __init__(self, file, read_storage):
        super().__init__(file)
        self._read_storage = read_storage

    def find_class(self, module, name):
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor_v2
        if module == "torch" and name in _DTYPES:
            return name  # dtype marker: just the storage class name
        if module == "torch" and name == "device":
            return lambda *a, **k: None
        if module in ("collections", "builtins", "__builtin__"):
            return super().find_class(module, name)
        if module.startswith("torch"):
            raise pickle.UnpicklingError(
                f"unsupported torch object in checkpoint: {module}.{name}"
            )
        return super().find_class(module, name)

    def persistent_load(self, pid):
        # ('storage', StorageClassName, key, location, numel)
        if isinstance(pid, tuple) and pid and pid[0] == "storage":
            _, dtype_name, key, _location, numel = pid
            if not isinstance(dtype_name, str):
                dtype_name = getattr(dtype_name, "__name__", str(dtype_name))
            return _Storage(dtype_name, str(key), int(numel))
        raise pickle.UnpicklingError(f"unsupported persistent id {pid!r}")


def load_pt(path: str):
    """Load a torch.save'd zip checkpoint with numpy only.

    Returns the deserialized object with tensors as numpy arrays.
    """
    zf = zipfile.ZipFile(path)
    names = zf.namelist()
    pkl_name = next(n for n in names if n.endswith("/data.pkl") or n == "data.pkl")
    prefix = pkl_name[: -len("data.pkl")]

    def read_storage(st: _Storage) -> bytes:
        return zf.read(f"{prefix}data/{st.key}")

    up = _TorchUnpickler(io.BytesIO(zf.read(pkl_name)), read_storage)
    obj = up.load()

    def resolve(x):
        if isinstance(x, _StubTensor):
            return x.to_numpy(read_storage)
        if isinstance(x, dict):
            return {k: resolve(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            t = [resolve(v) for v in x]
            return t if isinstance(x, list) else tuple(t)
        return x

    return resolve(obj)


def convert_pt_lens(pt_path: str, gguf_path: str, *, base_model: str = "", dtype=np.float16) -> JacobianLensGGUF:
    """Convert a reference JacobianLens ``.pt`` file to the GGUF lens format."""
    try:
        import torch  # noqa: F401

        ckpt = _load_with_torch(pt_path)
    except ImportError:
        ckpt = load_pt(pt_path)

    if "J" not in ckpt:
        raise ValueError(
            f"{pt_path} does not look like a JacobianLens file "
            f"(keys: {sorted(ckpt)}; is it a fit() checkpoint?)"
        )
    jacobians = {int(l): np.asarray(J, dtype=np.float32) for l, J in ckpt["J"].items()}
    lens = JacobianLensGGUF(
        jacobians,
        d_model=int(ckpt["d_model"]),
        n_prompts=int(ckpt.get("n_prompts", 0)),
        fit_method="jacobian",
        base_model=base_model,
    )
    lens.save(gguf_path, dtype=dtype)
    return lens


def _load_with_torch(pt_path: str) -> dict:
    import torch

    ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.float().numpy()
        if isinstance(x, dict):
            return {k: to_np(v) for k, v in x.items()}
        return x
    return to_np(ckpt)
