"""GGUF tensor-edit + weight orthogonalization tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_tuner.lens.bake import Direction, _orthogonalize_output_rows
from quant_tuner.lens.gguf_edit import (
    copy_gguf_with_tensor_edits,
    list_gguf_tensor_names,
    load_gguf_tensor,
)
from quant_tuner.paths import ensure_gguf_py


def _write_synthetic_gguf(path, tensors: dict[str, np.ndarray], kv: dict):
    ensure_gguf_py()
    from gguf import GGUFWriter

    w = GGUFWriter(str(path), arch="llama")
    for k, v in kv.items():
        if isinstance(v, str):
            w.add_string(k, v)
        elif isinstance(v, list):
            w.add_array(k, v)
        else:
            w.add_uint32(k, int(v))
    for name, arr in tensors.items():
        w.add_tensor(name, arr.astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def test_copy_preserves_unedited_tensors_and_kv(tmp_path):
    src = tmp_path / "src.gguf"
    rng = np.random.default_rng(0)
    tensors = {
        "blk.0.attn_output.weight": rng.standard_normal((6, 4)).astype(np.float32),
        "blk.1.ffn_down.weight": rng.standard_normal((6, 4)).astype(np.float32),
        "token_embd.weight": rng.standard_normal((10, 6)).astype(np.float32),
    }
    _write_synthetic_gguf(src, tensors, {
        "llama.block_count": 2, "llama.embedding_length": 6,
        "general.name": "synthetic", "some.array": [1, 2, 3],
    })

    dst = tmp_path / "dst.gguf"
    new = np.zeros((6, 4), np.float32)
    copy_gguf_with_tensor_edits(src, dst, {"blk.0.attn_output.weight": new})

    # unedited tensor byte-identical
    np.testing.assert_array_equal(
        load_gguf_tensor(src, "blk.1.ffn_down.weight"),
        load_gguf_tensor(dst, "blk.1.ffn_down.weight"),
    )
    # edit applied
    np.testing.assert_array_equal(load_gguf_tensor(dst, "blk.0.attn_output.weight"), new)
    # names preserved
    assert set(list_gguf_tensor_names(dst)) == set(tensors)


def test_copy_missing_edit_target_raises(tmp_path):
    src = tmp_path / "src.gguf"
    _write_synthetic_gguf(src, {"blk.0.ffn_down.weight": np.ones((4, 4), np.float32)},
                          {"llama.block_count": 1, "llama.embedding_length": 4})
    with pytest.raises(KeyError, match="not found"):
        copy_gguf_with_tensor_edits(src, tmp_path / "d.gguf", {"blk.9.nope.weight": np.zeros((4, 4))})


def test_orthogonalize_removes_output_direction():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((6, 4)).astype(np.float32)
    v = rng.standard_normal(6).astype(np.float32)
    v /= np.linalg.norm(v)
    We = _orthogonalize_output_rows(W, v)
    # no input can produce a component along v
    assert np.abs(v @ We).max() < 1e-5
    # orthogonal component preserved for a random input
    x = rng.standard_normal(4).astype(np.float32)
    y0 = W @ x
    np.testing.assert_allclose(We @ x, y0 - (v @ y0) * v, atol=1e-4)


def test_direction_save_load(tmp_path):
    v = np.arange(8, dtype=np.float32)
    d = Direction(v=v, layers=[10, 11], method="loop_mean_diff", source={"run_id": "abc"})
    p = tmp_path / "dir.npz"
    d.save(p)
    d2 = Direction.load(p)
    np.testing.assert_array_equal(d2.v, v)
    assert d2.layers == [10, 11]
    assert d2.method == "loop_mean_diff"
    assert d2.source["run_id"] == "abc"
