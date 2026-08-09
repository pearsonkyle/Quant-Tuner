"""Unit tests for drafter window/config logic (no torch, no model files)."""

import json
from pathlib import Path

import pytest

from quant_tuner.drafter.train import TrainConfig, load_windows
from quant_tuner.drafter.windows import WindowConfig


class FakeTok:
    def encode(self, text, add_special_tokens=True):
        return list(range(len(text.split())))

    def apply_chat_template(self, *a, **k):
        return "w " * 100


def test_load_windows_chunks_to_max_len(tmp_path):
    p = tmp_path / "w.jsonl"
    p.write_text(json.dumps({"input_ids": list(range(20000))}) + "\n")
    chunks = load_windows(p, max_len=8192)
    assert [len(c) for c in chunks] == [8192, 8192, 3616]
    assert chunks[0][0] == 0 and chunks[1][0] == 8192


def test_load_windows_drops_singletons(tmp_path):
    p = tmp_path / "w.jsonl"
    # length 8193 -> 8192 + a 1-token sliver that must be dropped
    p.write_text(json.dumps({"input_ids": list(range(8193))}) + "\n")
    chunks = load_windows(p, max_len=8192)
    assert [len(c) for c in chunks] == [8192]


def test_train_config_validate_missing_windows(tmp_path):
    cfg = TrainConfig(
        target_model="t", drafter_model="d",
        windows=tmp_path / "nope.jsonl", out_dir=tmp_path / "o",
    )
    with pytest.raises(ValueError, match="not found"):
        cfg.validate()


def test_train_config_validate_ok(tmp_path):
    w = tmp_path / "w.jsonl"
    w.write_text(json.dumps({"input_ids": [1, 2, 3]}) + "\n")
    TrainConfig(target_model="t", drafter_model="d", windows=w, out_dir=tmp_path).validate()


def test_window_config_rejects_bad_stride(tmp_path):
    cfg = WindowConfig(logs=tmp_path / "l", out=tmp_path / "o", max_len=1024, stride=0)
    with pytest.raises(ValueError, match="stride"):
        cfg_iter(cfg)


def cfg_iter(cfg):
    from quant_tuner.drafter.windows import iter_windows

    return iter_windows(cfg, FakeTok())


def test_window_config_rejects_maxlen_below_minlen(tmp_path):
    cfg = WindowConfig(logs=tmp_path / "l", out=tmp_path / "o", max_len=100, min_len=256)
    with pytest.raises(ValueError, match="max_len"):
        cfg_iter(cfg)


def test_package_imports_without_torch():
    import quant_tuner.drafter as d

    assert callable(d.write_windows)
