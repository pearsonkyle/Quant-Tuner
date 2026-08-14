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
    # each chunk is (ids, gen_start); legacy windows -> gen_start 0
    assert [len(ids) for ids, _ in chunks] == [8192, 8192, 3616]
    assert all(gs == 0 for _, gs in chunks)
    assert chunks[0][0][0] == 0 and chunks[1][0][0] == 8192


def test_load_windows_drops_singletons(tmp_path):
    p = tmp_path / "w.jsonl"
    # length 8193 -> 8192 + a 1-token sliver that must be dropped
    p.write_text(json.dumps({"input_ids": list(range(8193))}) + "\n")
    chunks = load_windows(p, max_len=8192)
    assert [len(ids) for ids, _ in chunks] == [8192]


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


def test_finephrase_best_generation_filters():
    from quant_tuner.drafter.finephrase import best_generation

    # too short -> None
    assert best_generation([{"finish_reason": "stop", "text": "hi",
                             "usage": {"completion_tokens": 3}}], min_tokens=10) is None
    # degenerate empty -> None
    assert best_generation([{"finish_reason": "stop", "text": "  ",
                             "usage": {"completion_tokens": 50}}], min_tokens=10) is None
    # picks the longest finished one
    rr = [{"finish_reason": "stop", "text": "short", "usage": {"completion_tokens": 20}},
          {"finish_reason": "stop", "text": "longer one", "usage": {"completion_tokens": 40}}]
    assert best_generation(rr, min_tokens=10) == "longer one"
    # unfinished (not stop/length) -> skipped
    assert best_generation([{"finish_reason": "error", "text": "x",
                             "usage": {"completion_tokens": 99}}], min_tokens=10) is None


def test_finephrase_config_validate():
    import pytest

    from quant_tuner.drafter.finephrase import FinePhraseConfig
    with pytest.raises(ValueError, match="unknown configs"):
        FinePhraseConfig(out=Path("/x"), configs=("faq", "bogus")).validate()
    with pytest.raises(ValueError, match="token_budget"):
        FinePhraseConfig(out=Path("/x"), token_budget=10, max_len=2048).validate()


def test_load_windows_masks_gen_start(tmp_path):
    import json as _json

    from quant_tuner.drafter.train import load_windows
    p = tmp_path / "op.jsonl"
    p.write_text(
        _json.dumps({"input_ids": list(range(100)), "gen_start": 40}) + "\n"
        + _json.dumps({"input_ids": list(range(20))}) + "\n"
    )
    chunks = load_windows(p, max_len=1024)
    assert chunks[0] == (list(range(100)), 40)      # on-policy window kept whole w/ gen_start
    assert chunks[1] == (list(range(20)), 0)        # legacy window -> gen_start 0


def test_onpolicy_config_validate(tmp_path):
    import pytest

    from quant_tuner.drafter.onpolicy import OnPolicyConfig
    w = tmp_path / "w.jsonl"; w.write_text('{"input_ids":[1,2,3]}\n')
    OnPolicyConfig(base_url="http://x/v1", model="m", out=tmp_path/"o", prompt_windows=w).validate()
    with pytest.raises(ValueError, match="not found"):
        OnPolicyConfig(base_url="http://x/v1", model="m", out=tmp_path/"o",
                       prompt_windows=tmp_path/"nope").validate()
