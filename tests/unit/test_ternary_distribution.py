"""The step-0 code census, which the report's distribution-shift panel is read against.

A census that silently reads nothing is worse than one that raises: the panel renders
empty and a reader concludes the codes did not move.
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from scripts.ternary_distribution import census


def _write(tmp_path, *, sharded: bool):
    """A two-tensor stand-in for a decoder, laid out either way."""
    w = {"model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(256, 256),
         "model.language_model.layers.0.mlp.gate_proj.weight": torch.randn(256, 256)}
    if not sharded:
        save_file(w, str(tmp_path / "model.safetensors"))
        return
    keys = list(w)
    for i, k in enumerate(keys, 1):
        save_file({k: w[k]}, str(tmp_path / f"model-{i}.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {' + ", ".join(
            f'"{k}": "model-{i}.safetensors"' for i, k in enumerate(keys, 1)) + "}}")


@pytest.mark.parametrize("sharded", [True, False])
def test_census_reads_both_single_file_and_sharded_checkpoints(tmp_path, sharded: bool):
    """gemma-4-E4B-it-qat-q4_0-unquantized ships 15.9 GB as ONE model.safetensors with
    no index. Requiring the index made the census unusable for it."""
    _write(tmp_path, sharded=sharded)
    rows = census(tmp_path, None, want_all=True)
    assert len(rows) == 2, rows
    assert {r["kind"] for r in rows} == {"q_proj", "gate_proj"}
    for r in rows:
        assert r["neg"] + r["zero"] + r["pos"] == r["numel"]
        assert 0.0 < r["zero_frac"] < 1.0


def test_census_says_what_is_missing_rather_than_reading_nothing(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(SystemExit, match="neither"):
        census(tmp_path, None, want_all=True)
