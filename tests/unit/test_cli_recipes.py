"""Tests for the recipe-loading helpers in quant_tuner.cli.

Covers the YAML → ``RunConfig`` round-trip, the bare-name → packaged-recipe
resolution, and the PLACEHOLDER override semantics. Doesn't actually run any
pipelines.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from quant_tuner.cli import _load_recipe, _resolve_recipe
from quant_tuner.config import RunConfig

PACKAGED = Path(__file__).resolve().parents[2] / "src" / "quant_tuner" / "recipes"


def test_resolve_bare_name_finds_packaged_recipe():
    p = _resolve_recipe(Path("q4_k_m_imatrix"))
    assert p == PACKAGED / "q4_k_m_imatrix.yaml"


def test_resolve_full_filename_finds_packaged_recipe():
    p = _resolve_recipe(Path("q4_k_m_none.yaml"))
    assert p == PACKAGED / "q4_k_m_none.yaml"


def test_resolve_explicit_path(tmp_path: Path):
    custom = tmp_path / "my.yaml"
    custom.write_text(
        "name: t\nmodel: x\nworkspace: ./w\n"
        "calibration: {method: none}\nquantize: {type: Q4_K_M}\n"
    )
    assert _resolve_recipe(custom) == custom


def test_resolve_missing_raises():
    with pytest.raises(typer.BadParameter):
        _resolve_recipe(Path("does_not_exist"))


def test_load_with_overrides(tmp_path: Path):
    cfg = _load_recipe(
        Path("q4_k_m_imatrix"),
        model="org/repo",
        logs=tmp_path / "logs.jsonl",
        workspace=tmp_path / "work",
    )
    assert isinstance(cfg, RunConfig)
    assert cfg.model == "org/repo"
    assert cfg.data.logs == tmp_path / "logs.jsonl"
    assert cfg.workspace == tmp_path / "work"
    assert cfg.calibration.method == "imatrix"
    assert cfg.calibration.variant == "hybrid_custom"


def test_load_no_calibration_doesnt_require_logs(tmp_path: Path):
    cfg = _load_recipe(
        Path("q4_k_m_none"),
        model="org/repo",
        logs=None,
        workspace=tmp_path / "work",
    )
    # PLACEHOLDER logs is fine when method=none.
    assert cfg.calibration.method == "none"


def test_load_placeholder_model_rejected(tmp_path: Path):
    with pytest.raises(typer.BadParameter, match="model"):
        _load_recipe(Path("q4_k_m_imatrix"), model=None,
                     logs=tmp_path / "x.jsonl", workspace=None)


def test_load_placeholder_logs_rejected_when_calibrating(tmp_path: Path):
    with pytest.raises(typer.BadParameter, match="logs"):
        _load_recipe(Path("q4_k_m_imatrix"), model="org/repo",
                     logs=None, workspace=tmp_path / "w")
