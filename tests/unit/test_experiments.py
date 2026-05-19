"""Tests for the experiment-runner helpers."""

from __future__ import annotations

from pathlib import Path

from quant_tuner.experiments import step


def test_step_runs_when_output_missing(tmp_path: Path) -> None:
    out = tmp_path / "artifact.txt"
    calls = []

    def build():
        out.write_text("hi")
        calls.append(1)
        return "built"

    result = step("make artifact", out, build)
    assert result == "built"
    assert calls == [1]
    assert out.read_text() == "hi"


def test_step_skips_when_output_present(tmp_path: Path) -> None:
    out = tmp_path / "artifact.txt"
    out.write_text("already here")
    calls = []

    def build():
        calls.append(1)
        return "built"

    result = step("make artifact", out, build)
    assert result is None
    assert calls == []
    assert out.read_text() == "already here"


def test_step_requires_all_outputs_present_to_skip(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("a")  # b missing → should still build

    calls = []

    def build():
        b.write_text("b")
        calls.append(1)

    step("make both", [a, b], build)
    assert calls == [1]
    assert b.read_text() == "b"
