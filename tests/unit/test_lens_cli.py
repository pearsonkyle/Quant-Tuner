"""CLI wiring tests for `quant-tuner lens` (no model load required)."""

from __future__ import annotations

import re

from typer.testing import CliRunner

from quant_tuner.cli import app
from quant_tuner.lens.cli import _parse_layers, lens_app

runner = CliRunner()

# Typer's rich-rendered help inserts ANSI color codes *inside* long option
# names when the terminal is colored (e.g. a CI runner with a wide COLUMNS),
# splitting "--lens-csv" into "-"/"lens"/"-csv" segments. Strip the escapes
# before asserting so the check is independent of the color state.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


def test_lens_subapp_registered():
    result = runner.invoke(app, ["lens", "--help"])
    assert result.exit_code == 0
    for cmd in ("fit", "capture", "diff", "serve", "replay-toolcalls", "loop",
                "probe", "study", "bake", "report", "build-server"):
        assert cmd in result.output


def test_each_command_has_help():
    for cmd in ("fit", "capture", "diff", "serve", "replay-toolcalls", "loop",
                "probe", "study", "bake", "report", "convert-pt", "inspect"):
        result = runner.invoke(lens_app, [cmd, "--help"])
        assert result.exit_code == 0, f"{cmd} --help failed: {result.output}"


def test_parse_layers():
    assert _parse_layers(None) is None
    assert _parse_layers("0,2,5") == [0, 2, 5]
    assert _parse_layers("0-3") == [0, 1, 2, 3]


def test_leaderboard_has_lens_csv_option():
    result = runner.invoke(app, ["leaderboard", "--help"])
    assert result.exit_code == 0
    plain = _plain(result.output)
    if "--lens-csv" not in plain:
        print("\nDEBUG len(output)=", len(result.output),
              "len(plain)=", len(plain))
        print("repr(plain):", repr(plain[:800]))
    assert "--lens-csv" in plain
