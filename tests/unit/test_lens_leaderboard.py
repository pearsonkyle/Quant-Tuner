"""merge_lens sidecar join into the leaderboard (no model load required)."""

from __future__ import annotations

import csv

from quant_tuner.leaderboard.aggregate import merge_lens


def _write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_merge_lens_joins_by_basename(tmp_path):
    sidecar = tmp_path / "lens.csv"
    _write_csv(sidecar, [
        {"model": "Q2_K-imatrix.gguf", "lens_gold_rank": "3.2",
         "lens_emerge_layer": "42.0", "lens_decision_kld": "0.015",
         "lens_n_decisions": "50"},
    ], ["model", "lens_gold_rank", "lens_emerge_layer", "lens_decision_kld", "lens_n_decisions"])

    rows = [
        {"model": "gemma Q2_K-imatrix", "quant_path": "/x/y/Q2_K-imatrix.gguf"},
        {"model": "other", "quant_path": "/x/y/other.gguf"},
    ]
    merged = merge_lens(rows, sidecar)
    assert merged[0]["lens_gold_rank"] == "3.2"
    assert merged[0]["lens_emerge_layer"] == "42.0"
    assert merged[0]["lens_decision_kld"] == "0.015"
    assert "lens_gold_rank" not in merged[1]


def test_merge_lens_missing_file_is_noop(tmp_path):
    rows = [{"model": "x", "quant_path": "/a/b.gguf"}]
    assert merge_lens(rows, tmp_path / "nope.csv") == rows
