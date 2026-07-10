"""Unit tests for bench.runner.bench_one wiring (no llama.cpp binaries)."""

from pathlib import Path

import pytest

from quant_tuner.bench import runner
from quant_tuner.bench.kld import KLDMetrics


@pytest.fixture()
def patched(monkeypatch):
    calls: list[tuple[Path, Path]] = []

    def fake_evaluate(quant, dataset, baseline, *, ctx, n_tokens=None, log=None):
        calls.append((dataset, baseline))
        return KLDMetrics(
            ppl=2.0 if len(calls) == 1 else 3.0,
            ppl_ratio=1.0,
            mean_kld=0.1 if len(calls) == 1 else 0.2,
            median_kld=0.05,
            same_top_p=90.0 if len(calls) == 1 else 80.0,
            rms_dp=1.0,
        )

    monkeypatch.setattr(runner.kld, "evaluate", fake_evaluate)
    monkeypatch.setattr(runner.bpw_mod, "size_gib", lambda p: 1.5)
    monkeypatch.setattr(runner.bpw_mod, "bpw", lambda p, n: 4.0)
    return calls


def test_bench_one_tools_suite_populates_tools_columns(patched, tmp_path):
    row = runner.bench_one(
        tmp_path / "q.gguf", "m",
        reference_n_params=10,
        eval_dataset=tmp_path / "eval.txt",
        eval_baseline=tmp_path / "baseline.kld",
        tools_dataset=tmp_path / "tools.txt",
        tools_baseline=tmp_path / "baseline-tools.kld",
        suite="kld",
    )
    assert len(patched) == 2
    assert patched[1] == (tmp_path / "tools.txt", tmp_path / "baseline-tools.kld")
    assert (row.mean_kld, row.same_top_p) == (0.1, 90.0)
    assert (row.ppl_tools, row.mean_kld_tools, row.same_top_p_tools) == (3.0, 0.2, 80.0)


def test_bench_one_without_tools_leaves_tools_columns_none(patched, tmp_path):
    row = runner.bench_one(
        tmp_path / "q.gguf", "m",
        reference_n_params=10,
        eval_dataset=tmp_path / "eval.txt",
        eval_baseline=tmp_path / "baseline.kld",
        suite="kld",
    )
    assert len(patched) == 1
    assert row.mean_kld_tools is None and row.same_top_p_tools is None


def test_bench_one_rejects_mismatched_tools_pair(patched, tmp_path):
    with pytest.raises(ValueError, match="must be set together"):
        runner.bench_one(
            tmp_path / "q.gguf", "m",
            reference_n_params=10,
            eval_dataset=tmp_path / "eval.txt",
            eval_baseline=tmp_path / "baseline.kld",
            tools_dataset=tmp_path / "tools.txt",
            suite="kld",
        )
