"""Unit tests for quant_tuner.eval.reps — aggregation + CSV writers."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from quant_tuner.eval.reps import (
    ModelReps,
    RepResult,
    _seed_for,
    aggregate_metrics,
    sampling_extra_cols,
    write_aggregated_csv,
    write_csvs,
    write_per_rep_csv,
)
from quant_tuner.eval.toolcall import Sampling


def _rr(rep: int, metrics: dict[str, float], **kw) -> RepResult:
    return RepResult(rep=rep, seed=1000 + rep, metrics=metrics, **kw)


class TestAggregateMetrics:
    def test_single_rep_stdev_is_zero(self):
        agg = aggregate_metrics([_rr(0, {"acc": 0.8})])
        assert agg["acc"]["mean"] == 0.8
        assert agg["acc"]["stdev"] == 0.0
        assert agg["acc"]["n"] == 1

    def test_mean_and_stdev_known_values(self):
        # values: 0.6, 0.7, 0.8 → mean=0.7, sample-stdev (n-1)=0.1
        rrs = [_rr(i, {"acc": v}) for i, v in enumerate([0.6, 0.7, 0.8])]
        agg = aggregate_metrics(rrs)
        assert math.isclose(agg["acc"]["mean"], 0.7, rel_tol=1e-9)
        assert math.isclose(agg["acc"]["stdev"], 0.1, rel_tol=1e-9)
        assert agg["acc"]["n"] == 3

    def test_uses_sample_stdev_not_population(self):
        # values: 1.0, 2.0  (mean 1.5)
        # sample (n-1) variance     = ((-0.5)² + (0.5)²) / 1 = 0.5  → stdev √0.5 ≈ 0.7071
        # population (n)  variance  = 0.5 / 2                = 0.25 → stdev 0.5
        rrs = [_rr(0, {"x": 1.0}), _rr(1, {"x": 2.0})]
        agg = aggregate_metrics(rrs)
        assert math.isclose(agg["x"]["stdev"], math.sqrt(0.5), rel_tol=1e-9)

    def test_errored_reps_are_skipped(self):
        rrs = [
            _rr(0, {"acc": 0.9}),
            _rr(1, {}, error="server died"),
            _rr(2, {"acc": 0.7}),
        ]
        agg = aggregate_metrics(rrs)
        assert agg["acc"]["n"] == 2
        assert math.isclose(agg["acc"]["mean"], 0.8, rel_tol=1e-9)

    def test_metric_missing_in_some_reps(self):
        # Some reps report `b`, some don't.
        rrs = [
            _rr(0, {"a": 1.0, "b": 10.0}),
            _rr(1, {"a": 2.0}),  # no b
            _rr(2, {"a": 3.0, "b": 20.0}),
        ]
        agg = aggregate_metrics(rrs)
        assert agg["a"]["n"] == 3
        assert agg["b"]["n"] == 2
        assert math.isclose(agg["b"]["mean"], 15.0)

    def test_non_numeric_metric_silently_dropped(self):
        # Strings can sneak in via dict[str, float] type erasure; the aggregator
        # should skip them rather than blow up.
        rrs = [
            _rr(0, {"acc": 0.8, "note": "good"}),  # type: ignore
            _rr(1, {"acc": 0.9, "note": "ok"}),    # type: ignore
        ]
        agg = aggregate_metrics(rrs)
        assert "note" not in agg
        assert agg["acc"]["n"] == 2

    def test_empty_input(self):
        assert aggregate_metrics([]) == {}

    def test_all_errors_yields_empty(self):
        rrs = [_rr(0, {}, error="x"), _rr(1, {}, error="y")]
        assert aggregate_metrics(rrs) == {}


class TestSeedDerivation:
    def test_default_base_seed(self):
        assert _seed_for(1000, 0) == 1000
        assert _seed_for(1000, 9) == 1009

    def test_alternate_base_seed(self):
        assert _seed_for(42, 5) == 47


class TestCsvWriters:
    def _make_model_reps(self) -> list[ModelReps]:
        mr1 = ModelReps(
            model="model-f16.gguf",
            n_reps=3,
            reps=[
                _rr(0, {"acc": 0.80, "f1": 0.75}),
                _rr(1, {"acc": 0.82, "f1": 0.78}),
                _rr(2, {"acc": 0.81, "f1": 0.77}),
            ],
        )
        mr1.aggregate = aggregate_metrics(mr1.reps)
        mr2 = ModelReps(
            model="Q4_K_M-none.gguf",
            n_reps=2,
            reps=[
                _rr(0, {"acc": 0.60, "f1": 0.55}),
                _rr(1, {"acc": 0.62, "f1": 0.57}),
            ],
        )
        mr2.aggregate = aggregate_metrics(mr2.reps)
        return [mr1, mr2]

    def test_per_rep_writes_one_row_per_rep(self, tmp_path: Path):
        path = tmp_path / "reps.csv"
        write_per_rep_csv(self._make_model_reps(), path)
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 5  # 3 + 2
        assert rows[0]["model"] == "model-f16.gguf"
        assert rows[0]["rep"] == "0"
        assert rows[0]["seed"] == "1000"
        assert rows[0]["acc"] == "0.8"
        assert rows[3]["model"] == "Q4_K_M-none.gguf"

    def test_per_rep_includes_extra_cols(self, tmp_path: Path):
        path = tmp_path / "reps.csv"
        write_per_rep_csv(
            self._make_model_reps(), path,
            extra_cols={"temperature": 0.6, "top_p": 0.95},
        )
        rows = list(csv.DictReader(path.open()))
        assert rows[0]["temperature"] == "0.6"
        assert rows[0]["top_p"] == "0.95"

    def test_aggregated_one_row_per_model_with_mean_stdev(self, tmp_path: Path):
        path = tmp_path / "agg.csv"
        write_aggregated_csv(self._make_model_reps(), path)
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 2
        # Mean/stdev columns produced for every metric.
        for r in rows:
            assert r["acc_mean"] != ""
            assert r["acc_stdev"] != ""
            assert r["f1_mean"] != ""
        # First model has 3 reps in n column.
        assert rows[0]["acc_n"] == "3"
        assert rows[1]["acc_n"] == "2"

    def test_write_csvs_writes_both(self, tmp_path: Path):
        per_rep = tmp_path / "reps.csv"
        agg = tmp_path / "agg.csv"
        write_csvs(self._make_model_reps(), per_rep=per_rep, aggregated=agg)
        assert per_rep.exists() and agg.exists()
        assert len(list(csv.DictReader(per_rep.open()))) == 5
        assert len(list(csv.DictReader(agg.open()))) == 2

    def test_errored_rep_still_appears_in_per_rep_csv(self, tmp_path: Path):
        mr = ModelReps(
            model="m.gguf", n_reps=2,
            reps=[_rr(0, {"acc": 0.5}), _rr(1, {}, error="boom")],
        )
        mr.aggregate = aggregate_metrics(mr.reps)
        path = tmp_path / "reps.csv"
        write_per_rep_csv([mr], path)
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 2
        assert rows[1]["error"] == "boom"
        assert rows[1]["acc"] == ""


class TestSamplingExtraCols:
    def test_renders_set_fields_and_blanks_unset(self):
        s = Sampling(temperature=0.6, top_p=0.95, top_k=20)
        cols = sampling_extra_cols(s)
        assert cols["temperature"] == 0.6
        assert cols["top_p"] == 0.95
        assert cols["top_k"] == 20
        # min_p, presence_penalty, repetition_penalty are None → blank string
        assert cols["min_p"] == ""
        assert cols["presence_penalty"] == ""
        assert cols["repetition_penalty"] == ""
