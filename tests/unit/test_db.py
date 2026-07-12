"""Unit tests for the quant-tracking database (in-memory SQLite)."""

from __future__ import annotations

import csv
import math

import pytest
from sqlmodel import SQLModel, select

from quant_tuner.bench.runner import CSV_COLUMNS, BenchRow
from quant_tuner.db import get_engine, init_db, session_scope
from quant_tuner.db.export import export_reps_csvs, export_results_csv
from quant_tuner.db.io import (
    add_mmlu_reps,
    add_toolcall_reps,
    get_or_create_dataset,
    quant_from_bench_row,
    upsert_quant,
)
from quant_tuner.db.models import MmluRep, Quant, ToolcallRep
from quant_tuner.db.query import aggregate_reps, list_quants, top_k
from quant_tuner.eval.reps import ModelReps, RepResult, aggregate_metrics
from quant_tuner.eval.toolcall import Sampling


@pytest.fixture()
def engine():
    eng = get_engine(":memory:")
    init_db(eng)
    return eng


def _bench_row(label: str, **kw) -> BenchRow:
    base = dict(
        model=label, size_gib=5.0, bpw=4.5, ppl=2.4, ppl_ratio=0.9,
        mean_kld=0.5, median_kld=0.0015, same_top_p=89.6, rms_dp=17.0,
        ppl_tools=None, mean_kld_tools=None, same_top_p_tools=None,
        prefill_tok_s=None, prefill_stdev=None, decode_tok_s=None,
        decode_stdev=None, ttft_2k_ms=None, ttft_stdev_ms=None,
        bench_repetitions=None, quant_path="/out/repo/Q4_K_M-imatrix.gguf",
    )
    base.update(kw)
    return BenchRow(**base)


def _mmlu_reps(model: str, accs: list[float]) -> ModelReps:
    reps = [
        RepResult(rep=i, seed=1000 + i, metrics={"accuracy": a, "math_accuracy": a - 0.1})
        for i, a in enumerate(accs)
    ]
    return ModelReps(model=model, n_reps=len(reps), reps=reps,
                     aggregate=aggregate_metrics(reps))


# ---------------------------------------------------------------------------


def test_init_db_creates_all_tables(engine):
    names = set(SQLModel.metadata.tables)
    assert {"dataset", "quant", "mmlu_rep", "toolcall_rep"} <= names
    # Tables are actually queryable.
    with session_scope(engine) as s:
        assert list(s.exec(select(Quant))) == []


def test_get_or_create_dataset_dedups(engine):
    with session_scope(engine) as s:
        a = get_or_create_dataset(s, "custom", path="/c.txt", token_count=100)
        b = get_or_create_dataset(s, "custom", path="/c.txt")
        assert a.id == b.id
        # token_count preserved from first create
        assert b.token_count == 100
    with session_scope(engine) as s:
        assert get_or_create_dataset(s, None) is None


def test_upsert_quant_is_idempotent_on_natural_key(engine):
    with session_scope(engine) as s:
        q1 = upsert_quant(s, model_repo="org/m", quant_type="Q4_K_M",
                          technique="imatrix", variant="hybrid_custom",
                          bpw=4.5)
        q1_id = q1.id
    with session_scope(engine) as s:
        upsert_quant(s, model_repo="org/m", quant_type="Q4_K_M",
                     technique="imatrix", variant="hybrid_custom",
                     bpw=4.6, same_top_p=90.0)
    with session_scope(engine) as s:
        rows = list(s.exec(select(Quant)))
        assert len(rows) == 1
        assert rows[0].id == q1_id
        assert rows[0].bpw == 4.6          # updated
        assert rows[0].same_top_p == 90.0  # new field set


def test_upsert_quant_distinct_keys_make_distinct_rows(engine):
    with session_scope(engine) as s:
        upsert_quant(s, model_repo="org/m", quant_type="Q4_K_M")
        upsert_quant(s, model_repo="org/m", quant_type="Q6_K")
        upsert_quant(s, model_repo="org/other", quant_type="Q4_K_M")
    with session_scope(engine) as s:
        assert len(list(s.exec(select(Quant)))) == 3


def test_upsert_quant_rejects_unknown_metric(engine):
    with session_scope(engine) as s, pytest.raises(KeyError):
        upsert_quant(s, model_repo="org/m", quant_type="Q4_K_M", bogus=1.0)


def test_quant_from_bench_row_maps_kld_fields(engine):
    with session_scope(engine) as s:
        q = quant_from_bench_row(
            s, _bench_row("org/m|imatrix|custom"),
            model_repo="org/m", quant_type="Q4_K_M",
            technique="imatrix", variant="hybrid_custom",
            dataset_name="custom", dataset_tokens=500_000, n_params=9_000_000,
        )
        assert q.kld_mean == 0.5         # mean_kld -> kld_mean
        assert q.kld_median == 0.0015
        assert q.same_top_p == 89.6
        assert q.n_params == 9_000_000
        assert q.dataset is not None and q.dataset.name == "custom"


def test_add_mmlu_reps_and_aggregate_matches_eval_reps(engine):
    accs = [0.6, 0.7, 0.8]
    with session_scope(engine) as s:
        q = upsert_quant(s, model_repo="org/m", quant_type="Q4_K_M")
        add_mmlu_reps(s, q, _mmlu_reps("Q4_K_M.gguf", accs),
                      sampling=Sampling(temperature=0.6, top_p=0.95, top_k=20))
        qid = q.id
    with session_scope(engine) as s:
        rows = list(s.exec(select(MmluRep).where(MmluRep.quant_id == qid)))
        assert len(rows) == 3
        assert rows[0].temperature == 0.6 and rows[0].top_k == 20
        assert rows[0].acc_math is not None
        agg = aggregate_reps(s, qid, "mmlu")
        # mean of [0.6,0.7,0.8] = 0.7, sample stdev = 0.1 (matches eval.reps math)
        assert math.isclose(agg["accuracy"]["mean"], 0.7, rel_tol=1e-9)
        assert math.isclose(agg["accuracy"]["stdev"], 0.1, rel_tol=1e-9)
        assert agg["accuracy"]["n"] == 3


def test_add_reps_replace_clears_prior(engine):
    with session_scope(engine) as s:
        q = upsert_quant(s, model_repo="org/m", quant_type="Q4_K_M")
        add_mmlu_reps(s, q, _mmlu_reps("m.gguf", [0.6, 0.7]))
        qid = q.id
    with session_scope(engine) as s:
        q = s.get(Quant, qid)
        add_mmlu_reps(s, q, _mmlu_reps("m.gguf", [0.9]))  # replace=True default
    with session_scope(engine) as s:
        rows = list(s.exec(select(MmluRep).where(MmluRep.quant_id == qid)))
        assert len(rows) == 1
        assert rows[0].accuracy == 0.9


def test_add_toolcall_reps(engine):
    reps = [RepResult(rep=i, seed=i, metrics={"tool_selection_acc": 0.5 + i * 0.1})
            for i in range(2)]
    mr = ModelReps(model="m.gguf", n_reps=2, reps=reps,
                   aggregate=aggregate_metrics(reps))
    with session_scope(engine) as s:
        q = upsert_quant(s, model_repo="org/m", quant_type="Q4_K_M")
        add_toolcall_reps(s, q, mr)
        qid = q.id
    with session_scope(engine) as s:
        rows = list(s.exec(select(ToolcallRep).where(ToolcallRep.quant_id == qid)))
        assert len(rows) == 2
        agg = aggregate_reps(s, qid, "toolcall")
        assert math.isclose(agg["tool_selection_acc"]["mean"], 0.55, rel_tol=1e-9)


def test_top_k_ranks_best_reps(engine):
    with session_scope(engine) as s:
        q = upsert_quant(s, model_repo="org/m", quant_type="Q4_K_M")
        add_mmlu_reps(s, q, _mmlu_reps("m.gguf", [0.5, 0.9, 0.7, 0.3]))
        qid = q.id
    with session_scope(engine) as s:
        best = top_k(s, qid, "mmlu", "accuracy", k=2)
        assert [r.accuracy for r in best] == [0.9, 0.7]


def test_top_k_rejects_unknown_metric(engine):
    with session_scope(engine) as s:
        q = upsert_quant(s, model_repo="org/m", quant_type="Q4_K_M")
        with pytest.raises(ValueError):
            top_k(s, q.id, "mmlu", "not_a_metric")


def test_list_quants_filters(engine):
    with session_scope(engine) as s:
        upsert_quant(s, model_repo="a/x", quant_type="Q4_K_M")
        upsert_quant(s, model_repo="a/x", quant_type="Q6_K")
        upsert_quant(s, model_repo="b/y", quant_type="Q4_K_M")
    with session_scope(engine) as s:
        assert len(list_quants(s, model_repo="a/x")) == 2
        assert len(list_quants(s, quant_type="Q4_K_M")) == 2
        with pytest.raises(KeyError):
            list_quants(s, bogus="z")


def test_export_results_csv_roundtrips_columns(engine, tmp_path):
    with session_scope(engine) as s:
        quant_from_bench_row(
            s, _bench_row("x"), model_repo="org/m", quant_type="Q4_K_M",
            technique="imatrix", dataset_name="custom",
        )
    out = tmp_path / "results.csv"
    with session_scope(engine) as s:
        n = export_results_csv(s, out, label_format="dataset_quant")
    assert n == 1
    with out.open() as f:
        reader = csv.DictReader(f)
        assert list(reader.fieldnames) == CSV_COLUMNS
        row = next(reader)
        assert row["model"] == "org/m|custom|Q4_K_M"
        assert float(row["mean_kld"]) == 0.5
        assert float(row["bpw"]) == 4.5


def test_export_reps_csvs_layout(engine, tmp_path):
    with session_scope(engine) as s:
        q = upsert_quant(s, model_repo="org/m", quant_type="Q4_K_M",
                         quant_path="/out/Q4_K_M-imatrix.gguf")
        add_mmlu_reps(s, q, _mmlu_reps("ignored", [0.6, 0.8]),
                      sampling=Sampling(temperature=0.6, top_p=0.95))
    agg = tmp_path / "mmlu_agg.csv"
    per = tmp_path / "mmlu_per.csv"
    with session_scope(engine) as s:
        export_reps_csvs(s, "mmlu", per_rep=per, aggregated=agg)
    with agg.open() as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        assert "accuracy_mean" in fields and "accuracy_stdev" in fields
        assert "temperature" in fields
        row = next(reader)
        # model key is the GGUF basename from quant_path
        assert row["model"] == "Q4_K_M-imatrix.gguf"
        assert math.isclose(float(row["accuracy_mean"]), 0.7, rel_tol=1e-9)
