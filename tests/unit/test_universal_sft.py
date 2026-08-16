"""Tests for the universal SFT mixture dataset builder and its private-only guard."""

from __future__ import annotations

import gzip
import json

import pytest

from quant_tuner.datasets import publish, universal_sft
from quant_tuner.datasets.registry import DatasetSpec, SplitSpec, get_spec

ROWS = [
    {"id": "a", "source": "logs", "split": "train", "messages": [{"role": "user", "content": "x"}],
     "n_chars": 1, "system_scrub": {"dropped": 3}},
    {"id": "b", "source": "broad-instruct", "split": "holdout", "messages": [], "n_chars": 2},
    {"id": "c", "source": "logs", "split": "test", "messages": [], "n_chars": 3},
]


@pytest.fixture
def sft_file(tmp_path, monkeypatch):
    p = tmp_path / "sft.jsonl.gz"
    with gzip.open(p, "wt") as fh:
        for r in ROWS:
            fh.write(json.dumps(r) + "\n")
    monkeypatch.setenv("QT_SFT_JSONL", str(p))
    return p


# ------------------------------------------------------------------------- path resolution
def test_env_var_overrides_default_build_dirs(sft_file):
    assert universal_sft.resolve_sft_path() == sft_file


def test_explicit_path_wins_over_env(tmp_path, sft_file):
    other = tmp_path / "other.jsonl.gz"
    with gzip.open(other, "wt") as fh:
        fh.write("{}\n")
    assert universal_sft.resolve_sft_path(other) == other


def test_missing_export_is_an_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_SFT_JSONL", str(tmp_path / "nope.jsonl.gz"))
    with pytest.raises(FileNotFoundError):
        universal_sft.resolve_sft_path()


def test_no_export_anywhere_explains_how_to_build_one(monkeypatch):
    monkeypatch.delenv("QT_SFT_JSONL", raising=False)
    monkeypatch.setattr(universal_sft, "DEFAULT_BUILD_DIRS", [])
    with pytest.raises(FileNotFoundError, match="build_universal_corpus"):
        universal_sft.resolve_sft_path()


# ------------------------------------------------------------------------------- filtering
def test_records_are_filtered_to_their_split(sft_file):
    assert [r["id"] for r in universal_sft.iter_sft_records("train")] == ["a"]
    assert [r["id"] for r in universal_sft.iter_sft_records("holdout")] == ["b"]
    assert [r["id"] for r in universal_sft.iter_sft_records("test")] == ["c"]


def test_build_bookkeeping_is_dropped_from_rows(sft_file):
    """`system_scrub` describes what the build did, not the conversation."""
    rec = next(iter(universal_sft.iter_sft_records("train")))
    assert "system_scrub" not in rec
    assert rec["messages"] == [{"role": "user", "content": "x"}]


def test_unknown_split_is_rejected(sft_file):
    with pytest.raises(ValueError, match="unknown split"):
        list(universal_sft.iter_sft_records("validation"))


# --------------------------------------------------------------------- private-only guard
def _spec(**kw) -> DatasetSpec:
    return DatasetSpec(name="t", title="t", summary="t",
                       splits=[SplitSpec("train", lambda: iter(()))], **kw)


def test_private_only_refuses_a_public_push():
    """The logs are not ours to publish; remembering --private is not a control."""
    with pytest.raises(ValueError, match="private_only"):
        publish.push(_spec(private_only=True), version="0.1.0", private=False, dry_run=True)


def test_private_only_allows_a_private_push():
    publish.push(_spec(private_only=True), version="0.1.0", private=True, dry_run=True)


def test_ordinary_datasets_are_unaffected():
    publish.push(_spec(), version="0.1.0", private=False, dry_run=True)


def test_the_sft_mixture_is_registered_private_only():
    assert get_spec("universal-sft-mixture").private_only is True
