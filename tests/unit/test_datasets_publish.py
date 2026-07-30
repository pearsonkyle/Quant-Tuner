"""Tests for the dataset publishing machinery (no network, no model loads).

Pins the properties that make "add a dataset later and push a new version" safe:
  * semver bumping,
  * build writes one jsonl line per record and records rows/sha in the manifest,
  * the card carries HF frontmatter with a ``configs`` block per split (else the Hub
    viewer will not find the data),
  * a FAILED push must not advance the recorded published history.
"""

import json

import pytest

from quant_tuner.datasets.publish import (
    build,
    bump_version,
    push,
    read_manifest,
    render_card,
    write_card,
)
from quant_tuner.datasets.registry import DatasetSpec, SplitSpec


def _spec(tmp_path, rows_a=3, rows_b=2) -> DatasetSpec:
    spec = DatasetSpec(
        name="demo-set",
        title="Demo",
        summary="a demo dataset",
        splits=[
            SplitSpec("resolved", lambda: iter(
                [{"instance_id": f"i{i}", "resolved": True, "n_tool_calls": 4}
                 for i in range(rows_a)]), "verified"),
            SplitSpec("all", lambda: iter(
                [{"instance_id": f"j{i}", "resolved": i == 0, "n_tool_calls": 2}
                 for i in range(rows_b)]), "everything"),
        ],
        tags=["demo"],
        body="## Body\n\nprose",
    )
    object.__setattr__(spec, "name", "demo-set")
    # redirect staging into tmp_path
    import quant_tuner.datasets.registry as reg
    reg.DATASETS_DIR = tmp_path / "datasets"
    return spec


def test_bump_version():
    assert bump_version("0.0.0", "patch") == "0.0.1"
    assert bump_version("1.2.3", "minor") == "1.3.0"
    assert bump_version("1.2.3", "major") == "2.0.0"


def test_build_writes_jsonl_and_manifest(tmp_path, monkeypatch):
    import quant_tuner.datasets.registry as reg
    monkeypatch.setattr(reg, "DATASETS_DIR", tmp_path / "datasets")
    spec = _spec(tmp_path)
    manifest = build(spec)

    assert manifest["splits"]["resolved"]["rows"] == 3
    assert manifest["splits"]["all"]["rows"] == 2
    assert manifest["splits"]["all"]["resolved_rows"] == 1
    assert manifest["splits"]["resolved"]["mean_tool_calls"] == 4.0
    assert len(manifest["splits"]["resolved"]["sha256"]) == 64

    payload = (spec.stage_dir / "data" / "resolved.jsonl").read_text().strip().splitlines()
    assert len(payload) == 3
    assert json.loads(payload[0])["instance_id"] == "i0"


def test_card_has_frontmatter_and_configs_for_every_split(tmp_path, monkeypatch):
    import quant_tuner.datasets.registry as reg
    monkeypatch.setattr(reg, "DATASETS_DIR", tmp_path / "datasets")
    spec = _spec(tmp_path)
    manifest = build(spec)
    manifest["version"] = "1.4.0"
    card = render_card(spec, manifest)

    assert card.startswith("---\n")                       # HF frontmatter first
    assert "configs:" in card and "config_name: default" in card
    for split in ("resolved", "all"):                     # viewer needs a path per split
        assert f"- split: {split}" in card
        assert f"path: data/{split}.jsonl" in card
    assert "Version `1.4.0`" in card
    assert "## Body" in card                              # spec prose survives
    assert "| `resolved` | 3 |" in card                   # live stats table

    written = write_card(spec, manifest)
    assert written.exists() and written.name == "README.md"


def test_failed_push_does_not_record_a_publish(tmp_path, monkeypatch):
    import quant_tuner.datasets.registry as reg
    monkeypatch.setattr(reg, "DATASETS_DIR", tmp_path / "datasets")
    spec = _spec(tmp_path)
    build(spec)

    class _Boom:
        def __init__(self, *a, **k):
            pass

        def create_repo(self, *a, **k):
            raise RuntimeError("network down")

    monkeypatch.setattr("huggingface_hub.HfApi", _Boom)
    with pytest.raises(RuntimeError):
        push(spec, version="0.1.0", note="should fail")

    # version may be staged, but nothing must be recorded as published
    assert read_manifest(spec).get("published") == []


def test_unpublished_split_is_withheld_from_upload_and_card(tmp_path, monkeypatch):
    """publish=False must keep a split off the Hub AND out of the card."""
    import quant_tuner.datasets.registry as reg
    monkeypatch.setattr(reg, "DATASETS_DIR", tmp_path / "datasets")

    spec = DatasetSpec(
        name="demo-set",
        title="Demo",
        summary="s",
        splits=[
            SplitSpec("resolved", lambda: iter([{"resolved": True, "n_tool_calls": 1}]), "ok"),
            SplitSpec("all", lambda: iter([{"resolved": False, "n_tool_calls": 1}] * 4),
                      "local", publish=False),
        ],
    )
    manifest = build(spec)
    # built locally...
    assert (spec.stage_dir / "data" / "all.jsonl").exists()
    assert manifest["splits"]["all"]["publish"] is False

    # ...but absent from the card (no viewer config, no stats row)
    card = render_card(spec, manifest)
    assert "path: data/resolved.jsonl" in card
    assert "data/all.jsonl" not in card
    assert "| `all` |" not in card

    captured = {}

    class _Api:
        def __init__(self, *a, **k):
            pass

        def create_repo(self, *a, **k):
            pass

        def upload_folder(self, **kw):
            captured.update(kw)

        def create_tag(self, *a, **k):
            pass

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    push(spec, version="0.1.0", note="n")
    assert "data/all.jsonl" in captured["ignore_patterns"]
    assert "data/resolved.jsonl" not in captured["ignore_patterns"]


def test_dry_run_push_skips_upload_and_records_nothing(tmp_path, monkeypatch):
    import quant_tuner.datasets.registry as reg
    monkeypatch.setattr(reg, "DATASETS_DIR", tmp_path / "datasets")
    spec = _spec(tmp_path)
    build(spec)
    push(spec, version="0.2.0", note="dry", dry_run=True)
    m = read_manifest(spec)
    assert m["version"] == "0.2.0"      # card/manifest reflect what WOULD ship
    assert m.get("published") == []     # but no release recorded
