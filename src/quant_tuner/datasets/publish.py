"""Build, version, card, and push datasets declared in :mod:`quant_tuner.datasets.registry`.

Shared machinery so a new dataset only needs a registry entry. Responsibilities:

* **build** — run each split's builder into ``datasets/<name>/data/<split>.jsonl``, hashing
  every payload and recording row counts in ``manifest.json``.
* **card**  — render ``README.md`` with HF YAML frontmatter (``configs`` so the Hub's dataset
  viewer picks the splits up automatically) plus the spec's prose body and a live stats table.
* **push**  — upload the staging directory to the Hub, bump the version, append to
  ``CHANGELOG.md``, and tag the repo ``v<version>`` so older releases stay pinnable.

Versions are semver ``MAJOR.MINOR.PATCH``. ``bump_version`` is used by the push CLI; a version
is only written to ``manifest.json`` once the upload succeeds, so a failed push cannot leave
the repo claiming a release that does not exist on the Hub.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from quant_tuner.datasets.registry import DatasetSpec

MANIFEST = "manifest.json"
CHANGELOG = "CHANGELOG.md"


# ------------------------------------------------------------------------------- versioning
def bump_version(version: str, part: str = "patch") -> str:
    major, minor, patch = (int(x) for x in version.split("."))
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def read_manifest(spec: DatasetSpec) -> dict:
    p = spec.stage_dir / MANIFEST
    if p.exists():
        return json.loads(p.read_text())
    return {"name": spec.name, "version": "0.0.0", "splits": {}, "published": []}


def write_manifest(spec: DatasetSpec, manifest: dict) -> None:
    spec.stage_dir.mkdir(parents=True, exist_ok=True)
    (spec.stage_dir / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")


# ------------------------------------------------------------------------------------ build
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(spec: DatasetSpec) -> dict:
    """Materialize every split to jsonl and refresh the manifest's split stats."""
    data_dir = spec.stage_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(spec)
    manifest["name"] = spec.name
    manifest["hf_repo"] = spec.repo_id
    manifest["built_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    splits: dict[str, dict] = {}

    for split in spec.splits:
        out = data_dir / f"{split.name}.jsonl"
        n_rows = 0
        n_resolved = 0
        n_tool_calls = 0
        outcomes: dict[str, int] = {}
        models: set[str] = set()
        with out.open("w") as fh:
            for rec in split.builder():
                fh.write(json.dumps(rec) + "\n")
                n_rows += 1
                n_resolved += bool(rec.get("resolved"))
                n_tool_calls += int(rec.get("n_tool_calls") or 0)
                # red-team datasets label an `outcome` + a target `model` instead
                if rec.get("outcome"):
                    outcomes[rec["outcome"]] = outcomes.get(rec["outcome"], 0) + 1
                if rec.get("model"):
                    models.add(rec["model"])
        stats = {
            "file": f"data/{split.name}.jsonl",
            "description": split.description,
            "publish": split.publish,
            "rows": n_rows,
            "resolved_rows": n_resolved,
            "mean_tool_calls": round(n_tool_calls / n_rows, 1) if n_rows else 0.0,
            "bytes": out.stat().st_size,
            "sha256": _sha256(out),
        }
        if outcomes:  # red-team split: surface the outcome mix + model coverage
            stats["outcomes"] = dict(sorted(outcomes.items()))
            stats["models"] = sorted(models)
        splits[split.name] = stats
        print(f"[dataset] {spec.name}:{split.name}  {n_rows} rows  "
              f"({splits[split.name]['bytes'] / 1024**2:.1f} MB)"
              f"{'' if split.publish else '   [local only, not published]'}", flush=True)

    manifest["splits"] = splits
    write_manifest(spec, manifest)
    return manifest


# ------------------------------------------------------------------------------------- card
def render_card(spec: DatasetSpec, manifest: dict) -> str:
    """Dataset card: HF YAML frontmatter + stats table + the spec's prose."""
    # Only published splits appear on the Hub: they drive the viewer config, the stats
    # table and the size bucket. Local-only splits stay out of the card entirely.
    splits = {n: i for n, i in manifest.get("splits", {}).items() if i.get("publish", True)}
    cfg_lines = ["configs:", "- config_name: default", "  data_files:"]
    for name, info in splits.items():
        cfg_lines.append(f'  - split: {name}')
        cfg_lines.append(f'    path: {info["file"]}')

    fm = [
        "---",
        f"license: {spec.license}",
        "task_categories:",
        *[f"- {t}" for t in spec.task_categories],
        "tags:",
        *[f"- {t}" for t in spec.tags],
        "size_categories:",
        f"- {_size_bucket(sum(i['rows'] for i in splits.values()) or 0)}",
        *cfg_lines,
        "---",
        "",
    ]

    # Adaptive stats table: outcome-labeled (red-team) datasets report the
    # outcome mix; others keep the SWE verified/tool-call columns. When nothing
    # is published (e.g. a dual-use dataset withheld by default), say so plainly.
    if not splits:
        rows = ["_No splits are published for this dataset (withheld by default — "
                "see below). Build locally to materialize them._"]
    elif any(i.get("outcomes") for i in splits.values()):
        rows = ["| split | rows | complied | defended | errored | models | size |",
                "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for name, i in splits.items():
            o = i.get("outcomes", {})
            rows.append(f"| `{name}` | {i['rows']} | {o.get('complied', 0)} | "
                        f"{o.get('defended', 0)} | {o.get('errored', 0)} | "
                        f"{len(i.get('models', []))} | {i['bytes'] / 1024**2:.1f} MB |")
    else:
        rows = ["| split | rows | verified (tests pass) | mean tool calls | size |",
                "| --- | ---: | ---: | ---: | ---: |"]
        for name, i in splits.items():
            rows.append(f"| `{name}` | {i['rows']} | {i['resolved_rows']} | "
                        f"{i['mean_tool_calls']} | {i['bytes'] / 1024**2:.1f} MB |")

    _SWE_SCHEMA = (
        "| field | meaning |\n"
        "| --- | --- |\n"
        "| `instance_id`, `repo` | the upstream issue and its repository |\n"
        "| `messages` | the full session in chat format (`system`/`user`/`assistant`+`tool_calls`/`tool`) |\n"
        "| `tools` | tool schema the agent was given (`bash(command)`) |\n"
        "| `submission` | the final `git diff` the agent produced |\n"
        "| `resolved` | **true = the hidden tests passed** (verified solution) |\n"
        "| `patch_produced`, `patch_chars` | whether a non-empty patch was submitted, and its size |\n"
        "| `n_messages`, `n_tool_calls`, `tools_used`, `tool_errors` | session shape |\n"
        "| `n_fail_to_pass[_passed]`, `n_pass_to_pass[_passed]` | grading detail |\n"
        "| `prompt_tokens`, `completion_tokens`, `total_tokens`, `wall_sec` | cost |\n"
        "| `exit_status` | how the agent loop ended (`completed` / `max_turns` / …) |"
    )
    schema = ["## Row schema", "", spec.schema_md.strip() or _SWE_SCHEMA]

    return "\n".join([
        *fm,
        f"# {spec.title}",
        "",
        spec.summary,
        "",
        f"**Version `{manifest.get('version', '0.0.0')}`**"
        + (f" · built {manifest['built_at']}" if manifest.get("built_at") else ""),
        "",
        *rows,
        "",
        spec.body.strip(),
        "",
        *schema,
        "",
        "## Reproducing",
        "",
        "Generated with [Quant-Tuner](https://github.com/pearsonkyle/Quant-Tuner); see",
        "`docs/ternary_qat.md` for the end-to-end pipeline and",
        "`src/quant_tuner/datasets/` for the exact builder used to publish this.",
        "",
    ])


def _size_bucket(n: int) -> str:
    for lim, label in ((1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K"),
                       (1_000_000, "100K<n<1M")):
        if n < lim:
            return label
    return "1M<n<10M"


def write_card(spec: DatasetSpec, manifest: dict) -> Path:
    card = spec.stage_dir / "README.md"
    card.write_text(render_card(spec, manifest))
    return card


# ------------------------------------------------------------------------------------- push
def append_changelog(spec: DatasetSpec, version: str, manifest: dict, note: str) -> None:
    p = spec.stage_dir / CHANGELOG
    head = f"# Changelog — {spec.name}\n" if not p.exists() else ""
    rows = "\n".join(
        f"- `{n}`: {i['rows']} rows ({i['resolved_rows']} verified), {i['bytes'] / 1024**2:.1f} MB"
        for n, i in manifest.get("splits", {}).items())
    entry = (f"\n## v{version} — {time.strftime('%Y-%m-%d')}\n\n"
             f"{note.strip() or 'Dataset refresh.'}\n\n{rows}\n")
    with p.open("a") as f:
        if head:
            f.write(head)
        f.write(entry)


def push(spec: DatasetSpec, *, version: str, note: str = "", private: bool = False,
         dry_run: bool = False) -> str:
    """Upload the staging dir to the Hub, then record the version locally.

    The manifest/changelog are only updated after a successful upload, so a failed push never
    leaves the repo claiming a release that is not on the Hub.
    """
    manifest = read_manifest(spec)
    manifest["version"] = version
    write_manifest(spec, manifest)          # card should show the version being published
    write_card(spec, manifest)

    if dry_run:
        print(f"[dataset] DRY RUN — would push {spec.stage_dir} -> {spec.repo_id} as v{version}")
        return spec.repo_id

    # Splits marked publish=False are built locally but must never leave the machine.
    ignore = [CHANGELOG] + [
        i["file"] for i in manifest.get("splits", {}).values() if not i.get("publish", True)
    ]
    held_back = [n for n, i in manifest.get("splits", {}).items() if not i.get("publish", True)]
    if held_back:
        print(f"[dataset] withholding local-only split(s): {', '.join(held_back)}", flush=True)

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(spec.repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=spec.repo_id,
        repo_type="dataset",
        folder_path=str(spec.stage_dir),
        ignore_patterns=ignore,             # changelog is repo-side history; + local-only splits
        commit_message=f"v{version}: {note or 'dataset refresh'}",
    )
    try:
        api.create_tag(spec.repo_id, repo_type="dataset", tag=f"v{version}",
                       tag_message=note or f"v{version}")
    except Exception as e:                  # tag already exists / no permission -> not fatal
        print(f"[dataset] tag v{version} not created ({type(e).__name__})", flush=True)

    manifest.setdefault("published", []).append(
        {"version": version, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "note": note,
         "splits": {n: i["rows"] for n, i in manifest.get("splits", {}).items()}})
    write_manifest(spec, manifest)
    append_changelog(spec, version, manifest, note)
    print(f"[dataset] pushed https://huggingface.co/datasets/{spec.repo_id} (v{version})")
    return spec.repo_id
