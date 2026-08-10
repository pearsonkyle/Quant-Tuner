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
from collections import Counter
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
        n_rows = n_resolved = n_tool_calls = est_tokens = 0
        # Corpus-shaped datasets are described by distributions, not by pass/fail counts:
        # which topics, which registers, and which disjoint half each row belongs to.
        tally: dict[str, Counter[str]] = {
            k: Counter() for k in ("area", "register", "half", "prompt_source")}
        area_tokens: Counter[str] = Counter()
        area_subjects: dict[str, set[str]] = {}
        with out.open("w") as fh:
            for rec in split.builder():
                fh.write(json.dumps(rec) + "\n")
                n_rows += 1
                n_resolved += bool(rec.get("resolved"))
                n_tool_calls += int(rec.get("n_tool_calls") or 0)
                est_tokens += (tokens := int(rec.get("est_tokens") or 0))
                for key, counter in tally.items():
                    if rec.get(key):
                        counter[rec[key]] += 1
                if area := rec.get("area"):
                    area_tokens[area] += tokens
                    area_subjects.setdefault(area, set()).add(rec.get("subject", ""))
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
        if est_tokens:
            stats["est_tokens"] = est_tokens
        stats |= {f"by_{k}": dict(c.most_common()) for k, c in tally.items() if c}
        if area_tokens:
            stats["areas"] = {
                a: {"rows": tally["area"][a], "subjects": len(area_subjects[a]),
                    "est_tokens": tok}
                for a, tok in area_tokens.most_common()
            }
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

    def _table(header: list[str], body_rows: list[list],
               left: tuple[str, ...] = ()) -> list[str]:
        """A GFM table whose separator is derived from the header (first column left,
        the rest right-aligned) so the cell count can never drift out of sync.

        ``left`` names extra columns to left-align — a prose column like a description
        reads badly flushed right against the numbers.
        """
        sep = ["---" if i == 0 or h in left else "---:" for i, h in enumerate(header)]
        out = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
        return out + ["| " + " | ".join(str(c) for c in r) + " |" for r in body_rows]

    # Corpus-shaped datasets have no pass/fail notion: the useful summary is coverage —
    # rows and tokens per split, then how those tokens fall across topics and registers.
    if any(i.get("areas") for i in splits.values()):
        rows = _table(
            ["split", "rows", "tokens~", "size", "contents"],
            [[f"`{name}`", f"{i['rows']:,}", f"{i.get('est_tokens', 0):,}",
              f"{i['bytes'] / 1024**2:.1f} MB", i.get("description", "")]
             for name, i in splits.items()],
            left=("contents",),
        )
        ref = next(i for i in splits.values() if i.get("areas"))
        areas, total_tok = ref["areas"], max(1, ref.get("est_tokens", 0))
        rows += ["", "### Topic distribution", ""]
        rows += _table(
            ["area", "subjects", "samples", "tokens~", "share"],
            [[f"`{a}`", d["subjects"], f"{d['rows']:,}", f"{d['est_tokens']:,}",
              f"{100 * d['est_tokens'] / total_tok:.1f}%"] for a, d in areas.items()]
            + [["**total**", sum(d["subjects"] for d in areas.values()),
                f"**{ref['rows']:,}**", f"**{ref.get('est_tokens', 0):,}**", "100%"]],
        )
        if by_reg := ref.get("by_register"):
            rows += ["", "### Sample registers", ""]
            rows += _table(
                ["register", "samples", "share"],
                [[f"`{k}`", f"{v:,}", f"{100 * v / max(1, ref['rows']):.1f}%"]
                 for k, v in by_reg.items()],
            )
        if by_half := ref.get("by_half"):
            rows += ["", "### Disjoint halves", "",
                     "Every row carries `half`, a **deterministic, non-overlapping** "
                     "assignment (see below). Filter on it; do not re-split.", ""]
            rows += _table(
                ["half", "samples", "intended use"],
                [[f"`{k}`", f"{by_half.get(k, 0):,}", u] for k, u in
                 (("calib", "quantization calibration (imatrix / AWQ / GPTQ)"),
                  ("mtp", "MTP draft-head training"))],
                left=("intended use",),
            )
    else:
        rows = _table(
            ["split", "rows", "verified (tests pass)", "mean tool calls", "size"],
            [[f"`{name}`", i["rows"], i["resolved_rows"], i["mean_tool_calls"],
              f"{i['bytes'] / 1024**2:.1f} MB"] for name, i in splits.items()],
        )

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
    def _detail(i: dict) -> str:
        # corpus-shaped splits have no pass/fail notion; report tokens instead of
        # "0 verified", which reads as a failure rather than as not-applicable
        return f"~{i['est_tokens']:,} tokens" if i.get("est_tokens") else \
               f"{i['resolved_rows']} verified"

    rows = "\n".join(
        f"- `{n}`: {i['rows']} rows ({_detail(i)}), {i['bytes'] / 1024**2:.1f} MB"
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
