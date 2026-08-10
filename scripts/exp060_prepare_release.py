"""exp-060 release prep: assemble uploads/pearsonkyle/<repo>/ for the Qwen3.8-27B quants.

Stages the ladder under terminal-quant filenames (``…-IQ4_XS.gguf``, so Hugging Face
derives the Ollama ``:IQ4_XS`` tag), copies the universal calibration + eval corpora into
``calibration_data/``, writes the MTP note, and renders README.md from whatever numbers
exist. Missing measurements render as ``—`` on purpose: the package can be assembled and
reviewed before the slow agentic / MTP-acceptance runs finish.

Does NOT push. Publishing is a separate, manual step — and the base model's license must be
confirmed and written into the frontmatter first (it is emitted as ``PLACEHOLDER``).

    PYTHONPATH=src .venv/bin/python scripts/exp060_prepare_release.py
    PYTHONPATH=src .venv/bin/python scripts/exp060_prepare_release.py --run exp-060-dryrun \\
        --stem Qwopus3.6-27B-Coder --base-model Jackrong/Qwopus3.6-27B-Coder
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DEFAULT_STEM = "Qwen3.8-27B"
DEFAULT_BASE = "Qwen/Qwen3.8-27B"
DEFAULT_ROWS = ("IQ2_M", "IQ3_M", "IQ4_XS", "Q5_K_M")

CAL_FILES = [
    "corpus.cal.txt",
    "corpus.eval.txt",
    "corpus.eval.general.txt",
    "corpus.eval.tools.txt",
    "corpus.eval.agentic.txt",
    "corpus.eval.broad.txt",
    "corpus.eval.redteam.txt",
    "corpora_audit.json",
]


def _rows_by_quant(csv_path: Path) -> dict[str, dict]:
    """QUANT -> bench row, keyed off the ``|QUANT|`` token in the ``model`` label."""
    out: dict[str, dict] = {}
    if not csv_path.exists():
        return out
    with csv_path.open() as fh:
        for r in csv.DictReader(fh):
            parts = r.get("model", "").split("|")
            if len(parts) >= 2:
                out[parts[1].strip()] = r
    return out


def _f(row: dict | None, key: str) -> float | None:
    if not row:
        return None
    v = row.get(key, "")
    try:
        return float(v) if v not in ("", None) else None
    except ValueError:
        return None


def _fmt(v: float | None, spec: str = ".4f", suffix: str = "") -> str:
    return "—" if v is None else f"{v:{spec}}{suffix}"


def _link(repo_id: str, name: str) -> str:
    return f"[{name}](https://huggingface.co/{repo_id}/resolve/main/{name})"


def _stage(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy:
        shutil.copy2(src, dst)
        return
    try:                       # hardlink: these are 9-20 GiB each
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def render_readme(
    *, stem: str, base_model: str, repo_id: str, rows: list[str],
    ext: dict[str, dict], gen: dict[str, dict], sizes: dict[str, float],
    mtp_info: dict, audit: dict, llama_commit: str,
) -> str:
    cal = audit.get("calibration", {})
    share = cal.get("token_share", {})
    tools_n = cal.get("tool_calls", {}).get("tool_call_marker_total")
    has_mtp = bool(mtp_info.get("has_mtp"))

    head = "| | FP16 (reference) | " + " | ".join(rows) + " |"
    sep = "|---" * (len(rows) + 2) + "|"

    def line(label: str, fp16: str, cells: list[str]) -> str:
        return f"| **{label}** | {fp16} | " + " | ".join(cells) + " |"

    table = "\n".join([
        head, sep,
        line("File", "n/a", [_link(repo_id, f"{stem}-{q}.gguf") for q in rows]),
        line("Size (GiB)", "—", [_fmt(sizes.get(q), ".2f") for q in rows]),
        line("BPW", "16.000", [_fmt(_f(ext.get(q), "bpw"), ".3f") for q in rows]),
        line("PPL (code/math/tools)", "—",
             [_fmt(_f(ext.get(q), "ppl")) for q in rows]),
        line("KLD med (code/math/tools)", "0.00000",
             [_fmt(_f(ext.get(q), "median_kld"), ".5f") for q in rows]),
        line("top_p (code/math/tools)", "100.00%",
             [_fmt(_f(ext.get(q), "same_top_p"), ".2f", "%") for q in rows]),
        line("PPL (general)", "—", [_fmt(_f(gen.get(q), "ppl")) for q in rows]),
        line("KLD med (general)", "0.00000",
             [_fmt(_f(gen.get(q), "median_kld"), ".5f") for q in rows]),
        line("top_p (general)", "100.00%",
             [_fmt(_f(gen.get(q), "same_top_p"), ".2f", "%") for q in rows]),
    ])

    mtp_section = (
        f"""## ⚡ Bundled MTP (multi-token prediction)

The base model ships a trained **MTP draft head** ({len(mtp_info.get('mtp_layer_indices', []))}
nextn layer(s), `blk.{mtp_info.get('mtp_layer_indices', ['?'])[0]}`). llama.cpp runs it as
built-in speculative decoding (`--spec-type draft-mtp`): the head drafts, the trunk verifies
in parallel, accepted drafts skip a decode step.

The head is **pinned to Q8_0 in every quant** while the trunk goes low-bit — it is tiny
relative to the model, and a 2-bit draft head drafts badly. Acceptance and speedup: **—**
(run `scripts/bench_mtp_speed.py` and fill in).

```bash
llama-server --model {stem}-IQ4_XS.gguf --spec-type draft-mtp --spec-draft-n-max 1 \\
             --flash-attn on --n-gpu-layers 999
```
"""
        if has_mtp else
        """## ⚡ MTP (multi-token prediction)

**No draft head is bundled** — the base model ships none, and this release does not graft
one. Speculative decoding needs a separate draft model (`--model-draft`). If a
byte-identical sibling with a trained head appears, `scripts/exp045_graft_mtp.py` grafts it
and the ladder can be rebuilt with the head pinned at Q8_0.
"""
    )

    shares = ", ".join(f"{k} {v:.0%}" for k, v in share.items()) or "—"
    return f"""---
library_name: gguf
base_model:
- {base_model}
tags:
- gguf
- llama.cpp
- quantization
- imatrix
- importance-matrix
{"- mtp\n- speculative-decoding" if has_mtp else ""}
- tool-use
- function-calling
{chr(10).join(f"- {q.lower()}" for q in rows)}
license: PLACEHOLDER   # confirm the base model's license before publishing
language:
- en
pipeline_tag: text-generation
---

# {stem} — imatrix-calibrated GGUF ladder{" + bundled MTP" if has_mtp else ""}

Calibrated quantizations of [`{base_model}`](https://huggingface.co/{base_model}),
{len(rows)} rows from aggressive 2-bit to near-lossless. Calibration used a **hybrid
importance matrix** (`E[a²]` blended with `‖W[:, c]‖²·E[a²]`, ctx 4096) fit on a
multi-source corpus that is majority *real tool-calling text*. Plain GGUF, no custom runtime.

## 1. Files & measurements

{table}

> FP16 reference is not included; fetch it from
> [`{base_model}`](https://huggingface.co/{base_model}). `—` = not yet measured.

Each eval column is a **separate distribution with its own FP16 KLD baseline**:
code/math/tools and general English are external corpora
([`eaddario/imatrix-calibration`](https://huggingface.co/datasets/eaddario/imatrix-calibration))
that the calibration loop never sees, so these numbers measure generalization, not fit.

## 2. Calibration data

The importance matrix was fit on a **universal corpus** blending every source below,
interleaved proportionally so every span of the file mixes them (token shares: {shares}):

| Source | What it contributes |
|---|---|
| On-disk agentic sessions (Claude Code / opencode / qwen code) | real tool-call turns, rendered through this model's own chat template with the tool schemas attached |
| [`pearsonkyle/swe-agentic-trajectories`](https://huggingface.co/datasets/pearsonkyle/swe-agentic-trajectories) | long verified issue-solving trajectories (tests actually passed) |
| [`pearsonkyle/broad-domain-supplement`](https://huggingface.co/datasets/pearsonkyle/broad-domain-supplement) | hand-authored breadth across 9 areas / 192 subjects, so low-bit precision isn't spent exclusively on coding |
| Reasoning-terminal windows | the same sessions re-cut so a chain-of-thought turn lands last — the only position a chat template preserves reasoning in |
| Red-team attack prompts, **every response replaced by a generic refusal** | keeps refusal behavior represented at low bit-widths; the original (sometimes harmful) completions are never used |
| `wiki.test.raw` | general English prose |

The calibration corpus contains **{tools_n if tools_n is not None else '—'} structured tool-call markers** — verified by re-scanning the built file, not assumed. Calibration, the
validation slice and every eval corpus are disjoint by construction (asserted at build
time). All shipped under `calibration_data/`.

{mtp_section}

## 3. Usage

```bash
ollama run hf.co/{repo_id}:{rows[-1]}
# also: {" · ".join(f":{q}" for q in rows[:-1])}
```

```bash
llama-server --model {stem}-{rows[-1]}.gguf --ctx-size 16384 --n-gpu-layers 999 \\
             --flash-attn on --host 0.0.0.0 --port 1234
```

## 4. License & attribution

* Inherits its license from [`{base_model}`](https://huggingface.co/{base_model}).
  **Confirm the exact terms and replace the frontmatter `license:` before publishing.**
* Calibration + quantization performed with
  [**Quant-Tuner**](https://github.com/pearsonkyle/Quant-Tuner); vendored llama.cpp at
  `{llama_commit}`.
* Calibration logs scraped with [**LogMiner**](https://github.com/pearsonkyle/LogMiner).
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="exp-060")
    p.add_argument("--stem", default=DEFAULT_STEM)
    p.add_argument("--base-model", default=DEFAULT_BASE)
    p.add_argument("--repo-id", default=None,
                   help="default: pearsonkyle/<stem>-imatrix-MTP-GGUF")
    p.add_argument("--rows", nargs="+", default=list(DEFAULT_ROWS))
    p.add_argument("--copy", action="store_true", help="copy GGUFs instead of hardlinking")
    a = p.parse_args()

    root = REPO / "out" / a.run
    corpora = root / "corpora"
    repo_id = a.repo_id or f"pearsonkyle/{a.stem}-imatrix-MTP-GGUF"
    pkg = REPO / "uploads" / repo_id
    pkg.mkdir(parents=True, exist_ok=True)

    mtp_info = {}
    if (root / "mtp_report.json").exists():
        mtp_info = json.loads((root / "mtp_report.json").read_text())
    audit = {}
    if (corpora / "corpora_audit.json").exists():
        audit = json.loads((corpora / "corpora_audit.json").read_text())

    # --- stage the GGUFs ---------------------------------------------------------
    sizes: dict[str, float] = {}
    missing: list[str] = []
    for q in a.rows:
        name = f"{a.stem}-{q}.gguf"
        src = root / q.lower() / name
        if not src.exists():
            missing.append(str(src))
            continue
        _stage(src, pkg / name, a.copy)
        sizes[q] = (pkg / name).stat().st_size / 1024**3
        print(f"  staged {name}  ({sizes[q]:.2f} GiB)")
    if missing:
        print(f"  !! {len(missing)} quant(s) not built yet — rendering '—' for them:")
        for m in missing:
            print(f"     {m}")

    # --- calibration data --------------------------------------------------------
    for name in CAL_FILES:
        src = corpora / name
        if src.exists():
            _stage(src, pkg / "calibration_data" / name, copy=True)
    (pkg / "calibration_data").mkdir(parents=True, exist_ok=True)
    (pkg / "calibration_data" / "README.md").write_text(
        "# Calibration & evaluation corpora\n\n"
        "Exactly the files used to build this release.\n\n"
        "* `corpus.cal.txt` — the calibration corpus (agent logs + SWE trajectories + "
        "broad-domain supplement + wiki, interleaved). The imatrix was collected on this.\n"
        "* `corpus.eval.txt` / `corpus.eval.general.txt` — external held-out PPL/KLD "
        "corpora; the headline numbers.\n"
        "* `corpus.eval.tools.txt` / `corpus.eval.agentic.txt` / `corpus.eval.broad.txt` — "
        "in-distribution holdouts, disjoint from calibration. `llama-perplexity` has no "
        "`--parse-special`, so chat markers in these tokenize as plain BPE: use them for "
        "quant-vs-quant comparison, not absolute PPL.\n"
        "* `corpora_audit.json` — token counts, per-source shares, the tool-call marker "
        "scan and the chat-template report.\n"
    )

    # --- MTP note ----------------------------------------------------------------
    if mtp_info.get("has_mtp"):
        (pkg / "MTP").mkdir(exist_ok=True)
        (pkg / "MTP" / "README.md").write_text(
            "# Bundled MTP draft head\n\n"
            f"Draft layer(s): `{mtp_info['mtp_layer_indices']}` "
            f"({mtp_info['mtp_tensor_count']} tensors), pinned to **Q8_0** in every quant "
            f"via `--tensor-type` patterns `{mtp_info['pin']}`.\n\n"
            "Serve with `--spec-type draft-mtp --spec-draft-n-max 1`. One nextn layer means "
            "n-max=1 is optimal; higher values re-draft from the same head and don't help.\n\n"
            "Acceptance / speedup: run `scripts/bench_mtp_speed.py` and record it here.\n"
        )

    llama_commit = "unknown"
    try:
        import subprocess

        llama_commit = subprocess.run(
            ["git", "-C", str(REPO / "vendor" / "llama.cpp"), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        pass

    readme = render_readme(
        stem=a.stem, base_model=a.base_model, repo_id=repo_id, rows=a.rows,
        ext=_rows_by_quant(root / "results.csv"),
        gen=_rows_by_quant(root / "results.general.csv"),
        sizes=sizes, mtp_info=mtp_info, audit=audit, llama_commit=llama_commit,
    )
    (pkg / "README.md").write_text(readme)
    print(f"\n=== staged {pkg} ===")
    print("  README.md frontmatter carries license: PLACEHOLDER — set it before pushing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
