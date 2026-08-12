"""exp-045 release prep: assemble uploads/pearsonkyle/tmax-27b-imatrix-2bit-MTP-GGUF/.

Stages the four 2-bit+MTP GGUFs under terminal-quant names (so Ollama tags resolve),
copies the calibration_data corpora, writes the MTP note, and renders README.md with
the §1 table + a head-to-head comparison vs Qwopus3.6-27B-Coder. All numbers are read
from the exp-045 artifacts (results.csv, qwopus_compare.csv, mtp_bench_iq2m.json,
swebench/aggregated.csv); cells with no data yet render as "—" so the package can be
assembled before the slow agentic / MTP steps finish.

Does NOT push to HF — that is scripts/exp045_upload_release.py (run manually after
confirming the license).

    PYTHONPATH=src .venv/bin/python scripts/exp045_prepare_release.py
    # large GGUFs are hardlinked by default; pass --copy to force real copies
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
EXP45 = REPO / "out" / "exp-045"
PKG = REPO / "uploads" / "pearsonkyle" / "tmax-27b-imatrix-MTP-GGUF"

MODEL_STEM = "tmax-27b"
BASE_MODEL = "allenai/tmax-27b"
REPO_ID = "pearsonkyle/tmax-27b-imatrix-MTP-GGUF"
QWOPUS_REPO = "pearsonkyle/Qwopus3.6-27B-Coder-imatrix-2bit-MTP-GGUF"

RESULTS_CSV = EXP45 / "results.csv"
QWOPUS_CSV = EXP45 / "qwopus_compare.csv"
# All-quant agentic run (6 tmax quants + Qwopus IQ2_M, same 10 SWE-rebench instances).
SWEBENCH_AGG = EXP45 / "swebench_allquants" / "aggregated.csv"
MTP_SWEEP = EXP45 / "mtp_accept_sweep_iq4xs.json"

CORPORA = EXP45 / "corpora"
# Every GGUF bundles the grafted Qwopus nextn head at blk.64 (Q8_0). Sources are the
# -mtp quants under out/exp-045/<label>_mtp/; dest uses terminal-quant names for Ollama.
# label -> (quant, source gguf under out/exp-045/<label>/, terminal-quant dest name)
QUANTS = [
    ("q2k_plain_mtp", "Q2_K",   f"{MODEL_STEM}-Q2_K-mtp.gguf",   f"{MODEL_STEM}-Q2_K.gguf",   "plain"),
    ("iq2xs_mtp",     "IQ2_XS", f"{MODEL_STEM}-IQ2_XS-mtp.gguf", f"{MODEL_STEM}-IQ2_XS.gguf", "imatrix"),
    ("iq2m_mtp",      "IQ2_M",  f"{MODEL_STEM}-IQ2_M-mtp.gguf",  f"{MODEL_STEM}-IQ2_M.gguf",  "imatrix"),
    ("q2ks_mtp",      "Q2_K_S", f"{MODEL_STEM}-Q2_K_S-mtp.gguf", f"{MODEL_STEM}-Q2_K_S.gguf", "imatrix"),
    ("iq3m_mtp",      "IQ3_M",  f"{MODEL_STEM}-IQ3_M-mtp.gguf",  f"{MODEL_STEM}-IQ3_M.gguf",  "imatrix"),
    ("iq4xs_mtp",     "IQ4_XS", f"{MODEL_STEM}-IQ4_XS-mtp.gguf", f"{MODEL_STEM}-IQ4_XS.gguf", "imatrix"),
    ("q5km_mtp",      "Q5_K_M", f"{MODEL_STEM}-Q5_K_M-mtp.gguf", f"{MODEL_STEM}-Q5_K_M.gguf", "imatrix"),
]


# ---- parsing helpers ------------------------------------------------------

def _rows_by_quant(csv_path: Path) -> dict[str, dict]:
    """Map QUANT -> bench row dict, keyed off the '|QUANT|' token in `model`."""
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


def _fmt(x: float | None, spec: str) -> str:
    return format(x, spec) if x is not None else "—"


def _pct(x: float | None) -> str:
    return f"{x:.2f}%" if x is not None else "—"


def _swebench_pass() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not SWEBENCH_AGG.exists():
        return out
    with SWEBENCH_AGG.open() as fh:
        for r in csv.DictReader(fh):
            out[r.get("model", "")] = r
    return out


def _mtp_accept() -> list[tuple[int, float]]:
    """[(n_max, accept_pct), ...] from the grafted-MTP acceptance sweep."""
    if not MTP_SWEEP.exists():
        return []
    sweep = json.loads(MTP_SWEEP.read_text()).get("sweep", {})
    out: list[tuple[int, float]] = []
    for n in (1, 2, 3, 4):
        r = sweep.get(str(n)) or sweep.get(n)
        if r and r.get("accept") is not None:
            out.append((n, r["accept"] * 100))
    return out


# ---- staging --------------------------------------------------------------

def _stage_gguf(src: Path, dest: Path, *, copy: bool) -> None:
    if dest.exists():
        print(f"  exists: {dest.name}")
        return
    if not src.exists():
        print(f"  !! MISSING source GGUF (run exp045_quants_tmax.py): {src}")
        return
    if copy:
        shutil.copy2(src, dest)
        print(f"  copied: {dest.name} ({dest.stat().st_size/1024**3:.1f} GiB)")
    else:
        try:
            os.link(src, dest)
            print(f"  linked: {dest.name} ({dest.stat().st_size/1024**3:.1f} GiB)")
        except OSError:
            shutil.copy2(src, dest)
            print(f"  copied (link failed): {dest.name}")


def _stage_calibration_data() -> None:
    cd = PKG / "calibration_data"
    cd.mkdir(parents=True, exist_ok=True)
    for name in ("corpus.cal.txt", "corpus.eval.general.txt",
                 "corpus.eval.tools.txt", "corpora_audit.json"):
        src = CORPORA / name
        if src.exists():
            shutil.copy2(src, cd / name)
            print(f"  calibration_data/{name}")
        else:
            print(f"  !! MISSING corpus: {src}")
    (cd / "README.md").write_text(_CALIB_README)
    print("  calibration_data/README.md")


# ---- README rendering -----------------------------------------------------

def _render_readme(tmax: dict[str, dict], qwo: dict[str, dict],
                   swe: dict[str, dict], mtp_rows: list[tuple[int, float]]) -> str:
    def cell(rows, q, key, spec):
        return _fmt(_f(rows.get(q), key), spec)

    def topp(rows, q):
        return _pct(_f(rows.get(q), "same_top_p"))

    # agentic head-to-head (IQ2_M), sourced from the all-quant run
    tmax_swe = swe.get(f"{MODEL_STEM}-IQ2_M.gguf", {})
    qwo_swe = swe.get("Qwopus3.6-27B-Coder-IQ2_M.gguf", {})
    tmax_pass = _pct(_f(tmax_swe, "pass_rate") and _f(tmax_swe, "pass_rate") * 100)
    qwo_pass = _pct(_f(qwo_swe, "pass_rate") and _f(qwo_swe, "pass_rate") * 100)
    n_inst = (tmax_swe or qwo_swe).get("n_instances", "?")

    # per-quant agentic table (SWE-rebench, all 6 tmax quants)
    def _ag(q, key, spec, scale=1.0):
        v = _f(swe.get(f"{MODEL_STEM}-{q}.gguf"), key)
        return _fmt(v * scale if v is not None else None, spec)

    def _ag_resolved(q):
        r = swe.get(f"{MODEL_STEM}-{q}.gguf")
        if not r:
            return "—"
        return f"{r.get('n_resolved', '?')}/{r.get('n_instances', '?')}"

    QORDER = ["Q2_K", "IQ2_XS", "IQ2_M", "Q2_K_S", "IQ3_M", "IQ4_XS"]
    agentic_tbl = "\n".join(
        f"| {q} | {_ag(q,'pass_rate','.0%')} | {_ag(q,'patch_rate','.0%')} | "
        f"{_ag_resolved(q)} | {_ag(q,'mean_tokens',',.0f')} | "
        f"{_ag(q,'mean_steps','.1f')} | {_ag(q,'tool_error_rate','.0%')} |"
        for q in QORDER)

    # grafted-MTP acceptance-vs-n table rows
    mtp_tbl = "\n".join(
        f"| {n} | {acc:.1f}% |" for n, acc in mtp_rows) or "| 1 | — |"

    base_url = f"https://huggingface.co/{REPO_ID}/resolve/main"

    return f"""---
library_name: gguf
base_model:
- {BASE_MODEL}
tags:
- gguf
- llama.cpp
- text-generation
- quantization
- quantized
- imatrix
- importance-matrix
- low-bit
- 2-bit
- 3-bit
- 4-bit
- iq2_xs
- iq2_m
- q2_k_s
- iq3_m
- iq4_xs
- q5_k_m
- 5-bit
- mtp
- multi-token-prediction
- speculative-decoding
- qwen3
- 27b
- tool-use
- function-calling
license: apache-2.0
language:
- en
pipeline_tag: text-generation
---

# 🧊 {BASE_MODEL} — imatrix GGUF (2 → 5-bit) + MTP

A ladder of importance-matrix quantizations of **{BASE_MODEL}**, from aggressive 2-bit
up to near-lossless 5-bit, each calibrated with a **hybrid importance matrix** from real
agentic-coding usage logs + wiki text — and each shipping a **Multi-Token-Prediction
(MTP) draft head bundled in at Q8_0** for built-in speculative decoding (`--spec-type
draft-mtp`). tmax-27b is a Qwen3.6-27B derivative whose released checkpoint dropped the
MTP head; because it is byte-for-byte architecture-identical to Jackrong/Qwopus3.6-27B-Coder
(same Qwen3.6-27B base), we **graft Qwopus3.6's trained nextn head** onto it — it drafts
at high acceptance (see §2.2). The 2-bit row is benchmarked **head-to-head against
Qwopus3.6** below; pick a higher-bit row when you have the VRAM.

## 🧰 1. Files & comparison

Seven quants spanning ~2–5 bits-per-weight (plain Q2_K anchor + six hybrid-imatrix rows),
each with the grafted MTP head at Q8_0. Lower KLD / higher top_p = closer to FP16. FP16
reference is not included — fetch it from [`{BASE_MODEL}`](https://huggingface.co/{BASE_MODEL}).

|  | Q2_K (plain) | IQ2_XS | IQ2_M | Q2_K_S | IQ3_M | IQ4_XS | Q5_K_M |
|---|---|---|---|---|---|---|---|
| **File** | [Q2_K]({base_url}/{MODEL_STEM}-Q2_K.gguf) | [IQ2_XS]({base_url}/{MODEL_STEM}-IQ2_XS.gguf) | [IQ2_M]({base_url}/{MODEL_STEM}-IQ2_M.gguf) | [Q2_K_S]({base_url}/{MODEL_STEM}-Q2_K_S.gguf) | [IQ3_M]({base_url}/{MODEL_STEM}-IQ3_M.gguf) | [IQ4_XS]({base_url}/{MODEL_STEM}-IQ4_XS.gguf) | [Q5_K_M]({base_url}/{MODEL_STEM}-Q5_K_M.gguf) |
| **Technique** | plain | hybrid imatrix | hybrid imatrix | hybrid imatrix | hybrid imatrix | hybrid imatrix | hybrid imatrix |
| **Size (GiB)** | {cell(tmax,'Q2_K','size_gib','.2f')} | {cell(tmax,'IQ2_XS','size_gib','.2f')} | {cell(tmax,'IQ2_M','size_gib','.2f')} | {cell(tmax,'Q2_K_S','size_gib','.2f')} | {cell(tmax,'IQ3_M','size_gib','.2f')} | {cell(tmax,'IQ4_XS','size_gib','.2f')} | {cell(tmax,'Q5_K_M','size_gib','.2f')} |
| **BPW** | {cell(tmax,'Q2_K','bpw','.3f')} | {cell(tmax,'IQ2_XS','bpw','.3f')} | {cell(tmax,'IQ2_M','bpw','.3f')} | {cell(tmax,'Q2_K_S','bpw','.3f')} | {cell(tmax,'IQ3_M','bpw','.3f')} | {cell(tmax,'IQ4_XS','bpw','.3f')} | {cell(tmax,'Q5_K_M','bpw','.3f')} |
| **PPL (general)** | {cell(tmax,'Q2_K','ppl','.4f')} | {cell(tmax,'IQ2_XS','ppl','.4f')} | {cell(tmax,'IQ2_M','ppl','.4f')} | {cell(tmax,'Q2_K_S','ppl','.4f')} | {cell(tmax,'IQ3_M','ppl','.4f')} | {cell(tmax,'IQ4_XS','ppl','.4f')} | {cell(tmax,'Q5_K_M','ppl','.4f')} |
| **KLD med (general)** | {cell(tmax,'Q2_K','median_kld','.4f')} | {cell(tmax,'IQ2_XS','median_kld','.4f')} | {cell(tmax,'IQ2_M','median_kld','.4f')} | {cell(tmax,'Q2_K_S','median_kld','.4f')} | {cell(tmax,'IQ3_M','median_kld','.4f')} | {cell(tmax,'IQ4_XS','median_kld','.4f')} | {cell(tmax,'Q5_K_M','median_kld','.4f')} |
| **top_p (general)** | {topp(tmax,'Q2_K')} | {topp(tmax,'IQ2_XS')} | {topp(tmax,'IQ2_M')} | {topp(tmax,'Q2_K_S')} | {topp(tmax,'IQ3_M')} | {topp(tmax,'IQ4_XS')} | {topp(tmax,'Q5_K_M')} |

### Head-to-head vs Qwopus3.6-27B-Coder

Both models are Qwen3.6-27B derivatives benched on the **identical** general eval
corpus and the **same** llama.cpp build (lower KLD / higher top_p is better). Qwopus3.6
numbers are re-measured here, not quoted from its README, so the comparison is
apples-to-apples.

| Metric (IQ2_M) | **tmax-27b** | Qwopus3.6-27B-Coder |
|---|---|---|
| KLD med (general) | {cell(tmax,'IQ2_M','median_kld','.4f')} | {cell(qwo,'IQ2_M','median_kld','.4f')} |
| top_p (general) | {topp(tmax,'IQ2_M')} | {topp(qwo,'IQ2_M')} |
| PPL (general) | {cell(tmax,'IQ2_M','ppl','.4f')} | {cell(qwo,'IQ2_M','ppl','.4f')} |
| Agentic pass_rate (SWE-rebench, n={n_inst}) | {tmax_pass} | {qwo_pass} |

> Agentic eval: {n_inst} lite SWE-rebench instances, one clean Docker container per
> instance, `pass_rate` = fraction whose patch makes the gold FAIL_TO_PASS tests pass.
> Both models run without speculative decoding (MTP only affects decode speed, not
> output quality), so the comparison is apples-to-apples.

### Agentic coding across quants (SWE-rebench)

Every quant run as a coding agent (mini-swe-agent) over the **same {n_inst} held-out
SWE-rebench instances**, one clean Docker container each. `pass_rate` = fraction whose
patch makes the gold FAIL_TO_PASS tests pass; `patch_rate` = fraction that produced a
non-empty diff. Token/step counts are per instance.

| Quant | pass_rate | patch_rate | resolved | mean tokens | mean steps | tool-err |
|---|---|---|---|---|---|---|
{agentic_tbl}

> Same {n_inst}-instance holdout as the head-to-head, no speculative decoding. At 2-bit
> the quants stay surprisingly capable agents; higher-bit rows trade size for headroom.

<div style="border:1px solid #fde68a;border-radius:12px;background:#fffbeb;padding:16px;margin:16px 0;color:#92400e;font-size:13px;line-height:1.7;">
  <b>⚠️ Caveat.</b> The 2-bit rows (Q2_K / IQ2_*) are aggressive — strong for their size
  but rougher on tmax than on Qwopus3.6 (see the head-to-head). Prefer <b>IQ3_M</b> or
  <b>IQ4_XS</b> when you have the VRAM; reach for the 2-bit rows only when memory is the
  binding constraint.
</div>

<div style="border:1px solid #bfdbfe;border-radius:12px;background:#eff6ff;padding:16px;margin:16px 0;color:#1e3a8a;font-size:13px;line-height:1.7;">
  <b>📋 Calibration scope — English &amp; Python, agentic coding.</b> The importance
  matrix (and the windowed packing that shaped it) was calibrated on <b>real
  agentic-coding sessions that are overwhelmingly English-language and
  Python-centric</b> (Claude Code, opencode, qwen code). Expect <b>weaker fidelity on
  other natural languages, non-Python ecosystems, and general-chat workloads</b>.
</div>

---

## 🔬 2. How they were made

**2.1 Hybrid importance matrix.** At 2 bits the quantizer must decide *where* to spend
its limited precision. An importance matrix measures, per input channel, how much each
channel drives a layer's output on a calibration corpus. This release blends activation
energy `E[a²]` with weight-column energy `‖W[:, c]‖² · E[a²]`, collected at ctx=4096.
Linear-attention / SSM tensors (Qwen3.6 is a hybrid architecture) pass through with raw
`E[a²]`. The output is a standard GGUF with no runtime overhead.

**2.2 Bundled MTP (grafted).** tmax-27b's released checkpoint dropped the Qwen3.6 MTP
draft head (its `mtp_num_hidden_layers` flag is vestigial). But tmax and
Jackrong/Qwopus3.6-27B-Coder are **byte-for-byte architecture-identical** finetunes of
the same Qwen3.6-27B base (hidden 5120, 64 layers, 24/4 GQA, head_dim 256, vocab 248320,
identical hybrid attention layout), so Qwopus3.6's trained nextn head is dimensionally
compatible. We **graft it onto tmax's trunk** as `blk.64` and keep it near-lossless at
**Q8_0** in every quant; llama.cpp serves it as built-in speculative decoding
(`--spec-type draft-mtp`).

Despite being a *cross-finetune* graft, it drafts at high acceptance — measured on the
IQ4_XS trunk over agentic-coding prompts (draft acceptance = fraction of drafted tokens
the trunk confirms):

| `--spec-draft-n-max` | draft acceptance |
|---|---|
{mtp_tbl}

**`--spec-draft-n-max 1` is recommended** (one nextn layer; higher `n` drafts longer
chains at falling acceptance). Speedup is GPU-bandwidth-dependent — spec decoding helps
most on memory-bound CUDA GPUs; the win is smaller on unified-memory/Metal. Acceptance
is also prompt-dependent (higher on predictable code, lower on open-ended text).

**2.3 Calibration & evaluation data.** Calibration and every eval corpus are disjoint by
construction — see `calibration_data/`. The general eval is broad English
(`combined_en_tiny` from `eaddario/imatrix-calibration`); the tools eval is the held-out
10% of usage sessions, windowed exactly like calibration but never seen by it.

---

## 🚀 3. Usage

### Quick start with Ollama

```bash
ollama run hf.co/{REPO_ID}:IQ2_M
# also: :Q2_K_S  ·  :IQ2_XS  ·  :Q2_K
```

### Running the server with MTP speculative decoding

```bash
./llama-server \\
    --model {MODEL_STEM}-IQ4_XS.gguf \\
    --ctx-size 16384 --n-gpu-layers 999 \\
    --spec-type draft-mtp --spec-draft-n-max 1 \\
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \\
    --host 0.0.0.0 --port 1234
```

> **MTP needs a recent llama.cpp** — `--spec-type draft-mtp` was merged in 2026-06. Drop
> the two `--spec-*` flags to run without MTP. `--spec-draft-n-max 1` is optimal (one
> nextn layer). If the model emits `<think>` reasoning, also pass
> `--chat-template-kwargs '{{"enable_thinking": false}}'` so the answer lands in
> `content`.

---

## 🪪 4. License & attribution

* **License: Apache-2.0** — inherited from the base model
  [`{BASE_MODEL}`](https://huggingface.co/{BASE_MODEL}). Ai2 asks that use follow its
  [Responsible Use Guidelines](https://allenai.org/responsible-use).
* Base weights: [`{BASE_MODEL}`](https://huggingface.co/{BASE_MODEL}) (Qwen3.6-27B
  derivative; released checkpoint is the text trunk only — vision + MTP head dropped).
* **MTP draft head grafted from** [`{QWOPUS_REPO}`](https://huggingface.co/{QWOPUS_REPO})
  (also a Qwen3.6-27B finetune; same head it ships natively). Also the head-to-head baseline.
* Calibration + quantization performed locally with
  [**Quant-Tuner**](https://github.com/pearsonkyle/Quant-Tuner).
"""


_MTP_README = """# MTP (Multi-Token Prediction) — bundled, not separate

Every GGUF in the parent directory bundles a **grafted** Qwen3.6 nextn/MTP draft head
directly at `blk.64` (Q8_0). There are no separate files here by design.

## Why grafted

allenai/tmax-27b's released checkpoint dropped its MTP head. But tmax and
Jackrong/Qwopus3.6-27B-Coder are byte-for-byte architecture-identical finetunes of the
same Qwen3.6-27B base (hidden 5120, 64 layers, 24/4 GQA, head_dim 256, vocab 248320), so
Qwopus3.6's trained nextn head is dimensionally compatible. We graft it onto tmax's trunk
and keep it at Q8_0 while the trunk is 2–4 bit. Despite being a cross-finetune graft it
drafts at high acceptance (see the parent `README.md` §2.2 for the acceptance-vs-n table).

## How to use it

```bash
./llama-server \\
    --model ../tmax-27b-IQ4_XS.gguf \\
    --spec-type draft-mtp --spec-draft-n-max 1 \\
    --n-gpu-layers 999 --ctx-size 16384
```

- **`--spec-draft-n-max 1`** is optimal: one nextn layer (higher `n` drafts longer chains
  at falling acceptance).
- Speedup is GPU-bandwidth-dependent — largest on memory-bound CUDA GPUs, smaller on
  unified-memory/Metal.
- Requires a llama.cpp built after the 2026-06 `--spec-type draft-mtp` merge.
"""


_CALIB_README = """# Calibration data

Produced by `scripts/build_corpora.py` + `scripts/exp045_quants_tmax.py` under
`out/exp-045/corpora/`, reproduced here for transparency. Usage-log text is packed with
a **stub + multi-window** scheme (system-prompt prose trimmed to a stub, each session
sliced into windows of real tool-call turns) so the imatrix sees the agentic decision
channels rather than boilerplate.

| File | Role | Source |
|---|---|---|
| `corpus.cal.txt` | hybrid imatrix collection | all of `wiki.test.raw` + ~500k tokens from the logtrain **train** slice (windowed) |
| `corpus.eval.general.txt` | **§1 general columns** (PPL · KLD · top_p) | `combined_en_tiny` (broad English) from `eaddario/imatrix-calibration` |
| `corpus.eval.tools.txt` | in-distribution tools eval (quant-vs-quant) | logtrain **holdout** slice (10%), windowed like calibration but disjoint from it |

`corpora_audit.json` records token/session counts and per-source breakdown, and asserts
the logtrain train/test/holdout fingerprints are disjoint. Every eval corpus is disjoint
from calibration, so §1 measures generalization, not fit. `corpus.eval.tools.txt` is
chat-templated and `llama-perplexity` lacks `--parse-special`, so its KLD is for
quant-vs-quant comparison (all rows vs FP16 on the identical file), not absolute PPL.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", action="store_true",
                    help="force real file copies instead of hardlinks for the GGUFs")
    args = ap.parse_args()

    PKG.mkdir(parents=True, exist_ok=True)

    print("== staging GGUFs ==")
    for label, _quant, src_name, dest_name, _method in QUANTS:
        _stage_gguf(EXP45 / label / src_name, PKG / dest_name, copy=args.copy)

    print("== staging calibration_data ==")
    _stage_calibration_data()

    print("== staging MTP/README.md ==")
    (PKG / "MTP").mkdir(parents=True, exist_ok=True)
    (PKG / "MTP" / "README.md").write_text(_MTP_README)

    print("== rendering README.md ==")
    tmax = _rows_by_quant(RESULTS_CSV)
    qwo = _rows_by_quant(QWOPUS_CSV)
    swe = _swebench_pass()
    mtp_rows = _mtp_accept()
    for src, rows in (("results.csv", tmax), ("qwopus_compare.csv", qwo)):
        print(f"  {src}: {len(rows)} quant rows {sorted(rows)}")
    print(f"  swebench aggregated: {'present' if swe else 'MISSING (—)'}")
    print(f"  mtp acceptance sweep: {mtp_rows or 'MISSING (—)'}")

    (PKG / "README.md").write_text(_render_readme(tmax, qwo, swe, mtp_rows))
    # auditable dump of the numbers that went into the README
    (EXP45 / "release_metrics.json").write_text(json.dumps(
        {"tmax": tmax, "qwopus": qwo, "swebench": swe, "mtp_accept": mtp_rows},
        indent=2, default=str))

    print(f"\nDONE → {PKG}")
    print("  Review README.md (license caveat!), then push with")
    print("  PYTHONPATH=src .venv/bin/python scripts/exp045_upload_release.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
