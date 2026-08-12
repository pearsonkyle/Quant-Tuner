# New-session prompt — re-quant the gemma-4-31B release on the new windowed corpus

> Paste everything below the line into a fresh Claude Code session in this repo.
> It mirrors what we just did for exp-041 (Qwopus3.6), applied to the gemma-4-31B
> HF release. Drafted 2026-06-16.

---

## Goal

Rebuild **every GGUF in `uploads/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/`** from
scratch using the **new windowed calibration corpus** (the stub+multi-window packer
that became the default in `build_corpora.py` / `data.split.stratified_pack` on
2026-06-16), then repackage the upload dir and **push to HF after I confirm**.

The motivation: the corpus packer changed (tool-turn token share 88.9%, distinct
sessions up, system-prose boilerplate trimmed). Every imatrix and AWQ calibration in
the current gemma release predates it, so all 11 quants are stale and must be
re-derived against the new corpus. See memory `project_windowed_calibration_corpus`.

## Canonical artifacts (already in the repo — read them first)

- **Builder:** `scripts/exp034_release_v3.py` — reproduces the exact 11-row lineup
  shipped in the README (QAT IQ2_M is intentionally dropped — broken, PPL ~2e10).
  Rows: 4 vanilla baselines (Q2_K plain + IQ2_XS/IQ2_M/Q2_K_S imatrix), 3 vanilla
  AWQ cv-gate (IQ2_XS/IQ2_M/Q2_K_S), 2 QAT imatrix (IQ2_XS/Q2_K_S), 2 QAT AWQ cv-gate
  (IQ2_XS/Q2_K_S). AWQ params baked in: proxy auto (q2k_b16 pinned for IQ2_M),
  `proxy_mix=None`, `proxy_tokens=1024`, `ctx=4096`, `per_tensor_alpha=True`,
  `grid_radius=0.15`, `cv_strategy=gate`, `cv_weight=1.0`, **`rmsnorm_plus_one=False`
  (gemma!)**, `sanity_max_rel=1.20`. Do not change these — they're the tuned release
  settings (exp-026/027/030/031).
- **Input-builder template:** `scripts/run_exp020_awq_mmmu_validation.py` — contains
  the `build_corpora` + vanilla-imatrix + `baseline.kld` steps gemma uses. Corpus call:
  `cal_tokens=500_000, val_tokens=0` (MMMU only, no logtrain test slice),
  `eval_tokens_per_domain=30_000, seed=42, val_supplement_override=MMMU_VAL`,
  `imatrix ctx=4096`, `eval ctx=4096`.
- **MTP drafter:** `scripts/add_mtp_drafter.py` — downloads Unsloth's prebuilt
  `gemma4-assistant` drafter. **Corpus-independent — do NOT rebuild it.** The existing
  `mtp-gemma-4-31B-it.gguf` + `MTP/` in the upload dir stay as-is.
- **Release-asset patcher:** `scripts/update_release_assets_exp020.py` — shows the
  README §3 table format + how GGUFs/corpora get copied into the upload dir. Reuse its
  table-rendering approach; note it only handled the 3 awq-cv-gate rows, so the README
  update here is broader (all 11).

## What exp034 consumes but does NOT build (you must rebuild these first)

exp034 reads pre-built corpora + imatrices + baseline from exp-020/exp-022 — it will
**silently reuse the stale ones** unless you regenerate them. Current on-disk state:

| Input | Path | State |
|---|---|---|
| Vanilla F16 + HF | `out/exp-009/google__gemma-4-31B-it/` | **MISSING** (cleaned) — exp034 Phase 0 re-fetches ~60 GB + converts |
| QAT F16 + HF | `out/exp-022/google__gemma-4-31B-it-qat-q4_0-unquantized/` | present |
| Corpora (cal/val/eval) | `out/exp-020/google__gemma-4-31B-it/corpora/` | **STALE** (old packer) |
| Vanilla imatrix | `out/exp-020/.../imatrix-cal.gguf` | **STALE** |
| QAT imatrix | `out/exp-022/.../imatrix-cal.gguf` | **STALE** |
| baseline.kld | `out/exp-020/.../baseline.kld` | **STALE** |

## Step-by-step plan

**Phase 0 — preflight.** Confirm env (`.venv/bin/python`, run scripts with
`PYTHONPATH=src .venv/bin/python …` — `uv run` picks the wrong interpreter; see memory
`reference_local_venv`). Confirm the vendored llama.cpp build is **post-2026-06-07**
(needed for the `gemma4-assistant` MTP arch). Read `scripts/exp034_release_v3.py` end to
end before touching anything.

**Phase A — rebuild inputs on the new windowed corpus.** Delete the 5 stale inputs above
(corpora dir, both `imatrix-cal.gguf`, `baseline.kld`) so `experiments.step()` (existence-
based) actually regenerates them — this is the #1 gotcha. Then:
1. Get the vanilla gemma model back. Easiest: run exp034's Phase 0 (it fetches vanilla HF
   + converts to F16). The vanilla imatrix and baseline.kld both need that F16.
2. Rebuild the windowed corpora into `out/exp-020/.../corpora/`. `build_corpora.build()`
   defaults to the windowed packer now; reuse exp-020's exact call (cal 500k / val 0+MMMU
   / eval 30k-per-domain / seed 42). **Tokenizer:** pass a gemma HF dir as `model_dir`
   (the QAT HF at exp-022 has the same gemma tokenizer and is on disk — fine to use).
   This now also writes **two extra standalone eval corpora** (added 2026-06-16):
   `corpus.eval.general.txt` (external `combined_en_tiny`, broad-English) and
   `corpus.eval.tools.txt` (logtrain **holdout** slice, windowed with the same packer —
   the in-distribution tool-log eval `corpus.eval.txt` can't provide). Both are tunable via
   `--general-eval-tokens` / `--tools-eval-tokens` (default 30k). They are **separate** from
   `corpus.eval.txt` and benched independently (see Phase C). ⚠️ `corpus.eval.tools.txt` is
   chat-templated and llama-perplexity has no `--parse-special`, so use it for
   **quant-vs-quant** comparison only, not absolute PPL.
3. Rebuild the **vanilla** imatrix (on vanilla F16) and the **QAT** imatrix (on QAT F16),
   both on the new `corpus.cal.txt`, `ctx=4096`, `--parse-special` on (default — never
   disable it for chat corpora).
4. Rebuild **three** F16 baselines (vanilla F16, `ctx=4096`), one per eval corpus:
   `baseline.kld` ← `corpus.eval.txt`, `baseline.general.kld` ← `corpus.eval.general.txt`,
   `baseline.tools.kld` ← `corpus.eval.tools.txt`. (exp034 itself only consumes the first;
   the other two are for the Phase C independent benches.)
5. Sanity-check the new corpus: open `corpora_audit.json`, confirm tool-turn share is high,
   the `eval_general`/`eval_tools` entries are present, and timestamps are today.

**Phase B — re-quant the 11 rows.** Delete any stale exp034 output GGUFs + its `results.csv`
first (same step() gotcha). Then run `PYTHONPATH=src .venv/bin/python
scripts/exp034_release_v3.py` **in the background** (`nohup … &`, log to a file) — it's
**~15-18 h on Metal**, and each AWQ row auto-deletes its ~115 GB intermediates after bench.
Watch the log; bench rows dedup against the CSV so a killed run resumes.

**Phase C — independent KLD on the new eval corpora.** exp034 only benches each quant
against `corpus.eval.txt`. For the two new holdouts, run a **separate** bench pass per quant:
`corpus.eval.general.txt` vs `baseline.general.kld`, and `corpus.eval.tools.txt` vs
`baseline.tools.kld`. `runner.bench_one` takes a single `eval_dataset`/`eval_baseline`, so
either call it once per corpus or extend it to emit per-corpus columns. The **tools** KLD is
the key signal — it's the in-distribution windowed-packer A/B (quant-vs-quant only, since
llama-perplexity lacks `--parse-special`). Record these next to `results.csv`. Compare the
new IQ2_M/Q2_K_S tools-KLD against the old release's to decide whether the windowed corpus
actually earns the re-ship.

**Phase D — regenerate plots.** The upload dir ships `performance_comparison.png`,
`performance_scatter.png`, `mmlu_heatmap.png`, `radar_comparison_subset.png`,
`awq_cv_gate_release.png`. Re-run the relevant `scripts/plot_*.py` against the new
`results.csv` so the figures match the new numbers. (MMLU heatmap needs the MMLU-Pro eval
— check whether that needs re-running too, or is corpus-independent.)

**Phase E — repackage `uploads/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/`.**
- **Hardlink** (not copy — saves ~hundreds of GB) the 11 new GGUFs from their
  `out/exp-034/{vanilla,qat}/` paths into the upload dir, overwriting the old filenames.
  Match the exact published filenames (e.g. `gemma-4-31B-it-IQ2_M-awq-cv-gate.gguf`,
  `gemma-4-31B-it-qat-Q2_K_S-imatrix.gguf`, …). The old ones are hardlinks — `rm` the
  upload-dir copy first, then `ln` the new one.
- Refresh `calibration_data/{corpus.cal.txt, corpus.val.txt, corpus.eval.txt,
  corpus.eval.general.txt, corpus.eval.tools.txt, corpora_audit.json}` from the new corpora.
- Update `README.md`: the §3 comparison table with the new `results.csv` numbers, and the
  figures. Reuse `update_release_assets_exp020.py`'s table format; refuse to clobber if
  anchor strings can't be found.
- Leave the MTP drafter (`mtp-gemma-4-31B-it.gguf`, `MTP/`) untouched.

**Phase F — push to HF (after explicit confirmation).** The repo
`pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF` is **live**; the remote is untouched by local
work until pushed. Show me the diff of what will upload (new GGUF sizes/bpw, README table
delta) and **wait for my explicit "yes" before the actual `HfApi`/`upload_folder` push.**

⚠️ **Ollama tag caveat (carries over from the Qwopus repo):** HF builds Ollama quant tags
from the *terminal* token of each GGUF filename. The gemma files (`…-IQ2_M-awq-cv-gate.gguf`,
`…-qat-Q2_K_S-imatrix.gguf`, …) put the quant mid-name, so `ollama run hf.co/…:IQ2_M` 400s.
Unlike Qwopus, gemma has **multiple variants per quant** (imatrix vs awq-cv-gate vs qat), so
you can't just make the quant terminal without collisions. Don't silently rename — surface
this and let me pick the tag scheme (e.g. fold the variant into the terminal token like
`…-awq_IQ2_M.gguf`, or split into per-variant repos) before any push.

## Gotchas / invariants (don't relearn these the hard way)

- **step() idempotency is existence-based.** Any stale output that still exists is reused
  silently. Delete before re-running — this applies to corpora, imatrices, baseline.kld,
  and every GGUF + results.csv.
- **AWQ on gemma needs `rmsnorm_plus_one=False`** (it's already set in exp034). Gemma uses
  `(1+γ)` norms but the AWQ apply expects the opposite convention here; flipping it
  collapses every 2-bit row. Verify via apply-rel + GGUF bench, not the sanity check alone
  (memory `reference_awq_arch_settings`, `project_awq_gemma_findings`).
- **QAT IQ2_M is excluded** — don't try to "fix" it into the lineup.
- **GGUFs are hardlinked** between `out/` and the upload dir. `rm` only frees space when
  every link is gone; `find <vol> -inum <N>` to locate siblings before deleting.
- Slice/source invariant: logtrain `train`+wiki → cal; MMMU → val; external
  code/math/tools → `corpus.eval.txt`; external `combined_en_tiny` → `corpus.eval.general.txt`;
  logtrain `holdout` → `corpus.eval.tools.txt` (windowed) **and** the agentic sessions. Don't
  repurpose slices (CLAUDE.md "Slice / source → corpus mapping").
- Run units of work in the **background** and poll the log; the full re-quant is many
  hours.

## Done criteria

11 fresh GGUFs in the upload dir (hardlinked from `out/exp-034`), refreshed
`calibration_data/` (incl. `corpus.eval.{general,tools}.txt`), per-quant KLD on all three
eval corpora (with the tools-KLD A/B decision recorded), README table + plots reflecting the
new `results.csv`, MTP drafter unchanged, Ollama tag scheme resolved with me, and — after my
explicit confirmation — the HF repo updated.
