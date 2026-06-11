# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`quant-tuner` takes a HuggingFace model plus a corpus derived from real prompt/response
logs and produces a GGUF quantization calibrated to that distribution. It then benchmarks
the result against an FP16 reference (KLD, perplexity, prefill/decode tok/s) and against
task-level metrics (tool-call accuracy on held-out sessions).

Status: beta. The `quant-tuner` CLI is recipe-driven and runs end-to-end
(`run`, `bench`, `leaderboard` all wired). The orchestrator
`scripts/reproduce_leaderboard.py` chains the OmniCoder study's stages and is
the canonical entry point for reproducing the published table.

## Setup and common commands

```bash
# One-time: fetch + build vendored llama.cpp (pinned commit 9e58d4d6)
git submodule update --init --recursive
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build -DGGML_METAL=ON   # Linux+CUDA: -DGGML_CUDA=ON
cmake --build vendor/llama.cpp/build -j

# Python env
uv sync

# Tests / lint / types
uv run pytest                                   # all unit tests
uv run pytest tests/unit/test_imatrix.py        # single file
uv run pytest tests/unit/test_imatrix.py::test_name
uv run ruff check src tests
uv run mypy src

# CLI (recipe-driven)
uv run quant-tuner run --recipe q4_k_m_imatrix --model org/repo --logs logs.jsonl --workspace out/run
uv run quant-tuner run --recipe q4_k_m_imatrix --model X --logs Y --workspace W --dry-run  # validate-only
uv run quant-tuner bench --quant Q.gguf --reference F16.gguf --eval EVAL.txt --out results.csv
uv run quant-tuner leaderboard --results out/<run>/results.csv --out LEADERBOARD.md
```

Override the llama.cpp build location with `LLAMA_CPP_DIR=/path/to/llama.cpp` if not
using the vendored submodule. `paths.llama_bin(name)` resolves binaries from
`$LLAMA_CPP_DIR/build/bin/`.

## Architecture

### Pipeline
`pipeline.run_pipeline(RunConfig)` is the canonical end-to-end flow; `cli.run`
is a thin shim over it. Stages (each idempotent via `experiments.step`):

```
HF model + usage-log JSONL
   → pipeline.extract_and_convert      (HF → F16 GGUF, one-time)
   → pipeline.prepare_corpora          (data.ingest + data.split + stratified_pack)
   → pipeline.calibrate                (method-dispatched: imatrix/awq/gptq/none)
   → pipeline.quantize_model           (F16 → Q* GGUF via llama-quantize)
   → pipeline.bench                    (BPW + KLD + PPL + llama-bench → CSV row)
   → leaderboard.aggregate             (CSV → markdown with SQS scoring; separate CLI step)
```

Pipeline behaviors worth knowing:
- The output GGUF is named `gguf/{quantize.type}-{method}[-{variant}].gguf`. The
  variant is part of the filename **on purpose**: `step()` idempotency is
  existence-based, so without it a re-run with a different variant would skip
  llama-quantize and bench the stale file under the new label.
- `extract_text_lm` pass-through (plain CausalLM, no vision tower) symlinks the
  HF snapshot into `ws.model_extracted` — every downstream stage reads from
  there, never from the HF cache path directly.
- The **AWQ branch collects its imatrix on the folded F16** (`calibration/
  imatrix-awq.gguf`), not the original: folding rescales per-channel
  activations, so an unfolded imatrix would over-weight exactly the channels
  AWQ already boosted. `params.imatrix_variant: <variant>` adds a second pass
  that re-weights it (`imatrix-awq-<variant>.gguf`) — this is how AWQ and
  `hybrid_custom` are stacked in one recipe.
- The **GPTQ branch verifies PPL against a measured reference** (cached in
  `eval/baseline-ppl.txt`); `ppl_max_ratio` defaults are bits-aware (see below).

### Three calibration methods (all produce a standard GGUF; no runtime cost)
- **imatrix** (`calibrate/imatrix.py`): consumes a base `imatrix.gguf` produced by
  `llama-imatrix` (via `models.llama_cpp.imatrix`), then re-weights per-tensor using one of
  several **variants**:
  - `analytic`, `mix_50`, `hybrid_custom` — closed-form, no model load. `hybrid_custom`
    is the published winner; it combines `E[a²]` with `‖W[:,c]‖²·E[a²]` per-tensor.
  - `outlier_l4`, `outlier_max` — require an HF forward pass to capture heavy-tail stats.
  - SSM tensors (Mamba etc.) always pass through with raw `E[a²]` — see
    `models.hf_gguf_map.is_ssm`; output-aware re-ranking is invalid for them.
- **awq** (`calibrate/awq.py`): activation-aware scaling folded into RMSNorm.
  - The α-search **proxy quantizer auto-matches `quantize.type`**
    (`proxy_for_quant_type`): codebook-aware `iq2_xxs`/`iq2_xs`/`iq2_s` for IQ2_*
    targets, `q2k_b16` for Q2_K/IQ1, `q3k_b16` for 3-bit, else `int4_g128`.
    Recipes can pin one via `params.proxy`.
  - The IQ2 proxies snap groups of 8 weights to llama.cpp's **exact E8-lattice
    codebooks** (256/512/1024 entries for XXS/XS/S+M) with the real scale form
    `db = d·(0.5+q4)·0.25` and the even-negatives-per-group sign parity for
    XXS/XS. Grids live in the generated `calibrate/_iq2_grids.py` — do not edit;
    regenerate with `scripts/gen_iq2_grids.py` when bumping the llama.cpp pin.
    IQ2_M selects the `iq2_s` proxy (M = the S grid + an IQ3 tensor mix that
    llama-quantize decides per-tensor; the α search can't influence the mix).
  - IQ1/IQ2/Q2_K targets additionally get a **per-member proxy mix**
    (`params.proxy_mix`, pipeline-defaulted to `quantize.type`): llama-quantize
    bumps attn_v → Q4_K (GQA/MoE ≥ 4), attn_output → IQ3_S (IQ2_S/M) / Q3_K
    (Q2_K), and ffn_down a tier up for the first eighth of layers (every layer
    for Q2_K), so `proxy_for_member` scores those members with the proxy
    matching their *real* target instead of the ftype's base grid. Pinning
    `params.proxy` disables both the auto-selection and the mix (set
    `proxy_mix` explicitly to stack them). `iq2_m_awq` pins `proxy: q2k_b16`:
    pure-`iq2_s` scoring regressed IQ2_M top_p — the codebook's steep α
    penalty plus v_proj's fictitious 2-bit error (really Q4_K) dragged the
    shared group α down.
  - The α grid search runs **on the model device** (weights are not copied to
    CPU); cheap precondition checks (`cv_strategy` needs `holdout_text` and
    `per_tensor_alpha`, unknown `proxy`) fail before the model load.
  - The final llama-quantize imatrix is collected on the **folded** F16 and can
    be re-weighted with any imatrix variant via `params.imatrix_variant`.
- **gptq** (`calibrate/gptq.py`): Hessian-based rounding with error compensation; has a
  `verify_perplexity` guardrail. The rounding grid auto-matches `quantize.type`
  (`grid_for_quant_type`): 2-3-bit targets → **asymmetric min+scale** per-16 block
  (all `2^n` levels; a symmetric 2-bit grid has only 3 usable levels and destroys the
  model), 4-bit+ → symmetric g32. PPL/logit guardrails are auto-relaxed at 2-3 bits
  (`ppl_max_ratio` 4.0/2.0, `sanity_max_rel` 1.0/0.75). `gptq.apply` runs on CPU by
  design (Cholesky); calibration honors `device`.
- **Recipe param routing**: calibrate-stage and apply/fold-stage kwargs live in the
  same `calibration.params` dict but go to different functions — the pipeline splits
  them via `_AWQ_APPLY_PARAMS` (`rmsnorm_plus_one`, `sanity_max_rel`, `sanity_tokens`)
  and `_GPTQ_APPLY_PARAMS` (`n_bits`, `group_size`, `dampen`, `actorder`, `sym`, …).
  When adding a new calibrator kwarg, decide which stage owns it and update the
  corresponding tuple, or the recipe will crash with a TypeError.
- All calibrators default to `device: "auto"` (cuda → mps → cpu via
  `calibrate/_device.resolve_device`); recipes only need `device` to pin one.
- IQ1/IQ2 targets require an imatrix; `pipeline.quantize_model` rejects them under
  `calibration.method: none` up front.

### Tensor naming
`models/hf_gguf_map.py` maps HF parameter names to GGUF tensor names. Anything that
crosses the HF↔GGUF boundary (imatrix variants, AWQ apply) goes through this mapping.

### Bench
`bench/runner.py` defines `BenchRow` and `CSV_COLUMNS`. Sub-modules:
- `bpw.py` — bits-per-weight from `n_params(f16)` and file size.
- `kld.py` — `build_baseline(f16, eval_ds)` produces a reference KLD file via llama.cpp;
  subsequent runs diff against it.
- `speed.py` — wraps `llama-bench` for prefill/decode tok/s + TTFT with N repetitions
  (mean ± stdev).

### Task-level evaluation (`eval/`)
Task-level eval lives in `src/quant_tuner/eval/`. Every benchmark follows the
same shape: a pure scoring layer + an orchestrator that returns a dataclass
of float metrics + a CLI shim. The multi-rep runner (`eval.reps`) is
benchmark-agnostic — anything that reduces to `dict[str, float]` plugs in.

- `eval/server.py` — `running_server(model_path)` context manager spawns/health-checks
  /tears-down `llama-server` via `paths.llama_bin`. Used by every eval orchestrator.
  `wait_for_health` raises immediately if the server process exits before becoming
  healthy (bad GGUF, port clash) instead of burning the startup timeout.
- `eval/toolcall.py` — tool-call benchmark. `Sampling` dataclass,
  `eval_per_turn`, and `run_toolcall_eval(holdout, model_path=… | base_url=…)`
  returning an `EvalSummary`. Sampling params (`T`, `top_p`, `top_k`, `min_p`,
  `presence`, `repeat_penalty`, `seed`) ride through `extra_body`.
  `run_toolcall_eval` **raises** when 0 of N turns scored (dead/unreachable
  server) rather than returning an all-zero summary — don't "fix" that by
  catching it in a reps loop, the zeros would silently drag the mean down.
- `eval/scoring.py` — pure type-aware comparators for tool-call params
  (`compare_value`, `param_score`, `is_schema_valid`). Fully unit-tested.
- `eval/mmlu_pro.py` — MMLU-Pro few-shot benchmark.
  `run_mmlu_pro_eval(holdout, model_path=… | base_url=…)` returns an
  `MmluProSummary` (overall + per-subject accuracy). `parse_answer` extracts
  the predicted letter from completions: the **last** in-range match wins
  (reasoning models revise mid-chain-of-thought), and the bare words "A"/"I"
  are not accepted mid-sentence (only as a letter-only line) — preserve both
  properties when touching the regexes. `build_messages` renders the K-shot
  chat prompt (system → K (user, assistant) demo pairs → target user turn).
- `eval/reps.py` — generic N-rep runner. `run_reps_for_models(models, eval_fn,
  reps=10, sampling, base_seed=1000)` spawns one server per model, runs
  `eval_fn(base_url, sampling, rep_idx)` `reps` times (per-rep seed =
  `base_seed + rep_idx`), aggregates mean ± stdev across reps. CSV writers
  emit one row per (model, rep) and one row per model. Used by both
  `scripts/run_toolcall_reps.py` and `scripts/run_mmlu_pro_reps.py`.

### Experiment scripts (`scripts/`)
The OmniCoder reproduction is here; the CLI handles ad-hoc runs.
- `gen_iq2_grids.py` — regenerates `calibrate/_iq2_grids.py` (the IQ2 E8-lattice
  codebooks) from llama.cpp's `ggml-common.h`. Run after bumping the submodule pin;
  it asserts the ksigns parity convention and records the commit in the header.
- `build_corpora.py` — **canonical text-corpus builder**. One pass, one seed (42),
  three corpora written to `--out`:
  - `corpus.cal.txt` — ALL of `wiki.test.raw` + ~500k tokens from the logtrain **train**
    split (stratified-packed). Feed to `llama-imatrix` and `awq.calibrate(cal_text=…)`.
  - `corpus.val.txt` — ~10k tokens from the logtrain **test** split + `calibration_supplement.txt`.
    Feed to `awq.calibrate(holdout_text=…)` for cv-mixed / cv-gate scoring. The supplement
    is deliberately *under-represented content* (Rust etc.) so that an α candidate winning
    on val genuinely generalizes beyond the cal distribution — not just a re-draw of the
    same sessions.
  - `corpus.eval.txt` — ~30k tokens each (~90k total) sampled deterministically from the
    external `eaddario/imatrix-calibration` dataset: `{code_small, math_small, tools_small}`
    parquet files, cached under `out/external/imatrix-calibration/`. Feed to the bench
    `eval_dataset` for PPL/KLD vs FP16. **Eval is intentionally not derived from logtrain
    or wiki** — both appear in the calibration corpus, so PPL on them would conflate fit
    with generalization. External code/math/tools text gives a clean third distribution.
  Also writes per-domain intermediates (`corpus.cal.logtrain.txt`,
  `corpus.eval.{code,math,tools}_small.txt`) and `corpora_audit.json` (token counts,
  session counts, per-source breakdown). Asserts logtrain `train`/`test`/`holdout`
  fingerprints are disjoint before returning.

  The logtrain `holdout` slice (10%) is *not* used by this script — it remains reserved
  for the tool-call eval sessions written by `pipeline.py`/`build_toolcall_holdout.py`.

  **Prefer this over the older one-off corpus builders** (`run_omnicoder_mixed_corpus.py`,
  `build_holdout_chunk.py`) when standing up a new model — those scripts predate this
  and build single corpora with overlapping cal/eval distributions.
- `reproduce_leaderboard.py` — orchestrator chaining 7 stages (extract → 3 calibration
  stages → holdout → speed rebench → tool-call reps → render). Each subprocess-isolated.
- `run_omnicoder_{q4_k_m,wiki_vs_custom,mixed_corpus}.py` — the three calibration stages,
  using `experiments.step()` for idempotency.
- `build_toolcall_holdout.py` — samples the 25-session tool-call holdout from the
  `test + holdout` slices of `logtrain.jsonl`.
- `eval_toolcall.py` — thin argparse CLI over `eval.run_toolcall_eval`. Pass `--base-url`
  to reuse a server across calls.
- `run_toolcall_all.py` — single-rep eval across the 8 OmniCoder GGUFs.
- `run_toolcall_reps.py` — N-rep version on top of `eval.reps`. Flags:
  `--models X.gguf [Y.gguf …] --reps N --base-seed S --results ... --aggregated ...`.
- `build_mmlu_pro_holdout.py` — samples N test questions per subject from
  `TIGER-Lab/MMLU-Pro` and picks K shots from the dev split.
  Default: 25 × {`computer science`, `math`}, 2-shot, seed=42.
- `eval_mmlu_pro.py` — single-rep MMLU-Pro CLI.
- `run_mmlu_pro_reps.py` — N-rep version, same shape as `run_toolcall_reps.py`.
  Default sampling T=0.6 / top_p=0.95 / top_k=20; pass `--temperature 0` for
  greedy + deterministic.
- `rebench_speed.py` — re-runs speed bench only (decode tok/s drifts with thermal state
  across long sessions — see README caveat).

### Workspace layout
`paths.Workspace(root)` is the canonical per-run output directory; `workspace.ensure()`
creates `model_extracted/`, `corpus/`, `calibration/`, `gguf/`, `eval/` and reserves
`results.csv` at the root. Experiment scripts write to `out/<run-name>/` following this layout.

### Recipes
`src/quant_tuner/recipes/*.yaml` declare end-to-end recipes. Loaded via
`RunConfig.from_yaml` and executed by `pipeline.run_pipeline`.
`cli._resolve_recipe` accepts both bare names and absolute paths;
`PLACEHOLDER` fields are rejected with a clear "pass `--model` / `--logs`" error.
- 4-bit baselines: `q4_k_m_imatrix`, `q4_k_m_awq`, `q4_k_m_gptq`, `q4_k_m_none`.
- Low-bit presets (2-3 bpw): `q2_k_awq`, `iq2_xs_awq`, `iq2_m_awq` (AWQ +
  `hybrid_custom` imatrix stacked, codebook proxies auto-selected),
  `q2_k_gptq`, `iq3_s_gptq` (asym grid + relaxed guardrails auto-derived).
- Model-specific: `q4_k_m_qwen3_5_4b`, `{q4_k_m,q5_k_s}_qwen3_6_mtp{,_awq,_none}`,
  `iq3_s_9b_mtp` (MTP heads kept via `extract.keep_mtp`).
A unit test (`test_all_packaged_recipes_parse`) requires every shipped recipe to
validate, and IQ1/IQ2 recipes to carry a calibration method — keep it green when
adding recipes.

## Conventions worth knowing

- GGUF linear weights are stored `[n_out, n_in]`. Summing `W²` over axis 0 gives
  `‖W[:, c]‖²` per input channel — used throughout `calibrate/imatrix.py`.
- `_load_base_imatrix` divides `*.in_sum2` by `*.counts` to recover `E[a²]`; new variants
  should preserve this normalization.
- `step()` (`experiments/runner.py`) skips work when its output files exist — when
  changing what a stage produces, change the output *filename* too (see the
  quant-GGUF naming note above), or stale artifacts will be silently reused.
- `calibrate/_iq2_grids.py` is generated — never hand-edit; regenerate with
  `scripts/gen_iq2_grids.py`. The proxies' parity rule (even negatives per group
  of 8 for IQ2_XXS/XS) is asserted against llama.cpp's ksigns table at
  generation time.
- Device strings: pass `"auto"` (the default) unless a recipe must pin a backend;
  resolution lives in `calibrate/_device.resolve_device` (cuda → mps → cpu).
- **Special-token consistency on chat-log corpora** (the corpus is chat-templated,
  full of `<|im_start|>` etc.):
  - *Calibration* parses them correctly on both stacks: `llama_cpp.imatrix`
    defaults `--parse-special` on (never disable it for chat corpora), and the
    HF-side passes (AWQ/GPTQ hooks, outlier forward stats) get the same behavior
    because `transformers` encodes in-text special tokens as single special IDs
    by default — never pass `split_special_tokens=True`.
  - *PPL/KLD eval* cannot: llama-perplexity has **no** `--parse-special`
    (verified against the pinned llama.cpp), so chat markers tokenize as plain
    BPE. Quant-vs-quant comparisons on the same file stay valid, but absolute
    numbers are off-distribution. Prefer a raw-text eval corpus —
    `build_corpora.py`'s external `corpus.eval.txt`, wired into a recipe via
    `bench.eval_corpus` — over the pipeline's default log-derived eval slice.
- `calibration.params.imatrix_ctx` (default 512) sets the llama-imatrix context
  length for all three methods; it is consumed by the pipeline and not forwarded
  to the calibrators.
- HF calibration forward passes go through `calibrate/_hf.forward_no_logits`,
  which runs the decoder trunk only — hooks still fire, but the `[ctx, vocab]`
  logits tensor and lm_head matmul are skipped. Keep using it for new
  hook-based stat collection; the AWQ/GPTQ *sanity checks* intentionally call
  `model(...)` because they compare logits.
- Speed numbers have a known thermal artifact: rows that run later in a session drift
  lower. SQS (which weights speed equally with compression) is noisier than KLD; for
  "which imatrix is best?" read **KLD and tool-call** columns.
- Calibration `train`, eval `test`, and `holdout` slices come from the same source
  (`logtrain.jsonl`) but are disjoint — preserve this invariant when adding new evals.
- **Slice / source → corpus mapping** (used by `scripts/build_corpora.py` and the
  convention every new experiment should follow):
  - logtrain `train` (80%) + wiki → **calibration** corpus → imatrix + AWQ proxy loss
  - logtrain `test`  (10%) + `calibration_supplement.txt` → **validation** corpus → AWQ
    cv-mixed / cv-gate α scoring. The supplement injects under-represented content so
    val is a genuine distribution shift from cal, not a re-draw of the same sessions.
  - external `eaddario/imatrix-calibration` {code_small, math_small, tools_small} →
    **eval** corpus (PPL/KLD). Eval is **not** drawn from logtrain or wiki — those
    appear in cal and would conflate fit with generalization.
  - logtrain `holdout` (10%) → tool-call eval sessions (via `pipeline.py` /
    `build_toolcall_holdout.py`). Untouched by `build_corpora.py`.
  Don't repurpose a slice or source without re-checking every eval that touches it.
