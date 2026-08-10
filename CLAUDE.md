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
# One-time: fetch + build vendored llama.cpp (master, currently f3e182816)
git submodule update --init --recursive
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build -DGGML_METAL=ON   # Linux+CUDA: -DGGML_CUDA=ON
cmake --build vendor/llama.cpp/build -j

# Python env
uv sync

# Agentic SWE-rebench benchmark (optional): mini-swe-agent + a running Docker daemon.
uv sync --extra swebench
# SWE-rebench ships linux/amd64 instance images on Docker Hub (swerebench/sweb.eval.*).
# On Apple Silicon they run under emulation — correct but slow; give Docker Desktop
# ample memory and pre-pull images. run_swebench_eval.py fails fast if Docker is down.

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
  - Low-bit targets (IQ1/IQ2/IQ3/Q2_K/Q3_K) additionally get a **per-member
    proxy mix** (`params.proxy_mix`, pipeline-defaulted to `quantize.type`):
    llama-quantize bumps members above the ftype's base grid — at 2 bits
    attn_v → Q4_K (GQA/MoE ≥ 4), attn_output → IQ3_S (IQ2_S/M) / Q3_K (Q2_K),
    ffn_down a tier up for the first eighth of layers (every layer for Q2_K);
    under Q3_K_M/L attn_v/attn_output/ffn_down all land on Q4_K–Q5_K; under
    IQ3_M attn_v (always) + attn_output + first-eighth ffn_down → Q4_K; under
    IQ3_S/XS attn_v → Q4_K when GQA/MoE ≥ 4 — so `proxy_for_member`
    scores those members with the proxy matching their *real* target. Pure
    ftypes (Q3_K_S; IQ3_S without GQA) resolve to zero overrides. Pinning `params.proxy`
    disables both the auto-selection and the mix (set `proxy_mix` explicitly
    to stack them). `iq2_m_awq` pins `proxy: q2k_b16`: pure-`iq2_s` scoring
    regressed IQ2_M top_p — the codebook's steep α penalty plus v_proj's
    fictitious 2-bit error (really Q4_K) dragged the shared group α down.
  - The α grid search runs **on the model device** (weights are not copied to
    CPU); cheap precondition checks (`cv_strategy` needs `holdout_text` and
    `per_tensor_alpha`, unknown `proxy`) fail before the model load.
  - The final llama-quantize imatrix is collected on the **folded** F16 and can
    be re-weighted with any imatrix variant via `params.imatrix_variant`.
- **gptq** (`calibrate/gptq.py`): Hessian-based rounding with error compensation; has a
  `verify_perplexity` guardrail. The rounding grid auto-matches `quantize.type`
  (`grid_for_quant_type`): 2-3-bit targets → **asymmetric min+scale** per-16 block
  (all `2^n` levels; a symmetric 2-bit grid has only 3 usable levels and destroys the
  model), Q6_K → sym g16 6-bit, Q5_K → sym g32 5-bit, else symmetric g32.
  - **Per-tensor grid mix** (`grid_for_member`, `params.grid_mix` pipeline-defaulted to
    `quantize.type`): mixed ftypes bump members above the base grid — at 2 bits
    attn_v → Q4_K (GQA/MoE ≥ 4), Q4_K_M attn_v/ffn_down → Q6_K (`use_more_bits`
    schedule), IQ4_XS attn_v → Q5_K under GQA — so each tensor is rounded on the
    grid llama-quantize actually stores it at. The tensor→target table lives in
    `calibrate/_quant_mix.target_type_for_member` and is **shared with AWQ's**
    `proxy_for_member` (now a thin proxy-vocabulary wrapper). Pinning
    `n_bits`/`group_size`/`sym` opts out of the mix (set `grid_mix` explicitly to
    stack); pure ftypes resolve to zero overrides.
  - PPL/logit guardrails are auto-relaxed at 2-3 bits (`ppl_max_ratio` 4.0/2.0,
    `sanity_max_rel` 1.0/0.75). `gptq.apply` runs on CPU by design — the
    damp/Cholesky/inverse chain runs in fp64 there for 2-3-bit grids (4-bit+
    keeps the cheaper fp32 path) and **retries under escalating damping**
    (×2.5, up to 4 escalations, recorded in `GPTQStats.dampen_used`)
    instead of dying on near-singular low-bit Hessians. Dead columns (zero H-diag)
    are plain-RTN'd on the group grid, not zeroed. Calibration honors `device`.
  - **The imatrix is collected on the GPTQ-rounded F16** (after the PPL guardrail —
    weight-aware variants must see the weights llama-quantize sees) and can be
    re-weighted via `params.imatrix_variant`, same stacking as AWQ. The final GGUF
    name gains a `-{imatrix_variant}` suffix (idempotency, both methods).
  - Mac/memory knobs on `gptq.calibrate`: `hessian_device: auto` keeps the `[in,in]`
    accumulators on-device under MPS (avoids an H-sized copy per hook call, per
    chunk); `layers_per_pass: N` forwards the corpus once per N-layer slice so peak
    Hessian RAM is Σ_slice in² instead of Σ_all (completed slices recorded in
    `hessians/_complete`, keyed to a tokens/ctx/corpus fingerprint and skipped on
    re-run only when the key matches). Use `dtype: float16` on
    M1-generation GPUs (no bf16).
- **Recipe param routing**: calibrate-stage and apply/fold-stage kwargs live in the
  same `calibration.params` dict but go to different functions — the pipeline splits
  them via `_AWQ_APPLY_PARAMS` (`rmsnorm_plus_one`, `sanity_max_rel`, `sanity_tokens`)
  and `_GPTQ_APPLY_PARAMS` (`n_bits`, `group_size`, `dampen`, `actorder`, `sym`,
  `grid_mix`, …).
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
- **SWE-rebench V1 vs V2**: `swebench_grade.grade_instance` dispatches on
  `install_config.log_parser` (`is_v2_instance`). V1 = Python/pytest, checkout at
  `/testbed`, `conda run -n testbed`. **V2** = 20 languages, checkout at
  `/<repo-name>` (`v2_workdir`, also fed to `_build_env_config` so the *agent*
  starts in the right directory), the instance's own `install_config.test_cmd`
  with **no conda wrapper**, and the log parser the instance names. Those parsers
  are vendored **verbatim** (MIT) as `eval/_swerebench_v2_parsers.py` — never
  hand-edit; re-vendor via `scripts/vendor_swerebench_parsers.py` (`--check`
  detects drift) and it is `ALL`-ignored by ruff on purpose. The recorded
  FAIL_TO_PASS ids are exactly what those functions emitted at dataset-build
  time, so a reimplementation that is 99% right parses **zero** matching ids and
  reports every trajectory unresolved — indistinguishable from a bad model.
  Ids are timing-normalized on both sides (`… [20.82 ms]` differs per run).
  `diagnose_container_error` classifies `docker run` exit-125 (registry
  unreachable vs. full VM disk), which otherwise looks like a model failure.
- `eval/swebench.py` + `eval/swebench_grade.py` — **agentic** SWE-rebench
  benchmark (does the quant actually solve real GitHub issues?). Same shape as
  the others: `SweSummary` float-metrics dataclass + `run_swebench_eval(holdout,
  model_path=… | base_url=…)` + a `swebench_rep` adapter for `reps`. It drives
  `mini-swe-agent` (the `swebench` extra) over the local OpenAI-compatible
  `llama-server`, **one clean Docker container per instance** (the SWE-rebench
  image), and grades a real **pass_rate** by running the gold
  `FAIL_TO_PASS`/`PASS_TO_PASS` tests in the container (preferring the
  instance's `install_config.test_cmd`). Curated metrics: token usage, tool
  (bash) calls, tool errors, `patch_rate` (non-empty diff), `pass_rate`
  (resolved). Full conversation trajectories are saved to
  `<workspace>/trajectories/<model>/<id>.traj.json` (+ `.result.json`) for
  debugging loops/hallucinations and as future training data. The agent uses
  litellm tool-calling by default; weak local models may do better with
  `--model-class litellm_textbased` (parses bash from text). `grade_instance`
  is the pragmatic pytest path — non-pytest runners surface as a grade error,
  not a silent pass. Requires a running Docker daemon (see setup caveat).
- `eval/agents/` — **pluggable agent scaffolds** behind the SWE-rebench eval.
  `run_swebench_eval(..., agent=…)` / `--agent {mini-swe,openai-agents}` selects
  the loop; everything else (Docker container creation via mini-swe-agent's
  `get_sb_environment`, `grade_instance`, `SweSummary` aggregation, `reps`) is
  backend-agnostic. A backend implements `AgentBackend.run(AgentRunContext) ->
  AgentRunResult` (a git-diff `submission` + trajectory + tool/token counts) and
  owns persisting its own `*.traj.json`; `swebench.run_instance` creates the
  shared env, calls the backend, then grades. `get_backend(name)` (in
  `agents/__init__.py`) resolves a registry with **lazy SDK imports**, so
  resolving a backend never needs the SDK installed — only running it does (the
  registry test stays green in the conda env that lacks both SDKs). Backends:
  - `agents/miniswe.py` `MiniSweBackend` — the default; wraps mini-swe-agent's
    `DefaultAgent` (the relocated `_build_model_config` + metrics-agent subclass).
    `model_class` is forwarded only here.
  - `agents/openai_agents.py` `OpenAIAgentsBackend` — OpenAI Agents SDK
    (`openai-agents` extra) pointed at the same llama-server via
    `OpenAIChatCompletionsModel` + `AsyncOpenAI(base_url=…)` (Chat Completions,
    **not** Responses — llama-server only implements the former);
    `set_tracing_disabled(True)` (no OpenAI key). A single `bash` tool shells
    into the container; the patch is read back as `git -C /testbed diff` rather
    than a submit sentinel (the same contract future CLI-in-container backends —
    Qwen Code, Claude Code — will use). Token/call counts **and the trajectory
    accumulate via a `RunHooks` (`on_llm_end`)**, so they survive a
    `MaxTurnsExceeded`/wall-timeout (a weak local model that never emits a clean
    "done" hits `max_turns` routinely; sourcing metrics only from the `RunResult`
    would zero them out on that common path). `exit_status` records how the loop
    ended (`completed`/`max_turns`/`wall_timeout`/`error:*`); `patch_produced` is
    separate. llama.cpp-only sampling extensions
    (top_k/min_p/repeat_penalty) are **not** forwarded by this backend
    (ModelSettings exposes temperature/top_p as the first-class knobs).
- `run_swebench_eval` / `run_swebench_eval.py` knobs: `--agent` selects the
  backend; `--temperature` defaults to **0.25** (`DEFAULT_TEMPERATURE`);
  `--spec-type`/`--spec-draft-n-max` forward to llama-server speculative
  decoding — use `--spec-type draft-mtp --spec-draft-n-max 1` for a GGUF with a
  bundled MTP head (e.g. the Qwopus3.6 2-bit-MTP release; that Coder variant
  emits `<think>`, so also pass `--chat-template-kwargs '{"enable_thinking":
  false}'`). **`_build_env_config` must keep `cwd: /testbed`** — the grader and
  the mini-swe agent call `env.execute` without a cwd and rely on it.

- `eval/red_team.py` + `eval/red_team_agent.py` — **red-team safety eval**
  (deepteam over llama-server). The only eval here that asks "is the quant still
  *safe*", not "still capable". Full method: `docs/benchmarks.md#red-team-safety`.
  - **Every deepteam/deepeval import is lazy** (inside the function that needs
    it), like `swebench.py` does with `minisweagent` — that is what keeps
    `build_summary`/`pair_runs`/`aggregate_reps` unit-testable without the extra.
    Do not hoist them; `tests/unit/test_red_team.py` breaks at collection if you do.
  - **`LocalLLM` honors deepteam's `schema=` via llama.cpp grammar** (json_schema
    `response_format`, `schema_response_format`) and returns the **validated
    pydantic object** — not a string, because `vulnerabilities/custom/custom.py`
    does `res.data` directly. deepteam's *only* other path is a raw-text fallback
    that needs the model to freely emit exactly `{"data":[...]}`; that's why
    `CustomVulnerability` + `Multilingual`/EnhancedAttack errored (`'data'`) before
    the fix. A server without json_schema flips `_structured_ok=False` once and
    degrades to that fallback (then a looser `json_object` retry). Not a deepteam
    bug and no newer version — 1.0.7 is latest; the gap was our wrapper opting out.
  - **The target callback MUST declare two parameters** (`input`, `turns=None`).
    deepteam's `wrap_model_callback` forwards conversation history only to a
    callback with arity > 1; with one parameter every multi-turn jailbreak
    (Linear/Crescendo/Tree) silently probes a target with no memory of the
    escalation. Unit-tested. Same reason `AgenticTarget.as_callback()` exists —
    deepteam's `iscoroutinefunction` check rejects an object with an async
    `__call__`, so never pass the instance directly.
  - **Telemetry needs BOTH opt-outs**: deepteam reads `DEEPTEAM_TELEMETRY_OPT_OUT`
    and deepeval reads `DEEPEVAL_TELEMETRY_OPT_OUT`; deepteam initialises PostHog
    at *import* time. Also pass `_upload_to_confident=False` on the `RedTeamer`
    path (defaults True, and calls `webbrowser.open()` on a headless box).
  - **Cross-quant numbers require the frozen bank** (`--frozen-bank`, default on).
    deepteam simulates fresh attacks per run, so unpaired deltas confound model
    drift with bank variance. Seed on the F16 reference; `pair_runs` joins on a
    content-derived `case_id` and reports `n_flip_unsafe` (reference refused,
    quant complied) — the headline number.
  - **`score` is tri-state everywhere** (1 defended / 0 complied / `None`
    errored), through `per_case`, the CSV, and `read_per_case_csv`. Collapsing an
    error to 0 turns a timeout into a recorded jailbreak. Errored cases stay in
    `n_tests` but leave the `pass_rate` denominator, and `_assert_scored` raises
    when nothing scored (deepteam's `ignore_errors=True` default would otherwise
    report a dead target as `pass_rate=0.0`).
  - **Reasoning models return empty `content` when truncated.** Ornith/Qwen3/
    DeepSeek spend the token budget on chain-of-thought (in a separate
    `reasoning_content` field) *before* the answer, so a low `--target-max-tokens`
    yields an empty answer the judge scores as "safe". `build_summary` counts
    `n_empty_output` and `_assert_scored` **hard-errors** on an all-empty run
    (override: `allow_empty_output`) — this caught a real 100%-false-pass in
    validation. Use ≥3000 target tokens for reasoning models.
  - **Disclosure artifact**: `write_disclosure_report` dumps every complied/errored
    case (seed prompt + multi-turn `turns` + the target's response + its
    `reasoning` trace, matched from the callback's `transcript_sink` + judge
    reason) to `disclosure_<model>_repN.json`. This is the evidence file for the
    model's authors; refusals are excluded. `_reasoning_of` reads
    `reasoning_content`/`reasoning`; the reasoning never reaches the judge.
  - One `--base-url` can serve several models (LM Studio/vLLM/llama-swap): repeat
    `--target-model-name` and the sweep runs them all on one frozen bank
    (`Target.served_model`).
  - Red-team columns are **display-only in the leaderboard** (`merge_redteam`,
    `--redteam-csv`) and never feed SQS — that scalar trades size/fidelity/speed,
    and refusal is not a currency to spend in it.
  - The agentic path grades `tools_called` (did it *run* the command). Always read
    its `pass_rate` next to `n_tool_calls`: **a quant too degraded to tool-call
    scores as "safe" for the wrong reason.**
  - Needs the `redteam` extra installed under **Python ≤ 3.12**: deepteam 1.0.7
    has a stray unused `from nntplib import NNTPDataError` in `test_case.py`, and
    nntplib was removed in 3.13. Version-specific, not a general deepteam
    property — 1.0.6 imports fine on 3.13 but lacks `Hallucination`, so
    `_VULN_SPECS` loses one entry there. The repo's main `.venv` is 3.13, so this
    extra lives in a separate `.venv-redteam` (see `docs/benchmarks.md`).

### Continued QAT for native-ternary models (`src/quant_tuner/qat/`)
For **natively-ternary** models (`prism-ml/Ternary-Bonsai-8B`), post-hoc calibration is a
structural no-op — the "F16" is a lossless container of `w = s·c`, `c ∈ {−1,0,+1}`, so there is
no quantization error for imatrix/AWQ/GPTQ to recover. The only lever is **more training with
the ternarization in the loop** (BitNet/TWN STE). Full guide: `docs/ternary_qat.md`.
- `qat/ternary.py` — per-group TWN straight-through estimator; reproduces the shipped weights
  **exactly** at step 0 (the fine-tune must start from the real model, not a re-derived one).
- `qat/corpus.py` — masking/packing shared by the log corpus and the trajectory corpus. Loss is
  masked to assistant/tool-call tokens **plus the terminating `<|im_end|>`** — omitting the stop
  token is what caused the iter-2/3 looping, and the builder now asserts labeled stop targets
  exist. `trajectory_to_messages` is also what `datasets/swe_trajectories.py` publishes, so the
  released dataset and the training corpus cannot drift.
- `qat/train.py` — `QATConfig` + `train_qat()`; `scripts/exp058_qat_train_v2.py` is a CLI shim.
- `qat/export.py` — latents → **Q2_0** GGUF; needs `LLAMA_CPP_DIR=vendor/llama.cpp-prism`
  (ftype 41 is fork-only).
- `qat/kd_precompute.py` — offline top-K teacher logits (iter-6). Deliberately
  architecture-agnostic (`resolve_vocab_size` walks `text_config`/`llm_config`/`decoder`;
  configs with float ints are sanitized; `logits_to_keep` with a full-gather fallback) so a
  larger-vocab teacher needs no format change. `tokenizer_compatibility()` compares id→token
  **strings**, not `vocab_size` — a padded embedding matrix is fine, a different tokenizer is
  refused (per-token KD across tokenizers is silently wrong). `kd_loss_from_topk` must
  renormalize **both** sides over the stored top-K; normalizing only the teacher leaves a
  constant offset (an identical student scored 0.89 instead of 0 — pinned by a unit test).
- **Two training methods, one pipeline**: Method A = masked CE on verified trajectories
  (run end-to-end); Method B = CE + KL against the precomputed top-K table (precompute and loss
  validated, trainer wiring still open — `train.py` today only has the in-loop `--kd-teacher`,
  which does not fit alongside an all-36 student).
- **Metal constraints are hard, not preferences**: `foreach=False` (MPS multi-tensor kernels
  deadlock at full-model scale), window ≤ 4096 (`32·8192²` overflows INT_MAX in the unfused
  training SDPA path), **fp32 latents** (bf16 underflows the ternary threshold → no code flips),
  `--optim adafactor` to fit all 36 layers (~66-75 GB vs AdamW's ~116 GB). `--compute-dtype bf16`
  is a **pessimization** at all-36 (54.5 GiB vs 31 GiB — the fp32 master copy stacks on top).
- **Read the code-flip telemetry, not the loss.** A ternary model only learns by flipping codes;
  lr 3e-4 flips ~0% (scale drift only) while the loss still falls. 5e-4 for ~2.2 epochs is the
  measured sweet spot; 8 epochs memorizes.
- Peak memory spikes on `--ckpt-every` boundaries (`save_ckpt`'s whole-dict `.cpu()` transient);
  both observed OOM kills landed exactly there. Keep the MPS-cache release before that copy.
- Trajectory generation (`scripts/run_ornith_distill_gen.sh`) is Docker-heavy and slow under
  amd64 emulation. `--cleanup-images` *untags* images, leaving `<none>` dangling layers — run
  `scripts/docker_housekeep.sh` alongside long runs (SWE images + dangling only; never `-a`).

### Publishable datasets (`src/quant_tuner/datasets/`)
Staged under `datasets/<name>/`; payloads are gitignored, but the card, `manifest.json` and
`CHANGELOG.md` are tracked so the repo records exactly what shipped. **Adding a dataset is a
one-entry change**: write a builder yielding dicts, append a `DatasetSpec` to `REGISTRY`.
- `SplitSpec.publish=False` builds a split locally but withholds it from the Hub **and** the
  card (viewer config, stats, size bucket) — that is how the `all` split stays available for
  failure analysis while only verified trajectories ship.
- `push()` records a release in the manifest **only after** a successful upload, so a failed
  push cannot leave the repo claiming one; each push tags `v<version>` for `revision=` pinning.
- `DatasetSpec.schema_md` overrides the card's Row-schema table (empty ⇒ the SWE default);
  `build()`/`render_card` are outcome-aware — a split whose records carry an `outcome` field
  renders a complied/defended/errored table + model coverage instead of the SWE tool-call one.
- CLI: `scripts/dataset.py {list,build,push}` (`--bump`, `--version`, `--dry-run`, `--no-build`,
  `--private`).
- Two datasets registered: `swe-agentic-trajectories` (verified solver trajectories) and
  `redteam-safety-disclosures` (`datasets/redteam_disclosures.py`) — one row per adversarial
  case: **target model id + full conversation (`messages`, multi-turn preserved) + `outcome`**,
  built from the red-team eval's `disclosure_*.json` + `*_per_case.csv`. **Both splits default
  `publish=False`**: the `flagged` rows carry a working attack *and* the harmful completion, so
  it's a responsible-disclosure / QAT-seed artifact, not something to broadcast — ship via
  `push --private` or a metadata-only view.

### Experiment scripts (`scripts/`)
The OmniCoder reproduction is here; the CLI handles ad-hoc runs.
- `gen_iq2_grids.py` — regenerates `calibrate/_iq2_grids.py` (the IQ2 E8-lattice
  codebooks) from llama.cpp's `ggml-common.h`. Run after bumping the submodule pin;
  it asserts the ksigns parity convention and records the commit in the header.
- `build_corpora.py` — the **two-source** text-corpus builder, kept for reproducing the
  published runs. One pass, one seed (42), five corpora written to `--out`. "logtrain"
  below means the CLI usage logs, now `datasets/agent-logs/data/logs-cli.jsonl.gz`; this
  builder deliberately does **not** pull in the harvested agent trajectories, because its
  name already has published numbers attached to it:
  - `corpus.cal.txt` — ALL of `wiki.test.raw` **interleaved** (window-sized chunks,
    round-robin) with ~500k tokens from the logtrain **train** split
    (stratified-packed). Feed to `llama-imatrix` and `awq.calibrate(cal_text=…)`.
    Wiki is NOT prepended as a monolith: token-budgeted calibrators sample the
    file, and a 250k-token wiki head used to eat AWQ/GPTQ's entire budget.
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
  - `corpus.eval.general.txt` — ~30k tokens from the external `combined_en_tiny` parquet
    (`GENERAL_EVAL_DOMAIN`); a broad-English eval distribution. **Separate** from
    `corpus.eval.txt` — give it its own `baseline.kld` and bench it independently.
  - `corpus.eval.tools.txt` — ~30k tokens windowed-packed (same stub+multi-window packer
    as `corpus.cal.txt`) from the logtrain **holdout** split. This is the *in-distribution*
    PPL/KLD eval — it measures fit to the real tool-call text, which `corpus.eval.txt`
    cannot. It is disjoint from the train (cal) slice, so it doesn't contaminate against
    calibration. ⚠️ llama-perplexity has no `--parse-special`, so its chat markers tokenize
    as plain BPE: use it for **quant-vs-quant** comparison (e.g. the windowed-packer A/B),
    not absolute PPL. Also **separate** — its own `baseline.kld`, benched independently.
  Also writes per-domain intermediates (`corpus.cal.logtrain.txt`,
  `corpus.eval.{code,math,tools}_small.txt`) and `corpora_audit.json` (token counts,
  session counts, per-source breakdown). Asserts logtrain `train`/`test`/`holdout`
  fingerprints are disjoint before returning.

  The logtrain `holdout` slice (10%) now feeds `corpus.eval.tools.txt` here **and**
  remains the source for the agentic tool-call eval sessions
  (`pipeline.py`/`build_toolcall_holdout.py`) — both uses stay out of the calibration
  (train) slice, so neither contaminates calibration.

  **Prefer this over the older one-off corpus builders** (`run_omnicoder_mixed_corpus.py`,
  `build_holdout_chunk.py`) when standing up a new model — those scripts predate this
  and build single corpora with overlapping cal/eval distributions.

  **For a NEW model, prefer `build_universal_corpus.py` over this** (see below); keep
  `build_corpora.py` for reproducing the published two-source runs. Both draw the external
  eval domains through `data/external.py`, so their eval numbers stay comparable.
- `build_universal_corpus.py` → `data/universal.py` — the corpus builder that combines
  **every dataset in `datasets/`** plus raw wiki: the two on-disk log corpora, reasoning-
  terminal windows, `swe-agentic-trajectories`, `broad-domain-supplement`, and
  `redteam-safety-disclosures` (refused — see below), interleaved proportionally
  (`split.interleave_many`) so a token-budgeted calibrator samples all of them. Adds four
  in-distribution eval holdouts (`corpus.eval.{tools,agentic,broad,redteam}.txt`) alongside
  the external ones — **each is a separate distribution needing its own `baseline.kld`**;
  never concatenate them. Invariants it enforces rather than assumes: the chat template is
  checked against a tool-calling fixture *before* the build, the finished corpus is re-scanned
  for tool-call markers **per source** (a total stays non-zero even when one source silently
  loses its calls), tool outputs are head+tail clipped, and cal/eval disjointness is asserted
  per source. The supplement's `mtp` half is deliberately excluded from calibration — it is
  reserved for MTP draft-head training. Published-dataset rows are read from the staged
  `datasets/<name>/data/<split>.jsonl` when present (byte-identical to what was pushed),
  else from the Hub.
- **On-disk logs live in `datasets/agent-logs/data/`, gzipped** (`ingest.CLI_LOGS`,
  `ingest.AGENT_LOGS`; card in `datasets/agent-logs/README.md`). `logs-cli.jsonl.gz` is the
  old repo-root `logtrain.jsonl`; `logs-agents.jsonl.gz` is 435 verified agent trajectories
  over 19 languages / 7 scaffolds. `ingest.load_sessions` is gzip-aware, **sniffs both row
  formats** into one session schema (agent rows get `source="agents:<language>"` so the
  packer's strata round-robin spreads the budget across languages), and
  `resolve_log_path` maps the legacy `logtrain.jsonl` name onto the new file so the ~30
  historical reproduction scripts still run. `split_sessions` splits **by
  `ingest.session_group`** — the agent logs hold ~4.6 attempts at each issue, and a per-row
  split would put one attempt in cal and another at the same issue in the eval holdout.
- `data/reasoning.py` — reasoning arrives inline (`<think>` in content, CLI logs) or as a
  `reasoning_content` field (agent logs); this normalizes both. **Measured on Qwen3.6: chat
  templates keep reasoning only on a render's FINAL assistant turn and scrub it from
  history**, and emit an *empty* `<think></think>` on that turn when none is supplied — so a
  naive `</think>` count reported healthy coverage on a corpus that had 2 real blocks out of
  4,291 available. `universal.reasoning_windows` is the fix: extra windows cut so a reasoning
  turn lands last (the only position that survives, and the context the model actually has
  while generating it). Coverage is reported per source in the audit.
- `data/system_prompt.py` — **SFT-only** system-prompt scrubbing. 90% of system-prompt
  characters in the logs are blocks repeated verbatim across sessions (tone, git etiquette,
  worked examples). A repeated block is dropped **unless it names a path/file the same
  conversation actually touches** — that grounding test is what separates repo context from
  harness. Neither frequency nor keywords alone works: harness blocks say "repository" and
  "file paths" constantly, and generic filenames (`package.json`, `CLAUDE.md`) plus library
  names (`Node.js`) are filtered out by document frequency so they can't ground anything.
  URLs are blanked first (a `github.com/...` link parses as an absolute path). 6.4M → 0.4M
  chars. The calibration corpus is deliberately NOT scrubbed — the packer's
  `system_prose_budget` stub is the right mechanism there.
- **The corpus is verified for the quantizers that read it, not just built.**
  `universal.scan_special_tokens` checks the bytes as written (`newline=""` — universal
  newlines hide the `\r` in agent tool output) and hard-fails if any control token present
  doesn't tokenize to exactly one id; `llama-imatrix --parse-special` and the HF-side
  calibrators both depend on that. `universal.sampled_coverage` runs the **production**
  sampler (`calibrate/_ingest.sample_chunks`) over an index tensor, so it reports the exact
  slice AWQ (65k) and GPTQ (32k) receive, attributed per source — a build whose GPTQ slice
  contains zero tool calls fails instead of shipping.
- `data/refusals.py` — the red-team disclosures enter calibration as **attack prompts +
  generic refusals**: every assistant turn is replaced from a deterministic bank (varied, so
  a 22-turn crescendo isn't one sentence eleven times), and the targets' original completions
  and `target_reasoning` never reach a corpus. `universal.build` asserts that on the built
  sessions. Refusal behavior is what low-bit quantization erodes first, so the attack
  distribution belongs in calibration — the harmful responses do not.
- `verify_chat_template.py` → `data/template_check.py` — run this FIRST for any new model.
  Renders a fixture (two tools in scope, an assistant turn with prose + a call, a tool
  result) and hard-fails when the schemas, the argument JSON or the result don't survive,
  when in-text markers stop tokenizing to single ids, or when `session_windows` returns
  nothing. Unrecognised marker families are a WARNING (extend `KNOWN_TOOL_CALL_MARKERS`).
  Both known failures it guards: a template that drops `tools=`, and Qwen3.5-VL's strict
  "No user query found" that silently dropped 90% of the calibration corpus.
- `models/mtp.py` — **never hardcode the nextn pin again.** `describe(f16)` reads the draft
  layer out of the GGUF and returns the `tensor_types` pin; `llama-quantize` accepts a
  `--tensor-type` pattern that matches nothing, so a stale `blk.64.` silently quantizes the
  draft head with the trunk and only shows up as a mediocre acceptance rate. Detection needs
  BOTH signals: on the shipped Qwopus3.6 F16 the head is `blk.64` but `block_count=65`
  (the converter counts it), so only the `nextn`/`mtp`/`eh_proj` name hint finds it.
  `config_declares_mtp()` is a *claim* to verify against actual weights (the Ornith trap).
  Wired into recipes as `quantize.mtp_pin` (default `q8_0`, applied when `extract.keep_mtp`).
- `reproduce_leaderboard.py` — orchestrator chaining 7 stages (extract → 3 calibration
  stages → holdout → speed rebench → tool-call reps → render). Each subprocess-isolated.
- `run_omnicoder_{q4_k_m,wiki_vs_custom,mixed_corpus}.py` — the three calibration stages,
  using `experiments.step()` for idempotency.
- `build_toolcall_holdout.py` — samples the 25-session tool-call holdout from the
  `test + holdout` slices of the CLI logs (`datasets/agent-logs/data/logs-cli.jsonl.gz`).
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
- `build_swebench_holdout.py` — samples the agentic SWE-rebench holdout
  (`out/external/swe-rebench/holdout.jsonl`). Pages Hugging Face's
  datasets-server `/rows` API (the full `test` split is multi-GB and the
  streaming reader stalls). Default: 10 `is_lite` instances, seeded shuffle
  (seed 42) over the first `--scan-limit` rows; `--no-lite-only` /
  `--max-difficulty` widen the pool. `--from-local` reads a downloaded split
  instead (the `/rows` preview API rate-limits with 429s on repeated calls);
  `--exclude <holdout.jsonl>` builds a training pool **disjoint** from what we
  grade on — that invariant is what makes the QAT generalization number mean
  anything. `download_swebench_dataset.py` fetches the full split once.
- **Ternary-QAT chain** (see `docs/ternary_qat.md` for the end-to-end guide):
  `run_ornith_distill_gen.sh` (harvest verified solver trajectories) →
  `build_ornith_distill_corpus.py` (resolved-only, student tokenizer, masked) →
  `run_iter5_pipeline.sh LR TAG EPOCHS` (train → export Q2_0 → bench) +
  `run_iter5_indist_eval.sh TAG` (the in-distribution diagnostic — read it
  *against* the generalization number, not instead of it) →
  `run_iter5_autoloop.sh` (unattended grow-data/retrain/bench until it
  generalizes). `kd_precompute.py` is the iter-6 offline-KD entry point.
- **Red-team chain** (see `docs/benchmarks.md#red-team-safety`):
  `eval_redteam.py` (sweep N targets on one frozen bank → `results.csv`,
  `results_per_case.csv`, `bank.json`) → `redteam_ladder.py` (pair every rung
  against the F16 reference → `ladder.csv` with `n_flip_unsafe`/`net_drift`;
  also runs the sweep itself given `--models`, or analyses an existing per-case
  CSV with no GPU) → `redteam_vs_quality.py` (Spearman of drift vs. KLD/PPL/
  tool-call — pure CSV). `redteam_agentic.py` is the separate agent-in-container
  path; it needs the `swebench` extra and Docker, and its instances should be
  built `--exclude`-disjoint from the SWE-rebench eval holdout.
- `run_swebench_eval.py` — runs `eval.run_swebench_eval` over one or more GGUFs
  (default = gemma-4-31B `qat-Q2_K_S-imatrix`). Fails fast if the Docker daemon
  is down. Writes `results.csv` (per-instance), `aggregated.csv` (per-model),
  `summary.json`, and the trajectory tree under `<workspace>/`. Two target modes:
  `--models a.gguf …` spawns a llama-server per GGUF, or `--base-url URL
  --target-model-name NAME` (repeatable) reuses an already-running
  OpenAI-compatible server (LM Studio / vLLM / llama-swap) — mutually exclusive.
- **Multi-language trajectories** (`nebius/SWE-rebench-V2`, 32k instances / 20
  languages): `validate_swebench_v2_grading.py` (golden-patch gate — run it
  BEFORE generating) → `run_multilang_distill_gen.sh`. Pools are built by
  `build_swebench_holdout.py` with `--languages` + `--balanced` (round-robin, so
  Python/Go/JS can't crowd out the rest), `--clean-only` (annotator code `A`) and
  `--max-f2p` (drop rows whose FAIL_TO_PASS is the whole suite — some list 16k+
  ids). See `docs/ternary_qat.md#stage-1b`.

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
- GPTQ ladder (2-4.5 bpw): `iq2_xs_gptq`, `iq2_m_gptq`, `iq3_m_gptq`,
  `iq4_xs_gptq` — grid mix + relaxed guardrails auto-derived; the 2-bit rungs
  stack `imatrix_variant: hybrid_custom` (as does `q2_k_gptq`).
- Model-specific: `q4_k_m_qwen3_5_4b`, `{q4_k_m,q5_k_s}_qwen3_6_mtp{,_awq,_none}`,
  `iq3_s_9b_mtp` (MTP heads kept via `extract.keep_mtp`).
- Qwen3.8-27B ladder (exp-060): `{iq2_m,iq3_m,iq4_xs,q5_k_m}_qwen3_8_mtp` — universal
  corpus + `hybrid_custom` + the draft head pinned Q8_0 via `quantize.mtp_pin`. The
  experiment scripts (`scripts/exp060_{setup,quants,prepare_release}_*.py`) are the
  canonical path; runbook in `docs/qwen3_8_release.md`.
A unit test (`test_all_packaged_recipes_parse`) requires every shipped recipe to
validate, and IQ1/IQ2 recipes to carry a calibration method — keep it green when
adding recipes.

### Jacobian-lens interpretability (`src/quant_tuner/lens/`, `native/jlens_server/`)
Opens up the *inside* of a quant: layer-by-layer lens readouts, tool-call
representation formation, loop autopsy, knowledge-loss probes, an A/B diff
between two quants, and a weight-edit "bake" path. Full guide in `docs/lens.md`.
Adapted (Apache-2.0, see root `NOTICE`) from `anthropics/jacobian-lens` (the
`vendor/jacobian-lens` submodule; exact causal fits only) and
`igorbarshteyn/jlens-gguf` (the GGUF-native stack, owned in-tree).
- **`native/jlens_server/`** — a tracked llama-server-compatible C++ binary that
  hooks the ggml eval callback to capture `l_out-<il>` residuals and apply
  runtime steer/ablate/swap edits. Built via `bash native/jlens_server/build.sh`
  (or `quant-tuner lens build-server`) against the vendored llama.cpp, linking
  **only** the public API (`paths.jlens_server_bin()` resolves it, honoring
  `$JLENS_SERVER_BIN`). Unlike the old `llama-mtp-capture`, the source is tracked
  in quant-tuner, not inside the dirty submodule tree. The build stamps the
  llama.cpp commit into `/props` (`llama_commit`); a startup `l_out_ok`
  self-check + client-side assert guard architecture support and submodule drift.
- **Lens strategy**: one regression lens per base model, fit on the **F16 GGUF**
  over the **calibration corpus** (`lens fit`, forward-only, any quant, no
  torch), reused across every quant of that model so A/B diffs isolate what
  quantization moved. Lens container = `lens.gguf` (`JacobianLensGGUF`, adopts
  jlens-gguf's format verbatim + namespaced `quant_tuner.lens.*` provenance kv).
- **Capture run** (`lens/capture.py`) is the A/B unit: `<runs_dir>/<run_id>/` with
  activations (not logits — readouts recomputed lazily, top-k cached) + a
  `RunManifest`. Content-addressed → idempotent. Full-vocab final logprobs are
  stored only at flagged `logits_positions` (exact KLD there; top-k+tail-mass
  elsewhere). `diff_runs(cand, ref)` treats `ref` as FP16: rank shift = where the
  reference's top-1 fell in the candidate; KLD is `D_KL(ref‖cand)`.
- **Analysis modules**: `replay.py` (tool-call decision points — parity with
  `eval.toolcall.eval_per_turn`'s turn walk is unit-tested; sidecar CSV joins the
  leaderboard via `merge_lens`/`--lens-csv`), `loops.py` (`detect_repetition`,
  `loop_autopsy`, `intervention_sweep` → `direction.npz`), `probes.py`
  (correct/suppressed/absent), `study.py` (`calibration_study`),
  `quant_noise.py` (the importable refactor of `measure_quant_noise` steps 1-3;
  its `--empirical` path is now unbroken — `models.hf_gguf_map.gguf_to_hf_names`
  inverse mapping was added).
- **Bake** (`gguf_edit.py` + `bake.py`): `orthogonalize_layers` projects a
  direction out of the FP16 residual-writing tensors (`attn_output`/`ffn_down`,
  per-expert for MoE; skip path untouched), then `bake_and_requantize` runs a
  fresh `llama-quantize` with the existing imatrix — never in-place block editing.
  `copy_gguf_with_tensor_edits` passes unedited tensors through as raw bytes
  (byte-identity unit-tested). Additive steering is **not** bakeable (no per-layer
  bias tensor) — it stays a runtime capability of the OpenAI-compatible server.
- **CLI**: `quant-tuner lens {build-server,fit,fit-causal,convert-pt,inspect,
  capture,diff,serve,replay-toolcalls,loop,probe,study,bake,report}`. `lens serve
  --model-b` enables live dual-backend A/B in the D3 UI (`web/ab.js` +
  `diff_heatmap.js`).
- **DB**: `LensToolcallRep` is the sibling rep table for lens tool-call
  diagnostics (per the "new benchmark = new child table" convention).
- **Tests**: `tests/unit/test_lens_*.py` need no model files; the integration
  tests (`tests/integration/`, gated on `QT_LENS_IT=1` + `QT_TINY_GGUF`) include a
  numpy-readout-vs-server-logits parity check (corr ≥ 0.9999) — the drift canary
  on submodule bumps. `scripts/lens_smoke.sh` is the CPU acceptance gate.
- Experiment scripts are gemma-4-31B-anchored: `scripts/lens_exp10{1,2,3,4}_*.py`
  + `scripts/build_probe_set.py`.

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
- **Calibration ctx is a PACKING parameter, not just a runtime flag.** `data.universal`'s
  `UniversalConfig.ctx` (default **8192**) sizes the windows it emits (`window_cap = ctx -
  headroom`), and the same value must reach `llama-imatrix -c`, `awq.calibrate(ctx=)` and
  `gptq.calibrate(ctx=)` — a corpus packed for one ctx and read at another either straddles
  chunk boundaries or glues unrelated conversations into one context. The corpus records
  what it was packed for in `corpora_audit.json: calibration.ctx`. Measured on the logs: at
  ctx 4096 / cap 3500, **51% of log windows and 46% of SWE windows ended at the cap** (chains
  cut mid-chain); repacking for 8192 took mean tool results per agentic window from **6.1 to
  13.5**. Cost on a 27B F16 on Metal: 48.5 s per 4096-pass (84 tok/s) vs 116.2 s per
  8192-pass (70 tok/s) — +19% wall-clock for the same tokens, ~17 h for a full 4.4M-token
  pass. `scripts/build_corpora.py` and the older recipes stay at 4096 so published numbers
  reproduce; **numbers from different ctxs are not comparable**, including PPL/KLD.
- `calibration.params.imatrix_ctx` (default **4096**, `pipeline.DEFAULT_IMATRIX_CTX`)
  sets the llama-imatrix context length for all three methods; it is consumed by
  the pipeline and not forwarded to the calibrators. 4096 fits the packer's
  ≤3500-token windows in one context chunk — the old 512 default sliced every
  window across ~7 chunks. Numbers produced at 512 are not comparable.
- **HF-side calibrators sample the WHOLE corpus** (`calibrate/_ingest.sample_chunks`):
  AWQ (`tokens` default 65536), GPTQ (32768), and the outlier forward stats
  stride their token budget evenly across the file instead of reading its head.
  The AWQ α-search activations `X` accumulate evenly-spaced rows from **every**
  sampled chunk (not the first chunk's head), so α is chosen on the corpus
  distribution rather than on the leading system prompt. Run
  `scripts/audit_calibration_coverage.py` to see what a corpus+budget yields.
- The packer dedups **tool schemas** like system prose (`tool_schema_quota`,
  default 1): full schemas render in the first window of the first session per
  unique schema set; every other window gets `stub_tools` (name + first
  description line, empty parameters). The pack audit's `boilerplate_tokens`
  (prose + schemas + wrapper, measured by a body-less prefix render) is what
  `tool_turn_token_share` is computed from; `system_prose_tokens` is the
  prose-only subset kept for continuity.
- `bench.eval_tools_corpus` (optional) adds a second, in-distribution KLD suite
  (own `eval/baseline-tools.kld`, `*_tools` CSV columns, display-only in the
  leaderboard). CLI: `quant-tuner bench --eval-tools corpus.eval.tools.txt`.
  Quant-vs-quant only — same no-`--parse-special` caveat as above.
- HF calibration forward passes go through `calibrate/_hf.forward_no_logits`,
  which runs the decoder trunk only — hooks still fire, but the `[ctx, vocab]`
  logits tensor and lm_head matmul are skipped. Keep using it for new
  hook-based stat collection; the AWQ/GPTQ *sanity checks* intentionally call
  `model(...)` because they compare logits.
- Speed numbers have a known thermal artifact: rows that run later in a session drift
  lower. SQS (which weights speed equally with compression) is noisier than KLD; for
  "which imatrix is best?" read **KLD and tool-call** columns.
- Calibration `train`, eval `test`, and `holdout` slices come from the same source
  (the on-disk logs under `datasets/agent-logs/data/`) but are disjoint — preserve this
  invariant when adding new evals. The split is by GROUP, not by row (`ingest.session_group`).
- **Slice / source → corpus mapping** (used by `scripts/build_corpora.py` and the
  convention every new experiment should follow):
  - logtrain `train` (80%) + wiki → **calibration** corpus → imatrix + AWQ proxy loss
  - logtrain `test`  (10%) + `calibration_supplement.txt` → **validation** corpus → AWQ
    cv-mixed / cv-gate α scoring. The supplement injects under-represented content so
    val is a genuine distribution shift from cal, not a re-draw of the same sessions.
  - external `eaddario/imatrix-calibration` {code_small, math_small, tools_small} →
    **eval** corpus (`corpus.eval.txt`, PPL/KLD). Eval is **not** drawn from logtrain
    or wiki — those appear in cal and would conflate fit with generalization.
  - external `eaddario/imatrix-calibration` `combined_en_tiny` → **general** eval holdout
    (`corpus.eval.general.txt`, separate baseline.kld, benched independently).
  - logtrain `holdout` (10%) → BOTH (a) the **tools** PPL/KLD eval holdout
    (`corpus.eval.tools.txt`, windowed by `build_corpora.py` — in-distribution but
    disjoint from cal-train) AND (b) the agentic tool-call eval sessions (via
    `pipeline.py` / `build_toolcall_holdout.py`). Both stay out of the cal (train) slice.
  - external `nebius/SWE-rebench` `test` split → **agentic** eval holdout (via
    `build_swebench_holdout.py`). A wholly separate source from the
    calibration/eval corpora — its issues never touch logtrain/wiki, so a quant
    solving them is genuine generalization, not fit.
  Don't repurpose a slice or source without re-checking every eval that touches it.
