# exp-062: rebuilding the Qwen3.8-27B low-bit rungs on a tool-dense corpus

Rebuild of the IQ2_M / IQ3_M / IQ4_XS rungs of
`pearsonkyle/Qwen3.8-27B-imatrix-MTP-GGUF` via AWQ + `hybrid_custom` imatrix, on a
new calibration corpus cut from `sft_chat_train.jsonl.gz`, with the fixed
`qwen3_8_safe_v2` chat template baked into every output.

Status legend: **[V]** measured on this run · **[P]** predicted, not yet measured.

## State at end of session (2026-08-16 04:30)

**NOTHING WAS PUBLISHED.** No upload to Hugging Face was performed; the live repo
is untouched and still serves the original four imatrix rungs.

| item | state |
|---|---|
| IQ2_M AWQ rung | built, audited, benched (2 evals) — `out/exp-062-awq-iq2m/gguf/IQ2_M-awq-best-hybrid_custom-mtp.gguf` |
| IQ3_M AWQ rung | built, audited, benched (2 evals) — regressed, see §3.2 |
| IQ4_XS AWQ rung | **never built** — recipe + workspace + gated runner staged only |
| tool-call smoke | IQ2_M pair complete (§3.3); IQ3_M/IQ4_XS **not run** |
| SWE-mimic smoke | **never run** |
| 6-eval KLD top-up | **never run** — `scripts/exp062_kld_all_evals.py` written and validated |
| model card | **not updated** — still describes the superseded imatrix IQ2_M |
| HF upload | **not performed** — `scripts/exp062_ship_iq2m.py` correctly BLOCKS on the stale card |

**To resume the IQ2_M ship**, in order:

1. `PYTHONPATH=src .venv/bin/python scripts/exp062_kld_all_evals.py --quant out/exp-062-awq-iq2m/gguf/IQ2_M-awq-best-hybrid_custom-mtp.gguf --out out/exp-062-32k/eval/kld_all_evals.csv` (~12 min) — the card publishes six eval distributions and only two are measured for this file.
2. `MODELS_FILE=models_exp062.txt PORT=18080 RBUDGET=2048 TAG=_exp062 bash run_all_quants.sh` in `/workspace/swe-mimic` — the card's 🤖 SWE rows.
3. Re-run the tool-call pair at multiple seeds — see §3.3; the single-seed result is inside the noise band and does not settle the ship question.
4. Update `out/exp-060-32k/release/README.md` (IQ2_M column + a "why the 2-bit is AWQ" section + the new corpus/template in methodology).
5. `PYTHONPATH=src .venv/bin/python scripts/exp062_ship_iq2m.py` (dry run), then `--push`.

⚠️ **Read §3.3 before shipping.** The distributional case is strong and the
task-level case is a wash, so this is a judgement call rather than a
measurement-driven conclusion.

> **Three variables moved at once** relative to the shipped ladder: the AWQ fold,
> the calibration corpus, and the embedded template. Any win therefore belongs to
> "the new recipe", not to AWQ specifically. The clean attribution experiment, if
> it is ever wanted, is the new corpus with `calibration.method: imatrix` (no AWQ)
> — one imatrix pass, ~3.4 h. It has not been run.

---

## 1. What changed, and why

### 1.1 The corpus **[V]**

Built by `scripts/exp060_repack_cal_32k.py` (which gained `--chat-template`,
`--drop`, `--budget`; its defaults still reproduce the pinned exp-060 corpus, and
`tests/unit/test_exp062_corpus_recipes.py` pins that property):

```bash
PYTHONPATH=src .venv/bin/python scripts/exp060_repack_cal_32k.py \
    --sft /workspace/sft_chat_train.jsonl.gz --run exp-062-32k --ctx 32768 \
    --chat-template data/chat_templates/qwen3_8_safe_v2.jinja \
    --drop redteam-refusals --drop reasoning \
    --budget logs=2800000 --budget swe-trajectories=800000 \
    --budget broad-supplement=300000
```

| | exp-060 (shipped) | exp-062 (new) |
|---|---|---|
| tokens | 4,255,761 | 3,744,101 |
| windows | 3,436 | 1,738 |
| tool-call markers | 6,817 | **6,955** |
| tool-response markers | 5,861 | **6,532** |
| tool calls per 1M tokens | 1,602 | **1,858** (+16%) |
| agentic share (logs+swe) | 63% | **92%** |

Fewer tokens, more tool traffic, and roughly double the tokens per window — i.e.
the 32K packing is actually being used.

### 1.2 Two strata dropped **[V]**

* **`redteam-refusals`** — 305 rows, **zero** tool calls, 0.5% of the source file.
  ⚠️ **This is a real tradeoff, not a free win.** Refusal behavior is what low-bit
  quantization erodes first (`data/refusals.py`), so these rungs are *not*
  calibrated on the attack distribution. If a release cares about that, re-add it
  at a small budget and re-measure; do not assume it is unchanged.
* **`reasoning`** — this stratum re-cut the *same* log sessions so a reasoning turn
  landed last. It existed because chat templates historically scrubbed reasoning
  from history. **Measured on this file: 3,158 of 3,220 reasoning turns (98%)
  already survive in place**, so the stratum was only double-weighting a subset of
  `logs`. Its 1M-token budget went to `logs` instead.

### 1.3 The template does NOT change the corpus **[V]**

The premise going in was that the new template "handles reasoning better" and
would therefore improve calibration. It does not — not here:

> All **6,100** rows of `sft_chat_train.jsonl.gz` render **byte-identical** under
> the stock template and `qwen3_8_safe_v2`. Zero render errors on either side.
> Identical non-empty `<think>` counts (3,158 vs 3,158). Total rendered chars
> differ by **0**.

The template's value is entirely at **inference** time. That byte-identity is also
the safety argument for baking it: the swap cannot change quality, only stop the
crashes.

Verified by extracting `tokenizer.chat_template` from **the built GGUFs
themselves** (not the source `.jinja`) and rendering each case under jinja2 — so
this tests the shipped artifact, not the intent **[V]**:

| case | shipped IQ2_M | new IQ2_M | status |
|---|---|---|---|
| `reasoning_effort="high"` (the OpenAI-standard value) | **raises** → HTTP 400 | renders | fixed |
| JSON-string tool `arguments` (what OpenAI-compatible servers return) | **raises** `TypeError: Can only get item pairs from a mapping` | renders, args preserved | fixed |
| leading `tool` message | **malformed** (see below) | wrapped in `<\|im_start\|>user` | fixed |
| bare string in a content list | raises `Unexpected item type in content.` | raises, clearer message | **not a bug** |

The leading-`tool`-message defect is worth seeing concretely — the shipped
template emits the `<tool_response>` with **no `<|im_start|>` header** and then
closes it with an orphaned `<|im_end|>`:

```
...<|im_end|>\n\n<tool_response>\nprior result\n</tool_response><|im_end|>\n<|im_start|>user\n...
```

⚠️ **Correction to `docs/patch_gguf_chat_template.md`**: that doc lists a fourth
bug — a bare string in a content list "emitting a vision token while DISCARDING
the text". **This does not reproduce.** Both templates *raise* on that input,
which is the safe behavior; the new one merely gives a better message. Three of
the four claimed bugs are real and fixed. (Tested under python jinja2; llama.cpp
serves templates through minja, so the exact message differs but the raise does
not.)

### 1.4 What the α search actually reads **[V]**

`calibration.params.tokens: 524288` is not arbitrary. Measured with the production
sampler (`calibrate/_ingest.sample_chunks`) on this corpus at ctx 32768:

| budget | chunks | tokens | corpus coverage | tool calls seen |
|---|---|---|---|---|
| **524,288 (this build)** | 16 | 500,069 | **13.4%** | **813** |
| 65,536 (awq default) | 2 | 41,317 | 1.1% | 66 |

At the default the α search would see 66 tool calls concentrated in two places.
The sampled span is `0..3744087` — the whole file, evenly strided — and the
α-search activation matrix `X` takes evenly-spaced rows from *every* sampled
chunk, so α is chosen on the corpus distribution rather than on a leading system
prompt.

---

## 2. Per-rung facts

### 2.1 The folds cannot be shared **[V]**

AWQ scores α candidates through a proxy quantizer chosen from `quantize.type`, so
each rung needs its own fold *and* its own imatrix pass (~3.4 h each). Measured:

| rung | proxy | per-member mix | α histogram | fold sanity rel |
|---|---|---|---|---|
| IQ2_M | `q2k_b16` (pinned) | `None` (pinning opts out) | 46 @ 0.25, 34 @ 0.50 | 1.892e-02 |
| IQ3_M | `q3k_b16` (auto) | `IQ3_M` | **10 @ 0.0**, 57 @ 0.25, 13 @ 0.50 | 1.503e-02 |

Only **49 of 80 groups share an α** across the two rungs.

Two observations worth acting on:

* **The 3-bit search declines to intervene on 10 groups** (α = 0.0). The 2-bit
  search never does. That is the expected shape — less quantization error at 3
  bits means AWQ's rescaling is sometimes not worth its own distortion — and it is
  evidence the search is discriminating rather than degenerate.
* **The grid's upper half is never used.** The grid is
  `(0.0, 0.25, 0.5, 0.75, 1.0)` and across both rungs *no* group selected 0.75 or
  1.0. All the action is in `[0, 0.5]`. **A refined grid over that range is the
  cheapest next lever** if a rung comes out a wash. **[P]**

### 2.2 A same-value sanity number is not a no-op **[V]**

The IQ2_M fold reported `rel=1.892e-02`, identical to the previous exp-060-32k-awq
attempt on the *old* corpus. That looks alarming and is not: comparing the two
`awq.pt` files, **all 80 scale tensors differ, max relative difference 2.006**,
while the α *histogram* is unchanged because the grid has only 5 coarse points.
The sanity number reflects fp-rounding of the fold on a fixed probe, and IQ3_M's
differing 1.503e-02 confirms it moves when the scales move enough.

### 2.3 Build audit (`scripts/validate_exp062_awq.sh`)

Every check guards a failure whose only other symptom is a mediocre benchmark
number hours later. IQ2_M **[V]**, all pass:

* template 27,850 bytes matching the repo `.jinja`; `general.name = Qwen3.8-27B`
* MTP head: 15 tensors at `blk.64`, 8 × Q8_0 + 7 × F32 (the pin matched real
  tensors — `llama-quantize` accepts a `--tensor-type` pattern matching nothing)
* imatrix coverage: the only tensors quantized blind are `token_embd.weight` and
  the 8 `blk.64.*` MTP tensors. Anything else in that list is a bug.
* imatrix itself: 994 entries / 497 tensors, **`output.weight` present**
  (`--process-output`), `token_embd` and `blk.64` correctly absent. Family counts
  confirm the hybrid split — 16 layers with `attn_k/q/v/output`, 48 with
  `attn_qkv`/`attn_gate`/`ssm_*`.
* `hybrid_custom`: 497 tensors, 144 SSM passthrough, 0 skipped.

---

## 3. Results

### 3.1 IQ2_M — distributional **[V]**

Identical bpw; eval corpora and FP16 baselines pinned to exp-060.

| external eval | shipped | new AWQ | delta |
|---|---|---|---|
| bpw | 3.0617 | 3.0617 | — |
| **ppl** | 56.540 | **34.530** | **−38.9%** |
| ppl_ratio vs FP16 | 2.881 | 1.760 | −38.9% |
| mean_kld | 1.2841 | 1.2284 | −4.3% |
| median_kld | 0.1242 | 0.1140 | −8.2% |
| same_top_p | 71.95% | 73.20% | +1.25pp |

| tools eval | shipped | new AWQ | delta |
|---|---|---|---|
| **ppl_tools** | 54.311 | **43.540** | **−19.8%** |
| mean_kld_tools | 2.3816 | 2.3220 | −2.5% |
| same_top_p_tools | 72.03% | 73.02% | +0.99pp |

Speed: 3,539 tok/s prefill, 107.1 tok/s decode (10 reps).

**Validity check**: the implied FP16 reference is 56.540/2.881 = 19.623 and
34.530/1.760 = 19.622 — identical, confirming both rows were scored on the same
eval corpus against the same baseline.

Note the PPL/KLD divergence: `mean_kld` 1.228 against a `median_kld` of 0.114 is a
~10× ratio, so a handful of outlier tokens dominate the mean. The quant got much
better at predicting real text while a few tokens still diverge sharply from FP16.

Timings, IQ2_M leg: calibrate 3.8 min · fold 1.2 min · convert 1.8 min ·
**imatrix 200.7 min** (114 chunks @ 105.5 s) · re-weight 2.0 min · quantize 5.8 min
· metadata bake 0.2 min · bench 2.4 min.

### 3.2 IQ3_M — distributional **[V]**

Identical bpw (3.81607 both). **Regressed on every metric except median_kld**, the
mirror image of IQ2_M:

| | shipped | new AWQ | delta |
|---|---|---|---|
| ppl (external) | **37.362** | 41.531 | +11.2% worse |
| mean_kld | **0.7394** | 0.7876 | +6.5% worse |
| median_kld | 0.035383 | 0.035385 | identical |
| same_top_p | **82.41%** | 81.65% | −0.75pp |
| ppl_tools | **21.695** | 23.322 | +7.5% worse |
| mean_kld_tools | **1.5354** | 1.5667 | +2.0% worse |
| same_top_p_tools | **80.52%** | 80.21% | −0.31pp |

Speed: 4,101 tok/s prefill, 92.3 tok/s decode.

**Working explanation for the 2-bit/3-bit split.** The new corpus is much narrower
(92% agentic; wiki, `reasoning` and `redteam-refusals` all dropped), and *both*
eval corpora are drawn from outside it. At 2 bits AWQ's channel protection is large
enough to swamp that narrowing; at 3 bits the quantization error is smaller, so the
narrowing dominates and shows up as a generalization cost. This is a hypothesis
consistent with the data, **not** a measured cause — testing it means rebuilding
IQ3_M with the wiki/broad strata restored. **[P]**

Timings, IQ3_M leg: calibrate 3.8 min · fold 1.1 min · convert 1.8 min ·
**imatrix 210.0 min** · re-weight 1.2 min · quantize 7.1 min · bake 0.2 min ·
bench 1.9 min.

### 3.3 Tool-call — IQ2_M pair complete; IQ3_M / IQ4_XS not measured

`scripts/smoke_exp062.py`, greedy, 25-session holdout, `stop_on_fail=False`.
⚠️ This harness scored **125** turns for IQ2_M where the published ladder scored
174, so **these numbers are not comparable to the card's 🤖 rows** — only to each
other, which is why every shipped rung is re-measured in the same sweep.

| model | tool_sel | param_acc | schema_valid | n |
|---|---|---|---|---|
| IQ2_M-awq-new | 0.504 | 0.267 | 0.832 | 125 |
| IQ2_M-shipped (control, same session) | 0.480 | 0.240 | 0.896 | 125 |
| **delta (new − shipped)** | **+0.024** | **+0.027** | **−0.064** | |

**Noise floor**: at p≈0.24, n=125, the binomial SE is **3.8pp**; the conservative
unpaired 1-SE bound on a difference is 5.4pp, so the "beyond noise" threshold is
roughly **±10.8pp**.

**Verdict on the decision metric: a WASH, not a win.** param_acc +0.027 is 2.7pp
against a ~10.8pp threshold — well inside noise, and nowhere near the gemma-order
+0.09 that would justify a swap on task grounds. tool_sel +0.024 is likewise
noise. schema_valid **−0.064** is also inside the band but points the wrong way on
a metric where down is unambiguously bad.

So the two halves of the evidence disagree:

* **Distributional: a large, unambiguous win.** ppl −38.9% is far outside any
  plausible noise, and every KLD/top_p metric improved.
* **Task-level: neutral, with a mild negative signal on schema validity.**

⚠️ **By the verdict rule in §5 this rung does NOT qualify for a swap** — the rule
requires a tool-call gain beyond noise *and* no schema regression, and neither
holds. The remaining arguments for shipping it anyway are real but are **not**
"the model got better at tool calls": the fixed chat template (the
`reasoning_effort="high"` → HTTP 400 fix), the corrected `general.name`, and a
much healthier PPL. That is a judgement call for the maintainer, not something the
measurements settle.

To resolve it properly rather than by judgement: re-run the pair at several seeds
(or on the full 174-turn footing) to shrink the interval, since the whole question
is whether a ~3pp gain and a ~6pp schema drop are real.

### 3.4 Agentic (SWE-mimic) — NOT MEASURED

Never run. `/workspace/swe-mimic` is staged with `models_exp062.txt` and the grader
truncation fix; `MODELS_FILE=models_exp062.txt PORT=18080 RBUDGET=2048 TAG=_exp062 \
  bash run_all_quants.sh` is the command.

Context for reading §3.3: KLD is a guardrail only. The gemma-4-31B precedent was
worse on median KLD (1.804 vs 1.571) and on top_p (43.9% vs 46.6%) and still won
by **+54%** on tool arguments (0.171 → 0.263). A −39% PPL does not by itself mean
better tool calls — and here it did not deliver them.

Shipped baselines to beat:

| rung | tool_selection | param_acc | schema_valid | source |
|---|---|---|---|---|
| IQ2_M | 0.4943 | **0.2601** | 0.9540 | full holdout, n≈174 |
| IQ3_M | 0.4540 | **0.2395** | 0.9310 | full holdout, n≈174 |
| IQ4_XS | 0.5029 | **0.2738** | 0.9306 | full holdout, n≈174 |
| F16 | 0.4943 | 0.2562 | 0.9310 | full holdout, n≈174 |
| IQ2_M | 0.650 | 0.3056 | 1.000 | 25-session smoke, n=92 |

**Noise floor**: greedy, n=92 smoke turns → binomial SE ≈ 4.8pp; n≈174 full
holdout → ≈ 3.3pp. A +0.01 "win" is noise. Ship on a gap of the gemma order
(+0.09) or run multiple seeds.

⚠️ **The shipped IQ3_M scores *below* IQ2_M on tool arguments** (0.2395 vs 0.2601)
despite 0.75 more bpw and roughly half the KLD. A rung better on every
distributional metric and worse on the task is the signature of a calibration
mismatch rather than of bit width — which is the specific defect this rebuild is
meant to repair. Watch that number.

**IQ4_XS expectation [P]**: small. AWQ protects salient channels from quantization
error, and that error is small at 4 bits; the α search there scores through plain
`int4_g128` rather than an E8-lattice codebook proxy; and the shipped IQ4_XS is
already the ladder's best tool-call rung (above even F16). The realistic good
outcome is "no regression", which still has value — it would let the repo ship on
one corpus and one template instead of a mix.

---

## 4. Harness notes

* **SWE-rebench proper cannot run in this container** — no docker binary, no
  `/var/run/docker.sock`, no `CAP_SYS_ADMIN`, and seccomp blocks `CLONE_NEWUSER`.
  Use `/workspace/swe-mimic` (Docker-free, real agent episode graded by running
  the instance's F2P/P2P tests). `MODELS_FILE=models_exp062.txt` selects the rungs.
* **Fixed in swe-mimic [V]**: the grader ran the P2P suite through `sh()`'s
  6,000-char cap while that suite emits 10,408 chars. Every `PASSED` line survived
  here by ~1.7×, but a larger suite would lose them and report a **false
  unresolved** — a pessimistic bias indistinguishable from a broken quant. The
  grading call now passes `limit=None`.
* **Shipped controls are re-measured in the same session**, not compared against
  recorded numbers — the exp-060 figures came from a different day and llama.cpp
  build, and an unpaired delta would fold server drift into the result.
* **Disk**: each rung's fold is ~103 GB (folded HF bf16 + folded F16 GGUF) and two
  cannot coexist here. `run_exp062_awq.sh` reclaims between rungs, and refuses to
  reclaim if the rung produced no GGUF.
* `scripts/report_exp062.py` assembles KLD + tool-call + SWE into one table and
  prints the noise bound next to every tool-call delta.

## 5. Verdict rule

Swap a rung **only** if tool-call `param_acc` improves by more than the noise
bound **and** schema validity does not regress **and** the SWE smoke still
completes. KLD moving the wrong way is acceptable — the gemma precedent did
exactly that and swapping was the right call.
