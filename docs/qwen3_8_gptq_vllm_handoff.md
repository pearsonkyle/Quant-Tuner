# Handoff — GPTQ W4A16 @ 32K for vLLM (Qwen/Qwen3.8-27B)

**Written 2026-08-15** at the end of the exp-060-32k GGUF session. Everything marked
**VERIFIED** was measured on this box in that session; everything marked **UNVERIFIED** is a
prediction you must check. Do not treat the two as equivalent — the traps in §3 are verified
and will silently corrupt the output if ignored.

---

## 0. Goal

Produce a **compressed-tensors W4A16 checkpoint that vLLM serves directly**, calibrated on
the **same 32K-packed corpus** as the GGUF ladder, then bench it on the **same two axes**:
KLD-vs-reference and the SWE agent bench.

This is the sibling of the GGUF ladder, not a replacement. The GGUF ladder ships
`IQ2_M / IQ3_M / IQ4_XS / Q5_K_M` with an MTP head pinned Q8_0; this path ships one
4-bit checkpoint for a vLLM deployment.

**Read `/workspace/CLAUDE.md` §"vLLM-native PTQ export" before starting** — it documents the
module you will be driving (`src/quant_tuner/vllm_export/w4a16.py`).

---

## 1. Machine requirements

| | Needed | This box (VERIFIED) |
|---|---|---|
| VRAM | ≥ 80 GB for a comfortable sequential run | 97,887 MiB (1× Blackwell, `ARCHS=1200`) |
| System RAM | ≥ 256 GB if you fall back to `--pipeline basic` (Hessian offload) | 1,511 GB total / 1,251 GB available |
| Disk | ≥ 120 GB free (52 GB source + ~16 GB output + scratch) | 174 GB free on `/workspace` |
| CPU | more is better for the fp64 Cholesky path | 384 cores |

`uv sync --extra vllm-ptq` installs `llmcompressor`. It is **not** in the default env —
`vllm_export.run_ptq` imports it lazily and raises a clear ImportError if missing.

---

## 2. Verified inputs (already on disk — do not rebuild)

### The calibration corpus — this is the irreplaceable asset

```
/workspace/Quant-Tuner/out/exp-060-32k/corpora/corpus.cal.txt
```

- **16,883,152 bytes / 4,255,761 tokens**, packed for **ctx 32768** (`window_cap 32076`)
- Source mix: `logs 47.0% · swe-trajectories 16.0% · reasoning 15.0% · broad-supplement
  12.6% · wiki 7.0% · redteam-refusals 2.4%`
- **6,817 tool-call markers**; audit at `out/exp-060-32k/corpora/corpora_audit.json`
  (top-level `ctx` / `window_cap` record what it was packed for)
- Built by `scripts/exp060_repack_cal_32k.py` from `/workspace/sft.jsonl.gz`

> **Calibration ctx is a PACKING parameter, not just a flag.** This corpus's windows are
> sized to fill one 32,768-token context. Reading it at a different ctx either straddles
> window boundaries or glues unrelated conversations into one sequence. **Pass `--ctx 32768`
> and nothing else.** Numbers produced at a different ctx are not comparable to the GGUF
> ladder's.

Per-source files (`corpus.cal.logs.txt` etc.) sit beside it if you want to reweight the mix.

### The bf16 source

```
/workspace/Quant-Tuner/out/exp-060/model_extracted     # symlink farm into the HF cache
```
52 GB, `dtype: bfloat16`. `AutoConfig` resolves it as `Qwen3_5Config` / `model_type qwen3_5`,
and **VERIFIED**: `qwen3_5` is present in transformers 5.12.1's `MODEL_FOR_CAUSAL_LM_MAPPING`,
so `AutoModelForCausalLM.from_pretrained` (what `run_ptq` calls) will load it.

### Architecture (VERIFIED from config.json + the safetensors index)

| | |
|---|---|
| Architecture | `Qwen3_5ForConditionalGeneration` (**multimodal wrapper**, `text_config` nested) |
| Layers | 64 — **48 `linear_attention` + 16 `full_attention`**, `full_attention_interval: 4` |
| hidden / intermediate / head_dim | 5120 / 17408 / 256 |
| attn heads / kv heads | 24 / 4 |
| Vocab | 248,320 · `tie_word_embeddings: false` |
| MTP | `mtp_num_hidden_layers: 1` |
| Tensors | 1,199 total = 850 `model.*` + **333 `model.visual.*`** + **15 `mtp.*`** + 1 `lm_head` |

---

## 3. Three traps — all VERIFIED, all silent

### 3.1 `DEFAULT_IGNORE` does NOT match this model's vision tower

`vllm_export/w4a16.py:39` ignores `re:.*vision_tower.*`. Measured against this checkpoint's
weight map:

```
vision_tower           matches 0      <-- the default pattern is dead here
visual                 matches 333    <-- the tower is model.visual.*
lm_head                matches 1      ok
embed_tokens           matches 1      ok
per_layer              matches 0      (gemma-ism, harmless)
multi_modal_projector  matches 0
```

With defaults you would **quantize a 333-tensor vision tower to int4 using a text-only
calibration corpus.** It will not error. You must add `re:.*visual.*` to `ignore`.

### 3.2 The MTP head (`mtp.*`, 15 tensors) is also unignored

Top-level prefix `mtp.` — not under `model.layers`, so nothing in `DEFAULT_IGNORE` catches
it. In the GGUF ladder this head is deliberately pinned **Q8_0** because a low-bit draft head
drafts badly and it is tiny relative to the trunk. Do the equivalent here: **add `re:mtp.*`
to `ignore`** so it stays bf16. If you later want vLLM speculative decoding off this head,
you want it near-lossless anyway.

### 3.3 Hybrid linear attention may break the sequential tracer — UNVERIFIED

48 of 64 layers are `linear_attention` carrying recurrent state (`mamba_ssm_dtype: float32`).
This is structurally the same hazard that forced `--pipeline basic` on gemma-4 (cross-layer
shared KV broke the sequential tracer — see CLAUDE.md).

**Try `--pipeline sequential` first** (the default, and far cheaper in memory). If it fails
inside the tracer, fall back to `basic`. Do not start with `basic` "to be safe" — see the
Hessian arithmetic below.

---

## 4. Hessian memory — why the pipeline choice matters

GPTQ accumulates an `[in_features, in_features]` Hessian per linear. For this model:

| projection | in_features | Hessian @ fp32 |
|---|---:|---:|
| q/k/v/gate/up_proj | 5,120 | 105 MB |
| o_proj | 6,144 (24×256) | 151 MB |
| **down_proj** | **17,408** | **1.21 GB** |

≈ **1.9 GB per full-attention layer.**

- `sequential` — one layer resident at a time → **~2 GB**. Fine on any GPU here.
- `basic` — all layers' Hessians live at once → the modifier auto-enables `offload_hessians`,
  pushing them to CPU RAM. Order-of-magnitude **60–100 GB of system RAM**. This box has
  1.25 TB available so it is survivable, but it is slow and it is not the default for a
  reason.

---

## 5. The command

```bash
cd /workspace/Quant-Tuner
uv sync --extra vllm-ptq

PYTHONPATH=src .venv/bin/python scripts/run_vllm_ptq.py \
  --model out/exp-060/model_extracted \
  --corpus out/exp-060-32k/corpora/corpus.cal.txt \
  --out out/exp-060-w4a16-32k/checkpoint \
  --ctx 32768 \
  --budget-tokens 4194304 \
  --scheme W4A16 \
  --group-size 128 \
  --pipeline sequential
```

**`--budget-tokens 4194304`, not the 524,288 default.** At ctx 32768 the default is only
**16 sequences** — far too few distinct sequences for a stable Hessian. 4M tokens ≈ 128
sequences at 32K and is ~the whole corpus (4.26M tokens).

**The `ignore` list is not exposed as a CLI flag** (`run_vllm_ptq.py` has no `--ignore`). Per
§3 you must extend it. Either add the flag, or drive `PTQConfig` directly:

```python
from pathlib import Path
from quant_tuner.vllm_export import PTQConfig, run_ptq, DEFAULT_IGNORE

cfg = PTQConfig(
    model_id="out/exp-060/model_extracted",
    out_dir=Path("out/exp-060-w4a16-32k/checkpoint"),
    corpus_files=[Path("out/exp-060-32k/corpora/corpus.cal.txt")],
    ctx=32768,
    budget_tokens=4_194_304,
    scheme="W4A16",
    group_size=128,
    # the two additions that matter — see §3.1 / §3.2
    ignore=DEFAULT_IGNORE + ("re:.*visual.*", "re:mtp.*"),
    pipeline="sequential",
)
run_ptq(cfg)
```

Adding `--ignore` (repeatable, appended to `DEFAULT_IGNORE`) to `run_vllm_ptq.py` is the
cleaner fix and is worth doing — this trap will recur on the next multimodal model.

`run_ptq` **fails loudly** if the exported config lacks `quantization_config` (otherwise vLLM
would silently serve bf16), and writes `quant_tuner_ptq.json` with corpus SHA-256s, ctx,
budget and scheme. Check that file exists before believing the run.

---

## 6. KLD — the harness does NOT exist for this path. This is the real build task.

**VERIFIED**: `bench/kld.py` shells out to llama.cpp (`llama-perplexity
--kl-divergence-base`). It only consumes **GGUF**. A compressed-tensors safetensors
checkpoint cannot be fed to it, and there is **no HF-side KLD anywhere in `src/`** (the only
`kl_div` in the tree is `qat/train.py`'s distillation loss, which is a different thing).

So you must write it. Design that keeps the number comparable to the GGUF ladder:

1. **Reference = the bf16 HF model**, not the F16 GGUF. (The GGUF ladder's reference is the
   F16 GGUF; these are the same weights but a different runtime, so absolute values will not
   match the GGUF table. Say so in any write-up — do not present them in one column.)
2. **Same eval corpora, same chunking.** Six files, each its own distribution, **never
   concatenated**, each with its own reference:
   ```
   out/exp-060-32k/corpora/corpus.eval.txt           external  (349,092 B)
   out/exp-060-32k/corpora/corpus.eval.general.txt   general   (125,227 B)
   out/exp-060-32k/corpora/corpus.eval.tools.txt     tools     (592,904 B)
   out/exp-060-32k/corpora/corpus.eval.agentic.txt   agentic   (378,162 B)
   out/exp-060-32k/corpora/corpus.eval.broad.txt     broad     (525,489 B)
   out/exp-060-32k/corpora/corpus.eval.cal8k.txt     cal8k     (155k tok, a FIT probe — NOT a holdout)
   ```
   The GGUF ladder chunked these at **eval_ctx 8192**. Use 8192 so the *shape* of the
   comparison matches.
3. **Report median KLD and top-token agreement**, the two columns the ladder is judged on.
   Median (not mean) for robustness to per-token tails.
4. Unlike llama-perplexity, a torch-side implementation **can** tokenize special tokens
   correctly. That is an improvement, but it means `tools`/`agentic`/`broad`/`cal8k` numbers
   are *more* correct here and therefore **not** comparable to the GGUF card's. Flag it.

Sketch: load bf16 and quantized models, run the same token chunks through both,
`KLD = Σ p_ref · (log p_ref − log p_quant)` per position, take the median across positions;
top-token agreement = fraction where argmax matches. Do it in fp32, chunk the vocab
dimension — 248,320 vocab × 8192 positions is 8 GB per logits tensor in fp32.

**Reference numbers from the GGUF ladder** (median KLD / top-token %, for orientation only):

| eval | IQ2_M | IQ3_M | IQ4_XS | Q5_K_M |
|---|---:|---:|---:|---:|
| external | 0.12416 / 71.95 | 0.03538 / 82.41 | 0.01047 / 88.18 | 0.00360 / 91.75 |
| general | 0.21989 / 66.04 | 0.05685 / 79.91 | 0.01542 / 87.97 | 0.00623 / 90.92 |
| tools | 0.04981 / 72.03 | 0.01148 / 80.52 | 0.00323 / 85.90 | 0.00143 / 87.84 |
| agentic | 0.01688 / 72.35 | 0.00480 / 78.46 | 0.00132 / 84.15 | 0.00060 / 86.40 |
| broad | 0.26908 / 65.32 | 0.07021 / 76.90 | 0.02039 / 84.87 | 0.00758 / 88.40 |
| cal8k | 0.08395 / 72.34 | 0.02613 / 78.96 | 0.00622 / 85.98 | 0.00248 / 88.69 |

W4A16 is ~4.5 bpw-equivalent on quantized layers, so **expect it to land near the IQ4_XS
row** — that is your sanity check. An order of magnitude worse means something in §3 went
wrong.

---

## 7. SWE agent bench — reusable as-is

`/workspace/swe-mimic/` runs the Docker-free SWE-rebench mimic and **already takes
`--base-url`**, so pointing it at vLLM needs no code change:

```bash
vllm serve out/exp-060-w4a16-32k/checkpoint \
  --max-model-len 32768 --port 18080          # quantization auto-detected

cd /workspace/swe-mimic
.venv/bin/python run_agent.py \
  --base-url http://127.0.0.1:18080/v1 \
  --model-name out/exp-060-w4a16-32k/checkpoint \
  --label W4A16 --reasoning-budget 2048
```

**Read `run_agent.py`'s docstring before quoting any number.** It is a smoke test, not
SWE-rebench: no container isolation, host-resolved dependency versions, one instance
(`dask__dask-11393`), one repetition. With a single instance a resolve/miss swings the rate
by 100 points. It answers "can this checkpoint drive an agentic loop at all", nothing more.

Harness facts you inherit (all fixed in this session, all learned the hard way):

- **`tool_errors` counts only `malformed` + `timeout`.** A non-zero exit is *not* an error —
  the prompt tells the agent to run the failing test (exit 1 by construction) and to grep
  (exit 1 on no match). Those land in `nonzero_exits`, which is diagnostic. An earlier
  version counted every non-zero exit and reported a fictitious "9.1% error rate".
- **A failed `cd` IS counted** (as `malformed`) because everything after the `&&` never ran.
  Qwen3.8 reaches for `cd /testbed` — the real SWE-rebench Docker workdir — so the mimic's
  system prompt now names the actual repo root.
- **`max_tokens=8096`** per call, matching `eval.swebench.DEFAULT_MAX_TOKENS`. Without it a
  looping rung generates until the context is exhausted (~5.5 min/turn) and scores as "slow"
  rather than as degradation.
- **Port 18080, not 8080** — Jupyter binds `0.0.0.0:8080` on this image and a server started
  there health-checks green against the wrong process.
- Every command is logged to `work/<instance>/traj_<label>.json`, so an error rate is
  auditable instead of asserted. Read it before believing any count.

Reference (GGUF ladder, unbudgeted, 1 rep — behavioural only, does **not** rank rungs):
all five of IQ2_M/IQ3_M/IQ4_XS/Q5_K_M/F16 resolved the instance, 0 malformed commands each;
output tokens ran F16 1,514 → IQ3_M 3,902 → Q5_K_M 4,526 → IQ4_XS 5,636 → IQ2_M 7,590.

Anti-reward-hacking: audit the produced patch. The test file appearing in `git diff` is the
harness's own applied `test_patch` (applied, not committed) — confirm it is byte-identical
to `work/<instance>/test_patch.diff` and that only source files were modified.

---

## 8. Traps inherited from the GGUF session

- **`pgrep`/`pkill -f <pattern>` matches your own shell.** `pkill -9 -f llama-server` from a
  `bash -c` whose command line contains that string kills the shell. Bit us twice. Kill by
  PID.
- **`llama-imatrix` defaults `process_output=false`**, so `output.weight` was calibrated
  blind on the whole GGUF ladder. Fixed (`models/llama_cpp.py` now passes
  `--process-output`). GPTQ-side equivalent: confirm `lm_head` is *intentionally* in `ignore`
  (it is, and should stay — a quantized 248k-vocab head is the rare-token failure mode).
- **llama-imatrix SIGSEGVs above ~17k ctx on large-vocab models** — `imatrix.cpp:911` does
  `all_logits + first*n_vocab` in `int`, and 248,320 vocab overflows INT_MAX at
  `n_ctx > 2^31/248320 ≈ 17,296`. Workaround is `--no-ppl` (verified not to affect activation
  stats — `llama_decode` sits outside every `compute_ppl` guard). Irrelevant to GPTQ, but if
  you rebuild any GGUF for comparison you will hit it.
- **`step()` idempotency is existence-based.** Change what a stage produces → change the
  output filename, or stale artifacts get silently reused under a new label.

---

## 9. Deliverables

1. `out/exp-060-w4a16-32k/checkpoint/` with a valid `quantization_config` and
   `quant_tuner_ptq.json`.
2. A **new HF-side KLD harness** (§6) + a results CSV over all six eval distributions.
3. SWE agent bench row(s) with the corrected error classification.
4. A short note on whether W4A16 lands near the IQ4_XS row, and if not, which of §3's traps
   explains it.
5. Serving confirmation: `vllm serve` comes up, quantization auto-detected (not silently
   bf16), and a long-context retrieval check at ~30k tokens — the point of calibrating at 32K
   is long-context behaviour, so test it rather than assuming.

## 10. Open items from the GGUF session (context, not tasks)

- **Ladder B** (`out/exp-060-32k-r16`, reasoning share 21.7%): corpus built, imatrix not run
  (~4.2 h). The A/B question is what reasoning weight in the calibration mix buys.
- **HF upload** of the GGUF ladder is gated on the user's token;
  `out/exp-060-32k/release/upload_to_hf.py` is dry-run-by-default and refuses `--push` while
  any `*pending*` placeholder remains in the card.
- **AWQ** was requested and never started. `recipes/iq2_m_awq.yaml` etc. exist; the AWQ branch
  collects its imatrix on the *folded* F16 (see CLAUDE.md) — do not reuse an unfolded one.
