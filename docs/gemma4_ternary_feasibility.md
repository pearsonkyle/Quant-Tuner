# Ternary `gemma-4-E4B-it` — desk feasibility

Status: **desk phase complete; no training run yet.** Everything below was measured on
2026-08-17/18 against the checkpoints themselves, on CPU apart from a ~5 GB slice of the GPU
for the round-trip generation, while the card was busy with the Qwen KD chain. Companion to
`scripts/TERNARY_GEMMA4_E4B_RESEARCH_PROMPT.md`, which set the questions.

Scripts: `scripts/gemma4_ternary_damage.py` (weight-space), `scripts/gemma4_layer_damage.py`
(output-space), `scripts/measure_stop_baseline.py` (termination baseline). All CPU-only.

**Runbook: `docs/gemma4_ternary_reproduce.md`** — every command, in order, with the check
that must pass at each step. The measurement outputs behind every table below are tracked
under `docs/gemma4_ternary/` (the `out/` tree is gitignored, so those files are the record).

---

## Summary — the three findings that should shape the plan

1. **Only half of E4B can be ternarized at all.** 3.973 B of 7.941 B params are
   `nn.Linear`; 3.490 B are embedding tables, 2.819 B of that in the per-layer-embedding
   (PLE) table alone. Ternarizing *every* linear and leaving the rest bf16 yields an
   **8.8 GB** artifact — **1.96× larger than the 4.5 GB Q4_0 Google already ships**. The
   size case for this project is much weaker than "1.58-bit" suggests; see
   [Size economics](#size-economics).

2. **The QAT base helps, but by ~3%, not by a category.** `gemma-4-E4B-it-qat-q4_0-unquantized`
   is literally the Q4_0-rounded weights stored densely (≤16 distinct values per block of
   32). Its weight-space distance to the ternary grid is ~3% lower than the stock release's
   — real, consistent across every tensor kind, and free to adopt, but it does not change
   the difficulty class. Under TWN it is statistically **indistinguishable from Gaussian**.

3. **Weight-space damage carries no scheduling signal; output-space damage carries a lot.**
   Every tensor scores ~0.43 relative Frobenius error — the Gaussian value — so the
   "ternarize what moves least first" ordering is unreadable from weights. Measured in
   *output* space the spread is **139×**, and it inverts the weight-space ranking:
   `mlp.down_proj` looks middling on weights and is by far the worst on outputs.

4. **Damage compounds exponentially, and that — not the ordering — is the design
   constraint.** Ternarizing layers 6 at a time along the best ordering multiplies KLD by
   ~2 at every step (0.105 → 0.269 → 0.610 → 1.222 → 2.171 → 5.288 → **10.666**), ending at
   4.7% top-1 agreement with the dense model. The individual layer KLDs sum to 2.830, so
   the whole is **3.77× superadditive**. A progressive schedule therefore has to *train
   between stages*; re-ordering alone just changes where on the exponential you start. The
   good news is the top of that curve: **6 layers is nearly free** and 12 is cheap.

The serving question, which the research prompt flagged as possibly fatal, is **not** a
blocker — the round-trip has been run end to end: see [Serving path](#serving-path).

---

## Inventory

`Gemma4ForConditionalGeneration`, 2076 tensors, all bf16, 15.88 GB.

| group | params | share |
|---|---:|---:|
| **language_model — Linear (ternarization candidates)** | **3.973 B** | **50.0%** |
| language_model — `embed_tokens_per_layer` (PLE table, 262144×10752) | 2.819 B | 35.5% |
| language_model — `embed_tokens` (262144×2560, tied to lm_head) | 0.671 B | 8.4% |
| audio_tower | 0.305 B | 3.8% |
| vision_tower + embed_vision/audio | 0.173 B | 2.2% |
| norms | 0.011 B | 0.1% |

Ternarizable breakdown — the MLP is 83% of it:

| tensor | shape | count | params |
|---|---|---:|---:|
| `mlp.gate_proj` / `up_proj` | 10240×2560 | 42 each | 1.101 B each |
| `mlp.down_proj` | 2560×10240 | 42 | 1.101 B |
| `self_attn.q_proj` | 2048×2560 (4096×2560 on full-attn layers) | 42 | 0.257 B |
| `self_attn.o_proj` | 2560×2048 | 42 | 0.257 B |
| `self_attn.k_proj` / `v_proj` | 512×2560 | **24 each** | 0.037 B each |
| `per_layer_input_gate` / `per_layer_projection` | 256×2560 / 2560×256 | 42 each | 0.028 B each |
| `per_layer_model_projection` | 10752×2560 | 1 | 0.028 B |

Every `in_features` is divisible by 128, so the existing `qat/ternary.py` group-128 TWN
applies to all of them with no padding.

### Architecture facts that constrain a schedule

* **42 layers; KV is shared from layer 24 on.** `num_kv_shared_layers: 18` — layers 0–23
  own `k_proj`/`v_proj`/`k_norm`; layers 24–41 have **none** and consume earlier layers'
  KV. Any depth-ordered schedule crosses that boundary at layer 24, and the two halves are
  not the same kind of thing.
* 7 `full_attention` layers (5, 11, 17, 23, 29, 35, 41); the other 35 are `sliding_attention`
  at window 512. Full-attention layers have a wider `q_proj` (`global_head_dim` 512).
* `tie_word_embeddings: true`, `final_logit_softcapping: 30.0`, vocab 262,144.

### Checkpoint hygiene — a second reason to use the QAT repo

| repo | ckpt tensors | dropped on load |
|---|---:|---:|
| `gemma-4-E4B-it-qat-q4_0-unquantized` | 2076 | **0** |
| `gemma-4-E4B-it` (stock) | 2130 | **54** |

The stock release ships `k_proj`/`v_proj`/`k_norm` for layers 24–41 — the 18 KV-sharing
layers that have no such modules. `from_pretrained` discards them silently. This is the
`dropped_tensors()` failure mode from the `vllm_export` notes, present in the stock repo
and absent from the QAT one. Configs are otherwise byte-identical.

---

## The QAT checkpoint is the Q4_0 grid, dequantized

Measured on `layers.10.mlp.down_proj.weight` (2560×10240):

| block size | distinct values per block (mean / max) |
|---|---|
| 32 | **12.2 / 16** |
| 64 | 23.7 / 27 |
| 128 | 45.0 / 50 |

1,888 distinct values in 26.2 M elements; kurtosis 5.46. That is Q4_0 exactly — 16 levels
per block of 32 — so "unquantized" means *dequantized*, not *pre-rounding latents*. Google
did not release the QAT latent weights, and we start from a model that is already
discretized once.

---

## Weight-space damage: flat, and therefore useless for ranking

Per-group TWN (g128, thresh 0.7), relative Frobenius error `‖W−Ŵ‖/‖W‖`, averaged over
tensors of each kind:

| tensor kind | QAT base | stock base | frac zero (QAT) |
|---|---:|---:|---:|
| `self_attn.o_proj` | 0.4196 | 0.4343 | 0.411 |
| `mlp.up_proj` | 0.4363 | 0.4517 | 0.419 |
| `self_attn.q_proj` | 0.4381 | 0.4543 | 0.422 |
| `mlp.gate_proj` | 0.4381 | 0.4536 | 0.420 |
| `mlp.down_proj` | 0.4399 | 0.4560 | 0.421 |
| `self_attn.v_proj` | 0.4550 | 0.4388\* | 0.431 |
| `self_attn.k_proj` | 0.4562 | 0.4399\* | 0.430 |
| `per_layer_input_gate` | 0.4860 | 0.5070 | 0.443 |

\* the stock k/v averages pool 42 tensors including the 18 dead ones, so they are not
comparable to the QAT column's 24.

**Reference points measured with the same code:** a synthetic tensor already on the grid
round-trips at 2.4e-04 (the step-0 exactness property `qat/ternary.py` is built around);
i.i.d. **Gaussian scores 0.4350 with frac_zero 0.4224**; i.i.d. uniform scores 0.3327.

Every real tensor lands on the Gaussian value, and the observed zero-fractions land on the
Gaussian prediction too. So:

* Q4_0-QAT training left **no ternary-friendly structure** in the weights — it concentrated
  them onto a 16-level lattice, which is not the same as concentrating them onto three levels.
* The QAT base *is* consistently closer to the ternary grid than stock (lower in all six
  comparable kinds), by ~0.014 absolute / ~3% relative. Take it — it is free and the
  checkpoint is cleaner — but do not budget for it.
* **The spread is 0.42–0.49.** There is no ordering to extract here. A schedule built on
  "ternarize the tensors that change least first" cannot be driven by this metric.

---

## Output-space damage: this is where the schedule comes from

Ternarize one group of tensors, change nothing else, re-run the same tokens, and measure
`KLD(dense ‖ ternarized)`. Eval is 3 × 2048 tokens, one window from each of three distinct
held-out conversations of our own SFT corpus (`split=test`). Full data in
`out/gemma4-ternary/layer_damage.json`.

| tensor kind (all 42 layers) | KLD | top-1 agree | PPL |
|---|---:|---:|---:|
| `per_layer_model_projection` | **0.0086** | 0.977 | 4.22 |
| `self_attn.k_proj` | 0.0867 | 0.900 | 4.28 |
| `per_layer_input_gate` | 0.0890 | 0.910 | 4.27 |
| `self_attn.q_proj` | 0.1470 | 0.863 | 4.49 |
| `self_attn.o_proj` | 0.2665 | 0.832 | 4.89 |
| `mlp.up_proj` | 0.3444 | 0.803 | 4.90 |
| `self_attn.v_proj` | 0.3495 | 0.793 | 5.11 |
| `mlp.gate_proj` | 0.3703 | 0.787 | 4.83 |
| `per_layer_projection` | 0.4264 | 0.778 | 5.09 |
| **`mlp.down_proj`** | **1.1990** | **0.658** | **12.91** |

Two things to take from this.

**`mlp.down_proj` alone breaks the model.** At 1.199 it is **3.4× the next-worst kind**, and
the only one whose perplexity leaves the 4.2–5.1 band — it takes it to 12.91. This is the same
tensor llama.cpp's k-quant mixes bump a tier at low bit-width (`ffn_down` in
`_quant_mix.target_type_for_member`) — independent corroboration rather than a coincidence.

**Weight space would have ranked it fourth-safest.** `down_proj` (0.4399), `gate_proj`
(0.4381) and `up_proj` (0.4363) are indistinguishable on weights and span 0.71 → 2.39 in
output KLD. Any schedule ordered by weight movement would have ternarized `down_proj`
early, and would have looked fine right up until the model stopped working. **Order the
schedule on output-space damage, measured, not on weight movement.**

---

## Depth: the KV-donor layers are the fragile ones

Same measurement, one whole decoder layer at a time:

| | least damaging | | most damaging | |
|---|---|---:|---|---:|
| 1 | `layer.03` | 0.0087 | `layer.22` | **0.6255** |
| 2 | `layer.01` | 0.0102 | `layer.23` | **0.2705** |
| 3 | `layer.00` | 0.0125 | `layer.17` | 0.1376 |
| 4 | `layer.02` | 0.0137 | `layer.21` | 0.1245 |

**Layers 22 and 23 are 30–70× more damaging than layers 0–3**, and there is a mechanism:
they are the *last KV-donor layers*. Layers 24–41 have no `k_proj`/`v_proj` of their own and
consume the KV produced upstream, so error injected at 22–23 propagates into all 18 layers
below it. Layer 24 immediately falls back to 0.065. The `full_attention` layers (11, 17, 23)
also rank high, consistent with the same story — they are the ones whose KV is not
window-limited.

This is the depth order a schedule should follow, and it is not the one intuition suggests
(neither "shallow first" nor "deep first" — it is "away from the KV-sharing boundary first").

---

## The compounding is the real answer: damage doubles every 6 layers

Walking the least-damaging-first ordering, ternarizing 6 more layers at each step, with **no
training at any point**:

| layers ternary | KLD | top-1 agree | PPL | vs previous |
|---:|---:|---:|---:|---:|
| 6 | 0.105 | 0.899 | 4.10 | — |
| 12 | 0.269 | 0.836 | 4.71 | 2.57× |
| 18 | 0.610 | 0.755 | 6.18 | 2.27× |
| 24 | 1.222 | 0.651 | 10.67 | 2.00× |
| 30 | 2.171 | 0.517 | 25.42 | 1.78× |
| 36 | 5.288 | 0.208 | 541.77 | 2.44× |
| **42** | **10.666** | **0.047** | **102,989** | 2.02× |

Two numbers matter here.

**Damage is 3.77× superadditive.** The individual layer KLDs sum to 2.830; ternarizing all 42
at once costs 10.666. Errors do not just accumulate, they interact.

**The growth is close to exactly exponential in the layer COUNT** — every 6-layer step
multiplies KLD by ~2 (2.57, 2.27, 2.00, 1.78, 2.44, 2.02), i.e. `KLD ≈ 0.105 · 2^((n−6)/6)`
across two and a half orders of magnitude. That regularity is the most useful thing the
profile produced, and it has a direct design consequence:

> **A progressive schedule cannot just re-order the same one-shot damage — it has to train
> between stages.** If you ternarize monotonically without recovery, you are riding an
> exponential, and the ordering only buys you which end of it you start at.

The encouraging half is the top of the table: **6 layers is nearly free** (PPL 4.10, against
4.06–4.09 for any single layer alone) and 12 is cheap (4.71). So ~6 layers per stage, each
followed by enough training to pull KLD back down before the next stage, is the schedule the
data actually supports — not "all 42 with a good ordering".

The bottom of the table is why the untrained round-trip GGUF emits token soup: at 42/42 the
model retains 4.7% top-1 agreement with itself.

---

## Size economics

Q2_0 in the prism fork is `QK2_0 = 128` weights + one fp16 scale = **2.125 bits/weight**
(the 1.58-bit figure is the information-theoretic rate, not the stored one). Against the
5.15 GB Q4_0 GGUF Google ships (4.49 GB by the same accounting, excluding the mmproj):

| configuration | trunk | embeddings | towers | total | vs Q4_0 |
|---|---:|---:|---:|---:|---:|
| bf16 dense (what we start from) | 7.95 G | 6.98 G | 0.96 G | 15.90 G | 3.54× |
| **Q4_0 — what Google already ships** | 2.23 G | 1.96 G | 0.27 G | **4.49 G** | 1.00× |
| ternary trunk, everything else bf16 | 1.06 G | 6.98 G | 0.96 G | 9.00 G | **2.00×** |
| ternary trunk, `down_proj` kept Q4_0, embeddings Q4_0, towers Q8 | 1.38 G | 1.96 G | 0.51 G | 3.85 G | 0.86× |
| ternary trunk (all of it), embeddings Q4_0, towers Q8 | 1.06 G | 1.96 G | 0.51 G | 3.53 G | 0.79× |

**Read the third row.** Ternarizing every linear weight in the model, with the embeddings
left alone, produces an artifact *twice the size* of the quantization Google publishes. The
44% of E4B that is embedding tables does not care what we do to the trunk.

So the honest framing: **a fully ternary E4B is a ~21% size win over Q4_0, not a 2–4× one**,
and only if the embeddings are separately quantized to Q4_0 — and the PLE table is exactly
the rare-token-degradation risk the `vllm_export` notes record ("Pineple"). Keeping
`down_proj` at Q4_0 costs 0.33 GB of that and buys back the single most damaging tensor:
**0.86× vs 0.79×**, for what the damage table says is most of the quality.

This does not kill the project, but it changes what it is: a **quality-at-low-bit research
result** (and possibly a throughput one, if ternary kernels are faster), not a footprint win.

### E4B is the worst member of the family for this, and that is the argument for doing it

The 21% is an E4B fact, not a gemma-4 fact. Measured from both checkpoints' own tensor
shapes:

| | ternarizable linears | embeddings | towers | ternary trunk + Q4_0 emb, vs the same at Q4_0 |
|---|---:|---:|---:|---:|
| **E4B** | 3.918 B — **49.3%** | 3.496 B — 44.0% | 0.472 B | 3.51 G vs 4.67 G = **0.75×** |
| **31B** | 29.287 B — **93.6%** | 1.415 B — 4.5% | 0.570 B | 9.18 G vs 17.88 G = **0.51×** |

Half of E4B is embedding table that ternarization cannot touch; on the 31B it is 4.5%.
So the same technique that buys ~25% here buys ~2× there, and the case for running the
hard, cheap model first is that **it is the hard one** — a method that survives E4B's
economics is not being flattered by them.

Three things also get easier with scale, not harder:

* **The teacher question dissolves.** A 31B student's natural teacher is the dense 31B
  itself — self-KD, which is what quantization-aware distillation wants anyway. No
  separate model resident, no tokenizer gate to pass, and none of the
  different-model confound that makes "KLD vs dense" ambiguous at E4B scale.
* **No KV-donor cliff.** E4B's two worst layers are 22–23 (KLD 0.6255 / 0.2705) because
  they are the last KV donors for the 18 sharing layers above them, and layer 24 falls
  back to 0.065. The 31B declares `num_kv_shared_layers: 0`, so that structure does not
  exist and its depth profile should be flatter to schedule.
* **More redundancy.** E4B is a MatFormer-style efficiency model, already
  information-dense; there is less slack in it to give up than in a conventional dense
  31B.

**The obstacle, which the schedule happens to solve.** fp32 latents for 29.3 B params are
**117 GB**, over a 95 GB card — and fp32 is not optional here, since bf16 underflows the
TWN threshold and no codes flip. But only *trainable* layers need fp32 latents: at 6 of
60 layers that is ~11.7 GB fp32 plus the frozen remainder in bf16, roughly 70 GB, which
fits. `--dtype` is model-wide today, so mixed-dtype freezing is real work — but it means
the progressive schedule is not only a quality strategy at 31B scale, it is what makes
the run possible at all.

Everything else transfers as MECHANISM and nothing as CONSTANT: re-measure the
output-space damage ordering, the per-kind ranking, the stop-probe baseline and the lr,
exactly as this study had to.

---

## Serving path

Better than the research prompt assumed. `vendor/llama.cpp-prism` already has **both** halves:

* **Architecture** — `conversion/gemma.py` defines `Gemma4Model`, `Gemma4UnifiedModel`,
  `Gemma4AssistantModel`, `Gemma4VisionAudioModel`; `src/models/gemma4.cpp` and
  `gemma4-assistant.cpp` implement it; `llama-arch.cpp` carries the PLE tensors
  (`per_layer_token_embd`, `blk.%d.inp_gate`, `blk.%d.proj`, …).
* **Format** — `GGML_TYPE_Q2_0 = 42`, `GGML_FTYPE_MOSTLY_Q2_0 = 28`, `QK2_0 = 128` with one
  `ggml_half` scale per block. That is **exactly** the grouping and scale dtype
  `qat/ternary.py` quantizes to, so the training forward and the deployed kernel agree by
  construction.

So `qat/export.py`'s latents → F16 GGUF → Q2_0 path is architecturally reusable. One change
is required: it currently forces `--token-embedding-type Q2_0` and
`--output-tensor-type Q2_0`, which for E4B would put the 262k×10752 PLE table on the ternary
grid. Given the size table above (that saves 1.2 GB) and the rare-token risk, this needs to
become a flag, with Q4_0 the default for embeddings.

### Round-trip: measured, and it works

Done end to end on CPU + a 5.5 GB slice of the GPU, with no training:

```
convert_hf_to_gguf.py <qat snapshot> --outtype f16            -> 666 tensors, 14,236 MiB
llama-quantize --token-embedding-type q4_0 \
               --tensor-type per_layer_token_embd=q4_0  … Q2_0 ->        2,926 MiB
```

Both overrides landed where intended — `per_layer_token_embd` 5376 → 1512 MiB and
`token_embd` 1280 → 360 MiB at Q4_0, while `blk.*` went to Q2_0 at the expected 2.125 bpw
(`attn_q` 10.00 → 1.33 MiB). The model **loads and generates** on the prism fork.

The control is what makes the result readable. Same F16 GGUF, same converter, same fork,
same prompt, same greedy settings — only the trunk type differs:

| trunk | embeddings | size | BPW | output for *"What is the capital of France?"* |
|---|---|---:|---:|---|
| **Q4_0** | Q4_0 | 4,043 MiB | 4.54 | *"The capital of France is Paris."* |
| **Q2_0 (ternary)** | Q4_0 | 2,926 MiB | 3.29 | `земeling บ GUNኝነት वइसटू गोवा porn memungkinkan…` |

Three things follow.

1. **The serving path is real.** gemma-4 + per-layer embeddings + Q2_0 converts, quantizes,
   loads and runs on `vendor/llama.cpp-prism`. Nothing about the format or the architecture
   blocks this project.
2. **The garbage is the ternarization, not the plumbing.** A one-variable A/B is what
   licenses that claim; without the Q4_0 arm the token soup would equally well be a broken
   converter.
3. **The size prize, measured rather than estimated: 2,926 vs 4,043 MiB — 28% smaller than
   the same-pipeline Q4_0.** (Against Google's shipped 5.15 GB Q4_0 GGUF it is 40%, but
   that file makes different choices about embedding precision, so the like-for-like number
   is 28%.) Of the 2,926 MiB, **1,872 is embeddings and only ~1,054 is the ternary trunk** —
   the artifact is two-thirds embedding table even after ternarizing every linear weight.

This is the expected starting point, not a failure: hard-ternarizing a dense model with no
training destroys it, which is the whole premise of doing QAT. What the round-trip buys is
that the deliverable end of the pipeline is no longer an unknown.

---

## Corpus and probe port — the trap that will bite silently

**gemma-4 renders an entire tool-calling exchange as ONE `model` turn**, with the tool
results embedded inside it. Measured render of a two-round tool loop:

```
<|turn>model
<|tool_call>call:bash{command:<|"|>ls<|"|>}<tool_call|><|tool_response>response:bash{value:<|"|>a.py b.py<|"|>}<tool_response|>Looking now.
<|tool_call>call:bash{...}<tool_call|><|tool_response>response:bash{...}<tool_response|>Now grep.Fixed it.<turn|>
```

Consequences, all of which differ from Qwen:

1. **A `<|turn>model … <turn|>` span is NOT a supervised span.** It contains
   `<|tool_response>`…`<tool_response|>` blocks, which are environment output. Porting
   `corpus.py`'s `_ASST_RE = r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>"` to
   `r"<\|turn>model\n(.*?)<turn\|>"` — the port the research prompt suggests — would
   **train the model on tool output**. The mask has to be built on token ids: inside a model
   turn, minus every `[50 … 51]` span.
2. **Assistant prose renders *after* the tool call and its response**, not before. Our logs
   are prose-then-call. The reordering is on-distribution (it matches the inference loop:
   model emits the call, harness injects the response, model continues in the same turn) but
   it is a real transformation of our data and should be stated, not discovered.
3. **Consecutive assistant messages concatenate with no separator** ("Now grep.Fixed it.").
   The template does the merging `merge_consecutive_assistant` was written to do — but it
   also means the empty-assistant defect (`drop_empty_assistant`) cannot produce a
   stop-token-only turn the way it did on Qwen. Re-measure, do not assume either way.

**This is now implemented**, not just described. `qat/dialect.py` holds the rule per
family: `QwenChatDialect` keeps the character-span regex verbatim (published corpora
fingerprints depend on it byte-for-byte) and `Gemma4ChatDialect` walks token ids —
supervise from after the 3-token `<|turn>model\n` header through the terminating `<turn|>`
inclusive, minus every `[50 … 51]` span. `qat.dialect.detect(tok)` picks by vocabulary
rather than by model name, and **refuses** an unknown family instead of guessing.
`tests/unit/test_qat_dialect.py` pins it, including a test that asserts the naive regex
port would have supervised the tool result.

All control tokens are single ids: `<|turn>` 105, `<turn|>` **106** (the stop token,
`eos_token_id: [1, 106]`), `<|tool>` 46, `<tool|>` 47, `<|tool_call>` 48, `<tool_call|>` 49,
`<|tool_response>` 50, `<tool_response|>` 51, `<|"|>` 52, `<|think|>` 98, `<|channel>` 100,
`<channel|>` 101, `<bos>` 2, `<eos>` 1. As the prompt warned, `<end_of_turn>` and
`<start_of_turn>` both resolve to id **3** (`<unk>`) — anything grepping for them matches
nothing.

Reasoning renders as `<|channel>thought\n…<channel|>` and is gated to assistant turns
**after the last user turn** unless `preserve_thinking` — the same behaviour
`data/reasoning.py` documents for Qwen3.6, so `universal.reasoning_windows` transfers.

---

## The stop probe does not port — and gemma-4 has no sharp stop point

Termination is the failure mode this pipeline broke every previous time, and the in-training
stop probe is the instrument that catches it. Porting it is not a marker swap.

Measured on the shipped E4B (`scripts/measure_stop_baseline.py`, fp32, CPU), P(`<turn|>`) at
each point of the probe prefix:

| probe point | shipped E4B | Qwen/Bonsai analogue |
|---|---:|---:|
| `start` | 0.00000 | — |
| `mid_sentence` | 0.00000 | — |
| **`sentence_period`** (DIAGNOSTIC — stopping here is wrong) | **0.00274** | 0.0092 |
| `sentence_newline` | 0.01335 | — |
| `after_tool_call` | **0.00004** | **0.99995** |
| `after_tool_response` | 0.00021 | *no equivalent* |
| **`answer_after_tool`** (CONTROL — stopping here is right) | **0.07032** | — |

**`after_tool_call` means the opposite thing in the two families.** Qwen's assistant turn
*ends* at its tool call, so 0.99995 there is a clean "can still stop when it should"
control. gemma's template instead hands over to the harness at that point (it emits an
opening `<|tool_response>` as the generation prompt), and the shipped model reads 0.00004 —
correctly. Reusing Qwen's control would have inverted the test: a healthy gemma would look
like a model that had lost the ability to terminate.

So gemma-4 gets its own probe points, and `PROBE_SPECS` is now per-family (points,
diagnostic, control, baseline) rather than a single shared list.

**The honest limitation:** gemma-4 has no position where stopping is strongly preferred.
After a complete answer it puts 0.275 on `\n\n` and only 0.070 on `<turn|>`, and 0.070 was
the *highest* of every candidate tried (`after_tool_response` 0.00021, an answer with no
tool use 0.026, mid-answer 0.000). The control therefore has ~25× of headroom over the
diagnostic where Qwen's had ~10⁴. It still detects the looping direction — 0.070 → ~0 is a
real signal — but it is a weaker instrument than the one the curriculum doc was written
against, and a gemma run that moves it should be checked against a trajectory rather than
trusted on the probe alone.

---

## The Bonsai anchor ladder settled the loss stack (2026-08-19)

Between this document's desk phase and now, the Bonsai side of the pipeline completed its
first full-schedule run that survived with termination intact (`anchor6`, 613 steps, probe
diagnostic 0.0000 / control 1.0000 at all 24 readings, val loss equal to the dense
endpoint), and the ladder that got there answered the question this project was always
going to hit next: **what stabilizes ternary training when CE alone does not.** A gemma
stage should start from that stack, not rediscover it.

The pieces, and why each exists (full arc: `docs/ternary_qat_curriculum.md`):

| term | flag | what it is for |
|---|---|---|
| tail-bucket KD KL | `--kd-table … --kd-alpha 0.5` | the *stabilizer*: an every-position pull toward a dense teacher; renormalized KL is blind to out-of-support mass, the K+1-bucket form is not |
| forced-stop support | `--include-ids <stop id>` at precompute | makes the KL an exact per-position constraint on P(stop) — the stop token is outside the teacher's top-64 almost everywhere |
| one-sided stop anchor | `--stop-anchor 0.2` + per-side margins | direction-aware hinge on the stop logit; the *symmetric* form collapsed the control under Bonsai's 176:1 continue:stop imbalance |
| termination steering | `--steer-weight 0.1` | probe-family contexts as an every-step gradient, with the probe prompts themselves held out (the Goodhart guard) |
| repetition steering | `--steer-rep-weight` | one-sided hinge on P(verbatim previous command); trains away the loop failure the mimic exposed |
| grad clip | `--clip-norm 0.25` | damps the objective×data-order oscillation waves; measured, not folklore |
| per-tensor lr | `--lr-scale group-scale` | lr ∝ median TWN group scale — Adafactor's normalized step starves small-scale tensors |
| dual probe abort | `--probe-abort … --probe-abort-control … --probe-abort-patience 2` | both failure directions, with hysteresis so one oscillation trough does not kill a recoverable run |

**The finding that matters most here is about CE, and it is worse for gemma.** The dense
control experiment decomposed the Bonsai collapse into (a) a slow objective-driven leak,
(b) oscillation waves from the objective × the fixed data order, and (c) ternary
amplification: the lr a ternary model needs to flip codes at all (5e-4) **diverges the
same model run dense in 10 steps**. On Bonsai the quantizer low-pass filters those
dynamics and the stack above damps what leaks through. A from-scratch gemma stage inverts
the geometry: freshly-ternarized layers sit at ~0.43 relative error from their dense
selves and need *large* early motion, while the still-dense layers are exactly the
divergently-hot regime the control exposed — and `--dense-kind down_proj` puts dense
tensors inside every trainable layer. Expect the oscillations to be worse than Bonsai's,
run the KD KL from step 1 (it is the only term that pulls *every* position back toward a
healthy distribution), and treat a stage's untouched layers as the built-in dense control
when reading the telemetry.

What transfers as **mechanism** vs. what must be **re-measured**:

* Transfers: every row of the table above; the report/watcher/telemetry chain
  (`scripts/run_kd_anchor_qat.sh` is now fully parametric — `TABLE`, `OUT`,
  `TEACHER_PROBE`, `REP`, plus the stage flags appended); code-flip telemetry as the
  primary signal; `kd_precompute` (architecture-agnostic, and its fp32 softmax is now
  chunked over positions — the whole-tensor path OOM'd a 95 GiB card on a 32B teacher, so
  a 31B gemma teacher with a 262k-vocab softmax needs the chunked path, already default).
* Re-measured: **every number.** lr and clip were tuned where step-0 quantization error is
  zero. Anchor margins (1.0/0.1 nats) and abort thresholds (0.09/0.95) encode Bonsai's
  baseline; gemma's own baseline (§stop probe) is diagnostic 0.00274 / control 0.070, so
  the abort pair must be scaled from those (e.g. diagnostic abort at ~10× baseline ≈ 0.03;
  a control floor has only ~25× of headroom here, not 10⁴ — gate on a trajectory, not the
  probe alone). The steering batches are the one piece that is NOT yet
  ported: the stop id now flows from the dialect automatically (train.py passes the
  probe's), but `SteerBatch`/`RepBatch` bodies hardcode Qwen's `<tool_call>` /
  `<|im_start|>` markers — and, worse than markers, `SteerBatch`'s control class teaches
  "stop after emitting a tool call", which is *correct Qwen and inverted gemma* (the same
  after_tool_call trap as the probe, §stop probe): gemma's stop-is-right position is
  `answer_after_tool`. Port the context classes per `PROBE_SPECS`, don't just swap
  markers; until then run gemma stages with `--steer-weight 0` and lean on the anchor +
  KD, which are dialect-clean.
* And one is **harsher**: the built train corpus has 6,300 labeled `<turn|>` targets in
  ~6.5 M supervised tokens — **one stop decision per ~1,030 "keep going"**, 5.9× more
  imbalanced than the 176:1 that broke the symmetric anchor on Bonsai. The one-sided form
  is not optional here.

Two provenance notes for anyone reading the tracked artifacts: the corpora are built and
fingerprinted (train `0c70d992882d29a7` / val `16177b9a361cbdd7`), but
`docs/gemma4_ternary/corpus_build_train.log` predates the bfc30c2 audit fix — its
"tool-calls 57/26,389 kept" lines are the *counter bug* that commit fixed (Qwen markers
counted on a gemma render; 25,772 calls are present and supervised), not a corpus defect.
The val log postdates the fix and shows the true 100% survival. And `stop_baseline.json`
is now tracked there, so the probe is interpretable from step 1.

---

## What is built

The pipeline is wired for gemma-4 up to the point where it needs a GPU.

* **`qat/dialect.py`** (new) — per-family supervised-span rule, id-based for gemma-4, with
  `detect()` refusing an unknown family. Qwen's rule is byte-identical to before.
* **`qat/train.py` — `decoder_layers()`** replaces the hard-coded `model.model.layers`,
  which does not exist on `Gemma4ForConditionalGeneration` (its decoder is at
  `model.language_model.layers`). `AutoModelForCausalLM` *does* resolve correctly for
  gemma-4, so the `--model-class` trap from `vllm_export` does not apply here — verified.
* **`qat/train.py` — `--ternary-layers` and `--dense-kind`** (new) — the third weight state
  a progressive schedule needs. Until now a layer was either *trainable and ternary* or
  *frozen and ternary*; on a natively-ternary model those are the only two that exist. A
  dense model needs **"still bf16, not on the grid yet"**, or every layer is ternarized at
  step 0 and there is no schedule. `--dense-kind down_proj` keeps the one catastrophic
  tensor off the grid everywhere. Weights left dense inside a *trainable* layer still get
  gradients — letting them adapt to their ternarized neighbours is most of why a partial
  schedule should beat all-at-once — and that is pinned by a test, because they are plain
  `Linear.weight`s that the name-based `requires_grad` pass cannot see.
* **`qat/stop_probe.py`** — dialect-aware markers, and `PROBE_SPECS[...].vanilla` is `None`
  for gemma until measured, so the log prints "no measured baseline" rather than quoting
  Bonsai's 0.0092 next to a gemma reading.
* **`scripts/measure_stop_baseline.py`** (new) — produces that baseline, forward-only.
* **`scripts/build_sft_qat_corpus.py --model`** — render our SFT corpus with any
  tokenizer instead of the hard-wired Bonsai one.

## Next steps

1. **Measure the gemma-4 stop-probe baseline** (`scripts/measure_stop_baseline.py`, CPU,
   minutes). The probe is uninterpretable until this exists. Note the CONTROL point does
   not mean the same thing here: after `<tool_call|>` gemma's template hands to the harness,
   so a low reading there is correct rather than a regression.
2. **Build the 32K gemma corpus** from `sft.jsonl.gz` with `--model
   google/gemma-4-E4B-it-qat-q4_0-unquantized`, then audit one window with
   `inspect_corpus_window.py --audit` before trusting it.
3. **Finish the output-space damage profile** (running): per layer as well as per kind, plus
   the cumulative walk. The cumulative curve is what answers "is fully ternary reachable at
   all" — if damage compounds superlinearly, no ordering saves it.
4. **60-step A/B on the damage-ordered schedule vs. all-at-once**, stop probe on from step 1,
   watching code flips and termination together. This is the first step that needs the GPU.
   Run both arms with the full anchor-ladder loss stack (previous section) from step 1 —
   Bonsai's ladder already paid for the evidence that CE-only arms waste the GPU time.

### Open design questions, stated rather than resolved

* **Which 31B to distil from.** Both `gemma-4-31B-it` and its QAT variant pass
  `tokenizer_compatibility()` on all 262,144 ids. Precompute is forward-only and the teacher
  never enters the training loop, so both are cheap to try. (An earlier draft claimed the
  table scales with the 1.7× vocab — it does not: the table stores top-K ids+logprobs per
  *position*, so size tracks supervised positions, and this corpus's ~6.5 M supervised
  tokens land within ~10% of the Qwen table's 2.2 GB at top-64. What the bigger vocab does
  cost is the per-window [K, 262k] fp32 softmax at precompute time — covered, see the
  loss-stack section.)
* **Does `down_proj` stay Q4_0?** The damage table says it is worth 0.33 GB to keep it there.
  Whether QAT can recover it is precisely what step 4 should test — with `down_proj` as the
  last thing the schedule turns on, if at all.
* **Where does the schedule cross layer 24?** Depth ordering has to reckon with the KV-sharing
  boundary; layers 24–41 consume KV they do not produce.
