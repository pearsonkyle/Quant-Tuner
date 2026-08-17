# Ternary `gemma-4-E4B-it` — desk feasibility

Status: **desk phase, GPU-free.** Everything below was measured on CPU on 2026-08-17
against the checkpoints themselves, while the card was busy with the Qwen KD chain.
Companion to `scripts/TERNARY_GEMMA4_E4B_RESEARCH_PROMPT.md`, which set the questions.

Scripts: `scripts/gemma4_ternary_damage.py` (weight-space), `scripts/gemma4_layer_damage.py`
(output-space). Both run on CPU with no corpus preprocessing and no GPU.

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
   *output* space the spread is **27×**, and it inverts the weight-space ranking:
   `mlp.down_proj` looks middling on weights and is catastrophic on outputs.

The serving question, which the research prompt flagged as possibly fatal, is **not** a
blocker: see [Serving path](#serving-path).

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

Ternarize one tensor kind across all 42 layers, change nothing else, re-run the same
tokens, and measure `KLD(dense ‖ ternarized)` on held-out conversations from our own SFT
corpus. Smoke run (512 tokens, `split=test`); the full 3×2048 run is in
`out/gemma4-ternary/layer_damage.json`.

| tensor kind | KLD | top-1 agree | PPL |
|---|---:|---:|---:|
| `per_layer_model_projection` | **0.035** | 0.957 | 19.20 |
| `self_attn.k_proj` | 0.206 | 0.846 | 17.90 |
| `per_layer_input_gate` | 0.330 | 0.834 | 19.12 |
| `self_attn.q_proj` | 0.352 | 0.795 | 19.50 |
| `self_attn.o_proj` | 0.581 | 0.740 | 20.19 |
| `self_attn.v_proj` | 0.603 | 0.719 | 18.91 |
| `mlp.gate_proj` | 0.706 | 0.689 | 17.69 |
| `mlp.up_proj` | 0.887 | 0.680 | 22.51 |
| `per_layer_projection` | 0.922 | 0.699 | 20.64 |
| **`mlp.down_proj`** | **2.393** | **0.484** | **148.75** |

Two things to take from this.

**`mlp.down_proj` alone breaks the model.** It is the only kind whose perplexity leaves the
~17–22 band, and it takes it to 148.75 while halving top-1 agreement. This is the same
tensor llama.cpp's k-quant mixes bump a tier at low bit-width (`ffn_down` in
`_quant_mix.target_type_for_member`) — independent corroboration rather than a coincidence.

**Weight space would have ranked it fourth-safest.** `down_proj` (0.4399), `gate_proj`
(0.4381) and `up_proj` (0.4363) are indistinguishable on weights and span 0.71 → 2.39 in
output KLD. Any schedule ordered by weight movement would have ternarized `down_proj`
early, and would have looked fine right up until the model stopped working. **Order the
schedule on output-space damage, measured, not on weight movement.**

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

Not verified yet: that the prism converter accepts *this* checkpoint end to end, and that a
Q2_0 gemma-4 actually loads and generates. That is the first thing to test, and it needs no
training — see Next steps.

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

## Next steps

Ordered cheapest-falsifying-first; nothing below needs the GPU until step 4.

1. **Finish the output-space damage profile** (running): per-layer as well as per-kind, plus
   a cumulative walk along the ranking. The cumulative curve is the one that answers "is
   fully ternary reachable at all" — if damage compounds superlinearly, no schedule saves it.
2. **Round-trip a Q2_0 gemma-4 through the prism fork with no training** — convert, quantize
   the trunk to Q2_0 with embeddings at Q4_0, load, generate. Forward-only, and it either
   validates the serving path or kills it in an afternoon.
3. **Port `corpus.py`'s mask to token ids** and `stop_probe.py` to `<turn|>`/`model`, with a
   unit test that a tool-response span is excluded from the loss. This is the piece that
   fails silently if it is wrong.
4. **60-step A/B on the damage-ordered schedule** vs. all-at-once, stop probe on from step 1,
   watching code flips and termination together.

### Open design questions, stated rather than resolved

* **Which 31B to distil from.** Both `gemma-4-31B-it` and its QAT variant pass
  `tokenizer_compatibility()` on all 262,144 ids. Precompute is forward-only and the teacher
  never enters the training loop, so both are cheap to try — but the KD table scales with the
  1.7× vocab, and the ~2.2 GB Qwen table at top-64 becomes ~3.7 GB here. Re-derive, do not assume.
* **Does `down_proj` stay Q4_0?** The damage table says it is worth 0.33 GB to keep it there.
  Whether QAT can recover it is precisely what step 4 should test — with `down_proj` as the
  last thing the schedule turns on, if at all.
* **Where does the schedule cross layer 24?** Depth ordering has to reckon with the KV-sharing
  boundary; layers 24–41 consume KV they do not produce.
