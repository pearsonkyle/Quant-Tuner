# New-session prompt — a ternary/QAT pipeline for `google/gemma-4-E4B-it`

> Paste everything below the line into a fresh Claude Code session in this repo.
> Pure research: we want to know what *could* work, not to ship a model.
> The desk-work facts in "Verified before you start" were measured on 2026-08-17;
> re-check anything you intend to lean on.

---

## Goal

Work out whether we can produce a usable **ternary** (or near-ternary) `gemma-4-E4B-it`,
and by what route. Two candidate paths, neither decided:

* **Path A — logit distillation.** Ternarize E4B and train it against a 31B teacher's
  output distribution (offline top-K KD).
* **Path B — gradual ternarization.** Walk E4B onto the ternary grid on our own training
  corpus, then use logit distillation as a repair/polish stage.

**Start from `google/gemma-4-E4B-it-qat-q4_0-unquantized`, not the stock release** — see
"Verified before you start". It is the QAT-trained distribution in dense bf16, which is a
strictly easier ternarization target and costs nothing to adopt.

I want a reasoned design, the cheapest experiments that would falsify it, and one GPU
validation run at the end. **Negative results are valuable** — "path A is dead because X"
delivered in an hour beats a plausible plan that dies after a day of training.

---

## THE FRAMING THAT MATTERS MOST

Our existing ternary pipeline (`src/quant_tuner/qat/`, `docs/ternary_qat.md`,
`docs/ternary_qat_reproduce.md`) trains `prism-ml/Ternary-Bonsai-8B`, which is **natively
ternary** — it ships as `w = s·c`, `c ∈ {−1,0,+1}`, and `qat/ternary.py` reproduces those
weights *exactly* at step 0. Every tuned number we have assumes the model starts **on** the
grid and we are nudging codes across thresholds.

gemma-4-E4B is dense bf16 — even the QAT checkpoint, whose weights were *shaped by* 4-bit
quantization but are not stored on any ternary grid. So this is still "ternarize a dense
model", a strictly harder problem than continued QAT, and our headline constants (lr 5e-4,
~0.7% code flips as the sweet spot, the ~2.2-epoch schedule) **do not transfer**. Treat
them as inspiration. If you find yourself reusing one, say so explicitly and justify it.

The QAT base helps with the *distance* to the grid, not with the *kind* of problem.

Expect the naive approach — hard-ternarize everything at step 0, then fine-tune — to
destroy the model. The interesting design space is *how to get there gradually*.

---

## Verified before you start (desk work already done — build on it, don't redo it)

### START FROM THE QAT CHECKPOINT, NOT THE DENSE RELEASE

Google's **Gemma 4 QAT Q4_0** collection ships three flavours per size. The one to use is
`*-qat-q4_0-**unquantized**`: those are the **QAT-trained weights stored densely in bf16**,
with **no `quantization_config`**. Verified:

| repo | arch | dtype | quantization_config | vocab / hidden / layers |
|---|---|---|---|---|
| `google/gemma-4-E4B-it-qat-q4_0-unquantized` | `Gemma4ForConditionalGeneration` | **bf16** | **absent** | 262,144 / 2,560 / 42 |
| `google/gemma-4-31B-it-qat-q4_0-unquantized` | `Gemma4ForConditionalGeneration` | **bf16** | **absent** | 262,144 / 5,376 / 60 |

Why this is the right base, and materially easier than the dense `gemma-4-E4B-it`:

* Its weight distribution has already been **shaped by quantization-in-the-loop**, so it
  is far more tolerant of coarse rounding than a stock bf16 release. Google paid for that
  training; take it.
* It is **dense**, so it loads as an ordinary model and `qat/ternary.py`'s TWN applies
  directly — no int4 unpacking, no dequantization step to get wrong.
* The alternative, `gemma-4-E4B-it-qat-w4a16-ct`, is `compressed-tensors` /
  `pack-quantized`: symmetric **int4, group strategy, group_size 32**, `targets:
  ['Linear']`, with **250 ignored modules** (vision tower, etc.). Ternarizing from there
  means unpacking to dense first *and* inheriting a group-32 lattice that may fight a
  ternary per-group scale. Use it as a **reference for which modules Google considered
  quantizable** — that `ignore` list is a free answer to "what stays bf16" — but not as
  the starting weights.
* There are also `*-qat-q4_0-gguf` (llama.cpp Q4_0) and `*-unquantized-assistant` (the MTP
  drafters) variants, plus `E2B`, `12B` and a `26B-A4B` MoE if you want a smaller pilot or
  a different teacher.

### KD between the sizes is FEASIBLE — the gating question, and it passes

Both pairings verified with `kd_precompute.tokenizer_compatibility()`:

```
student = E4B-it-qat-q4_0-unquantized
  teacher = gemma-4-31B-it-qat-q4_0-unquantized   OK  all 262144 ids agree
  teacher = gemma-4-31B-it                        OK  all 262144 ids agree
```

So `qat/kd_precompute.py` + `--kd-table` work as-is on the tokenizer front. **Which 31B to
distil from is an open design question worth stating in your write-up**: the dense
`gemma-4-31B-it` is the stronger teacher, while `gemma-4-31B-it-qat-q4_0-unquantized`
produces a distribution that a coarsely-quantized student may find easier to reach. Cheap
to test both — precompute is forward-only and the teacher never enters the training loop.

**gemma-4's chat markers are NOT gemma-2/3's, and this WILL silently break a ported probe.**

```
<|turn>  = 105        <turn|> = 106        <bos> = 2      <eos> = 1
generation prompt:  '<bos><|turn>user\nhi<turn|>\n<|turn>model\n'
```

* The assistant role is **`model`**, not `assistant`.
* `<end_of_turn>` / `<start_of_turn>` (gemma-2/3) are **absent** — they tokenize to 7
  junk tokens each and `convert_tokens_to_ids` returns `<unk>` (3). Anything that greps
  for them silently matches nothing.
* The stop token for a turn is **`<turn|>` = 106**, the analogue of Qwen's
  `<|im_end|>` = 151645.

Two places hard-code the Qwen forms and must be ported, not reused:

* `qat/stop_probe.py` — `STOP_PIECE`, the probe prompts, the tool-call markers.
* `qat/corpus.py` — `_ASST_RE = r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>"` becomes
  something like `r"<\|turn>model\n(.*?)<turn\|>"`. Get this wrong and the loss mask
  silently covers the wrong spans, which no aggregate statistic will reveal.

**Vocab is 262,144 — 1.7× Qwen's 151,669.** Every logits-memory number in our code was
sized for the smaller vocab. The chunked `lm_head` path (`LOGIT_CHUNK`) and the KD top-K
table both scale with it; re-derive, don't assume.

---

## What to reuse (built, tested, and currently in use)

| module | what it gives you |
|---|---|
| `qat/ternary.py` | per-group TWN straight-through estimator |
| `qat/train.py` | masked-CE trainer, chunked lm_head, adafactor, TF32, `run_config.json` |
| `qat/stop_probe.py` | **in-training** P(stop) probe, `--probe-every` |
| `qat/kd_precompute.py` + `qat/kd_table.py` | offline top-K KD, **no teacher in GPU memory** |
| `scripts/run_kd_qat.sh` | precompute → train → A/B table, one command |
| `scripts/inspect_corpus_window.py --audit` | corpus structural audit |
| `scripts/verify_optimizer.sh`, `verify_corpus_fix.sh` | 60-step A/B harnesses |
| `scripts/qat_registry.py` | run ledger; joins config + curves + evals |

**Read `docs/ternary_qat_curriculum.md` first.** It records four levers we tested against a
termination collapse and eliminated by measurement — stop-weight, corpus defects, optimizer
choice, learning rate — plus why KD became the remaining candidate. That is a day of work
you do not need to repeat.

---

## Known traps — carried forward; verify each for gemma

1. **TERMINATION IS THE FAILURE MODE.** Every trained run we did broke the stop decision:
   P(stop | completed sentence) went 0.009 → ~0.95, and the agent then either stopped
   mid-task or looped forever (60 tool calls, the same command 58× in a row). **Masked-CE
   validation cannot see this** — one run's val was flat for 225 steps while it happened.
   Port the stop probe to `<turn|>`/`model` and run it from step 1. Highest-value
   instrument we have; a 60-step run now answers what once cost 11 hours.

2. **gemma-4 has cross-layer shared KV.** `shared_kv_states` flows from share-source layers
   into later sliding layers. It already broke llm-compressor's sequential pipeline
   (CLAUDE.md, `vllm_export`: gemma-4 needs `--pipeline basic`). Assume it interacts badly
   with any layer-wise ternarization schedule and with `train.py`'s `--trained-tail`
   prefix-context path. Test early and cheaply.

3. **E4B is not a plain CausalLM.** `Gemma4ForConditionalGeneration` — vision tower, audio
   tower, per-layer embeddings (PLE), and an MTP-style assistant drafter. Decide explicitly
   which tensors are ternarization candidates and which stay bf16. `lm_head` and embeddings
   almost certainly stay: see the `vllm_export` notes on the pruned-head rare-token
   disaster ("Pineple"). `AutoModelForCausalLM` may silently select a text-only class that
   matches none of the checkpoint's tensors — inspect the **live module tree**, and note
   `DEFAULT_IGNORE` in `vllm_export/w4a16.py` is already gemma-shaped.

4. **SCOPE THE SERVING PATH BEFORE TRAINING.** Decide early where a ternary gemma-4 would
   actually run, because a model we cannot serve is a research note, not a deliverable.
   Since we are on CUDA, **compressed-tensors + vLLM is the primary candidate**
   (`src/quant_tuner/vllm_export/`, already validated E2E on gemma-4-E4B at W4A16,
   ~68 tok/s) — and note vLLM has no native 1.58-bit kernel, so "ternary" may have to be
   *stored* in a supported low-bit container, or served via a custom kernel, or evaluated
   in-framework only. Our Q2_0 GGUF export is ftype 41, exists only in
   `vendor/llama.cpp-prism`, and was built for the Bonsai architecture; whether it can
   express a ternary gemma-4 is open. Answer "what would we serve this with?" in the desk
   phase, and say plainly if the honest answer is "nothing yet, this is an in-framework
   research result".

5. **Prior art in this repo — and why one piece of it probably does NOT apply.**
   `scripts/exp034_release_v3.py` drops **QAT IQ2_M as broken (PPL ~2e10)** on
   gemma-4-31B. Do not treat that as evidence against this project: it was a **llama.cpp
   GGUF k-quant** result on a *different size*, and the natural serving target here is
   CUDA via vLLM / compressed-tensors, where the IQ2_M block format is simply not
   involved. Worth a paragraph in your write-up establishing whether it was a format
   pathology or a gemma-at-low-bit pathology — the distinction decides how much it should
   worry us — but it should not gate the design. The genuinely encouraging prior is the
   other direction: Google ships QAT checkpoints for this exact model, so its tolerance
   for quantization-in-the-loop is established.

6. **`rmsnorm_plus_one=False` for gemma** (an AWQ-path gotcha already recorded). Any
   norm-folding you do needs the same care.

---

## Suggested ladder — cheapest falsifying experiment first

0. **Desk work, no GPU.** Module tree + tensor inventory of E4B; what fraction of params
   are actually ternarizable; whether the prism fork can express ternary gemma; re-confirm
   tokenizer compatibility. Kill either path here if you can.
1. **Ternarization damage probe.** Hard-ternarize E4B group by group (by depth, by tensor
   type) with **no training**, measuring perplexity + the stop probe after each. One
   forward pass per configuration. This maps where the model is fragile and is the single
   most informative cheap experiment available.
2. **Design the gradual schedule from (1)'s damage profile** — depth order, a ramped mixing
   coefficient, freezing the most-damaged tensors in bf16, or a partial-ternary target.
3. **Short KD runs (60 steps)** on the A/B harness, watching the stop probe *and* code
   flips together. Neither alone is sufficient: a run can hold termination and learn
   nothing (we measured exactly that at lr 2.5e-4), or learn and collapse.
4. **Only then** a long run.

---

## Constraints and etiquette

* **The GPU is likely BUSY with another experiment.** Do all desk work, code, unit tests
  and CPU-only analysis first. Queue GPU work behind an `nvidia-smi` guard — copy the one
  in `scripts/run_swe_mimic.sh`, which refuses rather than contending (a starved run reads
  as a bad model). Check `out/exp-058/` for what is running before you take the card.
* **Disk is the binding constraint, not VRAM.** A Bonsai checkpoint is 27.8 GB and the
  trainer writes the new one *before* pruning the oldest; gemma checkpoints will differ but
  the pattern holds. Check `df -h /workspace` and read the `require_disk` / `prune_round`
  helpers in `scripts/run_curriculum_qat.sh`.
* **No multi-hour run without a short A/B that justifies it.** Our rule: anything over an
  hour should have a 60-step version that would have caught the failure.
* Commit incrementally, with the reasoning in the message, not just the what.

---

## Deliverable

1. A design + feasibility write-up in `docs/` covering: which path looks viable and why,
   what the desk work ruled in or out, the damage profile from experiment (1), and a
   concrete recommended schedule with its open risks.
2. **One** GPU validation run testing the single riskiest assumption in that design, with
   the stop probe on and its result recorded.

If the honest answer is "neither path is viable, because X", write that up with the
evidence and stop. That is a successful outcome.
