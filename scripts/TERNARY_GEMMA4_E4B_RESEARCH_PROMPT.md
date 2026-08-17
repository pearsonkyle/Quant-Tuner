# New-session prompt — a ternary/QAT pipeline for `google/gemma-4-E4B-it`

> Paste everything below the line into a fresh Claude Code session in this repo.
> Pure research: we want to know what *could* work, not to ship a model.
> The desk-work facts in "Verified before you start" were measured on 2026-08-17;
> re-check anything you intend to lean on.

---

## Goal

Work out whether we can produce a usable **ternary** (or near-ternary) `gemma-4-E4B-it`,
and by what route. Two candidate paths, neither decided:

* **Path A — logit distillation.** Ternarize E4B and train it against
  `google/gemma-4-31B-it`'s output distribution (offline top-K KD).
* **Path B — gradual ternarization.** Walk E4B onto the ternary grid on our own training
  corpus, then use logit distillation as a repair/polish stage.

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

`gemma-4-E4B-it` is dense bf16. It is **not** on that grid. This is
"ternarize a dense model", a strictly harder problem, and our headline constants
(lr 5e-4, ~0.7% code flips as the sweet spot, the ~2.2-epoch schedule) **do not transfer**.
Treat them as inspiration. If you find yourself reusing one, say so explicitly and justify it.

Expect the naive approach — hard-ternarize everything at step 0, then fine-tune — to
destroy the model. The interesting design space is *how to get there gradually*.

---

## Verified before you start (desk work already done — build on it, don't redo it)

**KD between the two sizes is FEASIBLE.** This was the single gating question and it passes:

```
tokenizers agree on all 262144 shared ids (student 262144, teacher 262144)
```

| | architecture | vocab | hidden | layers |
|---|---|---|---|---|
| `google/gemma-4-E4B-it` (student) | `Gemma4ForConditionalGeneration` | 262,144 | 2,560 | 42 |
| `google/gemma-4-31B-it` (teacher) | `Gemma4ForConditionalGeneration` | 262,144 | 5,376 | 60 |

So `qat/kd_precompute.py` + `--kd-table` can be used as-is on the tokenizer front. Verify
with `kd_precompute.tokenizer_compatibility()` yourself before a long run; it is seconds.

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

4. **EXPORT MAY NOT EXIST — scope this before training.** Our Q2_0 export is ftype 41,
   which lives only in `vendor/llama.cpp-prism` and was built for the Bonsai architecture.
   Whether that fork can represent a ternary gemma-4 at all is **open, and possibly a hard
   blocker**. A model we cannot export is a research note, not a deliverable. The
   compressed-tensors/vLLM path (`src/quant_tuner/vllm_export/`) is the fallback and is
   already validated E2E on gemma-4-E4B (W4A16, ~68 tok/s).

5. **Prior art in this repo, both directions.** `google/gemma-4-E4B-it-qat-w4a16-ct` exists
   — Google ships a QAT checkpoint for this model, so it has known QAT tolerance and is a
   reference point. But `scripts/exp034_release_v3.py` drops **QAT IQ2_M as broken
   (PPL ~2e10)** on gemma-4-31B. Low-bit gemma has failed here before; find out why before
   assuming ternary will behave.

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
