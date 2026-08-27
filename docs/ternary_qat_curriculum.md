# The three-round ternary-QAT curriculum

A staged continued-QAT run for `prism-ml/Ternary-Bonsai-8B`, replacing the single-corpus
runs (`sft8k-full`, `sft32k`, `sft32k_sw1`) with an ordered curriculum. Companion to
`docs/ternary_qat.md` (method), `docs/ternary_qat_sft32k_study.md` (the observations that
set the hyper-parameters) and `docs/qat_32k_handoff.md` §10 (the CUDA port).

## The three rounds

| round | corpus | what it teaches | tokens | windows @32768 |
|---|---|---|---|---|
| 1 | `HuggingFaceH4/ultrachat_200k` | broad conversational grounding | 20.0M | 610 |
| 2 | `r0b0tlab/qwen3.8-max-…-distillation` | tools, agents, reasoning-in-context | 20.2M | 604 |
| 3 | our universal SFT | CLI logs + trajectories that resolve real issues | ~20M | 613 |

All three rounds are sized to ~20M tokens / ~610 windows, which is exactly the `sft32k`
run's shape — so each round is ~11 h and the curriculum is ~33 h.

Each round **continues from the previous round's latents**, so this is one long fine-tune
over a changing distribution, not three independent runs. The ordering is
general → capability → in-domain, so the last thing the model sees is the distribution it
is graded on. Catastrophic forgetting runs the other way, and that is the point.

Build and run:

```bash
python scripts/build_external_sft.py ultrachat   --out out/corpora/round1-ultrachat
python scripts/build_external_sft.py distill-mix --out out/corpora/round2-distill
bash scripts/run_curriculum_qat.sh curriculum 5e-4 1.0
```

Use `distill-mix`, not `distill`. See "the configs are overlapping views" below.

Round 3's corpus is the existing `scripts/build_universal_corpus.py` output; nothing new
is needed for it.

## Why each round is budgeted, not epoch-capped

ultrachat is ~300M tokens. At a 32768 window that is ~5,500 steps — **~96 h on its own**,
against the ~11 h the whole `sft32k` run took. Rounds are therefore capped by token budget
(`BUDGET_R1=ultrachat=20000000`), sized so each round is a comparable slice of wall-clock
and the full curriculum lands near ~33 h. Round 2 is capped at build time instead — its
domain budgets already sum to ~20M — and round 3 is ~20M as built.

Budgets are per-source `SOURCE=TOKENS` pairs — there is no global cap, because
`build_sft_qat_corpus.py` budgets each source separately so one source cannot crowd out
another.

## What is settled, and what this curriculum does not decide

Carried forward from the sft32k study, not re-litigated here:

* **fp32, never `--compute-dtype bf16`.** bf16 is 5.15× faster and diverged twice,
  non-reproducibly, at different steps, with every corpus source spiking at once and no
  anomalous gradient beforehand (Obs. 8). `GradSpikeGuard` cannot catch it: the weights are
  already broken by an update whose norm was *below* threshold, and the huge norms that
  follow are a symptom. `run_curriculum_qat.sh` does not expose the flag.
* **`--matmul-precision high` (TF32) IS safe and is the default here.** A different knob:
  latents, the TWN threshold and `ternarize_group` are elementwise fp32 and stay bit-exact,
  so the codes a step produces are unperturbed. Only matmul accumulation is reduced, and
  TF32 keeps 10 mantissa bits against bf16's 8. Worth ~1.38× at this window.
* **Read the code flips, not the loss.** A ternary model only learns by flipping codes; at
  lr 3e-4 the loss falls on scale drift alone with ~0% flipped.
* **gnorm magnitude is not the failure signal.** 90.96 at step 55 was survived without
  incident while 15.80 at step 265 diverged. NaN/Inf is the unambiguous one.

Still open, and this curriculum does **not** answer them:

* the lr 2.5e-4 control, which would separate "bf16 is causal" from "a 32K window at
  lr 5e-4 is simply fragile";
* whether the stop weight needs to be >1.0 — that is what `sft32k_sw1` measures, and its
  answer should set `STOP_WEIGHT` here before round 1 starts.

## The dataset traps

Encoded in `data/external_sft.py` and unit-tested in `tests/unit/test_external_sft.py`.
Each one produces a corpus that builds, trains and exports without error while teaching the
model something wrong.

**1. `sft_tools` and `sft_agent` are the same rows.** Byte-identical parquet (4,226,831
bytes) and an identical id-set hash over all 5,337 rows. The Hub's `/size` API reports every
one of the 24 configs with the same row count and byte size, which makes them look distinct
— they are not. Taking both double-weights the agentic half of round 2. `DISTILL_ALIASES`
records it; the builder skips the alias and says so.

**2. Tool schemas nest `parameters_json` as a JSON *string*.** The Qwen3 chat template
renders tools with `tool | tojson`, so an unconverted row emits the whole parameter schema
as one escaped blob inside `<tools>`. It templates cleanly, tokenizes cleanly, and teaches
a doubly-encoded format that no inference-time caller produces. `normalize_tool` converts
it; verified by rendering a raw row and confirming `parameters_json` leaks verbatim without
it.

**3. The per-row `split` field is `"train"` in every file**, including the validation and
test parquet. Trusting it puts graded rows into training. The *file* the rows came from is
the real split, so `convert_distill_rows` takes it as an argument: `UPSTREAM_SPLIT_MAP`
sends their `validation` → our `test` (the trainer's val corpus) and their `test` → our
`holdout`.

**4. The configs are overlapping views, not disjoint slices** — see the table below.
A mix assembled by combining configs double-counts heavily.

**5. Two schemas in one repo.** Per-config files carry `messages`/`tools` as lists;
`canonical` carries `messages_json`/`tools_json` as strings. Reading one drops the other
silently.

**6. Reasoning is inline `<think>` in content, never `reasoning_content`** — despite that
field existing on every message. `data.reasoning` normalizes both forms, so this works, but
*counting* it needs the non-empty check: the Qwen3 template emits a bare `<think></think>`
on the final assistant turn when no reasoning is supplied, which reported **200 reasoning
blocks on ultrachat, a corpus with exactly zero**.

## Measured compatibility

Both rounds through the real builder at the training window
(`--window 32768 --max-tool-tokens 12288 --min-density 0.05`):

```
round 1  ultrachat  16,857/166,341 convs  20,000,736 tok  77% masked -> 610 windows
         tool-calls 0 · reasoning 0 · labeled <|im_end|> 53,301
         density deciles 0.64 0.72 0.74 0.75 0.76 0.77 0.78 0.79 0.80 0.82 0.87

round 2  distill    10,867/10,867  convs  20,209,609 tok  49% masked -> 604 windows
         tool-calls 38,753 · reasoning 26,315/29,511 kept (89%) · labeled <|im_end|> 33,288
         density deciles 0.08 0.32 0.41 0.47 0.49 0.52 0.55 0.58 0.60 0.65 0.74
```

Reasoning survives at 89% because these are agentic trajectories — the template keeps
reasoning only on assistant turns *after the last user turn*, which is nearly the whole
conversation when there is one task turn followed by many tool turns. Round 1's much higher
density (0.77 vs 0.49) is the absence of tool outputs: nothing in an ultrachat window is
masked except the user turns.

## How much is there, and what we take

Measured with the student's tokenizer over each source's train split:

| source | available | taken | share |
|---|---|---|---|
| ultrachat `train_sft` | 166,341 convs / **202.6M tok** | 16,857 convs / 20.0M | 9.9% |
| distillation (canonical, deduped, benchmark-free) | 56,773 convs / **56.9M tok** | 10,867 convs / 20.2M | 36% |

So ultrachat is ~3.6x the distillation set by tokens, and the distillation set is the
scarcer resource — we take a third of it against a tenth of ultrachat. The agentic part is
scarcer still: **10.6M tokens is ALL of it**, so round 2 takes 100% of both agentic domains
and fills the remaining ~9.6M around them.

### The round-2 domain mix (52% agentic)

```
agent_tool                     5,445/5,445    10,058,226 tok   (agentic, taken whole)
executed_tool_recovery           395/395         520,692 tok   (agentic, taken whole)
grounded_long_context            447/1,140     3,999,405 tok
code                           3,263/13,566    3,999,942 tok
executable_code                  513/513         591,542 tok
stateful_dialogue                471/471         805,103 tok
verified_analysis_code_review    130/130         102,884 tok
iterative_instruction            203/203         131,815 tok
TOTAL                         10,867           20,209,609 tok  (52% agentic)
```

`reasoning`, `math` and `instruction` are deliberately absent — see contamination below.
Reasoning is still taught: the agentic trajectories are ~40% `<think>` by message, so the
round teaches reasoning *in the agentic context* rather than as multiple-choice.

### The configs are overlapping views, not disjoint slices

This is the trap that makes a hand-assembled config mix wrong. Measured id-set overlaps:

| | overlap |
|---|---|
| `sft_agent` vs `sft_tools` | **identical** (5,337/5,337) |
| `sft_glm_agent` ⊂ `sft_tools` | 4,791/4,791 (100%) |
| `sft_tools` ⊂ `sft_reasoning` | 5,300/5,337 (99.3%) |
| `sft_reasoning_specialist` ⊂ `sft_reasoning` | 44,151/44,151 (100%) |
| `sft_code` ∩ `sft_reasoning` | 11,138 (83.6% of `sft_code`) |
| `sft_math` ∩ `sft_reasoning` | 12,672 (93.6% of `sft_math`) |
| `sft_long_context` ∩ `sft_reasoning` | 996 (86.5%) |

`canonical`, `openai_messages`, `sft` and `sft_final` are all the same 98,455-row full set,
which itself holds only 56,773 distinct ids. So `build_external_sft.py distill-mix` builds
from the **deduped canonical set and selects by `domain`**, which is the only way to get a
mix whose weights mean what they say.

### Two schemas in one repo

The per-config files carry `messages`/`tools` as real lists; `canonical` and the other
full-set views carry `messages_json`/`tools_json` as **strings**. Reading only one drops
every row of the other silently — the conversation comes back empty and fails the
length check, so a 56,773-row build reports zero without raising. Unit-tested.

## Contamination: worse than just `sft_science`

The whole `reasoning` domain (10,366 rows) is public benchmark material and nothing else:
SciQ 3,905, CommonsenseQA 2,792, QASC 1,667, ARC-Easy 962, OpenBookQA 572,
ARC-Challenge 468. And the `code` domain carries **HumanEval (126) and MBPP (180)** — the
two standard code benchmarks — mixed in with synthetic Evol-Code and CodeAlpaca.

`distill-mix` drops all eight sources by default (`BENCHMARK_SOURCES`); `--keep-benchmarks`
opts back in. Keeping them means MMLU-Pro, HumanEval and MBPP stop being quotable for this
model, and this repo's leaderboard runs MMLU-Pro.

## The two termination failures, and why one number cannot see both

`scripts/analyze_swe_anomalies.py` re-reads the saved agent trajectories and names the
failure mode. Run over what exists today it separates the two ternary results into
*opposite* failures:

| model | mode | evidence |
|---|---|---|
| vanilla Q2_0 | **loop** | 19 tool calls, 4 distinct commands (21%); alternates `cat utils.py` with `cat utils.py#L437` — a GitHub line-anchor pasted into a shell path — failing 8 times and never adapting |
| sft32k (sw 6.0) | **mute** | 1 output token, 0 tool calls: emits its stop token immediately |

Every non-ternary rung resolves the same instance (F16, IQ2_M, IQ3_M, IQ4_XS, Q5_K_M,
W4A16), so the instance is solvable at 2 bits and this is the ternary model's problem,
not the harness's.

**The probe and the trajectory measure different things, and vanilla proves it.** Vanilla's
P(im_end) probe is textbook healthy — 0.0092 after a sentence, 0.99995 after a tool call —
and its trajectory still loops. A single-token probe at a fixed position cannot see a
multi-turn loop; conversely the trajectory cannot distinguish early stopping from plain
incapacity. `scripts/choose_stop_weight.py` therefore reads both, and when they disagree
it returns the natural rate and says so rather than picking a side.

## The stop-signal ratio, per round

Relevant to `--stop-weight`, since that is what `sft32k_sw1` is measuring:

| round | trainable tokens | labeled `<|im_end|>` | 1 stop per |
|---|---|---|---|
| 1 ultrachat | 15.4M (77% density) | 53,301 | 289 tokens |
| 2 distill | 9.9M (49% density) | 33,288 | 297 tokens |
| sft8k reference | 5.7M | 32,448 | 176 tokens |

Both new rounds are ~1.7x SPARSER in stop signal than the corpus whose stop sparsity was
diagnosed as the cause of sft8k-full's 97% loop rate. If `sft32k_sw1` shows the 32K window
alone did not fix termination, that is the number to act on — and it argues for carrying
the chosen stop weight into every round, not just round 3.

## The proposed super-dataset: sized, not built

The follow-on idea is one big run instead of three rounds — 75K ultrachat conversations,
50K distillation conversations, and all of our SFT — released as a `qwen3-coder` model.
`scripts/size_super_dataset.py` measures it against the local parquet caches:

| source | take | available | tok/conv | tokens |
|---|---|---|---|---|
| ultrachat | 75,000 | 166,341 | 1,170 | 87.7M |
| distillation | **46,101** | 46,101 | 1,132 | 52.2M |
| our SFT (all) | 6,170 | 6,170 | 2,749 | 17.0M |
| **TOTAL** | | | | **156.9M** |

Three things this changes about the proposal:

**1. "50K from distillation" is more than exists.** The benchmark-free deduped train pool
is **46,101** conversations. The ask silently becomes "all of it" — which is fine, but it
means the distillation half cannot be scaled up further without re-admitting the
benchmark rows (SciQ/ARC/CommonsenseQA/QASC/OpenBookQA, HumanEval, MBPP) and giving up
MMLU-Pro, HumanEval and MBPP as quotable numbers for this model.

**2. It costs ~86 h fp32 for ONE epoch** — 4,692 windows at 32768, at the measured 66
s/step. TF32 (`--matmul-precision high`, safe here) brings it to **~62 h**. For scale,
each curriculum round is ~610 windows / ~11 h, and the whole three-round curriculum is
~33 h. So the super-dataset is roughly **two curricula** of wall-clock.

**3. It DILUTES the agentic content, which is the scarce thing.** The agentic domains
total 10.6M tokens and that is all of them — the same 10.6M whether the run is 20M tokens
or 157M. So agentic share falls from **52% in round 2** to **~7% of the super-dataset**.
Per hour of training, the super-dataset teaches agentic behaviour ~7x less than round 2
does. If the agent benchmark is the target, that is the wrong direction, and the honest
alternative is to keep the staged curriculum and spend extra budget on more *epochs of
round 2*, or on harvesting more trajectories, rather than on more ultrachat.

Worth building only if the curriculum's agent benchmark shows the model is limited by
general fluency rather than by agentic exposure.

## The stop-weight ablation failed, and so did the corpus hypothesis

`sft32k_sw1` set `--stop-weight` back to 1.0 with the 32768 window as the only
intervention. Result, against the two controls:

| probe | vanilla | sft32k (sw 6.0) | **sft32k_sw1 (sw 1.0)** |
|---|---|---|---|
| start | 0.00002 | 0.12194 | 0.01105 |
| mid_sentence | <6.2e-06 | 0.00002 | <2.9e-05 |
| **sentence_period** | **0.00919** | **0.97435** | **0.95310** |
| sentence_newline | 0.00000 | 0.89579 | 0.12922 |
| after_tool_call | 0.99995 | 0.95285 | 0.81426 |

**A 6x change in the stop weight moved the diagnostic by 0.02.** The weight is not the
mechanism. It is not inert either — it fixed the model's ability to start (0.122 → 0.011)
and the newline case (0.896 → 0.129) — but the sentence-end collapse is untouched, and
`after_tool_call`, the control where stopping is CORRECT, got *worse* (0.953 → 0.814).

The agent benchmark says the same thing behaviourally: sw1 **loops harder than vanilla** —
60 tool calls, hitting max turns, the same command repeated **58 times in a row**. So the
three ternary models fail three different ways (vanilla loops mildly, sft32k goes mute,
sw1 loops severely), and `choose_stop_weight.py` classified it as CONTRADICTORY: stops too
early at a sentence *and* fails to stop after a tool call. **The model has not learned to
stop too much or too little — it has lost the position-dependence of the stop decision.**

### The corpus does not teach this either

`scripts/analyze_stop_context.py` measures the conditional directly, in the probe's exact
situation — a sentence end within the first 32 tokens of a fresh assistant turn:

| corpus | marginal | P(stop \| sentence end) | **P(stop \| sentence end, <32 tok into turn)** |
|---|---|---|---|
| ours (SFT) | 0.0062 | 0.0697 | **0.0555** |
| ultrachat | 0.0034 | 0.0665 | **0.0166** |
| distillation | 0.0033 | 0.0280 | **0.0025** |

The corpus teaches at most **0.055**; the trained model emits **0.95**. A **17x
over-shoot**, so this is not the model learning its data — it is the model collapsing a
mild positional tendency into a near-deterministic rule. Note the ordering also
contradicts a simple data explanation: the distillation corpus has the *lowest* rate at
the probe position (0.0025), 22x below ours.

What follows a sentence end when it is NOT a stop is the tell: in our corpus it is
` Let` (25%), ` The` (20%), `<tool_call>` (10%) — the continuations the agent needs and
the model no longer produces.

### What is still open

Neither the objective nor the data explains a 17x over-shoot, which leaves the *training*
itself: lr 5e-4 flips 4.2% of codes in the leading tensor (sft32k managed 1.8% and is
equally broken, vanilla flips 0% and is fine). A ternary model has three values per weight
and no room for a finely-conditioned decision boundary, so the working hypothesis is that
continued QAT at this lr coarsens the stop decision into a positional rule regardless of
what the corpus says. The curriculum tests it for free: three rounds, three corpora whose
probe-position rates span 22x, each probed straight after its export.

## The corpus defect, found and fixed

The 17x gap between what the corpus taught (0.055) and what the model emitted (0.95) sent
me to read the actual training windows rather than more statistics. `scripts/inspect_corpus_window.py`
prints a packed window with supervised targets bracketed, and the first one showed this:

```
<|im_start|>assistant
[[That's not right - I accidentally removed too much. Let me check the current state:<|im_end|>]]
<|im_start|>assistant
[[I made a mistake. Let me restore the file from git and redo this properly:<|im_end|>]]
<|im_start|>assistant
[[<tool_call>...</tool_call><|im_end|>]]
```

Three consecutive assistant turns with nothing between them. In the real session that was
ONE assistant message — prose followed by a tool call — but the log records an assistant
response's content blocks as separate messages, and each fragment rendered as its own
`<|im_start|>assistant … <|im_end|>` turn. So the corpus explicitly taught that a short
prose preamble is followed by the STOP token rather than by the tool call that came next.

Two defects, both specific to our ingestion:

| defect | ours | ultrachat | distillation |
|---|---|---|---|
| assistant msgs preceded by another assistant msg | **21.2%** (logs-agents), 9.0% (logs) | 0.0% | 0.0% |
| "Let me…" turns ending at their first sentence | **18.5%** | 0.0% | 0.0% |
| empty assistant turns (`<|im_start|>assistant\n<|im_end|>`) | **2,155** | 0 | 0 |

The empty turns are the sharper of the two: a supervised span whose ONLY trained token is
the stop token — the purest possible lesson in "emit `<|im_end|>` immediately". That is
what the `start` probe measures, and it moved 0.00002 → 0.12194 in the run trained on this
corpus.

### The fix, and what it costs

`merge_consecutive_assistant` + `drop_empty_assistant` in `qat/corpus.py`, applied before
truncation and before rendering. Merging LOSES NOTHING — verified by accounting over the
whole corpus:

```
tool_calls      33,572 -> 33,572   (zero delta)
content chars   79,090,289 -> 79,090,795   (+506: the "\n\n" paragraph joins)
reasoning chars  1,966,416 ->  1,966,468   (+52)
messages           85,643 ->     79,499    (-6,144 merged into their turn)
```

Only **assistant** messages merge. Tool results arrive as `user`-role messages under this
template, so merging consecutive user messages would fuse a real user turn with a tool
response — a different and worse corruption.

Ten conversations are dropped outright: they QUOTE a chat control token in their content
(our own past sessions debugging chat templates, where the assistant wrote
`rendered.find('<|im_end|>')` in a code block and special-token parsing turned it into a
real stop token inside supervised prose). Dropped rather than repaired — rewriting the
token would corrupt the code the message is about.

Rebuilt result, our SFT corpus at a 32768 window:

| metric | before | after |
|---|---|---|
| consecutive assistant turns | 511 | **0** |
| turns ending at their first sentence | 3.7% | **0.5%** |
| "Let me…" turns ending there | 18.5% | **0.0%** |
| empty assistant turns | 2,155 | **0** |
| labeled `<|im_end|>` targets | 35,359 | **29,843** |
| real supervised leakage | — | **0** |

The distillation corpus rebuilt to fingerprint `b36696c8ce45c1ca`, **byte-identical** to
before — the fix is provably a no-op where the defect was absent, which is the check that
makes the merge trustworthy on the corpus where it does fire.

`CORPUS_DIR` in `run_curriculum_qat.sh` now defaults to `out/exp-058/fixed`. Round 3 no
longer symlinks sw1's corpus, so **curriculum-r3 vs sft32k_sw1 is no longer a clean A/B** —
it differs in both training history and corpus. That is deliberate: preserving the A/B
would mean deliberately re-teaching the defect. The per-round probes separate the causes.

## Did the corpus fix work? A/B verification (`verify_corpus_fix.sh`)

Two 59-step runs from the same shipped weights, identical hyper-parameters, corpus the
only difference. Endpoint is the in-training stop probe.

| step | OLD `sentence_period` | NEW `sentence_period` | ratio | OLD `after_tool` | NEW `after_tool` |
|---|---|---|---|---|---|
| shipped | 0.0017 | 0.0017 | — | 0.99996 | 0.99996 |
| 10 | 0.01250 | 0.00430 | 2.9x | 0.9998 | 0.9998 |
| 20 | 0.01670 | 0.00530 | 3.2x | 0.9946 | 0.9997 |
| 30 | 0.03240 | 0.00950 | 3.4x | 0.9995 | 0.9995 |
| 40 | 0.06260 | 0.00920 | 6.8x | 0.9993 | 0.9998 |
| 50 | **0.06660** | **0.01020** | **6.5x** | 0.9993 | 0.9995 |

**Read this as a PARTIAL pass, against the criterion set before the data existed.** The
script's own pre-registered test was "the fix works if NEW stays near 0.002 while OLD
climbs". NEW reached 0.0102 — 6x the shipped model's 0.0017 — so it did not stay near
0.002, and moving that goalpost after the fact would make the test worthless.

What the data does support:

* The OLD corpus rose **39x above baseline in 50 steps and was still accelerating**
  (+93% over steps 30-40). It is on the trajectory that ended at 0.95.
* The NEW corpus rose **6x and flattened**: 0.0095 → 0.0092 → 0.0102 across the last 30
  steps is a plateau, not a climb.
* `after_tool_call`, the control where stopping is CORRECT, held ≥ 0.9993 on both arms —
  so neither short run reproduced the *second* half of the failure (sw1 ended at 0.81),
  and this test is only sensitive to the early-stopping half.

Two readings remain open and 59 steps cannot separate them: the residual is an early-
training transient that stays bounded near 0.01, or it resumes climbing later. Do not
claim the first without a longer run. What is settled is the comparison: the pre-fix
corpus was at 0.0666 and accelerating where the fixed corpus is at 0.0102 and flat.

**Operational consequence:** run every round with `--probe-every 25`. If the residual does
resume climbing it now surfaces within 25 steps rather than 11 hours, and the per-round
probes attribute it to a corpus.

## Is the OPTIMIZER the cause? No — and the direction is informative

Adafactor here runs `scale_parameter=False, relative_step=False` with `beta1=None`, i.e.
"Adam with a rank-1 second moment and NO MOMENTUM". A ternary latent only changes anything
when it crosses the ternarization threshold, and crossing needs pressure accumulated over
many steps — so no-momentum training was a plausible reason fine, context-dependent
structure would fail to form while a coarse always-on signal formed anyway.

Tested with `scripts/verify_optimizer.sh adamw8bit 60`: same corpus, same lr, 8-bit state
so real per-parameter moments + momentum fit (84.1 GiB peak measured, against
Adafactor's 70.6; full-precision AdamW would need ~126 GiB and Adafactor+beta1 ~98 GiB,
both OOM on a 95 GiB card).

| step | adafactor (no momentum) | adamw8bit (β1=0.9) | adafactor `after_tool_call` | adamw8bit |
|---|---|---|---|---|
| 10 | 0.0043 | 0.0043 | 0.9998 | 0.9999 |
| 20 | 0.0053 | 0.0089 | 0.9997 | 0.9980 |
| 30 | 0.0095 | 0.0318 | 0.9995 | 0.9980 |
| 40 | 0.0092 | 0.0192 | 0.9998 | 0.9987 |
| 50 | **0.0102** | **0.0326** | **0.9995** | **0.9985** |

**Momentum makes it worse on both measures** — 3.2x on the diagnostic, and 3x the
deviation from 1.0 on the control. Adafactor's factored state was not losing anything that
mattered; its lack of momentum was accidentally protective. Keep `--optim adafactor`.

The direction is the useful part. Momentum accumulates signals that are CONSISTENT across
batches. Had the stop decision needed fine per-weight discrimination, momentum would have
helped it; instead momentum amplified the drift. So what is being learned is a coarse
signal present in nearly every batch, not a subtle one the optimizer failed to resolve.

### Where the diagnosis stands

| candidate | verdict |
|---|---|
| `--stop-weight` | ruled out — a 6x change moved the diagnostic by 0.02 |
| corpus defect | REAL, worth 3-6x on our data, but not sufficient (clean ultrachat drifts to 0.0265 by step 25) |
| optimizer state / momentum | ruled out — momentum is 3.2x worse |

What all three share is **lr 5e-4 against a ternary weight space**, and that is not a free
knob: below ~3e-4 the codes do not flip at all (the run drifts fp16 scales, the loss falls
from 2.26 to ~1.0, and the exported GGUF is the shipped model unchanged). So the choice is
between two failure modes, not between "works" and "works better".

The 2.5e-4 arm is now a sharper experiment than it was, because flips and termination can
be read simultaneously — flips at each checkpoint, `sentence_period` every 25 steps:

* **flips ≈ 0, termination stable** → the floor is confirmed, 5e-4 is forced, and the fix
  has to come from data mix / KD rather than lr.
* **flips > 0, termination stable** → the window exists; take it.
* **flips ≈ 0, termination still drifts** → the drift is not code-flip-driven at all, which
  would point at scale drift and reframe the problem. No previous run could tell this apart.

## Is the LR the lever? No — there is no window

`LR=2.5e-4 bash scripts/verify_optimizer.sh adafactor 60`, same corpus, adafactor with
`beta1=None` exactly as the baseline. Both halves measured together for the first time.

**Termination holds — it is the only arm that never moved:**

| step | lr 5e-4 | **lr 2.5e-4** | adamw8bit |
|---|---|---|---|
| 10 | 0.0043 | 0.0029 | 0.0043 |
| 20 | 0.0053 | 0.0028 | 0.0089 |
| 30 | 0.0095 | 0.0030 | 0.0318 |
| 40 | 0.0092 | 0.0031 | 0.0192 |
| 50 | **0.0102** | **0.0033** | **0.0326** |

`after_tool_call` sat at **0.99990 at every single step** on the 2.5e-4 arm.

**But it does not learn.** Code flips at the final checkpoint, same 59 steps:

| tensor | lr 5e-4 | lr 2.5e-4 |
|---|---|---|
| `0.q_proj` | 0.0194% | **0.0024%** |
| `5.k_proj` | 0.0011% | 0.0000% (1 weight) |
| `15.o_proj` | 0.0029% | 0.0000% (2 weights) |
| `20.o_proj` | 0.0053% | 0.0000% (5 weights) |
| `35.down_proj` | 0.0039% | 0.0000% (4 weights) |
| `10.v_proj`, `25.gate_proj`, `30.up_proj` | non-zero | **0.0000% (zero weights)** |

Seven of eight tracked tensors flipped literally nothing, while scale drift ran 0.33-0.58%
and the loss fell to 0.7623. That is the documented trap verbatim: *the run only drifts
fp16 scales, the loss falls, and the exported model is unchanged.*

**Do not extrapolate the 8x flip ratio forward.** Flipping is a THRESHOLD phenomenon, not a
linear one: below some lr the latents equilibrate before ever crossing, which is why the
reference study measured ~0% flips at 3e-4 over a FULL run. 2.5e-4 would not train over
613 steps either.

### Four candidates, four eliminated

| candidate | verdict |
|---|---|
| `--stop-weight` | ruled out — a 6x change moved the diagnostic by 0.02 |
| corpus defect | REAL, worth 3-6x, but insufficient (clean ultrachat drifts to 0.0265) |
| optimizer / momentum | ruled out — momentum is 3.2x worse on the diagnostic AND the control |
| learning rate | **no window** — 5e-4 learns and breaks termination, 2.5e-4 preserves it and learns nothing |

The trade looks structural to plain masked-CE on this model: it only learns by flipping
codes, and flipping codes at any useful rate destroys the stop decision. Every lever tried
so far moves along that trade rather than off it.

**That is the argument for KD** (Method B, already scaffolded in `qat/kd_precompute.py`).
Hard CE supplies one target per position and says nothing about the SHAPE of the
distribution, so the model is free to collapse P(stop) anywhere the argmax survives. A KL
term against a teacher with correct termination constrains exactly the quantity that
drifts, while leaving the argmax free to move where the data demands — i.e. it attacks the
mechanism instead of trading against it.

## KD round one: the renormalized KL had a measured blind spot

The first KD A/B (60 steps, `SWE-Lego/SWE-Lego-Qwen3-8B` top-64, α=0.5, T=1, lr 5e-4 —
`out/exp-058/verify-opt-kd8b-a0.5`) landed *between* the two CE-only arms: diagnostic
0.0029 → 0.0065 by step 40, roughly 35% below CE-only at the same lr but clearly rising,
where lr 2.5e-4 stays flat at ~0.0030. KD slowed the drift; it did not pin it.

The mechanism was in the loss, and it is measurable, not speculative.
`kd_loss_from_topk` renormalized BOTH distributions over the teacher's stored top-K.
Inflating a logit *outside* that support shifts every support logprob by the same
logsumexp constant, which renormalization cancels **exactly** — the KL is mathematically
blind to any student mass placed outside the teacher's top-K. And on our corpus:

- `<|im_end|>` is in the teacher's top-64 at only **1.8%** of supervised positions
  (400k-position sample of the 5.80M-row table);
- so at **98.2%** of positions, the collapsing P(stop) was invisible to the KL — the one
  quantity KD was brought in to constrain, at almost every position where it drifts;
- the teacher's mean tail mass is 0.0062, i.e. the teacher itself keeps stop (and
  everything else outside its top-64) at well under a percent.

The fix costs nothing and needs no re-precompute: the KL is now taken over **K+1
buckets** — the stored top-K at their TRUE probabilities plus one tail bucket (teacher
side already stored per position; student side `1 − Σ support`, exact from the same
logits). A student pushing P(stop) toward 0.95 against a teacher tail of ~0.006 now pays
~0.95·log(0.95/0.006) ≈ 4.8 nats of KL at that position, where before it paid zero.
Identical-student-scores-0 still holds exactly. Pinned by
`test_tail_bucket_sees_out_of_support_mass`, which asserts the drifted student is
penalized by the new form AND scored ~0 by the old one.

Two escalations if the tail bucket is not enough, in order:
1. **Force the stop id into the stored support at precompute** (union top-K ∪
   {151645}), converting the aggregate tail cap into an exact per-position constraint on
   P(stop) specifically. Requires a re-precompute (~30 min for the 8B, hours for the
   32B) — worth folding into the 32B table build regardless, since that table is built
   once and reused.
2. **Raise α** (0.5 → 0.75). Only meaningful after the blind spot is closed; before it,
   a higher α just amplified a term that could not see the failure.

Also fixed while auditing: `--trained-tail` + `--kd-table` silently misaligned every KD
row (the table is validated against the FULL keep set, then `masked_forward` drops
prefix targets without dropping their teacher rows). No affected runs — `trained_tail=0`
everywhere so far — and the row count is now asserted at the point of use.

## KD round two: the smoke ladder that produced the full-run config

Six 59-step arms on the identical corpus/schedule (plus the teacher's own reading, which
defines the asymptote — measured on CPU with the student's chat template, i.e. exactly the
rendering KD feeds the teacher):

| arm | diagnostic @50 | control @50 | verdict |
|---|---|---|---|
| teacher (SWE-Lego-8B) | 0.0000 | 0.99999 | the target: textbook termination |
| CE only lr 5e-4 | 0.0102 rising | 0.9995 | learns, drifts |
| CE only lr 2.5e-4 | 0.0033 flat | 0.9999 | preserves, cannot learn |
| KD renormalized | 0.0073 rising | 0.9977 | blind spot: slowed, not stopped |
| KD tail-bucket | 0.0044 oscillating | **0.9867 falling** | diagnostic fixed, control traded |
| KD forced-stop | 0.0058 flat | 0.9991 flat | both held |
| KD tail-bucket + lr-scale | 0.0047 flat | 0.9995 flat | flattest of all; flips redistribute |
| **KD forced-stop + lr-scale** | **0.0048 flat** | **0.9995 flat** | **launch config** |

Findings that came out of the ladder, in causal order:

1. **The teacher's stopping policy is stricter than the student's** (0.0000 vs 0.0092 at
   the diagnostic), so KD arms converging BELOW vanilla is convergence, not anomaly.
2. **The tail bucket's control slide is support-composition-specific.** The tb arm's
   after_tool_call fell 0.9997 -> 0.9867 (accelerating); the fs arm — same loss, stop id
   forced into every support row — held 0.9990-0.9999 throughout. No separate tail-weight
   knob was needed.
3. **Scale-aware lr redistributes flips exactly as the scale analysis predicted** (tb vs
   tbls, same table and loss): v_proj 1 -> 23 flips, up_proj x14, gate x7, while q/down
   slow to match their 0.5x multipliers. Loss unchanged (0.680 vs 0.684). And it
   *stabilized* the probe — the tensors driving the drift were the small-scale ones the
   rule slows down.
4. **Every KD arm's KL fell ~0.63 -> 0.35** and flips matched CE-only arms — the
   constraint costs nothing measurable in learning at this horizon.

Full run launched with: forced-stop table + tail-bucket KL (alpha 0.5, T 1.0) +
`--lr-scale group-scale` + `--probe-abort 0.09`, 613 steps (1.0355 epochs), lr 5e-4.
Output: `out/exp-058/kd8b-full/`. The open question a 59-step arm cannot answer — does it
still LEARN at full depth (4.22% flips over 613 steps on the CE-only reference) — is what
the full run measures, followed by Q2_0 export, the GGUF probe, and SWE-rebench.

## The sustained-lr regime: what the smoke ladder could not see

The alpha-0.5 full run aborted at step 125 (diagnostic 0.0973, the pre-set threshold).
Two structural findings:

1. **A 59-step arm validates a config under a DECAYING lr, not the full run's.** The
   arms complete their whole cosine inside 59 steps — most of their life is below
   3e-4 — while the full 613-step schedule holds ~5e-4 for hundreds of steps. Under
   sustained peak lr the alpha-0.5 KL only *slowed* the collapse: diagnostic tripling
   per 25 steps (0.0020 → 0.0973), control 0.9998 → 0.8952. Both failure directions at
   once, i.e. position-dependence itself eroding.
2. **The probe-abort guard replaces the smoke protocol for full-schedule questions.**
   No short arm can reproduce sustained peak lr; the abort makes the full run its own
   gate — failure costs ~2 h and a saved checkpoint, survival is the finished artifact.

The trade is now measured on one run: at abort, learning was ON the CE-only reference's
pace (mean tracked flips ~0.084% at step 100; lr-scale redistribution holding — v_proj
0.049%, gate 0.105%) while termination drifted unbounded. KD at alpha 0.5 changes the
collapse's *rate*, not its boundedness, once lr stays at peak.

Escalation now running: **alpha 0.75** (KL:CE 3:1), everything else identical
(`out/exp-058/kd8b-full-a0.75`). If it aborts too: stop-aware KL weighting (the table
stores every position's teacher P(stop) — weight the KL harder where it is extreme),
then peak lr 4e-4.

## Alpha is not the lever — and the collapse has a second face

The alpha-0.75 run (`kd8b-full-a0.75`) was stopped by hand at step ~150. Its two series,
against alpha-0.5 at the same steps:

| step | diag a0.5 | diag a0.75 | ctrl a0.5 | ctrl a0.75 |
|---|---|---|---|---|
| 75  | 0.0109 | 0.0111 | 0.9934 | 0.9931 |
| 100 | 0.0435 | 0.0206 | 0.9500 | 0.9737 |
| 125 | 0.0973 (abort) | 0.0842 | 0.8952 | 0.9688 |
| 150 | — | 0.0435 | — | **0.8876** |

Three findings:
1. **Alpha delays, does not change, the collapse** (~25 steps for +0.25 alpha). Mechanism:
   the KL's restoring force on the stop logit is `P_s − P_t` — proportional to the drift
   itself, ~0.01 while the student drifts through 1e-2 against a ~1e-6 teacher. Tripling
   a vanishing force still vanishes. Same signature as the stop-weight null result.
2. **The collapse has a second face the diagnostic-only abort cannot see.** At step 150
   the diagnostic RETREATED below threshold (0.0842 → 0.0435) while the control fell
   monotonically 0.9998 → 0.8876 — the sft32k loop failure. `--probe-abort-control`
   (0.95) now guards that direction; both guards ride every run.
3. The oscillating diagnostic + monotone control decay = the model is losing
   position-DEPENDENCE of the stop decision, not just moving one number.

Now running: `--stop-anchor 0.2` (margin 1 nat) at alpha 0.5 — a per-position hinge on
|log P_s(stop) − log P_t(stop)| whose gradient is O(1) in log space at any drift
magnitude, computed from the forced-stop table inside the existing logit chunks. It
pushes DOWN at sentence boundaries and UP after tool calls with the same constant force,
i.e. it targets both faces at once. `out/exp-058/kd8b-full-anchor`.

## The anchor works — and its first version worked too hard in one direction

`kd8b-full-anchor` (symmetric L1 hinge, beta 0.2, margin 1 nat): the diagnostic was
PINNED at 0.0000-0.0002 through step 175 — through the entire zone that killed both
alpha runs — while learning ran ABOVE reference (mean tracked flips 0.099% at step 100
vs 0.084% for a0.5; the anchor's stop-policy work visibly concentrated in
35.down_proj at 0.334%). The O(1) log-space restoring force does what the KL's
vanishing P_s−P_t force cannot. At init the student sits ~7 nats from the teacher's
stop policy on-corpus (an=6.4) — a gap the KL literally could not see.

Then the control collapsed alone: 0.9987 → 0.9356 → 0.6974, with every no-stop probe
point still at 0.000x. Diagnosis: the symmetric hinge exerts O(1) force at EVERY
position in BOTH directions, and continue-positions outnumber stop-positions 176:1 —
the same ratio that broke stop-weight — so the aggregate trunk-level pressure on
P(stop) is massively net-downward. The anchor crushed early-stopping perfectly and
dragged the ability to stop down with it.

Fix (running as `kd8b-full-anchor2`): ONE-SIDED per position type — at
continue-positions only stopping MORE than the teacher is penalized, at stop-positions
only stopping LESS, zero force on the safe side — so the 176:1 imbalance has nothing
to push with.

Also caught here: commit 380984a *claimed* the control-abort guard while its
str.replace silently no-op'd (the target block had been refactored; str.replace does
not error). run_config recorded the threshold, ruff and --help passed, and nothing
enforced it. Guards must be verified by grep of the LOGIC, not the flag.

## Anchor iteration 2: the margin means opposite things on the two sides

`kd8b-full-anchor2` (one-sided hinge, single 1-nat margin): diagnostic 0.0000 at every
probe, learning the best yet (mean tracked flips 0.126% at step 100), anchor nearly
silent (an ≤ 0.03 from step 30 on) — and the control STILL collapsed
(1.0000 → 0.9930 → 0.9753 → 0.8431; control guard fired at 150, correctly this time).

The hole: `an` stayed ~0.01 through the whole collapse, i.e. the student never left
the anchor's legal band ON-CORPUS. A 1-nat margin at continue-positions is 1e-6 →
2.7e-6 (harmless); at stop-positions it is P(stop) 0.99999 → 0.37 — the control can
fall to uselessness entirely inside it. Iteration 3 (`kd8b-full-anchor3`, running):
`--stop-anchor-margin-hi 0.1` — stop-positions held to ≥ ~0.905·teacher, engaged 25x
earlier along the sag.

Open question from this run: val spiked 0.8234 → 1.4672 between steps 100 and 150
while train loss stayed 0.62-0.74 — a broad held-out regression not attributable to
stop positions (0.57% of labels). Watch anchor3's val at 150.

If anchor3 holds on-corpus but the PROBE control still collapses, the drift is
genuinely off-distribution and the next tool is anchor prompts: mix a small batch of
synthetic stop-discipline contexts (the probe family, varied) into training as extra
supervised stop-positions.

## Anchor iteration 3 + the first benchmark: the failure narrowed to a weak TAIL

`kd8b-full-anchor3` (per-side margins 1.0/0.1): diagnostic 0.0000 at every probe,
learning best-yet (0.142% mean flips at step 100), val stable — control OSCILLATING
with growing amplitude (0.99 -> 0.97 -> 0.99 -> 0.93), guard abort at step 175.

The in-distribution measurement (scripts/measure_indist_stop.py — P(stop) at REAL
im_end targets in held-out windows, ternarized forward) resolved the probe-vs-reality
question: the model did NOT uniformly lose stopping. At after-tool-call stop positions
(n=273) the MEDIAN is unchanged (0.9912 vs vanilla 0.9932) while the p10 collapsed
(0.9801 -> 0.0787) — a bimodal weak tail covering ~10-15% of positions. Prose-end
stops moved TOWARD the labels (0.08 -> 0.58 mean).

Exported (step-175) to Q2_0 and benchmarked (Docker-free mimic, dask__dask-11393):
- GGUF probe: diagnostic < 8e-5 at all no-stop points, control 0.9203 (rank 1).
- Agent: LOOPED — same find command 9x in a row, died at 20 steps
  (error:InternalServerError). 8% non-yield per call compounds to loop entry over an
  episode. Milder than sft32k_sw1 (60-step hard loop) but the same mode.
- No generation of this model has ever patched this instance (vanilla included) —
  the benchmark metric here is behavioral integrity, and the weak tail fails it.

Verdict: the anchor solved the early-stop face completely and improved learning; the
remaining defect is a CONTEXT-DEPENDENT weak tail on the stopping side that grows with
time at peak lr. Ladder: anchor4 (same config, peak lr 4e-4, running) -> anchor
prompts (train the failing context family directly) -> gemma pivot.

## The dense control: ternary's lr is past the continuous stability boundary

Question posed: is the ~step-150 control collapse a property of ternary discreteness,
the objective, or the implementation? The dense control (identical objective — KD +
one-sided anchor, per-side margins — with every projection trained as an ordinary fp32
Linear) answered in TEN STEPS at the ternary lr 5e-4: outright divergence (loss 2.16 ->
8.62, KL -> 7.27, gnorm -> 47). Not a subtle stop-face drift — the whole optimization
explodes, as plain dense SFT at 25x its normal lr should.

Reading: ternary training NEEDS 5e-4 to cross thresholds, and 5e-4 is far beyond the
stability boundary of the underlying continuous dynamics. The quantizer acts as a
low-pass filter — the function only moves at threshold crossings — which is why ternary
runs are stable for hundreds of steps at an lr that destroys the dense model in ten.
The recurring ~150-step control collapse is best read as the filtered echo of that same
instability, leaking through in threshold-crossing waves and surfacing first in the
least parameter-supported behavior (short-context stopping). This also disposes of the
kernel/driver hypothesis: the same failure at 5e-4 dense, with no ternary code in the
loop at all.

Running next: dense control v2 at lr 2e-5 (a stable dense lr) — does the OBJECTIVE
harm the stopping face at all when dynamics are stable? Then the decision, with the
new levers this diagnosis suggests for ternary: tighter grad clipping (the control
waves co-occur with gnorm spikes 2.5-4.8; clip is currently 1.0), the steering loss
(--steer-weight) giving the fragile class every-step support, and grad-accum > 1 to
average the too-hot per-step direction.

## Dense control v2 (lr 2e-5): the decomposition

274 steps, stable dynamics, guards off, probe every 10. The control face's full story:
slow erosion from 0.9999 to a ~0.97 plateau, two excursions (0.885 @160, 0.936 @180)
that RECOVER and dampen, then stability 0.966-0.977 through the end. Diagnostic 0.0000
throughout; val monotone to 0.744 (project best); loss 2.30 -> 0.57.

The decomposition of the ternary failure, now measured:
1. **A slow objective-driven leak** (present in dense): the KD+anchor objective erodes
   the short-context control face ~0.03 over 274 stable steps, plateauing ~0.97.
2. **Waves** (present in dense at 3-5x smaller amplitude, recovering): a property of
   the training process (objective x fixed data order), NOT of ternarization. Window
   stats at the wave steps are unremarkable (no poisoned block).
3. **Ternary amplification** (the killer): at the lr ternary needs (5e-4 — which
   diverges a dense model in 10 steps, loss 15), the quantizer low-pass filters the
   too-hot dynamics; the waves leak through at threshold-crossing bursts, deepen
   instead of recovering, and take the control face to 0.55-0.75.

Synthesis run (anchor6): anchor5's config + --steer-weight 0.1 (every-step gradient
support for the fragile context class, against #1) + --clip-norm 0.25 (damping what
leaks through the filter, against #3) + patience-2 guards.

## anchor6: the first complete artifact — and the frontier moves

The synthesis config (KD forced-stop table + one-sided per-side-margin anchor +
--steer-weight 0.1 + --clip-norm 0.25 + lr-scale + lr 5e-4 + patience guards) ran the
FULL 613 steps with a perfect probe record: all 24 probes at exactly 0.0000 diagnostic /
1.0000 control. Final flips 1.07-2.37% across every tracked tensor (v_proj alive at
1.07%); val 0.745 = the dense control's endpoint; guards never fired.

The three-tier evaluation of the Q2_0 export:
- GGUF probe: diagnostic < 5e-5 everywhere; control 0.9999975 (above vanilla's own).
- In-distribution (273 real stop positions): median 0.9950 (> vanilla 0.9932), mean
  0.9414, p10 0.8197 — the anchor3 weak tail improved 10x (0.079 -> 0.820), not fully
  closed (vanilla 0.980).
- Agent episode: 60 clean turns, zero tool errors — and the identical command repeated
  49x. NOT a termination failure: the model stopped correctly after every call, then
  chose the same action again. The frontier is now REPETITION (state-tracking), a
  different pathology that vanilla shares in gentler form.

**Serving-side mitigation works today**: llama-server `--repeat-penalty 1.3
--repeat-last-n 2048 --presence-penalty 0.8` (CLI defaults apply because the agents SDK
sends only temperature/top_p) turned the same model's episode into 10 clean steps,
self-terminated, zero loops — anomaly class **"worked, unresolved"**, the first trained
ternary model ever to earn it. No 2-bit variant (vanilla included) has ever resolved
this instance; capability at 2 bits is the remaining gap, not behavior.

**Training-side counterpart implemented** (`--steer-rep-weight`, qat/steer.py
RepBatch/repetition_loss): contexts of (command -> unhelpful result -> assistant
header) with the VERBATIM previous command teacher-forced; a one-sided hinge penalizes
its mean per-token log-prob above `--steer-rep-cap` (default 0.5) — only
near-deterministic copying is suppressed; re-running a command stays available.

## anchor7–anchor10: the repetition arc (2026-08-20/21)

Full detail in the runbook §11 and `out/exp-058/kd32b-full-anchor10/notes.md`; the
compressed findings, in the order they were forced:

- **anchor7** (32B teacher + rep hinge v1 at k=1): termination best-ever (in-dist p10
  0.8916) but rp= read 0.0000 all run — the hinge was defined where the pathology is
  not (P≈0.33 at k=1, under the 0.5 cap) — and the mimic looped 59x.
- **Escalation measurement**: trained models escalate P(repeat) with identical rounds
  (vanilla flat). anchor8 (hinge at k=2–5, synthetic) inverted its own curve below
  vanilla and moved real-material states by ~0.02: **synthetic states don't transfer**.
- **Real-material bank** (`build_rep_bank.py`): trained models read 0.78→0.98 there —
  the 56x-loop regime exposed. **anchor9** (bank hinge, cap 0.6 = vanilla's level)
  suppressed it to 0.08 held-out — and still looped 29x.
- **State-dependence nailed** (`measure_traj_repeat.py`): same anchor9 weights,
  reconstructed real episode = 0.96 at the FIRST re-issue (the looped command already
  appeared ~11x non-consecutively earlier — induction primed); truncating the real
  prefix to 3k tokens collapses it to 0.06. The loop state IS the full history.
- **The teacher endorses the repeat** (0.79–0.99 under forcing at every k, 0.80 mean
  even at k=1): rep teacher-KL withdrawn before it trained — it would teach copying.
- **anchor10** (bank + harvested full-prefix episode contexts, hinge-only): real-state
  P on an UNSEEN episode 0.96 → 0.53–0.59 flat; best val of the ladder (0.7411);
  24/24 probes. Still looped bare at T=0.25 — **sharpening rescues the loop from any
  state where repeat is merely the argmax**.
- **The 2x2 verdict**: anchor9@0.7 loops (11x) with 31/48 malformed; anchor10@0.25
  loops (43x); **anchor10@0.7 is clean** — streak 1, 0 malformed, self-terminated.
  Both levers necessary. Serving: T=0.7, top_p 0.95, no penalties.
- **Ops**: mid-run benching = CPU sidecar per checkpoint (GPU peak 91.4/95 GiB forbids
  concurrent GPU eval); three OOM classes fixed in the rep losses (full-vocab logits,
  whole-tensor fp32 softmax, float-mask math-path SDPA — span-only lm_head + unpadded
  per-row forwards); runner now propagates the trainer's rc.
