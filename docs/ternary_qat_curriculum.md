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
