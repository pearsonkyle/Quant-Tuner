# Reproducing the ternary QAT pipeline, end to end

Every command needed to go from raw logs to a benchmarked Q2_0 GGUF, in order, with the
check that must pass at each step. Companion to `docs/ternary_qat.md` (method),
`docs/ternary_qat_curriculum.md` (the staged curriculum and the corpus defect) and
`docs/ternary_qat_sft32k_study.md` (the observations that set the hyper-parameters).

Written after the termination collapse, so it front-loads the checks that would have
caught it: an 11 h run once produced a single number that a 60-step run now produces.

Paths assume the repo root. `PY=.venv/bin/python`, `PYTHONPATH=src`.

---

## 0. Prerequisites

```bash
uv sync
git submodule update --init --recursive
# Q2_0 is ftype 41 and exists ONLY in the prism fork. Mainline llama.cpp cannot read
# these GGUFs at all — not to quantize, not to serve.
cmake -S vendor/llama.cpp-prism -B vendor/llama.cpp-prism/build -DGGML_CUDA=ON
cmake --build vendor/llama.cpp-prism/build -j
```

The base model lives at `out/exp-057/model` (HF dir, shipped ternary weights) with its
chat template at `out/exp-057/chat_template.jinja`. Everything downstream reads the
tokenizer and template from there — never from a finetune's copy.

---

## 1. Build the SFT source (`sft.jsonl.gz`)

One file per corpus, all in the same schema (`messages`, `tools`, `source`, `split`).

```bash
# OURS — CLI logs + agent trajectories + SWE trajectories + breadth + refusals.
# Also writes the calibration corpora; the QAT path only needs sft.jsonl.gz.
PYTHONPATH=src $PY scripts/build_universal_corpus.py \
    --out out/corpora/qwen3-universal-v2 --model out/exp-057/model --ctx 8192

# ROUND 1 — ultrachat_200k, broad conversational grounding (no tools, no reasoning)
PYTHONPATH=src $PY scripts/build_external_sft.py ultrachat \
    --out out/corpora/round1-ultrachat

# ROUND 2 — the distillation mix. Use `distill-mix`, NOT `distill`: the repo's 24
# configs are overlapping VIEWS (sft_reasoning contains 99.3% of sft_tools, and
# sft_agent is byte-identical to sft_tools), so combining configs double-counts.
PYTHONPATH=src $PY scripts/build_external_sft.py distill-mix \
    --out out/corpora/round2-distill
```

`build_external_sft.py` drops public-benchmark rows by default (SciQ, ARC, CommonsenseQA,
QASC, OpenBookQA, HumanEval, MBPP). `--keep-benchmarks` opts back in and costs you
MMLU-Pro/HumanEval/MBPP as quotable numbers for the model.

---

## 2. Pack to training windows

```bash
FIXED=out/exp-058/fixed; mkdir -p $FIXED
pack() {  # sft split budget out
  local ba=(); [ -n "$3" ] && ba=(--budget "$3")
  PYTHONPATH=src $PY scripts/build_sft_qat_corpus.py --sft "$1" --split "$2" \
      --window 32768 --max-tool-tokens 8192 --min-density 0.05 "${ba[@]}" --out "$4"
}
pack out/corpora/qwen3-universal-v2/sft.jsonl.gz train ""                  $FIXED/corpus_ourssft_32768.pt
pack out/corpora/qwen3-universal-v2/sft.jsonl.gz test  ""                  $FIXED/corpus_ourssft_val_32768.pt
pack out/corpora/round1-ultrachat/sft.jsonl.gz   train ultrachat=20000000  $FIXED/corpus_ultrachat_32768.pt
pack out/corpora/round1-ultrachat/sft.jsonl.gz   test  ultrachat=2000000   $FIXED/corpus_ultrachat_val_32768.pt
pack out/corpora/round2-distill/sft.jsonl.gz     train ""                  $FIXED/corpus_distill_32768.pt
pack out/corpora/round2-distill/sft.jsonl.gz     test  distill=2000000     $FIXED/corpus_distill_val_32768.pt
```

**Budget the val corpora.** The trainer reads only `--val-windows` (4) of them, but an
unbudgeted test split builds a tensor LARGER than the training corpus and loads all of it:
ultrachat's test split packs to 754 windows against its own 610 training windows.

**`--max-tool-tokens` scales with the window** — 3072 at 8064, 8192 at 32768. At 1024 it
drops 28% of all conversation content.

The packer applies the corpus hygiene automatically (`qat/corpus.py`):

| fix | what it prevents |
|---|---|
| `merge_consecutive_assistant` | prose and its tool call rendering as two turns, teaching "preamble → `<|im_end|>`" |
| `drop_empty_assistant` | a turn whose only supervised token IS the stop token |
| `has_inline_control_tokens` | a log that QUOTES `<\|im_end\|>` becoming a real stop token in supervised prose |

The build prints all three counts per source. On our corpus expect ~4,726 merges, 5 empty
turns dropped, 10 conversations dropped.

### Check the pack before training on it

```bash
PYTHONPATH=src $PY scripts/inspect_corpus_window.py $FIXED/corpus_ourssft_32768.pt \
    --audit --max-windows 40
```

Must report: all control tokens single ids · only `user`/`assistant`/`system` roles ·
**0 supervised tokens in NON-assistant turns** · **0 stray `<|im_end|>`**. A non-zero
"carry-over from the previous window" is expected — packed windows start mid-turn.

Read a window as the model sees it, supervised targets bracketed:

```bash
PYTHONPATH=src $PY scripts/inspect_corpus_window.py $FIXED/corpus_ourssft_32768.pt \
    --window 5 --tokens 700
```

And confirm the corpus is not teaching the pathology:

```bash
PYTHONPATH=src $PY scripts/analyze_stop_context.py $FIXED/corpus_*_32768.pt --max-windows 60
```

`P(stop | sentence end, <32 tok into turn)` should be ~0.05 or below on every corpus.

---

## 3. Verify the corpus BEFORE a long run

Two ~60-step runs from the same shipped weights, identical hyper-parameters, differing
only in the corpus, compared on the in-training stop probe:

```bash
bash scripts/verify_corpus_fix.sh 60
```

~45 min per arm. The endpoint is `P(<|im_end|> | completed sentence)`, which starts at
**0.0017** on the shipped weights (torch fp32) and reached **0.95** in every run trained
on the pre-fix corpus. This is the check that replaces discovering the answer 23 h into a
curriculum.

---

## 4. Train

```bash
PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY -m quant_tuner.qat.train \
    --corpus $FIXED/corpus_ourssft_32768.pt \
    --val-corpus $FIXED/corpus_ourssft_val_32768.pt \
    --train-layers 36 --optim adafactor --dtype fp32 \
    --compute-dtype fp32 --matmul-precision high \
    --grad-accum 1 --epochs 1.0 --lr 5e-4 --warmup-frac 0.05 \
    --stop-weight 1.0 --grad-spike-factor 0 \
    --val-every 25 --val-windows 4 --probe-every 25 \
    --ckpt-every 50 --ckpt-keep 2 \
    --out out/exp-058/trained_myrun > out/exp-058/trained_myrun/train.log 2>&1
```

Non-negotiable settings, all measured:

* **fp32, never `--compute-dtype bf16`.** bf16 is 5.15x faster and diverged twice,
  non-reproducibly, with no anomalous gradient beforehand. `GradSpikeGuard` cannot catch
  it and this is not a knob to retry.
* **`--matmul-precision high` (TF32) IS safe** and worth ~1.38x: latents, the TWN
  threshold and `ternarize_group` are elementwise fp32 and stay bit-exact, so the codes a
  step produces are unperturbed.
* **`--optim adafactor`** to fit all 36 layers (~70 GB vs AdamW's ~116 GB).
* **`--probe-every 25`** — the termination telemetry. Five short forwards, 0.7 s.

The trainer writes `run_config.json` before step 0 (full config, corpus fingerprint,
argv, git commit), so the run directory can always say what produced it.

### What to watch, in order

1. **`STOPPROBE` lines.** `sentence_period` must stay near 0.002; `after_tool_call` near
   1.0. Losing either is a failure and it shows up within ~25 steps.
2. **Code flips**, printed at every checkpoint. A ternary model only learns by flipping
   codes — at lr 3e-4 the loss falls on scale drift alone with ~0% flipped.
3. **Not the loss, and not masked-CE validation.** sft32k's validation went FLAT for 225
   steps while its `sentence_period` went to 0.97.
4. **gnorm magnitude is not the failure signal**: 90.96 at step 55 was survived; 15.80 at
   step 265 diverged. NaN/Inf is the unambiguous one.

### Disk

A checkpoint is **27.8 GB** and the trainer writes the new one BEFORE pruning the oldest,
so a run needs `(ckpt_keep + 1) x 27.8 GB` free however many it keeps. Watch it — this is
the constraint that actually kills runs, and nothing in the metrics stream hints at it.

---

## 5. Report

```bash
bash scripts/qat_report_refresh.sh out/exp-058/trained_myrun "myrun — ternary QAT"
```

Chains parse-log → cached step-0 census → latest census → HTML. Safe beside training and
idempotent. The "Termination policy over training" panel is the one to read first.

---

## 6. Export, probe, benchmark

```bash
TAG=myrun
# Q2_0 export — needs the prism fork (ftype 41)
LLAMA_CPP_DIR=vendor/llama.cpp-prism PYTHONPATH=src $PY scripts/exp057_qat_export.py \
    --latents out/exp-058/trained_myrun/trained_latents.pt --tag $TAG

# ~50 GB of intermediates (HF checkpoint + F16) for a 2.1 GB deliverable
bash scripts/prune_export_intermediates.sh $TAG

# Termination endpoint on the exported GGUF (CPU, --ngl 0)
LLAMA_CPP_DIR=vendor/llama.cpp-prism PYTHONPATH=src $PY scripts/probe_stop_prob.py \
    --model out/exp-057/Ternary-Bonsai-8B-$TAG-Q2_0.gguf --label $TAG \
    --out out/exp-058/eval/stop_prob.csv --ngl 0

# Agentic smoke test: can it patch a real repo? (needs a free GPU)
bash scripts/run_swe_mimic.sh $TAG
PYTHONPATH=src $PY scripts/analyze_swe_anomalies.py --all
```

**Read the probe and the trajectory together.** Each is blind in one direction: the probe
scores a single token at a fixed position and cannot see a multi-turn loop, while the
trajectory cannot tell early stopping from plain incapacity. The shipped model proves the
gap — its probe is textbook healthy (0.0092 / 0.99995) and its trajectory loops.

`analyze_swe_anomalies.py` names the failure mode, because `resolved=0` is the same number
for a model that never emitted a token (**mute**), one that repeated a command 58 times
(**loop**), and one that genuinely tried (**worked, unresolved**) — and only the last is
about capability.

---

## 7. The staged curriculum

```bash
bash scripts/run_curriculum_qat.sh curriculum 5e-4 1.0
```

Three rounds, each `--resume` from the previous round's latents: ultrachat → distillation
→ ours. Reads `CORPUS_DIR` (default `out/exp-058/fixed`), exports + probes + reports after
every round, prunes checkpoints and export intermediates as it goes, and refuses to start
a round without disk headroom.

Fully unattended, including the evaluation either side:

```bash
bash scripts/run_full_chain.sh              # postflight -> agent bench -> curriculum -> agent bench
FROM=curriculum bash scripts/run_full_chain.sh    # resume at a stage
```

---

## 8. The run ledger

```bash
$PY scripts/qat_registry.py --print
```

Joins `run_config.json`, `train.log`, the probe CSV, the tool-call CSV and swe-mimic's
results into `docs/qat_run_history.md`. A run is modelled as a SEQUENCE of legs (one per
`train*.log`), because `trained_sft32k` is three — two bf16 attempts that diverged and an
fp32 leg resumed from step 350 — and reading only the surviving log reports a resumed
leg's mid-run loss as the run's starting loss.

---

## The short version

If you do nothing else from this document:

1. `inspect_corpus_window.py --audit` before training on a corpus.
2. `--probe-every 25` on every run.
3. Read `sentence_period` and the agent trajectory together, never the loss alone.

---

## 9. Offline KD (Method B) — when plain CE trades learning against termination

Four levers were tested against the termination collapse and all four failed or traded
against learning (`docs/ternary_qat_curriculum.md`): `--stop-weight`, the corpus defect,
the optimizer, and the lr. Hard CE supplies one target per position and says nothing about
the SHAPE of the distribution, so the model is free to collapse P(stop) anywhere the argmax
survives. A KL term against a teacher that terminates correctly constrains exactly that.

```bash
bash scripts/run_kd_qat.sh                                    # 8B teacher, 60-step A/B
STEPS=613 bash scripts/run_kd_qat.sh                          # full run
TEACHER=SWE-Lego/SWE-Lego-Qwen3-32B TAG=kd32b STEPS=613 bash scripts/run_kd_qat.sh
```

### The teacher requirement, which silently ruins a run

Per-token KD needs the teacher and student to share a tokenizer **id -> string** map.
`kd_precompute.tokenizer_compatibility` compares the strings, not `vocab_size`, and refuses
a mismatch — per-token KL across different tokenizers is *wrong*, not approximate.

| teacher | verdict |
|---|---|
| `SWE-Lego/SWE-Lego-Qwen3-8B` | **OK** — agrees on all 151,669 ids; hidden 4096, 36 layers (identical to the student) |
| `SWE-Lego/SWE-Lego-Qwen3-32B` | **OK** — agrees on all 151,669 ids; hidden 5120, 64 layers |
| `Qwen/Qwen3.8-27B` | **NO** — vocab 248,320, a different tokenizer family |

The 27B is the obvious teacher to reach for (it solves our SWE instance at IQ2_M) and it is
the one that cannot be used. A vocab_size difference alone is fine: the SWE-Lego models
declare 151,936 against the student's 151,669 and the extra rows are embedding padding,
which the checker slices away.

Both teachers need `load_tokenizer_tolerant` — their configs carry
`max_position_embeddings: 163840.0`, a float where transformers demands an int, and stock
`AutoTokenizer.from_pretrained` refuses outright.

### Why offline

`--kd-teacher` loads a DENSE teacher alongside the student and cannot fit an all-36 run.
`--kd-table` reads a precomputed top-K table, so KD costs **no GPU memory at all** — the
table lives on CPU (~1.7 GB at top-64 over our corpus) and a window's slice is ~3.6 MB.

Precompute is cheap: **2.8 s/window** for the 8B, so 592 windows is ~28 minutes.

### What to check at startup

```
[qat] KD KDTable(teacher=..., topk=64, positions=..., windows=592, coverage=0.998)
[qat] KD alpha=0.5 T=1.0; loss = 0.5*CE + 0.5*T^2*KL
[qat] step 1/59 loss=0.9086 kl=0.5443 ...
```

**`coverage` is the number that de-risks the whole approach.** It is the teacher
probability mass captured by the stored top-K; at 0.998 the KL constrains essentially the
full distribution rather than a truncated shadow of it. Below 0.8 the trainer warns —
a low-coverage table makes the KL a far weaker constraint than it looks, and that is a
property of the teacher's entropy on this corpus that nothing later can detect.

### Two guards that exist because the failures are silent

* **Corpus fingerprint.** A table built from a different pack still resolves — same window
  count, same position range — and would distil every position against the wrong
  distribution without erroring anywhere.
* **Position alignment.** The stored positions must equal the `keep_idx` the trainer
  recomputes from the labels. The dangerous case is *equal counts, different positions*,
  which misaligns every row; the error names it.
* **Full coverage.** A partial table (e.g. one built with `--max-windows` for a smoke test)
  would let uncovered windows train on plain CE while the rest train on CE+KL — the
  objective changing from window to window. Refused at startup.

### The KL's blind spot, and the two fixes (in escalation order)

The first A/B (renormalized-over-support KL) landed *between* the CE-only arms: it slowed
the diagnostic's drift ~35% and did not stop it. Mechanism, measured: `<|im_end|>` is in
the teacher's top-64 at only **1.8%** of positions, and a support-renormalized KL is
mathematically blind to student mass outside the support — the collapse was invisible to
it at 98.2% of positions.

1. **Tail bucket (default, no re-precompute).** `kd_loss_from_topk` now takes the KL over
   K+1 buckets — the stored top-K at true probabilities plus a tail bucket (student side
   `1 − Σ support`). Caps the student's total out-of-support mass at the teacher's
   (~0.006 mean). Automatic at T=1; T≠1 falls back to the old renormalized form.
2. **Force the stop id into the support** (`scripts/kd_precompute.py --include-ids
   151645`): every stored row then carries the teacher's true P(stop), making the KL an
   exact per-position constraint on exactly the quantity that collapses. Costs one
   re-precompute; fold it into any 32B table build, which is built once and reused.

### Stop the run when the probe says stop

`--probe-abort 0.09` (10x the vanilla 0.0092) aborts with exit 3 and a saved checkpoint
the moment the diagnostic crosses the threshold. Every observed collapse was visible by
~step 50 and monotone afterwards; this converts an 11-hour post-mortem into a 40-minute
one. Leave it OFF for short A/B arms (their whole point is recording the trajectory) and
ON for full runs.

### Prune a dead run's checkpoints — after extracting, always

A run's latents are only needed to RESUME it or RE-EXPORT it; for a run that failed
(collapse, abort, divergence) neither applies, and each one holds ~52 GB (two 26 GB
checkpoints — note `trained_latents.pt` is a HARDLINK to the newest step file, so `du`
under-reports and deleting one name may free nothing). Before deleting, confirm the
record is extracted: train.log + metrics.jsonl in place, telemetry parsed
(`parse_qat_log.py`), the final census taken if the report wants a distribution-shift
column, and the exported GGUF kept if one was benchmarked. Then `rm` every
`trained_latents*.pt` in the run dir. Measured 2026-08-18: four dead runs held 207 GB.

## 10. The anchor ladder — reproducing the full-schedule runs

Every full-schedule KD run is one invocation of `scripts/run_kd_anchor_qat.sh` (dual
probe guards on, so a collapsing config self-terminates in ~2 h with a checkpoint):

```bash
TAG=a05      BETA=0 ALPHA=0.5              bash scripts/run_kd_anchor_qat.sh  # abort @125
TAG=a075     BETA=0 ALPHA=0.75             bash scripts/run_kd_anchor_qat.sh  # stop  @150
# (the symmetric-L1 and single-margin variants are code states, not flags: commits
#  c8d407d and aa74309 — margins landed in 4933a1d)
TAG=anchor3                                bash scripts/run_kd_anchor_qat.sh  # abort @175
TAG=anchor4 LR=4e-4                        bash scripts/run_kd_anchor_qat.sh
```

Then the evaluation chain for any tagged checkpoint:

```bash
bash scripts/run_kd_export_bench.sh anchor3
```

which runs export -> GGUF stop probe -> `measure_indist_stop.py` (real corpus stop
positions; this is what exposed the bimodal weak tail the probe medians hide) -> the
Docker-free SWE mimic, and prunes the ~50 GB of export intermediates.

## 11. The working recipe (anchor6) and its serving parameters

```bash
TAG=anchor6 LR=5e-4 STEER=0.1 CLIP=0.25 bash scripts/run_kd_anchor_qat.sh
bash scripts/run_kd_export_bench.sh anchor6
```

Loss = 0.5·CE + 0.5·(tail-bucket KL, forced-stop table) + 0.2·(one-sided anchor,
margins 1.0/0.1) + 0.1·(termination steering) with clip-norm 0.25, group-scale lr, and
patience-2 dual probe guards. First full-schedule completion; perfect probe record.

**Serve the export with repetition penalties** — behavioral integrity at 2 bits needs
them (llama-server CLI defaults reach OpenAI-compat requests that omit the fields):

```
--repeat-penalty 1.3 --repeat-last-n 2048 --presence-penalty 0.8
```

Measured: same GGUF, same episode — without penalties 60 turns / one command 49x /
MaxTurns; with them 10 clean turns, self-terminated, "worked, unresolved".
`--steer-rep-weight 0.1` is the training-time counterpart for the next run.
