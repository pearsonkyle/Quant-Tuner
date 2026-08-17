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
