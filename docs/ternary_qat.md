# Continued QAT for native-ternary (1.58-bit) models

A reusable pipeline to **fine-tune a natively-ternary GGUF model** (e.g.
`prism-ml/Ternary-Bonsai-8B`) on a task-specific corpus and re-export a runnable
2-bit GGUF. Built for Metal (Apple Silicon); the same scripts run on CUDA.

**Jump to:** [why](#why-this-exists-and-when-post-hoc-calibration-wont-do) ·
[quickstart for a new model](#quickstart-adapting-to-a-new-model) ·
[Metal constraints](#hard-constraints-on-metal-learned-the-hard-way--see-the-audit-doc) ·
[**reproducing the trajectory + KD runs**](#reproducing-the-trajectory-runs-iter-5--iter-6)

## Why this exists (and when post-hoc calibration won't do)

A native-ternary model stores `w = s·c` exactly (`c ∈ {−1,0,+1}`, one fp16 scale
`s` per group). Its "F16" is a *lossless* container, so **imatrix / AWQ / GPTQ are
structural no-ops** — there is no quantization error to recover (proof + measurements
in `ternary_calibration_experiments.md`). The only lever is **more training with the
ternarization in the loop** (BitNet/TWN-style QAT). This pipeline is that lever.

## The four stages

| Stage | Script | Output |
|------|--------|--------|
| 0. Get the trainable checkpoint | (HF `snapshot_download` of the `*-unpacked` repo) | `out/<exp>/model/` (plain `*ForCausalLM`, ternary weights as fp16) |
| 1. Build the masked corpus | `scripts/build_qat_masked_corpus.py` | `masked_corpus_<win>.pt` — turn-aware, **loss masked to assistant/tool tokens**, tool **schemas rendered** |
| 2. Train (STE ternary QAT) | `scripts/exp058_qat_train_v2.py` | `trained/trained_latents.pt` (checkpointed, signal-save) |
| 3. Export → GGUF | `scripts/exp057_qat_export.py` | `*-Q2_0.gguf` (runs on the prism `llama.cpp` fork) |

All the logic lives in the package (`src/quant_tuner/qat/`); the `scripts/exp05*` entries are
thin CLI shims over it, so the pipeline is importable and unit-tested rather than trapped in
one-off scripts:

| Module | What it owns |
|---|---|
| `qat/ternary.py` | per-group TWN straight-through estimator — reproduces the shipped weights **exactly** at step 0, so the fine-tune starts from the real model with no drift |
| `qat/corpus.py` | masking + packing, shared by the log and trajectory corpora (`trajectory_to_messages` is also what the published dataset uses) |
| `qat/train.py` | `QATConfig` + `train_qat()` — the training loop, flip telemetry, checkpoint/resume |
| `qat/master_opt.py` | fp32-master optimizer wrapper (per-tensor; never `foreach`) |
| `qat/export.py` | trained latents → Q2_0 GGUF via the prism fork |
| `qat/kd_precompute.py` | offline top-K teacher logits + `kd_loss_from_topk` (Method B, below) |

## Quickstart (adapting to a new model)

```bash
# 0. trainable checkpoint -> out/<exp>/model/, and the F16 GGUF for the chat template
#    (extract tokenizer.chat_template from the shipped GGUF -> out/<exp>/chat_template.jinja)

# 1. masked, schema-rendered, turn-aware corpora (train + val; window <= 4096 on MPS)
PYTHONPATH=src .venv/bin/python scripts/build_qat_masked_corpus.py \
    --window 4096 --wiki-tokens 300000 --max-tool-tokens 1024 \
    --out out/<exp>/masked_corpus_4096_v2.pt
PYTHONPATH=src .venv/bin/python scripts/build_qat_masked_corpus.py \
    --window 4096 --wiki-tokens 0 --max-tool-tokens 1024 --split test \
    --out out/<exp>/masked_val_4096_v2.pt
# read the printed density deciles; re-run with --min-density if the low tail is fat

# 2. LR probe FIRST (3 x ~40 steps): at 5e-5 the expected code-flip count is ~zero —
#    pick the highest LR whose val masked-CE is stable and whose flip telemetry moves
for LR in 5e-5 3e-4 1e-3; do
  PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv/bin/python scripts/exp058_qat_train_v2.py \
    --corpus out/<exp>/masked_corpus_4096_v2.pt --val-corpus out/<exp>/masked_val_4096_v2.pt \
    --layers 0-35 --optim adafactor --epochs 0.08 --grad-accum 4 --lr $LR \
    --ckpt-every 20 --out out/<exp>/probe_$LR
done

# 3. the real run: all 36 layers via Adafactor (~66-75 GB), >=1 full epoch, resumable
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv/bin/python scripts/exp058_qat_train_v2.py \
    --corpus out/<exp>/masked_corpus_4096_v2.pt --val-corpus out/<exp>/masked_val_4096_v2.pt \
    --layers 0-35 --optim adafactor --epochs 1 --grad-accum 8 --lr <probe-winner> \
    --train-norms --out out/<exp>/trained
#   ... interrupted? continue with:  --resume out/<exp>/trained/trained_latents.pt
#   capability lever (adds ~16 GB): --kd-teacher <dense parent, e.g. Qwen/Qwen3-8B>
#     ^ in-loop KD; prefer the OFFLINE top-K path below (Method B) at all-36 — the resident
#       teacher does not fit next to a student already at the memory ceiling
#   speed lever (validate parity):  --compute-dtype bf16

# 4. export -> Q2_0 GGUF (prints code-flips vs shipped — ~0% means the run only
#    drifted scales: raise LR / train longer before burning a SWE-rebench eval)
LLAMA_CPP_DIR=vendor/llama.cpp-prism PYTHONPATH=src .venv/bin/python \
    scripts/exp057_qat_export.py --latents out/<exp>/trained/trained_latents.pt --tag mytune
```

To point at a different model, update `MODEL` / `CHAT_TEMPLATE` at the top of the
three scripts (currently `out/exp-057/model`). Tool schemas are read from the logs
(`session["tools"]` or `messages[0]["tools"]` via `data.split.session_tools`); only
log formats with no stored schemas fall back to `reconstruct_tools` stubs.

## Hard constraints on Metal (learned the hard way — see the audit doc)

- **`foreach=False` is mandatory.** MPS multi-tensor (foreach) kernels *deadlock* at
  full-model scale — the "step-5 hang." AdamW, Adafactor (per-tensor by construction),
  `clip_grad_norm_`, and the fp32-master wrapper all stay per-tensor.
- **The window ceiling is memory, not INT_MAX — and it is ~12288, not 4096.** torch 2.12
  has **no MPS training kernel for SDPA** (fused paths are inference-only), so training
  materializes the full `[B, heads, S, S]` scores tensor and MPSGraph refuses a tensor
  with > INT_MAX elements. That gives **`n_heads · S² < 2³¹`** → `S ≤ 8191` at 32 heads;
  seq 8192 fails by *exactly one element* (32·8192² = 2³¹), which is why it read as
  "8k is impossible". Measured: 4096 / 6144 / 7168 / 8064 / 8128 / 8191 all pass, 8192 is
  the sole failure.

  **`qat.attention.enable_chunked_sdpa()` removes that cap outright** (on by default;
  `--no-chunked-attention` opts out). It computes causal SDPA in query blocks, so the
  score tensor is `[heads, chunk, kv_len]` and never `[heads, S, S]`. Output is
  **bit-identical** to `is_causal=True` — unit-tested at max abs err 0.0, which is what
  makes long-window results comparable with everything produced below 8191. It patches
  the registered `"sdpa"` entry *in place* rather than adding a name, so transformers'
  mask fast path still returns `attention_mask=None` for a plain causal decoder (a custom
  name gets an eager `[1,1,S,S]` float mask — 1 GB at S=16128).

  With chunking on the wall is unified memory. Measured full-model fwd+bwd, all-36
  layers, fp32, Adafactor, gradient checkpointing, M4 Max 128 GB — **on an otherwise
  idle machine**, with the numbers from a machine also hosting a ~41 GB LM Studio model
  for comparison, because the difference is large enough to change the decision:

  | window | fwd+bwd | ms/token | driver alloc | ms/token w/ 41 GB resident |
  |---|---|---|---|---|
  | 8064 | 66 s | — | 93 GiB | 8.16 |
  | **12288** | **116 s** | **9.47** | **125 GiB** | 10.31 |
  | 16128 | 227 s | 14.09 | 137 GiB | 19.98 |
  | 20480 | 499 s | 24.36 | 137 GiB | 38.56 |
  | 24576 | — | **hard OOM** | (tried 170 GiB) | — |

  **24576 is the hard wall.** Below it the cost is smoothly superlinear — attention is
  quadratic and the allocator starts spilling past ~128 GiB. **12288 is the throughput
  sweet spot**: 16128 costs +49% per token and 20480 +157%, so at a fixed wall-clock
  budget 12288 sees ~1.5×/2.6× more data. Choose a longer window only when whole-session
  context matters more than tokens seen — see the length distribution below. Freeing
  other GPU consumers is worth ~30% at 16128 alone. Same math rules out B≥2; grad-accum
  is the equivalent lever.

  **Conversation-length distribution** (universal SFT train split, `--max-tool-tokens
  4096`) is strongly bimodal: 83% of *conversations* are short broad-instruct/refusal
  rows under 2k tokens but only 2.6% of *tokens*; **81% of tokens live in conversations
  longer than 20k** (median CLI-log session 32k, agent-log 16k, SWE trajectory 11.5k;
  longest 256k). Share of conversations that fit whole in one window:

  | source | 4096 | 8064 | 12288 | 16128 | 20480 | 32768 |
  |---|---|---|---|---|---|---|
  | logs | 3% | 5% | 9% | 19% | 24% | 51% |
  | logs-agents | 2% | 22% | 37% | 49% | 58% | 76% |
  | swe-trajectories | 8% | 27% | 52% | 68% | 81% | 97% |
  | *all tokens in fitting convs* | 2.9% | 5.7% | 9.5% | 14.6% | 19.2% | 35.9% |

  There is no cliff — it is a long tail, and even 32k only reaches 36% of tokens.
  Nothing is *lost* at a smaller window (sessions pack contiguously across boundaries);
  what a longer window buys is the model seeing the far end of a trajectory conditioned
  on its start. The strongest case is `swe-trajectories`, the source the SWE-rebench
  eval actually grades: 52% → 68% → 81% whole at 12288 → 16128 → 20480.
- **fp32 latents, not bf16.** bf16 either destabilizes (high lr) or *underflows* the
  ternary threshold so no codes flip (low lr) — a ~lr update is below one bf16 ulp of
  a ~1e-2 latent. `--compute-dtype bf16` is the supported middle path: fp32 masters
  own the latents (updates accumulate exactly), only forward/backward run in bf16.
- **Fitting all 36 layers**: fp32 AdamW needs ≈ 116 GB (model 32.8 + grads 27.8 +
  2 fp32 states 55.6) → swaps. **`--optim adafactor` fits all-36 at ≈ 66-75 GB**
  (factored second moment is ~MBs). Real budget is the MPS working-set cap
  (~75-80% of unified memory ≈ 96-102 GB on 128 GB), not the nominal 128;
  `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` lifts the allocator cap in a pinch.
  Partial-layer training remains available (`--layers` + the grad probe) and is now
  cheaper: frozen layers whose weights are provably on-grid are left unwrapped.

## Corpus / masking rules (why the training is "on tool" properly)

- **Loss is masked to assistant/tool-call tokens** (`<|im_start|>assistant … <|im_end|>`
  spans, tool_calls included) **plus the terminating `<|im_end|>`** — the stop/EOS
  decision. (Before this fix no position in the corpus had `<|im_end|>` as a CE
  target, so the model was never trained to end its turn — the mechanistic cause of
  the iter-2/3 looping. The builder now asserts labeled stop-token targets exist.)
  Everything else is `-100`.
- **The session's REAL tool schemas are rendered** (`tools=` → the `# Tools` block) so
  the model trains *schema-conditioned*, matching inference. logtrain stores them on
  the system message; `reconstruct_tools` name→arg-key stubs are only the fallback for
  schema-less log formats.
- **All-masked windows are dropped** (a 4096 chunk landing in a long tool output has 0
  trainable tokens → NaN CE); windows keep ≥8 trainable tokens, plus an optional
  `--min-density` floor. `--max-tool-tokens` head+tail-truncates giant tool outputs —
  density is the wall-time lever (attention is only ~17% of FLOPs at 4096, so shorter
  windows barely help; fewer masked tokens do).
- **Train slice only** (seed-42 split), disjoint from the eval/holdout slices;
  `--split test` builds the validation corpus for `--val-corpus` from the disjoint
  test slice.
- Known limitation: windows can straddle session boundaries (a window may start
  mid-conversation under the previous session's residual context). Accepted packing
  noise; a session-aligned packer is a possible follow-up.
- ⚠️ Corpus rebuilds change the content fingerprint; `--resume` refuses a checkpoint
  from a different corpus, and loss values across corpus versions are NOT comparable
  (the stop-token class and real schemas shift both loss level and density).

## What we found (Ternary-Bonsai-8B, the first target)

The pipeline **works end to end and measurably moves the model** (masked loss
2.26→1.0, tool-error 79→68%), but a light (0.5-epoch, partial-layer) fine-tune did
**not** raise SWE-rebench patch/pass at 2-bit — the resolution floor is a capability
wall. The follow-up audit (`docs/qat_optimization_audit.md`) found this outcome
over-determined: the corpus never trained the stop token (hence the looping), real
tool schemas were discarded for stubs, AdamW's default weight decay eroded latents,
and at lr 5e-5 the flip-distance math predicts ~zero ternary code flips — the loss
drop was likely scale drift, not capacity re-allocation. The infrastructure for the
fuller attempt is now in place: all-36 layers via `--optim adafactor` (~66-75 GB),
`--resume` for multi-day runs, code-flip telemetry + an LR probe, KD from the dense
parent (`--kd-teacher`), bf16 compute over fp32 masters (`--compute-dtype bf16`), and
the corrected corpus. Historical loss numbers are not comparable to post-fix runs.
Full write-up: `ternary_calibration_experiments.md`.

---

# Reproducing the trajectory runs (iter-5 / iter-6)

Two ways to teach the ternary student, sharing every stage except the loss:

| | **Method A — verified-trajectory SFT** | **Method B — teacher-logit KD** |
|---|---|---|
| Signal | the solver's *sampled tokens* (hard labels) | the teacher's *full next-token distribution* |
| Loss | masked cross-entropy on assistant/tool tokens | KL to the teacher's top-K + CE, blended by `kd_alpha` |
| Teacher at train time | none (data is pre-generated) | none either — logits are **precomputed offline** |
| Needs | verified trajectories | verified trajectories **+** a same-tokenizer dense teacher |
| Status | run end-to-end (iter-5, results below) | precompute + loss validated; trainer wiring is the open step |

Both consume the same masked corpus and the same export/bench tail, so switching methods
changes one training flag, not the pipeline.

```
   SWE-rebench split ──► instance pools (train pool ⟂ eval holdout)
                              │
                              ▼
              Ornith-9B solver in Docker, graded by the hidden tests
                              │   (keep only resolved=true)
                              ▼
                    masked corpus (.pt, student tokenizer)
                         │                    │
             Method A ◄──┘                    └──► kd_precompute.py ──► top-K table
             masked CE                                  │
                         └──────────► Method B: CE + KL ◄┘
                                          │
                                          ▼
                       export Q2_0 GGUF ──► dual SWE-rebench bench
```

## Stage 0 — instance pools (disjoint by construction)

```bash
# full split to disk once: the datasets-server /rows preview API rate-limits (429)
.venv/bin/python scripts/download_swebench_dataset.py   # -> out/external/swe-rebench/all_test.jsonl

# what we GRADE on
PYTHONPATH=src .venv/bin/python scripts/build_swebench_holdout.py --n 10 \
    --from-local out/external/swe-rebench/all_test.jsonl \
    --out out/external/swe-rebench/holdout.jsonl

# what we TRAIN on — --exclude keeps the eval holdout out of the pool
PYTHONPATH=src .venv/bin/python scripts/build_swebench_holdout.py --n 150 \
    --from-local out/external/swe-rebench/all_test.jsonl \
    --exclude out/external/swe-rebench/holdout.jsonl \
    --out out/external/swe-rebench/distill_train.jsonl
```

`--exclude` is the invariant that makes the headline number meaningful: the student is never
graded on an issue whose solution it was trained on. The in-distribution eval (below) is the
deliberate *violation* of that, run separately as a diagnostic.

## Stage 1 — generate verified trajectories

```bash
bash scripts/run_ornith_distill_gen.sh          # --resume safe; kill and restart freely
```

Drives **Ornith-1.0-9B (Q5_K_M)** — 100% patch / 60% resolved on the eval holdout — through
`run_swebench_eval.py --agent openai-agents --temperature 0.25 --max-steps 100`, one clean
Docker container per instance. Every trajectory is graded by **actually running the gold
`FAIL_TO_PASS`/`PASS_TO_PASS` tests**, so `resolved=true` means verified, not plausible.

Artifacts land at
`out/swe-rebench/ornith-distill-gen/trajectories/Ornith-1.0-9B-Q5_K_M/<instance>.{traj,result}.json`.

Practicalities, all learned the hard way:
- **Budget ~10-14 h per 100 instances** on Apple Silicon: the SWE-rebench images are
  `linux/amd64` and run under emulation.
- `--resume` skips anything already graded; `--cleanup-images` removes each instance's SWE
  image after grading. That *untags* rather than deletes, leaving `<none>` dangling layers, so
  run `bash scripts/docker_housekeep.sh` alongside long generations or the disk fills
  (78 GB of dangling images once stalled a run with a bare `exit 125`).
- Docker Desktop's VM disk is the real cap, and it is shared with everything else you run.

### Stage 1b — going multi-language (SWE-rebench-V2)

Stage 1 as written harvests **Python only** — SWE-rebench V1 is a Python/pytest dataset.
To stop the student overfitting to one language, generate from
[**SWE-rebench-V2**](https://huggingface.co/datasets/nebius/SWE-rebench-V2): 32,079
instances across 20 languages, same instance schema plus a `language` field.

```bash
# 0. full V2 split to disk (defaults follow the dataset: split=train, v2_all.jsonl)
.venv/bin/python scripts/download_swebench_dataset.py --dataset nebius/SWE-rebench-V2

# 1. a language-BALANCED eval holdout, then a training pool disjoint from it
LANGS=python,go,ts,js,rust,java,php,kotlin
PYTHONPATH=src .venv/bin/python scripts/build_swebench_holdout.py \
    --from-local out/external/swe-rebench/v2_all.jsonl \
    --languages $LANGS --difficulty medium --max-f2p 25 --n 24 --seed 42 \
    --out out/external/swe-rebench/holdout_multilang.jsonl
PYTHONPATH=src .venv/bin/python scripts/build_swebench_holdout.py \
    --from-local out/external/swe-rebench/v2_all.jsonl \
    --languages $LANGS --difficulty medium --max-f2p 25 --n 240 --seed 7 \
    --exclude out/external/swe-rebench/holdout_multilang.jsonl \
    --out out/external/swe-rebench/distill_train_multilang.jsonl

# 2. PROVE THE GRADER WORKS before burning hours (gold patches must resolve)
PYTHONPATH=src .venv/bin/python scripts/validate_swebench_v2_grading.py \
    --holdout out/external/swe-rebench/holdout_multilang.jsonl --per-language 1

# 3. generate — against a server you already have running (LM Studio etc.)
BASE_URL=http://localhost:1234/v1 MODEL=ornith-1.0-35b \
    bash scripts/run_multilang_distill_gen.sh
```

**Selection knobs that matter.** `--balanced` (auto-on with `--languages`) samples
round-robin, so Python/Go/JS — the three biggest buckets — can't crowd out the rest.
`--clean-only` (default) keeps only annotator-code `A` tasks; the `B1..B6` codes flag
underspecified issues and tests keyed to implementation details, which yield trajectories
that are either garbage or accidentally-correct. `--max-f2p 25` drops rows whose
FAIL_TO_PASS lists the **whole suite** (some have 16,000+ ids), where "resolved" would
demand every test in the repo pass.

**Three things differ per instance, and getting any of them wrong is silent:**

| | V1 (Python) | V2 (20 languages) |
|---|---|---|
| checkout dir | `/testbed` | `/<repo-name>` |
| test command | pytest + node ids, wrapped in `conda run -n testbed` | the instance's own `install_config.test_cmd`, no conda |
| log → statuses | our pytest `-rA` parser | the parser the instance names in `install_config.log_parser` |

The parsers are **vendored verbatim** from SWE-rebench-V2 (MIT) into
`src/quant_tuner/eval/_swerebench_v2_parsers.py` — 76 of them — because the recorded
FAIL_TO_PASS ids are literally whatever *that exact function* emitted at dataset-build
time. A near-miss reimplementation parses zero matching ids and marks every trajectory
unresolved, which reads exactly like "the model failed." Re-vendor with
`scripts/vendor_swerebench_parsers.py` (`--check` for drift), then re-run the golden
check. Some runners embed timings in the test name (`… [20.82 ms]`), so both the recorded
and the freshly-parsed ids are timing-normalized before comparison.

**Always run the golden-patch check first.** It applies each instance's own gold patch and
requires `resolved=True`; a failure there means the *harness* is wrong, not the model.
It also distinguishes the recurring `docker run` exit-125 (registry unreachable vs. full
Docker VM disk) instead of leaving an opaque `CalledProcessError`.

## Stage 2 — masked corpus from the resolved trajectories

```bash
PYTHONPATH=src .venv/bin/python scripts/build_ornith_distill_corpus.py \
    --traj-dir out/swe-rebench/ornith-distill-gen/trajectories/Ornith-1.0-9B-Q5_K_M \
    --max-tool-tokens 1024 --out out/exp-058/distill_corpus.pt
```

Keeps `resolved=true` sessions only, re-renders them through the **student's** tokenizer and
chat template, and masks the loss to assistant/tool-call tokens **plus the terminating
`<|im_end|>`** (see the masking rules above — omitting the stop token is what caused the
iter-2/3 looping). Flattening is `quant_tuner.qat.corpus.trajectory_to_messages`, shared with
the published dataset builder so the two cannot drift.

### Stage 2b — the universal SFT corpus (preferred for a new run)

`data.universal` writes an `sft.jsonl.gz` next to the calibration corpus: **full**
conversations (no windowing, no clipping, no stubbing, no chat template applied), real
tool schemas, system prompts already boilerplate-scrubbed, and a `split` field that
matches the calibration corpus — so training on `split=="train"` keeps every eval holdout
held out. `build_sft_qat_corpus.py` masks and packs it exactly like the other two corpora:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_sft_qat_corpus.py \
    --sft out/corpora/qwen3-universal/sft.jsonl.gz \
    --window 12288 --max-tool-tokens 4096 --min-density 0.05 \
    --out out/exp-058/sft_corpus_universal_12288.pt
# and the disjoint validation corpus for --val-corpus:
PYTHONPATH=src .venv/bin/python scripts/build_sft_qat_corpus.py \
    --sft out/corpora/qwen3-universal/sft.jsonl.gz --split test \
    --window 12288 --max-tool-tokens 4096 --min-density 0.05 \
    --out out/exp-058/sft_val_universal_12288.pt
```

**Every source is taken whole** (`SFT_DEFAULT_BUDGETS = {}`). This is the deliberate
difference from the calibration corpus, which *is* token-budgeted (~4.4M) because
llama-imatrix/AWQ/GPTQ each sample a fixed slice of it and an unbalanced mix skews
`E[a²]`. QAT spends its budget in **epochs**, not in tokens on disk — so put everything on
disk and use fractional `--epochs`. Cap a source with `--budget SOURCE=N` (`0` drops it).

Sources are packed **separately**, so a window never glues two of them together.

#### What the preprocessing costs (audited, printed on every build)

The builder prints per source, and stores in the blob's `per_source`: tool-calls
rendered / in source, reasoning turns rendered / in source, tokens dropped by tool-output
truncation as a **share of that source's conversation content**, and windows dropped by
the density floor. Measured on the Qwen3-universal SFT file:

- **Tool calls survive 1:1.** Nothing in the chain drops a `tool_calls` field.
- **Reasoning survives for agentic trajectories, not for chat logs.** The template keeps
  `reasoning_content` only on assistant turns *after the last user turn*. An agent
  trajectory is one task turn followed by dozens of assistant/tool turns, so **all** of it
  is kept (measured 74/74 on an agent-log sample); a multi-turn CLI chat log keeps only
  its tail segment. That is also what the model sees at inference, so it is the right
  distribution — but it means the ~383 reasoning turns in the CLI logs mostly don't reach
  the corpus, while the ~3,900 in the agent logs do.
- **Tool-output truncation is by far the biggest cut, and `--max-tool-tokens` is the
  knob.** At an 8064 window, measured over a 24-conversation sample:

  | `--max-tool-tokens` | conversation content dropped | windows kept (`--min-density` 0 / .05 / .10) |
  |---|---|---|
  | 1024 | **28%** | 74 / 74 / 69 |
  | 2048 | 20% | 83 / 78 / 74 |
  | 3072 | 15% | 87 / 81 / 74 |
  | **4096** | **13%** | **89 / 79 / 75** |
  | 0 (off) | 0% | 93 / 82 / 75 |

  1024 was right at a 4096 window and is far too aggressive above it. **Use 4096** at a
  12288 window — a single tool result still can't eat more than a third of a window, but
  realistic outputs survive nearly intact (the full-corpus build drops 13% of conversation
  content, down from 28% at 1024). Note the interaction with `--min-density`: at 0.10 the
  extra tool context just pushes windows below the floor and they get dropped anyway, so
  keeping more history buys nothing there. **0.05** is the setting that actually keeps it.

## Stage 3A — Method A: masked-CE training

```bash
bash scripts/run_iter5_pipeline.sh 5e-4 myrun 2.2      # LR / tag / epochs -> train, export, bench
```

which is `quant_tuner.qat.train` (CLI shim `scripts/exp058_qat_train_v2.py`) at:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv/bin/python scripts/exp058_qat_train_v2.py \
    --corpus out/exp-058/distill_corpus.pt \
    --layers 0-35 --optim adafactor --dtype fp32 \
    --epochs 2.2 --grad-accum 2 --lr 5e-4 \
    --ckpt-every 20 --flip-sample 12 --out out/exp-058/trained_myrun
```

**The LR is the whole ballgame**, because a ternary model only learns by *flipping codes*:

| LR | what actually happens |
|---|---|
| 3e-4 | ~0% code flips — the run only drifts fp16 scales. Loss falls; the model is unchanged. |
| **5e-4** | **~0.7% flips, loss ~0.5 at ~2.2 epochs — the measured sweet spot** |
| 1e-3 | real flips, but tool-calling degrades on small corpora |

Watch the **flip telemetry** (every 20 steps, and again at export vs the shipped weights).
A run reporting ~0% flips has learned nothing structural — fix that before spending 10 h on a
SWE-rebench eval. Also don't over-train: 8 epochs drove loss to 0.01, i.e. memorization.

## Stage 3B — Method B: teacher-logit KD (iter-6)

Hard labels waste most of what a teacher knows: only the sampled token is supervised. KD
supervises the whole distribution, which matters at 2 bits where the student cannot match the
teacher exactly and should at least be *wrong in the same direction*.

`train.py --kd-teacher` runs the teacher in-loop, but that holds ~16 GB resident next to a
student already at the MPS ceiling. So run the teacher **once, offline**, and keep only the
top-K:

```bash
.venv/bin/python scripts/kd_precompute.py \
    --teacher SWE-Lego/SWE-Lego-Qwen3-8B \
    --corpus out/exp-058/distill_corpus.pt \
    --topk 64 --out out/exp-058/kd_topk.pt
#   --max-windows 4   smoke test first: validates tokenizer + arch before the full pass
```

Measured on SWE-Lego-Qwen3-8B over the iter-5 corpus: **top-64 captures 99.8% of the teacher's
mass** (median 100%), teacher top-1 == gold label 83.7% of the time, gold inside the stored
top-64 at 98.9% of positions. Cost: ~0.4 KB per labeled position (**125 MB** for a 217-window
corpus vs ~16 GB resident), and training gets *cheaper* than plain CE since the teacher never
runs. The table stores `win/pos/idx(int32)/logp(fp16)/tail`, where `tail` is the log-mass
outside the top-K so the renormalization is exact.

**Choosing a teacher — the tokenizer, not the vocab size, is the constraint.** Per-token KD
across different tokenizers is silently wrong, so `tokenizer_compatibility()` compares actual
id→token *strings* and refuses divergent pairs. A padded embedding matrix is fine and common:
SWE-Lego-Qwen3-8B reports `vocab_size` 151936 vs Bonsai's 151669, but the tokenizers are
byte-identical over all 151669 ids, so logits are sliced to the shared prefix.

The precompute is deliberately **architecture-agnostic**, so switching to a newer/larger-vocab
teacher needs no format change:
- `resolve_vocab_size()` walks nested configs (`text_config` / `llm_config` / `decoder`), so
  multimodal wrappers (e.g. `BeeForConditionalGeneration`) resolve to their LM vocab.
- `load_teacher()` tolerates published-config quirks — SWE-Lego really does ship
  `max_position_embeddings: 163840.0` as a float, which strict validation rejects.
- The forward uses `logits_to_keep` where supported, with a full-logits gather fallback.
- Top-K storage is ids + values, so it is vocab-size independent.

`kd_loss_from_topk()` consumes the table. It renormalizes **both** sides over the stored top-K
— an early version normalized only the teacher, which left a constant offset and scored a
*identical* student at 0.89 instead of 0; `tests/unit/test_kd_precompute.py` pins that
identity-is-zero property.

> **Status**: precompute and loss are validated (unit tests + a real-weights smoke run);
> `train.py` still consumes only the in-loop `--kd-teacher`. Wiring the offline table into the
> training step is the remaining iter-6 task.

## Stage 4 — export and dual bench

```bash
LLAMA_CPP_DIR=vendor/llama.cpp-prism PYTHONPATH=src .venv/bin/python \
    scripts/exp057_qat_export.py --latents out/exp-058/trained_myrun/trained_latents.pt --tag myrun

# generalization: the disjoint 10-instance holdout (the number that counts)
PYTHONPATH=src .venv/bin/python scripts/run_swebench_eval.py \
    --models out/exp-057/Ternary-Bonsai-8B-myrun-Q2_0.gguf \
    --holdout out/external/swe-rebench/holdout.jsonl \
    --workspace out/swe-rebench/ternary-myrun-swe \
    --agent openai-agents --temperature 0.25 --max-steps 100 --resume --cleanup-images

# in-distribution: the instances it TRAINED on (diagnostic, expected inflated)
bash scripts/run_iter5_indist_eval.sh myrun
```

Read them together: in-dist up + generalization flat = it learned but overfit (get more data);
both up = real learning; both flat = the run was scale-drift, check the flip telemetry.

Q2_0 (ftype 41) requires the **prism llama.cpp fork** — hence `LLAMA_CPP_DIR=vendor/llama.cpp-prism`.

## Unattended: grow data until it generalizes

```bash
bash scripts/run_iter5_autoloop.sh
```

Per round: source a fresh batch of instances disjoint from both the eval holdout and everything
already tried → generate → rebuild corpus + in-dist holdout from *all* resolved trajectories →
retrain at the fixed sweet spot → export → dual-bench. **Stops** when generalization
`pass_rate > 0` or `patch_rate >= 0.60`, or at `MAX_ROUNDS`. Round artifacts are tagged
`iter5-rN`. It self-manages the dangling-image guard for the whole lifetime and auto-salvages
from the last checkpoint if a training run dies.

## What this produced

| verified trajectories | generalization patch | generalization pass |
|---:|---:|---:|
| 12 | 40% | 0% |
| 30 | 50% | 0% |
| 60 | 50% | 0% |

More verified data recovers the model's *ability to produce a patch* but has so far produced
**zero passes on unseen issues** — the 2-bit resolution floor looks like a capability wall,
not a data-quantity wall, which is the motivation for Method B. (A single 8% in-distribution
solve at 12 trajectories did **not** replicate at 30; treat it as noise.)

## Memory: the OOM you will otherwise hit

Both observed OOM kills landed **exactly on a `--ckpt-every` boundary** — peak training memory
plus `save_ckpt`'s ~28 GB whole-dict `.cpu()` transient. `save_ckpt` now releases the MPS cache
*before* that copy, the training loop calls `torch.mps.empty_cache()` every 25 steps to bound
working-set creep, and the auto-loop salvages from the last checkpoint if a run still dies.

Stay in **fp32**: `--compute-dtype bf16` is a *pessimization* at all-36 (measured 54.5 GiB vs
31 GiB), because `MasterOptimizer` keeps a full fp32 master copy on top of the bf16 model+grads.

## Publishing the trajectories as a dataset

Datasets are staged under `datasets/<name>/` and versioned independently of the code. The
payloads are gitignored (regenerable, and they live on the Hub); the card, `manifest.json` and
`CHANGELOG.md` are tracked so the repo records exactly what was published.

```bash
.venv/bin/python scripts/dataset.py list
.venv/bin/python scripts/dataset.py build swe-agentic-trajectories
.venv/bin/python scripts/dataset.py push  swe-agentic-trajectories --dry-run          # verify first
.venv/bin/python scripts/dataset.py push  swe-agentic-trajectories --bump minor -m "add round-3"
```

`push` rebuilds, bumps the version, uploads, and tags the Hub repo `v<version>` so earlier
releases stay pinnable via `load_dataset(..., revision="v0.1.0")`. The manifest records a
release **only after** a successful upload, so a failed push cannot claim one.

A split with `publish=False` is built locally but withheld from the Hub *and* from the card —
that is how the `all` split (including failures) stays available for failure analysis while
only verified solutions ship. Published: **[pearsonkyle/swe-agentic-trajectories](https://huggingface.co/datasets/pearsonkyle/swe-agentic-trajectories)**.

**Adding a new dataset is a one-entry change**: write a builder that yields dicts, then append
a `DatasetSpec` to `REGISTRY` in `src/quant_tuner/datasets/registry.py`. Staging layout, card
rendering, versioning, changelog and push are shared.
