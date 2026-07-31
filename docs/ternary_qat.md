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
- **Window ≤ 4096** (now enforced by the trainer). seq 8192 errors with *"MPSGraph
  tensor dims larger than INT_MAX"*: torch 2.12 has **no MPS training kernel for
  SDPA** (fused paths are inference-only), so training materializes the full
  `[B, heads, S, S]` scores tensor — and 32·8192² = 2³¹ overflows INT_MAX exactly.
  Same math rules out B≥4 at 4096; B=2 is legal but compute-bound → grad-accum is
  equivalent. Throughput is token-bound (~10 ms/token) regardless of window.
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
