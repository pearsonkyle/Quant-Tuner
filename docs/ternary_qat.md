# Continued QAT for native-ternary (1.58-bit) models

A reusable pipeline to **fine-tune a natively-ternary GGUF model** (e.g.
`prism-ml/Ternary-Bonsai-8B`) on a task-specific corpus and re-export a runnable
2-bit GGUF. Built for Metal (Apple Silicon); the same scripts run on CUDA.

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

The core quantizer is `src/quant_tuner/qat/ternary.py` — a **per-group TWN**
straight-through estimator that reproduces the shipped weights *exactly* at step 0
(so the fine-tune starts from the real model, no drift).

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

## Reproducing the iter-5 verified-distillation runs

Everything below is committed; `out/` artifacts are regenerated by these steps.

```bash
# 0. one-time: full SWE-rebench split to disk (the /rows preview API rate-limits at 429)
.venv/bin/python scripts/download_swebench_dataset.py       # -> out/external/swe-rebench/all_test.jsonl

# 1. instance pools, disjoint by construction
#    eval holdout (what we grade on) and a training pool that EXCLUDES it
PYTHONPATH=src .venv/bin/python scripts/build_swebench_holdout.py --n 10 \
    --from-local out/external/swe-rebench/all_test.jsonl \
    --out out/external/swe-rebench/holdout.jsonl
PYTHONPATH=src .venv/bin/python scripts/build_swebench_holdout.py --n 150 \
    --from-local out/external/swe-rebench/all_test.jsonl \
    --exclude out/external/swe-rebench/holdout.jsonl \
    --out out/external/swe-rebench/distill_train.jsonl

# 2. harvest trajectories from a strong solver, keep only VERIFIED (tests pass) ones
bash scripts/run_ornith_distill_gen.sh                      # Ornith-9B over the pool, --resume safe

# 3. corpus: resolved trajectories -> student tokenizer, assistant+stop-token masked
PYTHONPATH=src .venv/bin/python scripts/build_ornith_distill_corpus.py \
    --traj-dir out/swe-rebench/ornith-distill-gen/trajectories/Ornith-1.0-9B-Q5_K_M \
    --max-tool-tokens 1024 --out out/exp-058/distill_corpus.pt

# 4. train (the measured sweet spot) -> export Q2_0 -> dual-bench
bash scripts/run_iter5_pipeline.sh 5e-4 myrun 2.2           # lr / tag / epochs
#   or the unattended loop that grows data until it generalizes:
bash scripts/run_iter5_autoloop.sh

# 5. reusable dataset out of the trajectories (chat-template ready)
.venv/bin/python scripts/export_trajectory_dataset.py --resolved-only \
    --out out/datasets/swe_agentic_trajectories_resolved.jsonl
```

**Recipe that matters** (measured, see `ternary_calibration_experiments.md`): `lr 5e-4`,
~2.2 epochs, all 36 layers, Adafactor, **fp32** (`--compute-dtype bf16` is a pessimization at
all-36: 54.5 GiB vs 31 GiB, because `MasterOptimizer` adds a full fp32 master copy on top of
the bf16 model+grads). LRs of 3e-4 flip ~zero codes (scale drift only); 1e-3 flips codes but
wrecks tool use on small corpora.

**Memory**: both observed OOM kills landed exactly on a `--ckpt-every` boundary — peak
training memory plus `save_ckpt`'s ~28 GB whole-dict `.cpu()` transient. `save_ckpt` now
releases the MPS cache before that copy, and the loop auto-salvages from the last checkpoint
if a run still dies.

### KD (iter-6): offline top-K teacher logits

In-loop KD (`train.py --kd-teacher`) keeps a dense teacher resident (~16 GB) next to a student
already at the memory ceiling. Precompute instead:

```bash
.venv/bin/python scripts/kd_precompute.py \
    --teacher SWE-Lego/SWE-Lego-Qwen3-8B \
    --corpus out/exp-058/distill_corpus.pt \
    --max-windows 4 --out out/exp-058/kd_topk_smoke.pt      # drop --max-windows for the full pass
```

The teacher must share the student's **tokenizer**, not merely its `vocab_size`:
SWE-Lego-Qwen3-8B reports `vocab_size` 151936 vs Bonsai's 151669, but the tokenizers are
byte-identical over all 151669 ids — the extra rows are embedding padding, so logits are sliced
to the shared prefix. `tokenizer_compatibility()` verifies id→string equality and refuses
genuinely divergent tokenizers (per-token KD across different tokenizers is silently wrong).
`resolve_vocab_size()` walks nested configs so composite architectures (e.g.
`BeeForConditionalGeneration`) resolve to their LM vocab, and malformed configs (SWE-Lego ships
`max_position_embeddings: 163840.0`, a float) are sanitized on load.

### Publishing datasets

Datasets are staged under `datasets/<name>/` and versioned independently of the code. The
payloads are gitignored (regenerable, and they live on the Hub); the card, `manifest.json`
and `CHANGELOG.md` are tracked so the repo records exactly what was published.

```bash
.venv/bin/python scripts/dataset.py list
.venv/bin/python scripts/dataset.py build swe-agentic-trajectories
.venv/bin/python scripts/dataset.py push  swe-agentic-trajectories --bump minor -m "add round-3 trajectories"
.venv/bin/python scripts/dataset.py push  swe-agentic-trajectories --dry-run   # verify first
```

`push` rebuilds, bumps the version, uploads the folder, and tags the Hub repo `v<version>` so
earlier releases stay pinnable via `load_dataset(..., revision="v0.1.0")`. The manifest only
records a release **after** a successful upload, so a failed push cannot claim one.

**Adding a new dataset is a one-entry change**: write a builder that yields dicts, then append
a `DatasetSpec` to `REGISTRY` in `src/quant_tuner/datasets/registry.py`. Staging layout, card
rendering, versioning, changelog and push are shared.
