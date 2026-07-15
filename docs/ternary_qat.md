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

# 1. masked, schema-rendered, turn-aware corpus (window <= 4096 on MPS; see limits)
PYTHONPATH=src .venv/bin/python scripts/build_qat_masked_corpus.py \
    --window 4096 --wiki-tokens 300000 --out out/<exp>/masked_corpus_4096.pt

# 2. train — pick influential layers via the grad probe, not naive last-N
PYTHONPATH=src .venv/bin/python scripts/exp058_layer_importance.py \
    --corpus out/<exp>/masked_corpus_4096.pt --windows 24        # ranks layers
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv/bin/python scripts/exp058_qat_train_v2.py \
    --corpus out/<exp>/masked_corpus_4096.pt --layers 0-14,32,34,35 \
    --epochs 0.5 --grad-accum 4 --lr 5e-5 --dtype fp32

# 3. export -> Q2_0 GGUF (restores the original chat template, packs embd/output at Q2_0)
LLAMA_CPP_DIR=vendor/llama.cpp-prism PYTHONPATH=src .venv/bin/python \
    scripts/exp057_qat_export.py --latents out/<exp>/trained/trained_latents.pt --tag mytune
```

To point at a different model, update `MODEL` / `CHAT_TEMPLATE` at the top of the
three scripts (currently `out/exp-057/model`), and reconstruct the tool schemas for
your log format in `build_qat_masked_corpus.reconstruct_tools`.

## Hard constraints on Metal (learned the hard way — see `memory` / experiments doc)

- **`foreach=False` is mandatory.** MPS multi-tensor (foreach) kernels *deadlock* at
  full-model scale — the "step-5 hang." Both AdamW and `clip_grad_norm_` must pass it.
- **Window ≤ 4096.** seq 8192 errors with *"MPSGraph tensor dims larger than INT_MAX"*
  (no flash-attention → the `[heads × 8192²]` scores tensor is too big). Throughput is
  token-bound (~10 ms/token) regardless of window.
- **fp32 latents, not bf16.** bf16 either destabilizes (high lr) or *underflows* the
  ternary threshold so no codes flip (low lr). fp32 registers updates at any lr.
- **Fit by training a subset of layers.** Full-36 fp32 AdamW swaps (Adam's 2 fp32
  states ~56 GB over budget on 128 GB); ~18 layers fit at ~58 GB. Use the grad probe
  to pick *which* 18 (early layers 0-14 dominate the tool-call signal, not last-N).

## Corpus / masking rules (why the training is "on tool" properly)

- **Loss is masked to assistant/tool-call tokens** (`<|im_start|>assistant … <|im_end|>`
  spans, tool_calls included); everything else is `-100`. Uniform loss dilutes the
  signal with boilerplate.
- **Tool schemas are reconstructed and rendered** (`tools=` → the `# Tools` block) so
  the model trains *schema-conditioned*, matching inference. The logs don't store
  schemas; we synthesize them from the observed `name → arg-keys`.
- **All-masked windows are dropped** (a 4096 chunk landing in a long tool output has 0
  trainable tokens → NaN CE); windows keep ≥8 trainable tokens.
- **Train slice only** (seed-42 split), disjoint from the eval/holdout slices.

## What we found (Ternary-Bonsai-8B, the first target)

The pipeline **works end to end and measurably moves the model** (masked loss
2.26→1.0, tool-error 79→68%), but a light (0.5-epoch, partial-layer) fine-tune did
**not** raise SWE-rebench patch/pass at 2-bit — the resolution floor is a capability
wall. Likely under-trained (subsampled data, half the layers). The infrastructure is
here for the fuller attempt (more epochs, all layers via an fp32-master trick, more
data) and for the **next models we fine-tune in this style**. Full write-up:
`ternary_calibration_experiments.md`.
