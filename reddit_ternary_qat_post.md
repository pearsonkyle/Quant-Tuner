# Reddit post

**Title: You can fine-tune a native 1.58-bit (ternary) LLM on a Mac. Here is how, and the traps that cost me a week.**

Native-ternary models (BitNet style, weights are literally -1/0/+1 with a shared scale) are showing up on Hugging Face now, like prism-ml/Ternary-Bonsai-8B (a ternary Qwen3-8B). They run great at ~2 bits. But if you want to specialize one on your own data, there is a catch that took me a while to figure out.

**Post-hoc quant calibration does nothing to these models.** imatrix, AWQ, and GPTQ all exist to recover "quantization rounding error," and a native-ternary model has none. Its fp16 file is already exactly `scale * {-1, 0, +1}`, so the 2-bit GGUF is a lossless re-encode of it (I measured KLD ~ 0.000). There is no higher-precision original to fit toward. I checked this on my own GPTQ code: rounding the weights to the ternary grid changes them by literally 0.

**The only real lever is QAT.** Keep the ternarization inside the training loop (straight-through estimator, BitNet b1.58 / Ternary Weight Networks style) and continue-train the fp16 "latent" weights. Good news: the method is public, it is about 20 lines, and it runs on Metal. No CUDA required.

The traps, all Metal/MPS specific, all of which cost me real time:

- **foreach=False is mandatory.** MPS multi-tensor (foreach) optimizer and clip kernels deadlock at full-model scale. The classic symptom is "training hangs at step 5." Pass `foreach=False` to AdamW and to clip_grad_norm_.
- **Sequence length <= 4096.** seq 8192 errors out with "MPSGraph tensor dims larger than INT_MAX" (no flash attention on MPS, so the attention scores tensor is too large to index).
- **fp32 latents, not bf16.** bf16 either blows up at a useful learning rate, or the tiny updates underflow the ternary threshold so no weight ever flips sign. fp32 registers updates at any lr.
- **Train a subset of layers.** Full-model fp32 AdamW swaps even on 128GB (Adam keeps two fp32 state tensors, about 56GB on top of the model). ~18 layers fit at ~58GB. Pro tip: pick WHICH layers with a quick gradient-importance probe. The early layers carry most of the task signal, not the last N you would assume.

Corpus tip that mattered a lot: mask the loss to the assistant / tool-call tokens only, and render the tool schemas into the chat template, so you train the model to generate tool calls conditioned on the schema instead of on boilerplate.

Results on my first target (Ternary-Bonsai-8B, agentic SWE-rebench, 10 held-out issues):

The pipeline works end to end. Masked tool-call loss dropped 2.26 to 1.0 and the model still tool-calls correctly after training. But here is the honest scorecard, because it is more interesting than a clean win:

| run | patch rate | pass rate | behavior |
| --- | --- | --- | --- |
| base model (no fine-tune) | 50% | 0% | fine |
| QAT on the last 18 layers | 40% | 0% | LOOPED hard (repeated one command up to 553 times) |
| QAT on the most influential layers | 40% | 0% | looping fixed, clean runs |

Two things I learned that might save you time:

1. Pick which layers to train with a gradient-importance probe, do not just take the last N. On this model the early layers (block 0 dominates) carry almost all of the task gradient signal, and the middle layers are nearly dead. Training the wrong half made the model loop pathologically; training the right half fixed the looping and matched the baseline behavior.

2. A light fine-tune fixes behavior, not capability. Training cleaned up tool-call formatting and stopped the loops, but it did not raise the actual issue-resolution rate past the base model, and pass rate stayed 0% at 2 bits. That looks like a real 2-bit capability floor for a short fine-tune. The likely fixes are more training (full epochs, all layers, more data), not more clever sampling. That is the next experiment.

So: the infrastructure is proven and reusable, the gotchas are documented, and the "is 2-bit fine-tunable to agentic competence" question is still open pending a full-scale run.

Code and a reusable pipeline guide (docs/ternary_qat.md): [your repo link]

If you have an Apple Silicon Mac with enough unified memory and you want a task-specialized 1.58-bit model, this is a working starting point.

---

# X / Twitter thread

1/ You can fine-tune a native 1.58-bit (ternary) LLM on a Mac. No CUDA.

The thing nobody tells you: imatrix / AWQ / GPTQ do nothing to these models. They are already a lossless quant of themselves (KLD ~ 0). The only lever is QAT.

2/ QAT = keep the ternarize step inside the loop (straight-through estimator, BitNet b1.58 / TWN). About 20 lines, runs on Metal.

The traps that cost me a week, all MPS specific:

3/
- foreach=False, or AdamW and clip_grad_norm deadlock at scale (the "hangs at step 5")
- seq <= 4096 (8192 hits MPSGraph INT_MAX, no flash attn on MPS)
- fp32 latents (bf16 updates underflow the ternary threshold, no flips)
- train a layer subset, full net swaps. Pick layers by gradient importance, not last-N

4/ Bonus: mask the loss to tool-call tokens and render the tool schemas into the chat template, so it learns schema-conditioned tool use.

Loss dropped 2.26 to 1.0, model still tool-calls. Honest: the light run did not beat baseline patch rate yet, but the infra works.

Code: [your repo link]
