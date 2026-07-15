# Reddit post

**Title: You can fine-tune a native 1.58-bit (ternary) LLM on a Mac. Here is how, and the traps that cost me a week.**

## What these models are

A new family of "native ternary" (also called 1.58-bit) LLMs is showing up on Hugging Face. Instead of training in fp16 and quantizing afterward, these are trained from the start so that every weight is literally one of three values: -1, 0, or +1, with a single fp16 scale shared across a small group of weights. That is the BitNet b1.58 idea (Microsoft, 2024), and people are now releasing real ones.

My target was **prism-ml/Ternary-Bonsai-8B**. It is a **Qwen3-8B** under the hood (standard dense Transformer, 36 layers), but every projection weight is ternary. In practice that means an 8B model ships as a roughly 2GB GGUF and runs at about 2 bits per weight, with quality that is surprisingly close to the full model on general text. The catch is that the ternary format is new, so it needs a fork of llama.cpp to run (PrismML has one; upstream PR is in progress).

They run great. The interesting question is: can you **specialize** one on your own data. That is where it gets subtle.

## Post-hoc quant calibration does nothing here

imatrix, AWQ, and GPTQ all exist to recover "quantization rounding error," the gap between an fp16 original and the low-bit grid it gets squeezed onto. A native-ternary model has no such gap. Its fp16 file is already exactly `scale * {-1, 0, +1}`, so the 2-bit GGUF is a lossless re-encode of it (I measured KLD ~ 0.000 between them). There is no higher-precision original to fit toward. I even ran my own GPTQ code on it: rounding the weights to the ternary grid changes them by literally 0. So these calibration tools are a no-op on this class of model.

## How you actually train it (QAT)

The only real lever is quantization-aware training. You keep the ternarization inside the forward pass and continue-train the underlying fp16 "latent" weights:

1. Load the trainable (unpacked) checkpoint, whose fp16 weights are just the ternary values stored in a fat container.
2. In the forward pass, ternarize each weight on the fly (per group of 128 weights: pick a scale, snap each weight to -1/0/+1). The model computes with the ternary weights.
3. In the backward pass, use a straight-through estimator: pretend the ternarize step was the identity, so the gradient flows to the fp16 latent weight.
4. Train on your data. The latents drift, and each step re-ternarizes, so weights can flip sign when a latent crosses the threshold.
5. Re-ternarize the final latents and pack back to the 2-bit GGUF.

The whole ternarizer is about 20 lines (BitNet b1.58 / Ternary Weight Networks). The important part is that initializing the latents from the shipped weights reproduces the model exactly at step 0, so you are fine-tuning the real thing, not a degraded copy. And all of this runs on Metal. No CUDA required.

## The Metal / MPS traps (these cost me real time)

- **foreach=False is mandatory.** MPS multi-tensor (foreach) optimizer and clip kernels deadlock at full-model scale. The classic symptom is "training hangs at step 5." Pass `foreach=False` to AdamW and to clip_grad_norm_.
- **Sequence length <= 4096.** seq 8192 errors out with "MPSGraph tensor dims larger than INT_MAX" (no flash attention on MPS, so the attention scores tensor is too large to index).
- **fp32 latents, not bf16.** With bf16, either training blows up at a useful learning rate, or the tiny updates underflow the ternary threshold so no weight ever flips sign. fp32 registers updates at any lr.
- **Optimizer choice controls how many layers you can train.** Full-model AdamW keeps two fp32 state tensors, about 56GB on top of the model, and swaps even on 128GB. Adafactor uses factored state instead and fits all 36 layers in about 33GB. So Adafactor if you want the whole network, AdamW on a subset if you want the stronger optimizer.
- **Pick WHICH layers with a gradient-importance probe.** Do not just take the last N. On this model the early layers (block 0 dominates) carry almost all of the task gradient signal, and the middle layers are nearly dead. Training the wrong half made the model loop pathologically; training the right half fixed it.

Corpus tip that mattered a lot: mask the training loss to the assistant / tool-call tokens only, and render the tool schemas into the chat template, so you train the model to generate tool calls conditioned on the schema instead of on boilerplate.

## Results (preliminary, and encouraging)

Task: make the model a better agentic coder, measured on SWE-rebench (10 held-out real GitHub issues). Honest scorecard:

| run | patch rate | pass rate | behavior |
| --- | --- | --- | --- |
| base model (no fine-tune) | 50% | 0% | fine |
| QAT, last 18 layers | 40% | 0% | LOOPED hard (repeated one command up to 553 times) |
| QAT, gradient-influential layers | 40% | 0% | looping fixed, clean runs |

Two takeaways so far. First, choosing the right layers to train (by gradient importance, not position) is what fixed the pathological looping. Second, a light fine-tune fixes behavior (formatting, loops) but has not yet raised the actual issue-resolution rate past the base model.

But here is the reason I think this is worth continuing rather than a dead end: **every signal points to under-training, not a hard ceiling.** The masked training loss keeps dropping (2.26 to about 1.0), and a preliminary run that trains all 36 layers is driving the loss lower still than the partial-layer runs at the same point. The model clearly has room to absorb more. These first runs were half an epoch on a small, noisy corpus of scraped agent logs. The obvious next steps, more epochs, the full network, and cleaner training data, are exactly the things that should move the needle. So I read this as: the 2-bit model is fine-tunable, we just have not fed it enough yet.

## Try it

Code and a reusable pipeline guide are in the repo (docs/ternary_qat.md): [your repo link]

If you have an Apple Silicon Mac with enough unified memory and you want a task-specialized 1.58-bit model, this is a working starting point. And if you have compute to throw a full-data run at it, I would love to see whether it breaks the resolution ceiling.

---

# X / Twitter thread

1/ You can fine-tune a native 1.58-bit (ternary) LLM on a Mac. No CUDA.

These are BitNet-style models (every weight is -1/0/+1), like prism-ml/Ternary-Bonsai-8B, an 8B that ships as a ~2GB GGUF. The question: can you specialize one on your own data?

2/ First gotcha: imatrix / AWQ / GPTQ do NOTHING to these models. They are already a lossless quant of themselves (KLD ~ 0). There is no rounding error to recover. The only lever is QAT.

3/ QAT = keep the ternarize step inside the forward pass (straight-through estimator, BitNet b1.58 / TWN), continue-train the fp16 latent weights, re-pack to 2-bit. About 20 lines. Runs on Metal.

4/ The MPS traps that cost me a week:
- foreach=False, or AdamW/clip deadlock at scale ("hangs at step 5")
- seq <= 4096 (8192 hits MPSGraph INT_MAX, no flash attn)
- fp32 latents (bf16 updates underflow the ternary threshold)
- Adafactor to fit all 36 layers (AdamW swaps); pick layers by gradient importance, not last-N

5/ Bonus: mask the loss to tool-call tokens and render the tool schemas into the chat template, so it learns schema-conditioned tool use.

6/ Preliminary results on agentic coding: the fine-tune fixed behavior (killed a nasty command-repeat loop) but has not beaten the base model on issue-resolution yet.

BUT the loss keeps dropping and an all-layers run drops it further. This reads as under-trained, not capped. More epochs + cleaner data next.

Code: [your repo link]
