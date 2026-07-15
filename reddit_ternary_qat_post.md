# Reddit post

**Title: You can fine-tune a native ternary (sub-2-bit) LLM on a Mac. Here is how, plus the Metal traps that cost me a week.**

## What these models are

A new family of true sub-2-bit LLMs is out: PrismML's Bonsai line. Every weight is ternary (-1/0/+1) or binary (-1/+1), with one shared fp16 scale per group of 128 weights. Ternary lands at a true ~1.7 bits/weight, binary at ~1.1. That is genuinely sub-2-bit, unlike conventional "2-bit" GGUF quants, which are really ~2.8 bits once you average in the higher-precision tensors they keep.

Two things worth knowing:

- **They are not made the BitNet way.** BitNet pretrains a 1.58-bit model from scratch. Bonsai instead converts an existing pretrained model (a Qwen3) into ternary with a proprietary method, so you keep the model you already wanted. Their new flagship is a 27B; the release I used is an earlier 8B (Ternary-Bonsai-8B, a ternary Qwen3-8B). PrismML themselves note those earlier 1.7B-8B releases did not target reasoning or reliable tool use, which matters for my results below.
- **They run at ~2 GB for an 8B**, but the format is new, so you need PrismML's llama.cpp fork to run it (upstream PR pending).

The question I cared about: can you specialize one on your own data. Here is what I found.

## Post-hoc quant calibration does nothing here

imatrix, AWQ, and GPTQ exist to recover "quantization rounding error," the gap between an fp16 original and a low-bit grid. A native-ternary model has no such gap: its fp16 file is already exactly `scale * {-1,0,+1}`, so the 2-bit GGUF is a lossless re-encode (I measured KLD ~ 0.000 between them). I ran my own GPTQ on it: rounding the weights to the ternary grid changes them by 0. These tools are no-ops on this class of model.

## How to fine-tune it (QAT)

Bonsai's *creation* is proprietary, but *continuing to train* a released ternary model is standard QAT with a straight-through estimator, and that part is public and about 20 lines:

1. Load the unpacked checkpoint (fp16 weights = the ternary values in a fat container).
2. Forward pass: ternarize each weight on the fly (per 128-weight group). The model computes with ternary weights.
3. Backward pass: straight-through estimator, so gradients flow to the fp16 latent weights.
4. Train. Latents drift, re-ternarize each step, weights flip sign when a latent crosses the threshold.
5. Re-ternarize and re-pack to the 2-bit GGUF.

Init from the shipped weights reproduces the model exactly at step 0, so you fine-tune the real thing. All on Metal, no CUDA.

## The Metal / MPS traps (the useful part)

- **foreach=False is mandatory.** MPS multi-tensor (foreach) optimizer/clip kernels deadlock at full-model scale. Symptom: "training hangs at step 5." Pass it to AdamW and clip_grad_norm_.
- **Sequence length <= 4096.** seq 8192 throws "MPSGraph tensor dims larger than INT_MAX" (no flash attention on MPS).
- **fp32 latents, not bf16.** bf16 either blows up or the tiny updates underflow the ternary threshold, so no weight ever flips.
- **Optimizer sets how many layers fit.** AdamW keeps two fp32 states (~56GB extra) and swaps even on 128GB. Adafactor uses factored state and fits all 36 layers in ~33GB.
- **Pick which layers by gradient importance, not position.** On this model the early layers carry almost all the task gradient; the middle layers are nearly dead. Training the wrong half made it loop pathologically.
- **Mask the loss to tool-call tokens** and render tool schemas into the chat template, so you train schema-conditioned tool use, not boilerplate.

## Results (preliminary)

Task: make it a better agentic coder, measured on SWE-rebench (10 held-out real issues).

| run | patch rate | pass rate | training loss | notes |
| --- | --- | --- | --- | --- |
| base 8B (no fine-tune) | 50% | 0% | - | - |
| QAT, last 18 layers | 40% | 0% | ~1.0 | looped badly (one command repeated up to 553x) |
| QAT, gradient-influential layers | 40% | 0% | ~1.0 | looping fixed, clean runs |
| QAT, ALL 36 layers | 30% | 0% | 0.91 | best-behaved, worst patch rate |

Here is the twist, and it is the most useful thing I learned. The all-layers run drove the training loss LOWER than any other, and it produced the WORST patch rate. The lowest loss gave the least capable agent. It was the tidiest (fewest steps, no loops, cleanest tool calls) and it solved the fewest issues.

That kills the "just under-trained" idea. The problem is what the loss measures. My corpus is scraped agent logs (Claude Code, Gemini CLI), which are imitation data, not verified successful solutions. So minimizing the loss teaches the model to mimic the STYLE of those logs (be terse, emit clean tool calls, stop early), not to SOLVE. Lower loss = better log-mimicry = a neater agent that does less. The metric and the goal were misaligned.

So the real lever is not more training or more layers, it is better DATA:
- Distill from a strong solver: generate verified successful trajectories with a capable model and train on those (outcome data, not log-style data).
- Or reward actual test-pass with RL, instead of next-token imitation.
- And start from a stronger base. PrismML's own paper says this 8B was never built for reasoning or reliable tool use; their new capable model is a 27B.

The honest takeaway: imitation-fine-tuning a 2-bit agent on scraped logs cleans up its behavior but does not add problem-solving ability. The loss went down; the thing I cared about did not. The pipeline works, the negative result is clear, and the next attempt should change the data, not the training knobs.

## Try it

Code and a reusable pipeline guide (docs/ternary_qat.md): [your repo link]

Apple Silicon with enough unified memory is all you need. If you have compute for a full-data run, I would like to know whether it breaks the resolution ceiling.

---

# X / Twitter thread

1/ You can fine-tune a native ternary (sub-2-bit) LLM on a Mac. No CUDA.

PrismML's Bonsai models store every weight as -1/0/+1 (true ~1.7 bits/weight). An 8B ships as ~2GB. Made by converting a pretrained Qwen3 to ternary, not pretraining from scratch like BitNet.

2/ First gotcha: imatrix / AWQ / GPTQ do NOTHING to these. They are already a lossless quant of themselves (KLD ~ 0). No rounding error to recover. The only lever is continued QAT.

3/ QAT = ternarize inside the forward pass, straight-through gradient to the fp16 latent weights, re-pack to 2-bit. ~20 lines, runs on Metal.

4/ The MPS traps that cost me a week:
- foreach=False or AdamW/clip deadlock ("hangs at step 5")
- seq <= 4096 (8192 hits MPSGraph INT_MAX, no flash attn)
- fp32 latents (bf16 updates underflow the ternary threshold)
- Adafactor to fit all 36 layers; pick layers by gradient importance
- mask loss to tool-call tokens + render tool schemas

5/ Preliminary agentic-coding result: the fine-tune fixed behavior (killed a bad command-repeat loop) but has not beaten the base yet.

Not a wall though: this 8B was never built for tool use (PrismML says so), and the loss keeps dropping. Under-trained, not capped.

Code: [your repo link]
