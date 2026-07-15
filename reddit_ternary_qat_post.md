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
- **Optimizer sets how many layers fit.** AdamW keeps two fp32 states (~56GB extra) and swaps even on 128GB. Adafactor uses factored state and fits all 36 layers in ~70GB.
- **Pick which layers by gradient importance, not position.** On this model the early layers carry almost all the task gradient; the middle layers are nearly dead.
- **Mask the loss to tool-call tokens** and render tool schemas into the chat template, so you train schema-conditioned tool use, not boilerplate.
- **Label the stop token, or the model never learns to stop.** This one cost me the most. My masking labeled the assistant content but ended the span one token before the `<|im_end|>` terminator. Under the standard causal-LM label shift that means no position in the entire corpus ever had the stop token as a target, so the model got zero gradient toward ending its turn. That, not the training "teaching persistence," was the real cause of the pathological looping (a command repeated hundreds of times). If your fine-tuned model loops, check that your loss actually covers the end-of-turn token before you touch anything else.
- **Watch code flips, not just loss.** For a ternary net a weight only changes when its latent crosses the sign threshold. At a low LR the optimizer just nudges the per-group scales and never flips a single code, so the loss falls while the actual ternary weights barely move. Log how many codes flip per checkpoint. If it is near zero, your loss curve is lying to you and the LR is too low.

## Results (preliminary)

Task: make it a better agentic coder, measured on SWE-rebench (10 held-out real issues).

| run | patch rate | pass rate | training loss | notes |
| --- | --- | --- | --- | --- |
| base 8B (no fine-tune) | 50% | 0% | - | - |
| QAT, last 18 layers | 40% | 0% | ~1.0 | looped badly (one command repeated up to 553x) |
| QAT, gradient-influential layers | 40% | 0% | ~1.0 | looping fixed, clean runs |
| QAT, ALL 36 layers | 30% | 0% | 0.91 | best-behaved, worst patch rate |

Two caveats I found after the fact, and both point the same way. First, these runs trained on a corpus with the stop-token masking bug above, so some of the budget went to a broken target. Second, and more important: the all-layers run drove the training loss LOWER than any other and produced the WORST patch rate. The lowest loss gave the least capable agent. It was the tidiest (fewest steps, no loops, cleanest tool calls) and it solved the fewest issues.

That kills the "just under-trained" idea. The problem is what the loss measures. My corpus was scraped agent logs (Claude Code, Gemini CLI), which are imitation data, not verified successful solutions. So minimizing the loss teaches the model to mimic the STYLE of those logs (be terse, emit clean tool calls, stop early), not to SOLVE. Lower loss = better log-mimicry = a neater agent that does less. The metric and the goal were misaligned.

So the real lever is not more training or more layers, it is better DATA. That is what I am testing next.

## What is next (iter-5 and iter-6)

The plan is to replace the scraped logs with verified solutions, and to do it in the model's own vocabulary.

- **iter-5: distill from a strong solver.** I took a strong 9B agentic coder (a different model, 100% patch and 60% resolved on the held-out issues), let it solve a set of real GitHub issues, and kept ONLY the trajectories that actually passed the hidden tests. A patch that does not pass is the same mimicry trap, so it is filtered out. Those winning trajectories are re-rendered through the ternary model's tokenizer with the fixed masking, and it trains to imitate the SHAPE of a solution that works, not the style of a log. Crucially the issues are disjoint from the ones I grade on, so a gain is generalization, not memorization. This run is generating data now.
- **iter-6: distill the logits too.** The teacher above uses a different tokenizer, so I can only copy its behavior, not its probabilities. But the ternary model is a converted Qwen3-8B, and there are strong SWE fine-tunes of that exact base with the SAME vocabulary. Using one as a teacher lets me train on its full output distribution at every token (soft targets), which the low-bit-QAT literature consistently shows beats one-hot targets at 2 bits.

The honest takeaway so far: imitation-fine-tuning a 2-bit agent on scraped logs cleans up its behavior but does not add problem-solving ability. The loss went down; the thing I cared about did not. The pipeline works and the negative result is clear. The next attempt changes the data, not the training knobs.

## Try it

Code and a reusable pipeline guide (docs/ternary_qat.md): [your repo link]

Apple Silicon with enough unified memory is all you need. The open question is not whether a 2-bit model can be trained, it clearly can, but whether verified-solution data can push it past its base. That is what iter-5 and iter-6 are testing, and I will post the numbers.

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

5/ The trap that cost the most: if your fine-tune loops, check that your loss labels the stop token. Mine ended the masked span one token short of it, so the model got zero gradient toward ending its turn. That was the looping, not "training taught persistence."

6/ Also watch code flips, not just loss. A ternary weight only changes when its latent crosses the sign threshold. At a low LR the optimizer just drifts the scales and flips nothing, so loss falls while the weights barely move.

7/ Real result: driving the loss lower made the agent tidier but LOWERED its patch rate. Turns out the loss was rewarding log-mimicry, not problem-solving. My training data was scraped agent logs, not verified solutions. The metric and the goal were misaligned.

8/ So the fix is the data, not the knobs. Next up: distill a strong solver's VERIFIED winning trajectories (tests actually pass) into the ternary model, then add soft-label distillation from a same-vocab SWE fine-tune of its parent. Results soon.

Code: [your repo link]
