# Making 2-bit GGUFs usable for agents (AWQ vs imatrix, and why KLD lied to me)

**TL;DR:** 2-bit quants of agentic coders are usually junk, but it's the *calibration*, not the bit width. I found that AWQ-calibrated 2-bit tool-calls a lot better than plain imatrix, even though KLD/PPL say imatrix is the better build. Reshipped two models with AWQ 2-bit: a **10GB gemma-4-31B** and a **3.4GB Ornith-9B**. Links below.

**Grab them:**

```
ollama run hf.co/pearsonkyle/gemma4-31b-imatrix-mtp-GGUF:IQ2_M      # 10.2 GiB, gemma-4-31B
ollama run hf.co/pearsonkyle/Ornith-1.0-9B-imatrix-GGUF:IQ2_M       # 3.4 GiB, Ornith-9B
```

Both are plain GGUF (stock llama.cpp / Ollama / LM Studio), vision mmproj included, bigger IQ4/Q5 builds in the repos if you have the VRAM.

## the 3 ways to build a 2-bit

- **Q2_K, no calibration:** round everything uniformly. The naive baseline.
- **imatrix:** measure which channels matter on real data, tell the quantizer where to spend its bits. IQ2 formats *require* this to run at all.
- **AWQ:** also activation-aware, but it *rescales* channels before rounding and folds the inverse into the norms (identity in fp16, no runtime cost). Different lever: imatrix picks what to protect, AWQ reshapes the weights so the important stuff survives.

I calibrate on real agentic-coding logs (Claude Code / Qwen Code / opencode), not wikitext.

## how they actually tool-call

Static metrics (PPL / KLD / top-token match) measure average closeness to fp16. They do **not** measure whether the model still fills correct tool arguments, so I also replay 25 held-out tool-use sessions and score, per turn, whether it picks the right tool, fills the right arguments, and emits a valid call. 3 reps each.

**gemma-4-31B, ~2-bit**

| Metric | Q2_K | IQ2_M (imatrix) | **IQ2_M (AWQ)** |
|:-|:-|:-|:-|
| File | control | control | [IQ2_M.gguf](https://huggingface.co/pearsonkyle/gemma4-31b-imatrix-mtp-GGUF/resolve/main/gemma-4-31B-it-IQ2_M.gguf) |
| Quality | ❌ | ⭐⭐ | ⭐⭐⭐ |
| Method | none | imatrix | AWQ + imatrix |
| Size (GiB) | 11.10 | 10.17 | 10.17 |
| median KLD (vs fp16) | 5.21 | **1.57** | 1.80 |
| top-token match rate | 25.4% | **46.6%** | 43.9% |
| | | |
| correct tool selection rate | 0.00 | 0.454 | **0.492** |
| correct argument rate | 0.00 | 0.171 | **0.263** |
| valid tool-call rate | 0.00 | 0.805 | **0.823** |
| | | |
| SWE-rebench pass rate | 0%† | 40% | **40% ± 15%** |
| SWE-rebench patch rate | 0%† | 100% | **100%** |

*SWE-rebench = does the agent's patch actually resolve a real GitHub issue (10 held-out issues, gold tests). imatrix n=30 (3 reps); AWQ n=10 (1 rep, ± = binomial spread). †Q2_K not run — it emits 0 valid tool calls, so it can't drive the agent.*

**Ornith-9B, ~2-bit**

| Metric | Q2_K | IQ2_M (imatrix) | **IQ2_M (AWQ)** |
|:-|:-|:-|:-|
| File | control | control | [IQ2_M.gguf](https://huggingface.co/pearsonkyle/Ornith-1.0-9B-imatrix-GGUF/resolve/main/Ornith-1.0-9B-IQ2_M.gguf) |
| Quality | ❌ | ⭐⭐ | ⭐⭐⭐ |
| Method | none | imatrix | AWQ + imatrix |
| Size (GiB) | 3.56 | 3.36 | 3.36 |
| median KLD (vs fp16) | 2.03 | **0.11** | 0.12 |
| top-token match rate | 37.9% | **80.6%** | 79.7% |
| | | |
| correct tool selection rate | 0.026 | 0.306 | **0.536** |
| correct argument rate | 0.000 | 0.054 | **0.333** |
| valid tool-call rate | 0.026 | 0.851 | **0.930** |
| | | |
| SWE-rebench pass rate | 0%† | — | **0% ± 0%** |
| SWE-rebench patch rate | 0%† | — | **60% ± 15%** |

*Ornith AWQ makes plausible patches 60% of the time but resolves 0/10 real issues — a 9B is too weak to actually solve them at 2-bit, even though it tool-calls well. (imatrix SWE-rebench not run; †Q2_K can't tool-call.)*

Two takeaways:

1. **No calibration = dead.** The Q2_K builds are *bigger on disk* than the shipped IQ2_M and still useless. gemma's scores a flat 0.00 on every agentic metric (never emitted one valid tool call in 3 reps); the 9B gets 0% of arguments right. This is why IQ2 refuses to run without an imatrix.
2. **KLD picks the wrong build.** On both models imatrix has the better median KLD and top_p, yet AWQ fills correct tool arguments a lot more often (0.17 to 0.26 on gemma, 0.05 to 0.33 on the 9B) and is far more consistent run to run. If I'd trusted the static table I'd have shipped the worse one.

So: at 2-bit, benchmark the task you care about. Averaged-over-a-corpus metrics can't see the rare tokens that actually decide a tool call.

## Quick start

```
ollama run hf.co/pearsonkyle/gemma4-31b-imatrix-mtp-GGUF:IQ2_M
```

GPU + vision:

```
llama-server --model gemma-4-31B-it-IQ2_M.gguf \
    --mmproj mmproj-gemma-4-31B-it-Q8_0.gguf \
    --flash-attn on --n-gpu-layers 999
```

**Caveats:**

- Sub-3-bpw is for when VRAM is the constraint, not a replacement for Q4+ when you can afford it.
- Calibration was English + coding heavy; expect weaker fidelity on other languages / non-coding work.
- AWQ's edge showed up at 2-bit; at 4-bit+ plain imatrix is fine, so those ship imatrix.

📎 [gemma repo](https://huggingface.co/pearsonkyle/gemma4-31b-imatrix-mtp-GGUF) · [Ornith repo](https://huggingface.co/pearsonkyle/Ornith-1.0-9B-imatrix-GGUF) · [Quant-Tuner](https://github.com/pearsonkyle/Quant-Tuner) · [Log Miner](https://github.com/pearsonkyle/logminer) · [Calibration Data](https://huggingface.co/pearsonkyle/gemma4-31b-imatrix-mtp-GGUF/resolve/main/calibration_data/corpus.cal.txt)
