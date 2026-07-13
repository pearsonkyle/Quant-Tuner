# Ornith-9B vs Qwythos-9B-v2 — 2-bit agentic quant comparison

Two Qwen3.5-VL 9B agentic coders, same quant ladder, same SWE-rebench setup
(10 held-out issues, 1 rep, gold tests decide pass; reasoning-off,
repeat-penalty 1.1, T=0.25). Static metrics vs FP16 on the general eval. ± is
the binomial spread over the 10 issues.

## Head-to-head (the agentic bottom line)

| SWE-rebench | Ornith IQ2_M | Qwythos IQ2_M | Ornith IQ4_XS | Qwythos IQ4_XS | Ornith Q5_K_M | Qwythos Q5_K_M |
|:-|-:|-:|-:|-:|-:|-:|
| **Pass** | 10±9% | 0±0% | **60±15%** | 50±16% | **60±15%** | 40±15% |
| **Patch** | **60%** | 40% | 100% | 100% | 100% | 100% |

**Ornith wins at every bit-width.** Both hit 100% patch at IQ4_XS+ (the loop is
fixed in both base models), but Ornith resolves more real issues (60% vs 40–50%)
and degrades far more gracefully at 2-bit.

## Ornith-1.0-9B

| Metric | FP16 *(ref)* | Q5_K_M | **IQ4_XS** | IQ2_M |
|:---|---:|---:|---:|---:|
| Method | — | imatrix | imatrix | imatrix † |
| BPW | 16.0 | 5.78 | 4.64 | 3.22 |
| Size (GiB) | 17.9 | 6.0 | 4.8 | 3.4 |
| 🤖 Pass Rate | — | **60±15%** | **60±15%** | 10±9% |
| 🤖 Patch Rate | — | 100% | 100% | 60±15% |
| 🤖 Tool Errors | — | 12.7% | 11.3% | 19.3% |
| 🤖 Mean Tokens | — | 1017K | 997K | 2501K |
| 📐 PPL | 5.887 | 5.976 | 5.944 | 6.373 |
| 📐 KLD (med) | 0.000 | 0.0021 | 0.0096 | 0.1115 |
| 📐 same_top_p | 100.0% | 96.4% | 93.1% | 80.6% |

## Qwythos-9B-v2

| Metric | FP16 *(ref)* | Q5_K_M | **IQ4_XS** | IQ2_M |
|:---|---:|---:|---:|---:|
| Method | — | imatrix | imatrix | imatrix † |
| BPW | 16.0 | 5.78 | 4.64 | 3.22 |
| Size (GiB) | 17.9 | 6.0 | 4.8 | 3.4 |
| 🤖 Pass Rate | — | 40±15% | **50±16%** | 0±0% |
| 🤖 Patch Rate | — | 100% | 100% | 40±15% |
| 🤖 Tool Errors | — | 14.9% | 12.5% | 0.0% ‡ |
| 🤖 Mean Tokens | — | 858K | 1291K | 11K ‡ |
| 📐 PPL | ~5.88 | 5.897 | 5.888 | 7.226 |
| 📐 KLD (med) | 0.000 | 0.0018 | 0.0097 | 0.2290 |
| 📐 same_top_p | 100.0% | 96.6% | 93.3% | 73.9% |

† **AWQ tested at IQ2_M for both — no SWE-rebench benefit.** Ornith AWQ = 60%
patch / 0% pass (imatrix 60% / 10%); Qwythos AWQ = 40% / 0% (imatrix identical).
AWQ improved tool-argument *fidelity* on Ornith earlier, but that doesn't move
issue-resolution here — the ceiling is the model, not the calibration.
‡ Qwythos IQ2_M's tiny mean-tokens (11K vs 858K–2.5M) and 0% tool-errors are the
tell: episodes terminate almost instantly with garbled non-tool output — it
never engages the agent loop. Ornith IQ2_M still works through episodes (2.5M
tokens, 19% tool-errors) and even resolves one.

## Takeaways

1. **Ornith is the better agentic 9B at 2-bit** — its IQ2_M still functions
   (60% patch, 10% pass) where Qwythos IQ2_M is effectively dead (40% patch,
   0% pass, garbled output). Static KLD agrees: Ornith 2-bit 0.11 vs Qwythos 0.23.
2. **Both are solid at IQ4_XS+** — 100% patch, and this is the min-viable tier
   for agents on either model. Ornith edges Qwythos on resolve rate (60% vs 40–50%).
3. **The loop is a base-model property, and both fixed it** — no 32K rambles at
   any bit-width (unlike earlier runs); repetition penalty wasn't the bottleneck.
4. **AWQ is not a 2-bit rescue for issue-resolution** — it lifts tool-call
   fidelity but not SWE-rebench pass/patch; the size/precision floor dominates.

*SWE-rebench: nebius/SWE-rebench, 10 is-lite issues, OpenAI Agents SDK over a
local llama-server, real FAIL_TO_PASS grading. 1 rep (n=10).*
