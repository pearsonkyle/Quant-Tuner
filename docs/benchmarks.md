# Benchmarks

`quant-tuner` evaluates a quantized GGUF along three independent axes:
**fidelity** (does it produce the same outputs as the F16 reference?), **size**
(how compact is it on disk?), and **speed** (how fast is inference?).

The bench runner (`quant_tuner.bench.runner.bench_one`) produces one CSV row
per model with all of the metrics below.

## Metrics

### Fidelity — from `llama-perplexity --kl-divergence`

These compare the quantized model against an F16 reference on a held-out
text corpus. The reference is captured once via
`kld.build_baseline(reference_gguf, eval_dataset, baseline.kld)`; every
subsequent evaluation reuses that baseline.

| Column          | What it measures                                                                       |
| --------------- | -------------------------------------------------------------------------------------- |
| `ppl`           | Mean per-token perplexity of the **quantized** model on the eval set. Lower is better. |
| `ppl_ratio`     | `ppl(Q) / ppl(F16)`. 1.000 means quant matches reference exactly. Closer to 1 is better. |
| `mean_kld`      | Mean KL-divergence per token between Q's and F16's next-token distributions. Lower is better. |
| `median_kld`    | Median per-token KLD. Outlier-robust; dominates the mean when a quant has a few catastrophically off tokens. |
| `same_top_p`    | Percentage of tokens where Q and F16 agree on the top-1 prediction. Closer to 100% is better. |
| `rms_dp`        | RMS of the per-token difference in top-1 probability. Lower is better.                 |

KLD is the most informative of these for ranking quants — PPL ratio is
coarse-grained because PPL itself is an exponential of cross-entropy, so
small differences hide large ones. Mean KLD splits hairs that PPL ratio
can't.

### Size — from GGUF on-disk inspection

| Column     | Source                                              |
| ---------- | --------------------------------------------------- |
| `size_gib` | Filesystem byte size / 1024³                        |
| `bpw`      | `size_bytes · 8 / n_params` — effective bits/weight |

`n_params` is computed once per F16 reference (`bench.bpw.n_params(f16_gguf)`)
and reused; the quantized files don't store a separate count.

### Speed — from `llama-bench -o json`

| Column           | What it measures                                              |
| ---------------- | ------------------------------------------------------------- |
| `prefill_tok_s`  | Avg tok/s during a 2048-token prompt eval (no generation).    |
| `decode_tok_s`   | Avg tok/s during 128-token decode (no prompt).                |
| `ttft_2k_ms`     | Derived: `2048 / prefill_tok_s · 1000`. Time-to-first-token for a 2k-prompt request. |

Decode speed is the one users feel during a chat session; prefill matters for
agent workflows that send long contexts. Heavy K-quants (Q5_K_M, Q6_K) tend
to have lower decode tok/s than the F16 baseline on CPU/Metal because the
dequantize cost per matmul exceeds the I/O savings.

## Bench suites

`bench_one(quant, label, ..., suite=...)` selects what to compute:

| Suite          | Computes                       | Needs              |
| -------------- | ------------------------------ | ------------------ |
| `quick`        | size + BPW only                | nothing            |
| `kld`          | size + BPW + KLD               | eval_dataset, eval_baseline |
| `speed`        | size + BPW + llama-bench       | nothing            |
| `full`         | all of the above               | eval_dataset, eval_baseline |
| `leaderboard`  | alias for `full`               | "                  |

A `quick` row in CI/dev is enough to catch obvious regressions before paying
for the full KLD pass (which runs llama-perplexity on the full eval set and
takes minutes on a 1B model, tens of minutes on a 9B).

## Train / test / holdout splits

`quant_tuner.data.split.split_sessions(...)` partitions log sessions by
fingerprint into three disjoint sets:

* **train** — used by the calibrators (imatrix forward pass, AWQ activation
  capture, GPTQ Hessian accumulation). Default 80%.
* **test** — used to build the KLD eval dataset that scores every quant.
  Default 10%.
* **holdout** — reserved for end-to-end checks like the tool-call benchmark
  or human eval. Never touched by calibration or KLD scoring. Default 10%.

The split is deterministic given a `seed`. Splits are session-level (not
token-level), so a single session never appears in more than one split — this
prevents contamination from large multi-turn sessions whose tokens would
otherwise leak across the train/eval boundary.

## Comparing across models

KLD is computed against a *specific* reference GGUF (typically F16 of the
same model). To compare `quant-tuner` results across two different base
models, you'd need either:

1. A common eval set + a common reference (e.g. both models scored against
   a third "judge" model's distribution), or
2. Comparing only `same_top_p` / `rms_dp` numerically, since those depend
   only on the relative shape of each model's output distribution.

In practice for tuning one model, sticking with the per-model F16 reference
is what you want — every row's KLD is "how much fidelity did this quant
lose relative to my own F16?", which is the directly-actionable number.

## SQS — the leaderboard summary scalar

(Coming in Phase 5.) The leaderboard collapses all of the above into one
sortable number `SQS` (Squashed Quality Score) using weights `α, β, γ` over
KLD, top-1 agreement, and speed retention vs F16. Until the aggregator
lands, treat `mean_kld` as the primary ranking column and `decode_tok_s` as
the speed tie-breaker.
