# Calibration methods

All three methods produce a standard GGUF that runs on unmodified `llama.cpp`.
None changes the inference graph — they differ only in *how* they choose
which weights get more of the quantizer's precision budget.

## What "calibration" means in this context

When you quantize an FP16 weight matrix `W` to (say) Q4_K_M, llama.cpp splits
each row into 32-element blocks and picks one INT4 scale per block. That scale
must cover the largest |w| in the block — so a single outlier weight forces
neighbours into a coarse grid.

Calibration uses your real-data activations to decide *which* weights matter.
The three methods do this differently:

* **imatrix** tells the quantizer "channel `c` is important — keep its
  precision". The weights are unchanged; only the per-block grouping decision
  shifts.
* **AWQ** physically scales channels of `W` so that important columns get
  smaller magnitudes (and thus finer relative precision). The math is folded
  into the preceding RMSNorm so the F16 forward pass is unchanged.
* **GPTQ** rounds `W` itself toward an INT-N grid, while *compensating* for
  each column's rounding error in the columns that come after it. The result
  is then converted back to F16 (where it sits very close to the grid) and
  re-quantized by llama.cpp.

## imatrix (output-aware)

**Entry point:** `quant_tuner.calibrate.imatrix.calibrate(variant=..., ...)`

llama.cpp's standard imatrix file stores `E[a_c²]` — the mean squared
activation per input channel — and the quantizer uses it as a per-channel
priority weighting. `quant-tuner` produces five variants that combine that
signal with other priors:

| Variant         | Formula                                    | Needs forward pass? |
| --------------- | ------------------------------------------ | ------------------- |
| `analytic`      | `‖W[:,c]‖² · E[a_c²]`                     | No                  |
| `mix_50`        | √(L1-norm(`E[a²]`) · L1-norm(analytic))    | No                  |
| `hybrid_custom` | max(L1-norm(`E[a²]`), L1-norm(analytic))   | No                  |
| `outlier_l4`    | √`E[a_c⁴]` — L4 norm of channel activation | Yes (HF model)      |
| `outlier_max`   | `E[a_c²] · max|a_c|²`                      | Yes (HF model)      |

The `analytic` family combines the weight column norm with the activation
energy — channel `c`'s expected contribution to `‖y‖²` for `y = W a`. It needs
no model load; given an existing imatrix from `llama-imatrix` and the F16
GGUF, the calculation is closed-form and runs in seconds.

The `outlier_*` variants require an actual forward pass to capture
heavy-tail statistics (`E[a⁴]`, `max|a|`). They write a `ForwardStats` npz on
first run and reuse it on subsequent calls.

**SSM tensors are always passed through unchanged.** Mamba-style state-space
blocks (`blk.*.ssm_*`) don't have the `y = W a` linear-projection structure,
so the output-aware re-ranking is invalid for them. They keep their raw
`E[a²]` weighting.

## AWQ (activation-aware weight scaling)

**Entry points:** `quant_tuner.calibrate.awq.calibrate(...)`, `awq.apply(...)`

For each **scale group** (attention `q/k/v[/gate_proj]` sharing one
`input_layernorm`; MLP `gate_proj/up_proj` sharing
`post_attention_layernorm`), AWQ:

1. Captures `s_c = mean(|x_c|)` per input channel via forward pre-hooks on the
   group anchor.
2. Picks an exponent `α` (either fixed `0.5` per the AWQ paper, or grid-searched)
   that minimises a proxy loss using fake INT4 g128 round-to-nearest.
3. Sets `scale_c = s_c^α / geomean(s^α)` (normalized so the per-row quantizer
   scale isn't dominated by any one channel).

In `apply`, the scales are folded into the model in F32:

```
W'[:, c] = W[:, c] · scale_c                   for every member weight
γ'_c     = (1 + γ_c) / scale_c - 1             Qwen3.5 RMSNorm (plus_one=True)
γ'_c     = γ_c / scale_c                       Llama / Mistral / Qwen3
```

The F16 forward pass is mathematically unchanged: multiplying input channels
of `W` by `s` and pre-dividing the RMSNorm gain by `s` cancel exactly.
Only the *quantized* forward differs — and the channels AWQ has emphasized
land in finer per-block scales.

A sanity check after application rejects the run if F16 logits drift by more
than 3% (the bf16 noise floor sits at ~2%); a larger drift means the RMSNorm
fold has the wrong form (`plus_one` flag for Qwen3.5 vs everything else).

The leaderboard variants `awq_a050` (force `α=0.5`) and `awq_best`
(grid-searched) are reproduced by setting `force_alpha=0.5` vs leaving it
`None`.

## GPTQ (Hessian-based rounding with error compensation)

**Entry points:** `quant_tuner.calibrate.gptq.calibrate(...)`,
`gptq.apply(...)`, `gptq.verify_perplexity(...)`

GPTQ is structurally different from the other two: it actually **rewrites**
the weights, replacing each column with the nearest INT-N value while
compensating for that error in the columns that haven't been quantized yet.

1. **Calibrate.** For each target Linear (attention `q/k/v/o_proj`, MLP
   `gate/up/down_proj`), accumulate `H = Σ_t x_t x_t^T` over the calibration
   corpus. One file per tensor is snapshot-saved to disk every N chunks (atomic
   tmp + rename), so a crash mid-calibration is resumable. Memory: this stores
   one `[in × in]` FP32 Hessian per Linear — `down_proj` Hessians dominate at
   roughly `in_size² · 4 B`.

2. **Apply.** For each tensor, run the canonical Frantar 2023 algorithm:
   1. Damp `H` by `dampen · mean(diag(H))` (Tikhonov regularisation).
   2. Permute columns by descending `diag(H)` (act-order): round the
      high-importance columns first so the low-importance tail has more
      compensation budget left.
   3. Compute the upper-Cholesky `U` of `H⁻¹`. `U[j,j]` is the per-column
      step size that scales each column's residual before it spreads onward.
   4. Sweep columns left-to-right in groups of `group_size` (default 32, to
      match Q4_K_M's block structure). Inside each group: pick a symmetric
      INT-N scale per row, round each column, propagate the residual to
      later columns within the group and across to later groups.

3. **Save.** Write the rounded weights back into a new HF checkpoint. After
   HF → F16 GGUF conversion, the rounded values land close to llama.cpp's
   own K-quant grid; running `llama-quantize` then re-snaps them slightly,
   but the *inter-column* error compensation (which lives in the values of
   adjacent columns, not in the rounding of one column in isolation) survives.

**Two sanity gates** catch the failure mode where GPTQ produces unusable
weights (the source experiment's `gptq__IQ3_S` row hit PPL=3.6M):

* `apply()` runs a forward pass and raises if logit drift exceeds 50% by default.
* `verify_perplexity(f16_gptq, eval_ds, reference_ppl, max_ratio=1.5)` runs
  `llama-perplexity` on the F16 GGUF *before* calling `llama-quantize` and
  raises if PPL is more than 1.5× the baseline. Recommended before any final
  quantize.

GPTQ shines at higher bit budgets (Q4_K_M, Q5_K_M, IQ4_XS); at IQ3 and below
the rounding error per column outpaces the compensation capacity and the
result is unstable.

## Choosing a method

* **Start with imatrix.** It's the cheapest, requires no extra disk space, and
  often captures most of the recoverable gap on its own. Try
  `variant="hybrid_custom"` first.
* **AWQ adds gains when activation distributions are heavy-tailed** — coding
  models, instruction-tuned chat models, etc. Use `force_alpha=0.5` for the
  AWQ paper default; switch to grid search if you're chasing the last 1-2%.
* **GPTQ is the strongest method but the slowest and the most fragile.**
  Pair it with `verify_perplexity` and don't run it at low bit budgets without
  inspecting the result.

## Switching the target quant type

```python
gguf.quantize(f16, out, quant_type="Q4_K_M", imatrix=...)
gguf.quantize(f16, out, quant_type="IQ4_XS", imatrix=...)
gguf.quantize(f16, out, quant_type="Q5_K_M")
```

The `quant_type` string is passed through to `llama-quantize` unchanged.
Anything `llama-quantize --help` lists works; common ones:

| Tag       | Bits/weight | Notes                                                  |
| --------- | ----------: | ------------------------------------------------------ |
| `Q5_K_M`  | ~5.5        | Smallest quality loss; usually within 0.5% of F16 PPL  |
| `Q4_K_M`  | ~4.5        | Standard "good" target — the focus of the recipes     |
| `IQ4_XS`  | ~4.25       | Smaller than Q4_K_M with similar quality; slower CPU prefill |
| `IQ3_S`   | ~3.5        | Visible quality drop; calibration matters most here   |
| `IQ2_M`   | ~2.7        | Aggressive; only viable with strong calibration        |
