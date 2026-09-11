# Recipe reference

A recipe is a YAML file under `src/quant_tuner/recipes/` that declares one
**method × quant type** combination. The CLI resolves a bare name
(`--recipe iq2m_imatrix_32k`) against that directory; a real path works too.

```bash
# resolve + validate (no model load)
uv run quant-tuner run --recipe iq2m_imatrix_32k --model X --logs Y --workspace W --dry-run
```

## The 32K-ctx recipes (default for new models)

| Recipe | Method | Quant | ctx | Notes |
|---|---|---|---|---|
| `iq2m_imatrix_32k` | imatrix / `hybrid_custom` | IQ2_M | 32 768 | the 2-bit GGUF; requires an imatrix (llama-quantize refuses IQ2 without one) |
| `iq2m_gptq_32k` | gptq | IQ2_M | 32 768 | Hessian calibration + PPL guardrail (auto-relaxed to 4.0 at 2-bit) |
| `iq3m_imatrix_32k` | imatrix / `hybrid_custom` | IQ3_M | 32 768 | the 3-bit GGUF |
| `iq3m_gptq_32k` | gptq | IQ3_M | 32 768 | guardrail auto-relaxed to 2.0 at 3-bit |
| `iq4xs_imatrix_32k` | imatrix / `hybrid_custom` | IQ4_XS | 32 768 | the 4-bit GGUF |
| `iq4xs_gptq_32k` | gptq | IQ4_XS | 32 768 | default guardrail (1.5) |
| `q5km_imatrix_32k` | imatrix / `hybrid_custom` | Q5_K_M | 32 768 | the 5-bit GGUF (the "I have the VRAM" rung) |
| `q5km_gptq_32k` | gptq | Q5_K_M | 32 768 | default guardrail (1.5) |

All eight set `data.context_len: 32768` and `calibration.params.imatrix_ctx:
32768` (or `tokens: 32768, ctx: 32768` for GPTQ). The calibration corpus is
expected to be pre-packed at 32K windows — see
[`calibration_datasets.md`](calibration_datasets.md).

## The 4-bit baselines

`q4_k_m_{imatrix,awq,gptq,none}` — the original OmniCoder-9B study. The `none`
variant is the uncalibrated reference (the KLD you're trying to beat).

## Low-bit AWQ (2–3 bpw)

`q2_k_awq`, `iq2_xs_awq`, `iq2_m_awq` — AWQ + a folded-F16 imatrix re-weighted
with `hybrid_custom`. The α-search proxy **auto-matches `quantize.type`**
(`proxy_for_quant_type`): codebook-aware `iq2_xxs`/`iq2_xs`/`iq2_s` for
IQ2_* targets, `q2k_b16` for Q2_K/IQ1, `q3k_b16` for 3-bit, else `int4_g128`.
Recipes can pin one via `params.proxy`.

The IQ2 proxies snap groups of 8 weights to llama.cpp's **exact E8-lattice
codebooks** (256/512/1024 entries for XXS/XS/S+M) with the real scale form
`db = d·(0.5+q4)·0.25` and the even-negatives-per-group sign parity for
XXS/XS. The grids live in the generated `calibrate/_iq2_grids.py` — do not
edit; regenerate with `scripts/gen_iq2_grids.py` when bumping the llama.cpp
pin.

Low-bit targets (IQ1/IQ2/IQ3/Q2_K/Q3_K) additionally get a **per-member proxy
mix** (`params.proxy_mix`, pipeline-defaulted to `quantize.type`):
llama-quantize bumps members above the ftype's base grid — at 2 bits
attn_v → Q4_K (GQA/MoE ≥ 4), attn_output → IQ3_S (IQ2_S/M) / Q3_K (Q2_K),
ffn_down a tier up for the first eighth of layers (every layer for Q2_K); under
Q3_K_M/L attn_v/attn_output/ffn_down all land on Q4_K–Q5_K. `proxy_for_member`
scores those members with the proxy matching their *real* target. Pinning
`params.proxy` disables both the auto-selection and the mix.

`iq2_m_awq` pins `proxy: q2k_b16`: pure-`iq2_s` scoring regressed IQ2_M
top_p — the codebook's steep α penalty plus v_proj's fictitious 2-bit error
(really Q4_K) dragged the shared group α down.

## GPTQ grid mix

GPTQ's rounding grid **auto-matches `quantize.type`**
(`grid_for_quant_type`): 2-3-bit targets → **asymmetric min+scale** per-16
block (all `2^n` levels; a symmetric 2-bit grid has only 3 usable levels and
destroys the model), Q6_K → sym g16 6-bit, Q5_K → sym g32 5-bit, else symmetric
g32.

The **per-tensor grid mix** (`grid_for_member`, `params.grid_mix`
pipeline-defaulted to `quantize.type`): mixed ftypes bump members above the
base grid — at 2 bits attn_v → Q4_K (GQA/MoE ≥ 4), Q4_K_M attn_v/ffn_down →
Q6_K (`use_more_bits` schedule), IQ4_XS attn_v → Q5_K under GQA — so each
tensor is rounded on the grid llama-quantize actually stores it at. The
tensor→target table lives in `calibrate/_quant_mix.target_type_for_member` and
is **shared with AWQ's** `proxy_for_member`. Pinning `n_bits`/`group_size`/`sym`
opts out of the mix.

PPL/logit guardrails are auto-relaxed at 2-3 bits (`ppl_max_ratio` 4.0/2.0,
`sanity_max_rel` 1.0/0.75). `gptq.apply` runs on CPU by design — the
damp/Cholesky/inverse chain runs in fp64 there for 2-3-bit grids (4-bit+ keeps
the cheaper fp32 path) and **retries under escalating damping** (×2.5, up to 4
escalations, recorded in `GPTQStats.dampen_used`) instead of dying on
near-singular low-bit Hessians.

## Recipe param routing

Calibrate-stage and apply/fold-stage kwargs live in the same
`calibration.params` dict but go to different functions — the pipeline splits
them via `_AWQ_APPLY_PARAMS` (`rmsnorm_plus_one`, `sanity_max_rel`,
`sanity_tokens`) and `_GPTQ_APPLY_PARAMS` (`n_bits`, `group_size`, `dampen`,
`actorder`, `sym`, `grid_mix`, …). When adding a new calibrator kwarg, decide
which stage owns it and update the corresponding tuple, or the recipe will
crash with a TypeError.

All calibrators default to `device: "auto"` (cuda → mps → cpu via
`calibrate/_device.resolve_device`); recipes only need `device` to pin one.

## The vLLM W4A16 path (no recipe)

The W4A16 checkpoint is produced by `vllm_export.run_ptq` (not a `quant-tuner
run` recipe — it has no llama.cpp in the loop). The entry point is
`examples/gptq_w4a16.py`. Key knobs:

| Flag | Default | Meaning |
|---|---|---|
| `--ctx` | 32 768 | calibration sequence length (above the GGUF pipeline's 4 096 default — the serving target is long-context) |
| `--budget-tokens` | 524 288 | total calibration token budget, strided evenly across the corpus |
| `--scheme` | W4A16 | one of W4A16 / W8A8 / W8A16 / FP8_DYNAMIC |
| `--model-class` | None | pass the **conditional** class for multimodal checkpoints (e.g. `Qwen3_5ForConditionalGeneration`) — `AutoModelForCausalLM` silently picks the text-only class and matches none of the multimodal tensors |
| `--ignore` | (repeatable) | extra module patterns to keep in bf16 (the default ignore list is gemma-shaped; **audit it per model** with `scripts/run_vllm_ptq.py --dry-run-ignore`) |

## QAT for natively-ternary models (no recipe)

The `qat/` package is not recipe-driven — it is a trainer. The entry point is
`examples/ternary_qat.py`. The full method, the termination-probe invariants,
and the working recipe (anchor10: 32B forced-stop KD table + termination
steering + bounded repetition hinge, serve at T=0.7 / top_p=0.95) are in
[`ternary_qat.md`](ternary_qat.md).
