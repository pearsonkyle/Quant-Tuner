# Ternary QAT optimization audit (memory / technique / speed)

Audit of the continued-QAT pipeline for `prism-ml/Ternary-Bonsai-8B` (native-ternary
Qwen3-8B: 36 layers, hidden 4096, GQA 32/8, intermediate 12288, ~6.95 B linear params)
trained on an M4 Max / 128 GB via MPS. Goal: fit **all 36 layers**, train **≥1 full
epoch on more data** within budget, and make the training move *capability* (SWE-rebench
patch/pass), not just behavior. Every claim below was verified against first-hand file
reads, `logtrain.jsonl` inspection, a masking simulation, and source-level checks of the
pinned **transformers v5.8.1** and **torch v2.12.0** tags.

The changes this audit produced are implemented in this repo — see
"Prioritized change list" for what landed where, and `docs/ternary_qat.md` for the
updated runbook.

---

## TL;DR — the five findings that matter most

1. **The corpus never trained the stop token.** The assistant-span regex labeled
   group(1) (content only), so under HF's causal shift *no position in the entire
   corpus had `<|im_end|>` as a CE target*. The model received literally zero gradient
   toward ending its turn — the mechanistic cause of the pathological looping
   (912/953 duplicate tool calls). Fixed: spans now extend through `m.end(0)`;
   the builder asserts labeled `<|im_end|>` targets exist.
2. **Real tool schemas were being thrown away.** `reconstruct_tools` claimed "the logs
   store no schema block" and fabricated hollow schemas (only-called tools, all-`string`
   params, placeholder descriptions). In fact every logtrain session carries real
   schemas at `messages[0]["tools"]` (proper types, `required` lists, the full
   available-tool set) — `data.split.session_tools` already read them for the
   calibration corpus. Fixed: the QAT builder now uses them (stub fallback retained).
3. **AdamW's default `weight_decay=0.01` was active on the ternary latents** —
   decay pulls magnitudes toward the TWN threshold, silently eroding codes to 0 over
   long runs. Fixed: `--weight-decay` defaults to 0 for both optimizers.
4. **At lr 5e-5, ~zero code flips were possible.** Shipped latents sit at
   |w| ∈ {0, s}; the nearest flip boundary is 0.3–0.5·s ≈ O(1e-2) away; Adam-normalized
   steps move ~lr per step, so a flip needs hundreds of consistently-signed steps —
   the 261-step runs could flip almost nothing. The observed loss drop (2.26→~1.0) is
   consistent with continuous *scale* drift alone. This alone can explain
   "behavior changed, capability didn't." The trainer now reports **code-flip
   telemetry** every checkpoint and the export prints artifact-level flips vs shipped;
   run the LR probe (5e-5 / 3e-4 / 1e-3) before any long run.
5. **Adafactor makes full-36 fit.** all-36 fp32 AdamW needs ≈ 116 GB (swaps);
   Adafactor's factored second moment is ~MBs, so all-36 fp32 lands at ≈ 66-75 GB —
   comfortably inside the real MPS working-set budget (~96-102 GB, i.e.
   macOS `recommendedMaxWorkingSetSize`, not the nominal 128). Implemented as
   `--optim adafactor` (per-tensor loop — no foreach, no MPS deadlock risk).

---

## Memory math (8.19 B total params; 6.95 B trainable linear latents at all-36)

| Configuration | model | grads | optimizer state | activations† | total |
|---|---:|---:|---:|---:|---:|
| AdamW fp32, all-36 (old) | 32.8 | 27.8 | 55.6 (2× fp32) | ~8-10 | **~116 GB — swaps** |
| AdamW fp32, last-18 (old) | 32.8 | 13.9 | 27.8 | ~8-10 | ~57-74 GB ✓ |
| **Adafactor fp32, all-36** | 32.8 | 27.8 | ~0.01 (factored) | ~6-8‡ | **~66-75 GB ✓** |
| Adafactor + `--beta1 0.9` | 32.8 | 27.8 | 27.8 (fp32 momentum) | ~6-8 | ~95 GB (tight) |
| Adafactor + KD teacher fp16 | +16.4 teacher, +1-2 teacher activations | | | | ~88-90 GB (tight) |
| Adafactor + `--compute-dtype bf16` | 16.4 bf16 + 27.8 masters | 13.9 (bf16) | ~0.01 | ~3-4 | **~58-62 GB ✓** |

† Activation peak = per-layer checkpoint inputs (67 MB × 36) + the attention math-path
recompute (`[1,32,4096,4096]` fp32 ≈ 2.1 GB + grads) + the loss head.
‡ The masked-CE path (below) removes the full-logits [1,4096,151936] fp32 tensor
(~2.5 GB + autograd) from the peak.

Real budget note: PyTorch/MPS caps at the OS working-set recommendation (~75-80% of
unified memory). `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` disables the allocator cap if a
run sits right at the edge — prefer configs with headroom instead.

---

## The 12 questions

### Memory

**1. Adafactor integration — correct?** It didn't exist: the audited trainer was
AdamW-only (no `--optim` flag; "adafactor" had zero hits repo-wide). It has now been
added with the STE-correct configuration:

```python
Adafactor(params, lr=args.lr, scale_parameter=False, relative_step=False,
          warmup_init=False, beta1=None, weight_decay=0.0)
```

- `scale_parameter=False, relative_step=False, warmup_init=False` → the external
  warmup→cosine schedule (`lr_at`) stays the single source of LR truth. (With
  `relative_step=True` Adafactor ignores your schedule for a 1/√t internal one.)
- `beta1=None` (default off) → no first moment (that's the 27.8 GB you're saving).
  `--beta1 0.9` re-enables it if flip telemetry shows updates are too noisy — still
  fits (~95 GB) but leaves no KD headroom. timm's `AdafactorBigVision` keeps momentum
  in bf16 (+13.9 GB instead of +27.8) if that tradeoff is ever needed.
- Internal update-RMS clipping (`clip_threshold=1.0`) **composes** with the external
  `clip_grad_norm_`: they act at different stages (raw grads vs normalized update),
  and Adafactor's update is nearly invariant to uniform grad rescaling — the external
  clip just keeps transient spikes out of the EMA. Both are kept.
- **STE interaction: none adverse.** STE backward is identity, so latent grads are
  ordinary dense linear-layer grads — exactly the rank-1-structured `E[g²]` case the
  factored second moment was designed for (T5 linears). The known Adafactor weakness
  (row-sparse embedding grads) doesn't apply: embeddings/lm_head are frozen.
- Verified per-tensor loop, zero `torch._foreach_*`/fused calls in transformers 5.8.1 —
  **no MPS deadlock risk**. (Also verified: on torch 2.12, `foreach=None` never
  auto-selects foreach on MPS — "mps" isn't in the foreach device list — so the
  explicit `foreach=False` everywhere is belt-and-suspenders, kept deliberately.)
- Does it fit all-36 without swap? Yes: ~66-75 GB (table above).

**2. Gradient checkpointing — optimal?** Effectively yes. transformers 5.8.1's
`gradient_checkpointing_enable()` already defaults `use_reentrant=False` (verified in
`modeling_utils.py`), and per-decoder-layer granularity is right for this model. The
actual activation peaks were (a) the full-logits loss head — fixed by masked-CE — and
(b) the SDPA math path: **torch 2.12 MPS fused attention kernels are inference-only**;
training always materializes `[B,H,S,S]` on the tape (checkpointing confines it to the
recompute window). Finer-grained (sub-layer) checkpointing would shave ~1-2 GB of
recompute peak at +latency — not worth it. Activation *offload* is pointless on unified
memory (same physical pool).

**3. fp32-master + bf16-compute vs Adafactor?** It does **not** rescue AdamW at all-36:
masters 27.8 + bf16 weights 16.4 + bf16 grads 13.9 + AdamW fp32 states 55.6 ≈ 114 GB
(bf16 Adam states would fit but are an accuracy risk). It **does** solve the bf16
underflow *correctly*: updates accumulate in fp32 masters across steps; the bf16 copy
is compute-only, so sub-ulp updates are never lost (that was the "no codes flip"
failure — a ~lr update is below one bf16 ulp of a ~1e-2 latent). Important subtlety:
transformers' Adafactor internally upcasts bf16 params per step and writes back — a
*transient* upcast, NOT persistent masters, so it does not fix underflow; a real
wrapper was required (`qat/master_opt.py`, `--compute-dtype bf16`). The winning combo
is **Adafactor + fp32 latents** (fits already, zero new numerics) with bf16-compute as
the *speed* lever (~1.5-2× Metal matmuls, half-size activations, ~58-62 GB total).
Correctness for STE: codes are identical at step 0 (bf16 preserves sign/zero and the
threshold comparison), training-forward scales carry ≤0.4% bf16 rounding, and export
reads the fp32 masters — the artifact stays exact. Gate any long bf16-compute run on a
short loss-parity check vs fp32.

**4. Duplicated latents / transient copies?** `TernaryLinear` holds the original
`nn.Linear` — no latent duplication. Two real issues, both fixed:
(a) `save_ckpt`'s whole-dict `.detach().cpu()` is a +27.8 GB transient at all-36 —
safe at the post-step trough (~33-40 GB with Adafactor) but dangerous on a mid-accum
signal save with 27.8 GB of live grads; the trainer now drops the partial accum group
(`opt.zero_grad()`) before the final/signal save, writes atomically (tmp + `os.replace`
— a mid-write crash can no longer corrupt the only checkpoint), and frees + empties the
MPS cache after. (b) `ternarize_group` allocates ~5 W-sized transients per linear per
forward (×2 under checkpoint recompute) — frozen layers paid this for a **bit-exact
no-op** since shipped weights are already on the grid; `wrap_model` now proves
exactness (`torch.equal(ternarize(W), W)`) and skips wrapping frozen layers (wrapping
off-grid ones as a fallback). Within-microbatch caching of `w_hat` for *trainable*
layers was considered and rejected: ternarize is ~0.06% of window FLOPs and a stale
cache after `opt.step()` would be a silent-wrongness hazard.

**5. Train fewer params / adapters?** Current behavior verified: only
`.linear.weight` latents in trainable layers get grads — norms, embeddings, lm_head
frozen. Two artifact-preserving extensions: (a) **`--train-norms`** (implemented,
default off): RMSNorm + q/k-norm weights in trainable layers, ~1 M continuous params,
zero optimizer-state concern, and norms export as F32 GGUF tensors either way — the
cheapest capability-relevant knob given latent flips are scarce. (b) **LoRA-on-latent**
(documented, not implemented): train a low-rank delta on the *latent* (`W = W₀ + BA`)
with the same STE forward — optimizer state and stored grads shrink to MBs and the
export still just re-ternarizes latents, unlike output-side LoRA which breaks the pure
Q2_0 artifact. Constrains flip directions to rank-r; worth an experiment only if
Adafactor headroom ever becomes the binding constraint.

**6. ZeRO-style CPU offload?** Rejected. On Apple unified memory the "CPU" and "GPU"
share one physical pool — moving optimizer state to "CPU" frees nothing. The only
adjacent knob that matters is the MPS allocator watermark (see memory-math note).
Offload designs earn their complexity on discrete-VRAM systems only.

### Technique

**7. Why didn't the fine-tune raise capability? Is the recipe sound? LR? KD?**
The recipe (masked CE + STE + per-group TWN) is *sound but insufficient*, and the
failure is over-determined:
  1. **~zero code flips at lr 5e-5** (finding #4 above) — the fine-tune was mostly
     re-scaling groups, i.e. adjusting output magnitudes, not re-allocating ternary
     capacity. Peak LR for STE ternary continued-training plausibly wants to be
     5-20× higher (BitNet-family QAT runs at 1e-4–1e-3 with warmup); this must be
     probed, not assumed — hence flip telemetry + the 3-point LR probe in the runbook.
  2. **~2.4 M trainable tokens** (2090×4096 windows, ~30% density, 0.5 epoch) is tiny
     against an 8 B model's capability surface.
  3. **The corpus bugs** (stop token, hollow schemas) mean part of the budget trained
     the wrong distribution.
  4. **One-hot CE on the model's own domain adds little signal a 2-bit net can
     absorb** — the base model already assigns high probability to most of these
     tokens. **KD from the dense parent (Qwen3-8B fp16) is the strongest lever**:
     full-distribution targets at every labeled position carry far more usable
     information per token, and BitNet/low-bit-QAT literature consistently shows
     distillation dominating plain CE at ≤2 bits. Implemented:
     `--kd-teacher <hf-path> --kd-alpha 0.5 --kd-temp 1.0`,
     `loss = (1-α)·CE + α·T²·KL(teacher‖student)` computed only at labeled positions;
     the teacher is queried with `logits_to_keep=<index tensor>` so it never
     materializes full-vocab logits at unlabeled positions. Memory ≈ 88-90 GB with
     Adafactor — fits, tight; if it crowds, precompute top-K teacher logits offline
     (the corpus is only ~2.4 M labeled positions ≈ ~1-2 GB at top-64 fp16) — a
     documented follow-up.

**8. Is identity STE right?** Keep it for now — it's the proven baseline, the step-0
invariant depends on the exact TWN form, and the diagnosed problem is flip *pressure*
(LR), not estimator bias. Two principled upgrades staged behind flip telemetry:
(a) **gradient-through-scale**: TWN's scale is a differentiable function of the latents
(mean of kept magnitudes), but the whole-function STE discards that path — splitting
`w_hat = s_diff(W)·codes_detached + STE(codes)` gives exact scale gradients while
keeping flips on the identity path; (b) **clipped STE** (zero grad where |latent| ≫
threshold) keeps latents near decision boundaries so accumulated pressure produces
flips instead of runaway magnitudes. Soft/annealed ternarization is not recommended:
it breaks step-0 exactness and buys little for a *continued* fine-tune.

**9. Corpus audit + data scaling.** Bugs found (both fixed): the `<|im_end|>` masking
bug and the discarded real schemas (TL;DR #1/#2). Also fixed/added: `--window` default
was 8192 — above the documented MPS max (now 4096 + a trainer guard); tool-output
truncation `--max-tool-tokens N` (head+tail with an explicit marker — tool outputs
dominate the masked spans; p50 is small but the tail reaches ~40 k chars);
`--min-density` window filter chosen from the density histogram the builder now
prints; `--split test` builds the disjoint validation corpus for `--val-corpus`.
Known accepted limitation: windows straddle session boundaries (session B trained
under session A's residual context) — standard packing noise; a session-aligned packer
is a possible follow-up. **Scale**: the data is small (253 sessions; 202 in the train
slice). In order of leverage: (i) use the quality metadata already in `logtrain.jsonl`
(`score` 0.51-1.00, `label` good/maybe) for weighted sampling or `good`-only filtering;
(ii) KD multiplies signal per existing token (Q7); (iii) external agentic corpora
(SWE-smith/SWE-Gym-style trajectories, xlam/glaive function-calling) rendered through
the *same* chat template + masking — patch-writing supervision is precisely what the
patch-rate metric is starving for; (iv) keep the wiki full-loss mix (~2-8%) as the
anti-forgetting floor.

**10. TWN ↔ shipped-encoding match; train/export consistency.** Verified: every
128-group of the shipped F16 is single-magnitude (max/min = 1.0000, ~37% exact zeros),
TWN with 0.7 threshold reproduces it exactly (measured logit delta 0.0000e+00 across
all 252 wrapped linears), and the export re-ternarizes with the *same*
`ternarize_group` — trained layers get their final training-forward codes, frozen
layers are byte-identical. One gap found and closed: the export's F16 GGUF cast rounds
scales to fp16 while the training forward used fp32 scales — `ternarize_group` now
quantizes the scale to fp16 *inside* the quantizer, so training forward ≡ deployed
numerics exactly. This is provably a no-op at step 0 (shipped scales are fp16-native;
the fp32 mean of N identical fp16 values round-trips fp16 unchanged — unit-tested) and
it applies uniformly to trainer, probe, and export since all share the function.

### Speed (token-bound, ~8-10 ms/tok)

**11. What actually cuts wall-time per useful gradient?**
  1. **Density, not window length**: attention is only ~17% of window FLOPs at 4096,
     so shorter windows barely help — but every masked token costs full forward+
     backward for zero gradient. `--max-tool-tokens` + `--min-density` raise
     useful-gradient throughput roughly in proportion to the density gain.
  2. **Masked-CE loss head** (implemented): lm_head + CE only at labeled positions —
     ~7-9% of step FLOPs (verified: the HF path materializes full [1,4096,151936]
     logits in model dtype *plus* a full fp32 upcast copy) and −4-5 GB peak.
  3. **`--compute-dtype bf16`** (fp32 masters): ~1.5-2× on the Metal matmuls that
     dominate the 10 ms/token — the single biggest wall-time lever, gated on a parity
     check (Q3).
  4. Frozen-layer unwrap (implemented): removes the 5-transient no-op ternarize per
     frozen linear per forward (×2 under recompute) — matters for partial-layer runs.
  5. Not worth it: window-length reduction (see 1), finer checkpointing (Q2),
     `.item()` batching (one sync per ~40 s microbatch is noise — and the pre-backward
     isfinite guard needs the sync anyway).

**12. True batching (B>1) at seq 4096?** No. The INT_MAX arithmetic: the math-path
attention tensor has B·H·S² elements; 32·8192² = 2³¹ **exactly** overflows INT_MAX —
that is the observed seq-8192 failure — and B=4 at 4096 hits the same 2³¹. B=2 (2³⁰) is
representable, but MPS training attention is the composite math path (no fused/flash
kernel exists for training in torch 2.12) and the workload is compute-bound at these
shapes, so B=2 ≈ 2× the step time — grad-accum is equivalent and memory-cheaper. The
trainer now hard-errors on window > 4096 under MPS instead of crashing mid-run.
(A query-block-chunked attention could legalize 8192 windows for *data* reasons —
P99 single turn is 5.7 k tokens — at unchanged tokens/step cost; documented follow-up,
not implemented.)

---

## Prioritized change list (what landed)

| # | Tag | Change | Where |
|---|-----|--------|-------|
| 1 | memory | `--optim adafactor` (STE-correct config), `--weight-decay` default 0, `--beta1` | `scripts/exp058_qat_train_v2.py` |
| 2 | technique | `<|im_end|>` labeled; real schemas via `session_tools`; `--max-tool-tokens`; `--min-density`; `--split`; window default 4096; density/stop-token audit + fingerprint | `scripts/build_qat_masked_corpus.py` |
| 3 | memory+speed | masked-CE loss head (parity unit-tested vs HF full-logits path) | trainer `masked_forward` |
| 4 | infra | atomic checkpoints, grad-drop before signal save, `--resume` (step/data-order/Adafactor state, fingerprint-guarded) | trainer |
| 5 | speed | frozen-layer unwrap (bit-exactness proven at wrap time, fallback wrap) | trainer `wrap_model` |
| 6 | technique | fp16 scale quantization → train ≡ deploy numerics | `src/quant_tuner/qat/ternary.py` |
| 7 | technique | code-flip + scale-drift telemetry (per-ckpt; artifact-level at export) | trainer + `scripts/exp057_qat_export.py` |
| 8 | technique | accum-counter fix on non-finite loss; `--val-corpus/--val-every`; MPS window guard | trainer |
| 9 | technique | `--train-norms` (continuous capacity, artifact unchanged) | trainer |
| 10 | technique | `--kd-teacher/--kd-alpha/--kd-temp` online distillation | trainer |
| 11 | memory/speed | `--compute-dtype bf16` fp32-master wrapper | `src/quant_tuner/qat/master_opt.py` |
| 12 | docs | LR-probe runbook, memory table, data-scaling strategy | `docs/ternary_qat.md` |

Rejected with rationale: CPU-offload optimizer & activation offload (unified memory),
B>1 batching (INT_MAX + compute-bound), LoRA-on-output (breaks the pure-Q2_0 artifact),
any post-hoc calibration (proven structural no-op — `ternary_calibration_experiments.md`).

**Invariants preserved:** `foreach=False` on every optimizer/clip path (MPS deadlock);
window ≤ 4096 enforced (INT_MAX); fp32 latents/masters (bf16 underflow); step-0
`ternarize(shipped) == shipped` (unit-tested); train/export TWN consistency (single
shared `ternarize_group`).

**Comparability note:** the corpus fixes change the training distribution *on purpose*.
Historical loss values (e.g. 2.26→1.0) are not comparable to post-fix runs — a new
supervised token class (`<|im_end|>`) and real schemas shift both the loss level and
the density. Rebuild the corpus (new filename), and note `--resume` hard-fails across
corpus rebuilds by fingerprint.
