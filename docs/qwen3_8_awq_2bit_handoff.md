# Handoff: 2-bit AWQ for Qwen3.8-27B

**Goal.** Build an **IQ2_M AWQ** variant of Qwen3.8-27B and find out whether activation-aware
scaling beats the shipped plain-`hybrid_custom` imatrix 2-bit at the *same* 3.06 bpw. If it wins,
it replaces the IQ2_M rung in
[`pearsonkyle/Qwen3.8-27B-imatrix-MTP-GGUF`](https://huggingface.co/pearsonkyle/Qwen3.8-27B-imatrix-MTP-GGUF).

Facts below are marked **[V]** verified on this machine or **[U]** unverified/assumed. Do not
promote a **[U]** to a claim without checking it.

---

## 1. Why this is worth doing — and the decision criterion

The gemma-4-31B release made exactly this swap and the result was counter-intuitive: at identical
bpw the AWQ 2-bit was **worse on median KLD** (1.804 vs 1.571) and **worse on top_p** (43.9% vs
46.6%), yet **+54% on tool-argument accuracy** (0.171 → 0.263) with run-to-run variance collapsing
from ±.082 to ±.009. Perplexity nearly halved (1959 → 1040).

**So do not decide this on KLD.** The static table would have picked the wrong build for gemma.
The decision metrics, in priority order:

1. **Tool-call param accuracy** (25 held-out sessions, 174 turns) — the metric that moved for gemma.
2. **Tool-selection accuracy** and its **run-to-run spread** across seeds.
3. **SWE-mimic agent bench** — does it still drive a loop.
4. KLD / top_p / PPL — reported for completeness, *not* used to decide.

**Ship it only if param-accuracy improves by more than the noise floor.** At n≈174 the binomial SE
near 0.26 is **±3.3pp**; the whole shipped ladder spans 0.239–0.274, i.e. every rung is inside one
SE of every other. A +0.01 param-acc "win" is nothing. Gemma's +0.092 was real. Require a gap of
that order, or run multiple seeds (`scripts/run_toolcall_reps.py --reps 3`) and compare
distributions rather than points.

**Reference numbers to beat** (shipped IQ2_M, plain hybrid imatrix @ ctx 32768) **[V]**:

| | value |
|---|---|
| Size / BPW | 9.74 GiB / 3.062 |
| Tool-selection acc | 0.494 |
| **Param accuracy** | **0.260** |
| Schema-valid rate | 0.954 |
| KLD med / top_p (external) | 0.12416 / 71.95% |
| PPL (external) | 56.540 |
| SWE mimic | resolved, 20 steps, 0 malformed |

---

## 2. The central architectural question — already answered

Qwen3.8 is a **hybrid**: 64 layers, of which **48 are linear attention and 16 are full
attention** (`full_attention_interval: 4`, so layers 3, 7, 11, … are full). AWQ folds per-channel
scales into the preceding RMSNorm, which is only valid for the standard
`norm → linear` structure. The obvious worry is that AWQ can therefore only touch a quarter of
the model.

**That worry is mostly unfounded, and the code is already safe.** Verified by reading the
safetensors index **[V]**:

```
LINEAR-ATTN (48 layers)          FULL-ATTN (16 layers)
  input_layernorm                  input_layernorm
  linear_attn.in_proj_qkv          self_attn.q_proj / k_proj / v_proj
  linear_attn.in_proj_z            self_attn.o_proj
  linear_attn.in_proj_a            self_attn.q_norm / k_norm
  linear_attn.in_proj_b            post_attention_layernorm
  linear_attn.out_proj             mlp.gate_proj / up_proj / down_proj
  linear_attn.norm
  post_attention_layernorm
  mlp.gate_proj / up_proj / down_proj     <-- IDENTICAL in both
```

**Every one of the 64 layers has a standard MLP.** Only the attention block differs.

`calibrate/awq.py::_build_groups` gates the attention group on
`getattr(layer, "self_attn", None)` **and** `isinstance(getattr(attn, "q_proj", None),
torch.nn.Linear)` **[V]**. A linear-attention layer has no `self_attn` module at all, so it
silently contributes **no** attention group — no crash, and critically **no incorrect fold onto
SSM tensors**. The MLP branch keys off `mlp.gate_proj`, which every layer has, so it fires on all 64.

**Groups AWQ builds: 80** = 16 `L*_attn` + 64 `L*_mlp` **[V, measured 2026-08-15]**.

> ⚠️ **Corrected 2026-08-15.** This section previously claimed **160 groups / 92.4% coverage**
> and was marked [V]. It was not — it was an unrun prediction. The `*_out` groups
> (`L*_attn_out` = o_proj, `L*_mlp_out` = down_proj) are **never built**, because
> `awq._build_groups`' `include_output_proj` defaults **`False`** and neither this recipe nor
> the gemma-4-31B one sets it. The real numbers are below. **This does not invalidate the
> run** — gemma used the identical default, so the A/B is apples-to-apples — but it changes
> what §8 should conclude.

**Parameter coverage** (F16 GGUF tensor table, recomputed):

| bucket | params | AWQ |
|---|---:|---|
| MLP `ffn_gate` + `ffn_up` (all 64 layers) | 11.409 B | ✅ covered |
| full-attn `q/k/v_proj` (16 layers) | 0.671 B | ✅ covered |
| MLP `ffn_down` (all 64 layers) | 5.704 B | ❌ skipped (`include_output_proj`) |
| full-attn `o_proj` (16 layers) | 0.503 B | ❌ skipped (`include_output_proj`) |
| linear-attn projections (48 layers) | 1.536 B | ❌ skipped (no `self_attn`) |
| embed / output | 2.543 B | n/a |
| MTP head `blk.64` | 0.425 B | pinned Q8_0 |
| norms / conv1d / A_log (F32) | 4.027 B | not quantized |

**AWQ reaches 12.080 B of the 19.823 B of matmul projection weight — 60.9%.** The arithmetic
checks against the old table's own MLP bucket: 11.409 + 5.704 = 17.113 B. Everything skipped
falls back to plain imatrix.

> **[U] The cheap follow-up is now `include_output_proj: true`, not the SSM extension.**
> It is one recipe line and lifts coverage 60.9% → ~90.6% (`ffn_down` alone is 5.704 B —
> **3.7× more weight than the entire linear-attention bucket**), on tensors of exactly the
> structure AWQ is designed for. Try it before touching `linear_attn`.
>
> The linear-attention extension remains open but is both smaller and riskier:
> `linear_attn.in_proj_*` sits directly after `input_layernorm`, the structure AWQ folds into,
> and is skipped only because the module is named `linear_attn` rather than `self_attn` — but
> the SSM path applies gating and a `linear_attn.norm` between projection and output, so the
> fold's cancellation argument may not hold. `models.hf_gguf_map.is_ssm` exists precisely
> because output-aware re-ranking is invalid for state-space tensors. **Do the baseline run
> first** either way.

---

## 3. Machine state — what is on disk

Everything needed is present. **[V]** as of this handoff:

| path | size | notes |
|---|---:|---|
| `/workspace/.hf_home/hub/models--Qwen--Qwen3.8-27B` | 52 GB | bf16 source weights — **required**, AWQ needs HF forward passes |
| `out/exp-060/model_extracted/` | 4 KB | symlink farm → the cache above. **Do not delete the cache.** |
| `out/exp-060-32k/corpora/corpus.cal.txt` | 16.9 MB | 4,255,761 tokens, packed for **ctx 32768** (`window_cap` 32076) |
| `out/exp-060-32k/corpora/corpus.eval.*.txt` | ~2.6 MB | six eval holdouts |
| `out/exp-060-32k/baseline.*.kld` | **143 GB** | FP16 KLD baselines for all six evals — expensive to regenerate |
| `out/exp-060-32k/imatrix-hybrid_custom.gguf` | 14 MB | the shipped ladder's imatrix (provenance) |
| `out/exp-060-32k/iq2_m/` | 9.8 GB | **the A/B opponent** — keep |
| `out/exp-060/model-f16.gguf` | 51 GB | unfolded F16 |
| `vendor/llama.cpp/build/` | — | llama.cpp @ `f3e1828`, CUDA sm_120 |

Deleted during cleanup (recoverable): `iq4_nl/` (dead variant),
`mtp/mtp-Model_Extracted-F16.gguf` (intermediate). The IQ3_M / IQ4_XS / Q5_K_M GGUFs live on the
HF repo and may have been removed locally — re-download if you need them.

**⚠️ There is no `corpus.val.txt`.** **[V]** The universal corpus builder writes eval *holdouts*
(`corpus.eval.{tools,agentic,broad,general,cal8k,redteam}.txt`) rather than the
`corpus.val.txt` that `build_corpora.py` produced for older runs. AWQ's `cv_strategy` /
`per_tensor_alpha` paths **require** `holdout_text` and fail their precondition check before the
model loads if it is missing. Use **`corpus.eval.tools.txt`** (593 KB, in-distribution
tool-calling, disjoint from the calibration train split) as the holdout. Do not use
`corpus.eval.cal8k.txt` — it is a *fit* probe drawn from the previous calibration corpus.

**Disk budget.** An AWQ run adds: folded F16 GGUF (~51 GB) + new imatrix (~14 MB) + IQ2_M output
(~10 GB) ≈ **61 GB**. Check `df -h /workspace` before starting; free the IQ3_M/IQ4_XS/Q5_K_M
directories (they are on HF) if headroom is short.

---

## 4. The recipe

Start from `src/quant_tuner/recipes/iq2_m_awq.yaml` (written for gemma) and create
`iq2_m_qwen3_8_awq.yaml`. The pipeline order is: **AWQ calibrate → fold into HF weights → convert
folded to F16 GGUF → collect imatrix on the folded F16 → `hybrid_custom` re-weight →
llama-quantize**. The imatrix must be collected on the *folded* weights — folding rescales
per-channel activations, so an unfolded imatrix would over-weight exactly the channels AWQ boosted.

```yaml
name: iq2_m_qwen3_8_awq
model: /workspace/.hf_home/hub/models--Qwen--Qwen3.8-27B   # or out/exp-060/model_extracted
workspace: ./out/exp-060-32k-awq

calibration:
  method: awq
  variant: best
  params:
    proxy: q2k_b16            # see §5.1 — pinned, matching the gemma build
    proxy_tokens: 256
    imatrix_variant: hybrid_custom
    imatrix_ctx: 32768        # MUST match the corpus packing — see §5.2
    rmsnorm_plus_one: true    # [V] correct for Qwen3.8 — see §5.3
    sanity_max_rel: 0.03
    device: auto

quantize:
  type: IQ2_M
  mtp_pin: q8_0               # [V] required — see §5.4

extract:
  keep_mtp: true

bench:
  suite: full
  eval_ctx: 8192
```

Feed the corpora explicitly rather than letting the recipe rebuild them — the 32K packing is the
whole point of this ladder and a rebuild at a different ctx invalidates comparability:

```
cal_text     = out/exp-060-32k/corpora/corpus.cal.txt
holdout_text = out/exp-060-32k/corpora/corpus.eval.tools.txt
```

---

## 5. Traps

### 5.1 The proxy pin — inherited, and only partly transferable **[U]**
The gemma recipe pins `proxy: q2k_b16` with a documented reason: scoring every group member with
the pure `iq2_s` codebook regressed top_p, because the codebook's steep α penalty plus `v_proj`'s
*fictitious* 2-bit error (it is really Q4_K under GQA≥4, far smaller and nearly α-insensitive)
dragged the shared group α down.

That reasoning was established **on gemma**, where all 60 layers are standard attention. Here only
16 layers have a `v_proj` at all, so the distortion it corrects for applies to a quarter as many
groups. **Start with `q2k_b16` for comparability with the gemma precedent**, but the pipeline
default (`iq2_s` base + `proxy_mix: IQ2_M` per-member scoring) is a genuinely open A/B on this
architecture. If run 1 is a wash, that is the first knob to try.

### 5.2 Calibration ctx is a packing parameter, not a flag
`corpus.cal.txt` is packed for **ctx 32768** (`window_cap` 32076). The same value must reach
`awq.calibrate(ctx=)` **and** `llama-imatrix -c`. A corpus packed for one ctx and read at another
either straddles chunk boundaries or glues unrelated conversations into one context. **Numbers
produced at different ctx are not comparable** — including PPL/KLD, and including against the
shipped ladder.

### 5.3 `rmsnorm_plus_one: true` is correct here — verified **[V]**
`transformers 5.12.1` `Qwen3_5RMSNorm` computes `output = output * (1.0 + self.weight.float())`
with `weight = nn.Parameter(torch.zeros(dim))`. That is the **(1 + γ)** form, so the fold must use
`(1 + γ)/scale - 1`. Setting this `false` would silently produce a subtly wrong model that still
loads and generates — there is no crash to warn you.

### 5.4 The MTP head must stay pinned **[V]**
Qwen3.8 ships a trained MTP head that the converter maps to `blk.64` (15 tensors). It must be
pinned Q8_0 via `quantize.mtp_pin` — **and verified per output file**, because `llama-quantize`
silently accepts a `--tensor-type` pattern that matches nothing. `models/mtp.py::describe(f16)`
reads the draft layer from the GGUF rather than hardcoding it. The head sits outside the forward
pass so it receives no imatrix statistics and AWQ never touches it. Verify after quantizing:

```python
from gguf import GGUFReader                      # PYTHONPATH=vendor/llama.cpp/gguf-py
r = GGUFReader(out); mtp = [t for t in r.tensors if t.name.startswith("blk.64.")]
assert len(mtp) == 15 and sum(t.tensor_type.name == "Q8_0" for t in mtp) == 8
```

### 5.5 `llama-imatrix` segfaults above ~17k ctx on this model **[V]**
Its perplexity path computes `all_logits + first*n_vocab` with `first = n_ctx/2` in `int`
arithmetic (`tools/imatrix/imatrix.cpp:911`). With Qwen3.8's 248,320-token vocab that overflows
`INT_MAX` for any `n_ctx > 2³²/248320 ≈ 17,296`, and the process dies **after the first pass**.
**Pass `--no-ppl`.** It skips only the perplexity bookkeeping; the forward pass — and therefore
every activation statistic — is unchanged.

### 5.6 `--process-output` is now on by default **[V]**
`models.llama_cpp.imatrix` passes `--process-output`, so `output.weight` gets statistics.
`llama-imatrix` otherwise only collects tensors named `blk.*`. Audit `logs/quantize.log`: the
*expected* members of the `did not find weights for <tensor>` list are exactly
`token_embd.weight` (an embedding lookup, never collectable) and the `blk.64.*` MTP head.
**Anything else in that list is a bug.**

### 5.7 Vision tower **[V]**
The HF checkpoint carries **333 `model.visual.*` tensors**. The GGUF conversion drops them
(text tower only). If you use any llm-compressor / HF-side ignore lists, note that
`re:.*vision_tower.*` matches **zero** tensors here — the prefix is `model.visual.*`.

### 5.8 `step()` idempotency is existence-based
`experiments.step()` skips a stage when its output file exists. The GGUF name already encodes
`{type}-{method}-{variant}` for this reason. If you change a parameter without changing the output
filename, the stage is skipped and you bench a stale artifact under a new label.

### 5.9 pgrep self-match — bit us three times this project
`pkill -f llama-server` from a `bash -c` whose own command line contains that string kills the
invoking shell. Kill by PID.

---

## 6. Evaluation — reuse the existing baselines

The six FP16 KLD baselines already exist (143 GB, §3), so KLD needs no FP16 re-run. `bench/kld.py`
diffs a quant against the saved `baseline.<eval>.kld`.

```bash
# 1. KLD across all six evals (reuses baselines; eval_ctx 8192)
PYTHONPATH=src .venv/bin/python scripts/exp060_quants_qwen38.py \
    --run exp-060-32k-awq --ctx 32768 --eval-ctx 8192 \
    --evals external general tools agentic broad cal8k

# 2. Tool-call — THE decision metric. Multi-seed; a single rep cannot resolve the gap.
PYTHONPATH=src .venv/bin/python scripts/run_toolcall_reps.py \
    --models out/exp-060-32k-awq/.../IQ2_M-awq.gguf out/exp-060-32k/iq2_m/Qwen3.8-27B-IQ2_M.gguf \
    --reps 3 --results out/exp-060-32k-awq/eval/toolcall_reps.csv

# 3. Agent bench (swe-mimic; Docker-free)
cd /workspace/swe-mimic && PORT=18080 RBUDGET=2048 TAG=_awq bash run_all_quants.sh
```

Harness facts inherited from this ladder **[V]**:
- Tool-call eval: **greedy `temperature=0`**, `ctx=32768`, `--no-stop-on-fail` so every model is
  scored on the **identical 174 turns**. Never drop `--no-stop-on-fail` — the default halts a weak
  model early and scores it on fewer, easier turns, which makes models incomparable.
- swe-mimic: `temperature=0.25, top_p=0.95, max_tokens=8096`, reasoning budget 2048, step cap 60,
  `PORT=18080` (Jupyter owns 8080). Instance `dask__dask-11393`, 1 F2P + 34 P2P.
- swe-mimic classifies tool errors as `malformed` / `timeout` / `nonzero`. A non-zero exit is
  **not** an error — a `grep` with no match exits 1. Only `malformed` and `timeout` count.
- **Known latent bug**: `sh()` truncates command output to the last 6,000 chars while the P2P run
  emits 10,408. All 34 `PASSED` lines survive by ~1.7×, but a larger test set would lose them and
  report a false *unresolved*. It biases pessimistic — it cannot inflate a resolve. Fix before
  running instances with big suites.

---

## 7. If it wins

1. Rebuild the IQ2_M rung as AWQ, keeping the filename's terminal quant tag (`…-IQ2_M.gguf`) so HF
   still derives an Ollama `:IQ2_M` tag.
2. Set `general.name` to `Qwen3.8-27B` at quantize time — the converter otherwise writes the
   extraction directory name (`Model_Extracted`), which is a display-visible wart. **[V]**
   `gguf_set_metadata.py` cannot fix this after the fact: its `gguf_scalar_to_np` map has no
   STRING entry, so it refuses string fields. The only retrofit is
   `gguf.scripts.gguf_new_metadata --general-name` — a full file rewrite.
3. Update the card's unified table + add a "Why the 2-bit is AWQ" section mirroring the gemma card,
   and re-upload with `release/upload_to_hf.py --push` (dry-run by default; `--with-evidence` adds
   the provenance and SWE artifacts).
4. Squash the HF repo history afterward (`HfApi.super_squash_history`) — replacing a 9.74 GiB LFS
   file otherwise leaves both versions counting against repo storage.

## 8. If it loses or is a wash

Say so on the card. A negative result at 92.4% coverage is genuinely informative: it would suggest
AWQ's gemma win came from the attention-path scaling that the hybrid architecture mostly denies us
here, and the follow-up is §2's open question — extending `_build_groups` to `linear_attn.in_proj_*`
— rather than more α tuning.
