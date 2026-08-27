# W4A16 + fp8 KV cache for vLLM

How to produce a **compressed-tensors INT4 checkpoint vLLM serves directly**, with
**calibrated fp8 KV-cache scales baked in**, from the same corpora that calibrate the
GGUF ladder.

Module: `src/quant_tuner/vllm_export/w4a16.py` · CLI: `scripts/run_vllm_ptq.py` ·
Prior handoff: [`qwen3_8_gptq_vllm_handoff.md`](qwen3_8_gptq_vllm_handoff.md).

---

## Why fp8 KV is the interesting half

Quantizing weights is a **one-time** saving: a 27B bf16 trunk goes ~55 GB → ~20 GB and
stays there no matter how you serve it. Quantizing the KV cache is a **per-token,
per-sequence** saving, and that is what actually decides two things you care about at
deploy time:

- **how long a context fits** — the cache, not the weights, is what grows with context;
- **how many requests run concurrently** — vLLM's throughput is set by how many blocks
  fit in the KV pool.

On Qwen3.8-27B at its native 262,144-token context, bf16 KV is ≈17 GB and fp8 KV is
≈8.5 GB. That difference is what moves "262k fits, tightly, on a 48 GB card" to "262k
fits with ~14 GB of headroom for concurrency", and what makes YaRN extension to 524k
practical on the same card.

Only **softmax-attention layers have a KV cache at all.** On a hybrid model this is a
minority of layers: Qwen3.8 has 16 `self_attn` layers and 48 DeltaNet `linear_attn`
layers that carry recurrent state instead. Fewer scales than layers is correct, not a
bug — see [Verifying the export](#verifying-the-export).

## What "calibrated" means here

`--kv-cache-dtype fp8_e4m3` adds a **static, per-tensor, symmetric** fp8 scheme to the
*same* oneshot pass that quantizes the weights. compressed-tensors attaches observers to
the modules matching its `KV_CACHE_TARGETS` (`re:.*(self_attn|attention)$`), watches the
k/v activations over the calibration sequences, and writes a scalar `k_scale`/`v_scale`
onto each one.

`dynamic: False` is the entire point. A *dynamic* scheme defers the scale to runtime, so
nothing is calibrated and nothing is stored; the static scheme is what makes the
calibration corpus matter and what lets `vllm serve --kv-cache-dtype fp8_e4m3` pick the
scales up with no extra flags.

Because the scales come from the calibration corpus, they inherit the same property the
rest of this repo is built on: calibrate on **your** distribution and the fidelity lands
where you actually use the model.

## The recipe

The published INT4 cards do not use the `W4A16` preset. The preset is int4 / group-128 /
symmetric / minmax; the cards use:

| knob | preset | card | flag |
|---|---|---|---|
| group size | 128 | **32** | `--group-size 32` |
| symmetry | symmetric | **asymmetric** (zero-point) | `--asymmetric` |
| weight observer | minmax | **imatrix-mse** | `--observer imatrix-mse` |
| act-ordering | none | **static** | `--actorder static` |
| KV cache | none | **fp8-e4m3, calibrated** | `--kv-cache-dtype fp8_e4m3` |

`imatrix-mse` is a real observer in llm-compressor's registry (VERIFIED on 0.13.0) and is
the direct analogue of the GGUF ladder's imatrix: it weights the MSE clipping search by
per-input-channel activation importance collected during calibration, instead of taking
the raw range endpoints. GPTQ is already paying for those forward passes.

`--actorder static` permutes columns by activation magnitude once, at quantization time,
and folds the permutation into the saved weights — so serving pays no `g_idx`
indirection. Prefer it over `group` for anything you intend to deploy.

Deviating from the preset on **any** of group size / symmetry / observer / actorder
switches the run from `scheme="W4A16"` to an explicit `config_groups` spec
(`build_config_groups`). The two are mutually exclusive at the modifier; the module picks
for you and records the resolved group in `quant_tuner_ptq.json`.

### Full command

```bash
uv sync --extra vllm-ptq

uv run python scripts/run_vllm_ptq.py \
  --model out/exp-060/model_extracted \
  --corpus out/exp-060-32k/corpora/corpus.cal.txt \
  --out out/exp-060-w4a16-fp8kv/checkpoint \
  --ctx 32768 --budget-tokens 4194304 \
  --group-size 32 --asymmetric --observer imatrix-mse --actorder static \
  --kv-cache-dtype fp8_e4m3 \
  --ignore 're:.*visual.*' --ignore 're:mtp.*' \
  --pipeline sequential
```

Dry-run it first — this prints the ignore-match counts, the fully resolved recipe and the
tensors that would vanish from the export, and quantizes nothing:

```bash
uv run python scripts/run_vllm_ptq.py --model <hf-dir> --dry-run-ignore \
  --group-size 32 --asymmetric --observer imatrix-mse --kv-cache-dtype fp8_e4m3
```

### Two flags that are not optional in practice

- **`--ctx` must match the corpus's packing ctx** (top-level `ctx` in
  `corpora_audit.json`). It is a packing parameter, not just a runtime flag: a corpus
  packed for 32768 and read at 8192 straddles window boundaries, and numbers from
  different ctxs are not comparable.
- **`--budget-tokens` should be read as *sequences*.** The 524,288 default is 16
  sequences at ctx 32768 — far too few for a stable Hessian. Use ~4M (≈128 sequences).

### `--ignore` is the recurring trap

`DEFAULT_IGNORE` is gemma-shaped. On a checkpoint whose tower is `model.visual.*` the
default `re:.*vision_tower.*` matches **nothing** and the tower is quantized to int4
against a text-only corpus, with no error. A top-level `mtp.*` draft head is likewise
uncaught. Always `--dry-run-ignore` on a new model; a pattern matching zero modules is
flagged.

## Verifying the export

`run_ptq` calls `verify_export` before writing provenance, and refuses two failures that
otherwise produce a checkpoint that loads and serves perfectly while being wrong:

1. **no `quantization_config`** — vLLM serves it as bf16, at full size. The only symptom
   is that it did not get smaller.
2. **a requested KV scheme that wrote no scales** — the config claims fp8 KV and the
   `k_scale`/`v_scale` tensors are absent. vLLM falls back to an uncalibrated cache and
   nothing in the run log says so.

Read the counts, don't just trust the pass:

```python
from quant_tuner.vllm_export import count_kv_scales
count_kv_scales("out/exp-060-w4a16-fp8kv/checkpoint")
# {'k_scale': 16, 'v_scale': 16}  <- 16 self_attn layers, NOT 64
```

Compare that against the model's **softmax-attention layer count**, never its layer
count. On Qwen3.8, 16/16 is complete; 64 would mean something quantized a DeltaNet layer's
state as if it were a KV cache.

`quant_tuner_ptq.json` in the output dir records the resolved config groups, the KV
scheme, the ignore-match counts, the dropped tensors, and the corpus SHA-256s.

## Serving

The scales are in the checkpoint, so quantization is auto-detected and the cache dtype is
the only flag:

```bash
vllm serve out/exp-060-w4a16-fp8kv/checkpoint \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml
```

`--tool-call-parser qwen3_xml` is **required for tool calling on Qwen3.8** — it emits XML
tool calls, not JSON, and without the parser `tool_calls` is always empty, which reads as
a broken quant but is a serving flag.

If an MTP draft head survived the export, add speculative decoding:

```bash
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":2}'
```

## Benchmarking it

The GGUF bench cannot read this format — `bench/kld.py` shells out to
`llama-perplexity`, which is GGUF-only. Use the torch-side ladder instead:

```bash
uv run python scripts/run_hf_kld.py --help    # bench/kld_hf.py
```

Same *shape* as the GGUF ladder (six eval distributions, never concatenated, chunked at
`eval_ctx` 8192, median KLD + top-token agreement) but **not numerically comparable** to
it: the reference is the bf16 HF model rather than the F16 GGUF, and it tokenizes chat
control tokens to their real single ids. Report them in separate columns.

Two things worth measuring that only exist on this path:

- **fp8 KV is a second quantization.** Bench the checkpoint with and without
  `--kv-cache-dtype fp8_e4m3` at serve time; every published Δ that quotes a single number
  is measuring the *stack*, not the weights.
- **Long-context retrieval is where an fp8 cache would show up first.** Calibrating at
  32K and then only testing at 4K tests nothing. Run a needle check at the context length
  you intend to serve.

## Scheme support

| scheme | preset | custom weight grid |
|---|---|---|
| `W4A16` | ✅ | ✅ |
| `W8A16` | ✅ | ✅ |
| `W8A8` | ✅ | ❌ — also quantizes activations |
| `FP8_DYNAMIC` | ✅ | ❌ — also quantizes activations |

The activation-quantized schemes are supported at their preset settings only: a
hand-built config group for them would have to specify `input_activations` as well, and
silently dropping the override would be worse than refusing it. `--kv-cache-dtype` works
with all four.

Group size is restricted to `{-1, 32, 64, 128}` — what vLLM's packed-int4 kernels accept.
compressed-tensors would happily write any grouping, and the checkpoint would then fail at
`vllm serve`, long after the calibration is spent.
