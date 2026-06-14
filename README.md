# quant-tuner

**Calibrate your own GGUF quantizations from real usage logs — and push working models below 3 bits per weight.**

`quant-tuner` takes a HuggingFace model plus a corpus derived from your own
prompt/response logs and produces a GGUF quantization tuned to the distribution
your model actually sees. It then benchmarks the result against an FP16
reference (KL-divergence, perplexity, top-token agreement, prefill/decode
tok/s) and against task-level metrics (tool-call accuracy, MMLU-Pro). Every
output is a **standard GGUF** that runs on unmodified `llama.cpp` — calibration
changes *which* weights get the quantizer's budget, never the inference path.

---

## ⭐ Gemma-4-31B at ~2 bits per weight

| Feature | Details |
|---|---|
| **Base Model** | [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) |
| **QAT Base Model** | [`google/gemma-4-31B-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized) |
| **Hugging Face** | [`pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF`](https://huggingface.co/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF) |

### Benchmarks

| File | Quant | Technique | Size (GiB) | BPW | PPL | KLD (median) | top-p vs FP16 |
|---|---|---|---:|---:|---:|---:|---:|
| FP16 | FP16 | none (reference) | 57.20 | 16.005 | 277.89 | 0.000 | 100.0% |
| Q2_K-plain | Q2_K | plain (no imatrix, no AWQ) | 11.10 | 3.105 | 3370.57 | 5.147 | 25.8% |
| IQ2_XS-imatrix | IQ2_XS | imatrix only (baseline) | 8.88 | 2.484 | 12116.47 | 3.327 | 33.2% |
| **IQ2_XS-awq-cv-gate** | **IQ2_XS** | **AWQ cv-gate + imatrix** | **8.88** | **2.484** | **327.28** | **1.817** | **46.3%** |
| IQ2_M-imatrix | IQ2_M | imatrix only (baseline) | 10.17 | 2.845 | 2060.73 | 1.496 | 47.6% |
| **IQ2_M-awq-cv-gate** | **IQ2_M** | **AWQ cv-gate + imatrix** | **10.17** | **2.845** | **652.81** | **1.548** | **48.8%** |
| Q2_K_S-imatrix | Q2_K_S | imatrix only (baseline) | 10.22 | 2.861 | 1436.40 | 2.138 | 42.6% |
| **Q2_K_S-awq-cv-gate** | **Q2_K_S** | **AWQ cv-gate + imatrix** | **10.22** | **2.861** | **73.09** | **1.632** | **51.1%** |

#### QAT `google/gemma-4-31B-it-qat-q4_0-unquantized` source

| File | Quant | Technique | Size (GiB) | BPW | PPL | KLD (median) | top-p vs FP16 |
|---|---|---|---:|---:|---:|---:|---:|
| [Q4_0](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-gguf) | [Q4_0](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-gguf) | QAT (Google official, ref only) | 16.44 | 4.600 | 78.19 | 0.913 | 64.4% |
| qat-IQ2_XS-imatrix | IQ2_XS | imatrix only (from QAT) | 8.88 | 2.484 | 209.00 | 1.270 | 47.1% |
| **qat-IQ2_XS-awq-cv-gate** | **IQ2_XS** | **AWQ cv-gate + imatrix** | **8.88** | **2.484** | **108.65** | **1.151** | **48.9%** |
| qat-Q2_K_S-imatrix | Q2_K_S | imatrix only (from QAT) | 10.22 | 2.861 | 110.71 | 1.332 | 47.7% |
| **qat-Q2_K_S-awq-cv-gate** | **Q2_K_S** | **AWQ cv-gate + imatrix** | **10.22** | **2.861** | **88.18** | **1.081** | **49.4%** |

**Takeaways:**
- **AWQ cv-gate beats imatrix-only on PPL by 4–20×** at every working bit budget.
- **`qat-IQ2_XS-awq-cv-gate`** is the best size/quality pick at **8.88 GiB** (KLD 1.151, top-p 48.9%).
- Plain `Q2_K` (no calibration) loses ~25 absolute top-p points despite a *larger* file.

> ⚠️ These are sub-3-bpw quants of a 31B reasoning model. They are meaningfully
> better than the alternatives *at the same size*, but they are **not** a
> substitute for FP16 / Q4_K_M / Q5_K_M when you have the VRAM. Use them when
> memory is the binding constraint.

---

## How it works

Three calibration methods, all producing a standard GGUF with **zero inference cost**:

| Method   | What it does                                            | Pre-quant cost | Best for |
| -------- | ------------------------------------------------------- | -------------- | -------- |
| imatrix  | Per-tensor importance (`E[a²]`) steers the quantizer's bit budget | Low | 3–5 bpw baselines |
| AWQ      | Per-channel weight rescale folded into RMSNorm, flattening outliers | Medium | sub-3 bpw, where outliers wreck the codebook |
| GPTQ     | Hessian-based rounding with error compensation          | High           | when you want explicit PPL guardrails |

### Why AWQ wins at 2 bits

At 2-bit each weight has only ~4 codebook values, so a handful of outlier
channels can wreck quantization error for an entire layer. For every linear
`y = W · a`, AWQ picks a per-channel scale `s` and rewrites:

```
y  =  W · a  =  (W · diag(s))  ·  (diag(1/s) · a)
              └──────┬──────┘    └──────┬──────┘
              quantize this       fold into prev RMSNorm gain
```

Math-equivalent to the original layer, but the rescaled weight matrix has a
flatter per-channel range, so the 2-bit codebook fits it with far less error.
The inverse scale is absorbed into the preceding RMSNorm — **no runtime
overhead, the GGUF stays standard.** Imatrix and AWQ are complementary:
AWQ rewrites the weights, then a hybrid imatrix guides the final
`llama-quantize` pass.

### Three AWQ refinements that make sub-3-bpw work

1. **Per-tensor α refinement.** Each member of a group (q, k, v individually)
   nudges its α within a local grid around the shared group choice.
2. **Binary held-out gate (cv-gate).** A per-tensor α is only accepted if it
   *also* lowers proxy loss on a **disjoint validation slice** it never saw
   during the search. Without the gate, per-tensor refinement over-fits the
   calibration corpus at sub-3 bpw and PPL collapses on unseen text.
3. **Codebook-faithful proxy quantizers.** The α search scores candidates
   against bit-exact E8-lattice proxies (matching llama.cpp's
   `iq2xxs/xs/s_grid`), a `q2k_super` proxy for Q2_K_S, and `q2k_b16 + iq3_s`
   for IQ2_M's mixed tensors — so the optimum doesn't drift.

### Three disjoint corpora

| Slice | Source | Used for |
|---|---|---|
| **Calibration** | ~500k tokens usage-log + all of `wiki.test.raw` | imatrix collection + AWQ α search |
| **Validation** | out-of-distribution supplement (e.g. MMMU disciplines) | held-out gate for per-tensor α (binary accept/reject only) |
| **Eval** | ~90k tokens external code+math+tools ([`eaddario/imatrix-calibration`](https://huggingface.co/datasets/eaddario/imatrix-calibration)) | all PPL/KLD numbers; neither cal nor val appears here |

Validation never feeds the eval numbers, and eval is drawn from a third
distribution — so a winning α genuinely generalizes rather than re-fitting the
calibration mix.

---

## Quick start

```bash
# 1. Clone + fetch the pinned llama.cpp submodule
git clone <this repo> quant-tuner
cd quant-tuner
git submodule update --init --recursive

# 2. Build llama.cpp once. Metal on macOS; swap for -DGGML_CUDA=ON on Linux+NVIDIA.
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build -DGGML_METAL=ON
cmake --build vendor/llama.cpp/build -j

# 3. Install the Python package
uv sync
```

**Requirements:** Python ≥ 3.11 · [`uv`](https://github.com/astral-sh/uv) · a C++
toolchain for `llama.cpp` (Xcode CLT on macOS, build-essential on Linux).
Override the build location with `LLAMA_CPP_DIR=/path/to/your/build` to use a
system install.

## Running an end-to-end calibration

The CLI is recipe-driven. Each recipe under `src/quant_tuner/recipes/` declares
one calibration method × quant type:

```bash
# AWQ + hybrid imatrix at IQ2_XS (the Gemma sub-3-bpw recipe family)
uv run quant-tuner run \
    --recipe iq2_xs_awq \
    --model google/gemma-4-31B-it \
    --logs logtrain.jsonl \
    --workspace ./out/gemma_iq2xs

# Validate-only (resolves recipe + overrides, prints the merged config):
uv run quant-tuner run --recipe iq2_xs_awq --model X --logs Y --workspace W --dry-run

# Bench any GGUF against an FP16 reference → CSV row
uv run quant-tuner bench \
    --quant out/gemma_iq2xs/gguf/IQ2_XS-awq.gguf \
    --reference out/gemma_iq2xs/gguf/model-f16.gguf \
    --eval out/gemma_iq2xs/corpus/corpus.eval.txt \
    --out out/gemma_iq2xs/results.csv

# Aggregate to a markdown leaderboard with SQS scoring
uv run quant-tuner leaderboard --results out/gemma_iq2xs/results.csv --out LEADERBOARD.md
```

**Shipped recipes:**
- **4-bit baselines:** `q4_k_m_imatrix`, `q4_k_m_awq`, `q4_k_m_gptq`, `q4_k_m_none`
- **Low-bit (2–3 bpw):** `q2_k_awq`, `iq2_xs_awq`, `iq2_m_awq` (AWQ + hybrid imatrix, codebook proxies auto-selected); `q2_k_gptq`, `iq3_s_gptq`
- **Model-specific:** `q4_k_m_qwen3_5_4b`, `{q4_k_m,q5_k_s}_qwen3_6_mtp{,_awq,_none}`, `iq3_s_9b_mtp` (MTP heads retained)

A recipe is just YAML — copy any of them to a local file and pass the path to
`--recipe` to override the calibration variant, token budget, or sampling.

### Building the calibration corpora

`scripts/build_corpora.py` is the canonical one-pass builder: it writes
`corpus.cal.txt` (imatrix + AWQ α search), `corpus.val.txt` (held-out gate),
and `corpus.eval.txt` (external PPL/KLD), asserting the three slices are
disjoint before returning.

```bash
uv run python scripts/build_corpora.py --logs logtrain.jsonl --out out/corpora
```

## Task-level benchmarks

Two task evals report **mean ± stdev across N reps** (default 10), each spawning
one `llama-server` per model and reusing it across reps:

```bash
# Tool-calling on a held-out session corpus (built from logtrain.jsonl)
uv run python scripts/run_toolcall_reps.py \
    --models out/run/model-f16.gguf out/run/IQ2_XS-awq.gguf --reps 10

# MMLU-Pro CS + math (25 questions per subject, 2-shot)
uv run python scripts/build_mmlu_pro_holdout.py   # one-time, downloads dataset
uv run python scripts/run_mmlu_pro_reps.py \
    --models out/run/model-f16.gguf out/run/IQ2_XS-awq.gguf --reps 10
```

Plug a new benchmark in by writing a `dict[str, float]`-returning adapter on top
of `quant_tuner.eval.reps.run_reps_for_models`.

## Serving a quant

These are plain GGUFs — load them in `llama-server`, LM Studio, or anything that
reads GGUF:

```bash
./llama-server \
    --model gemma-4-31B-it-qat-Q2_K_S-awq-cv-gate.gguf \
    --ctx-size 16384 --n-gpu-layers 999 \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
    --host 0.0.0.0 --port 1234
```

```python
import json, urllib.request

def ask(content, max_tokens=256):
    body = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        # Gemma 4 is a thinking model — set this False (or raise max_tokens),
        # else the reply lands in reasoning_content and "content" is empty.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request("http://127.0.0.1:1234/v1/chat/completions",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())["choices"][0]["message"]["content"]

print(ask("What is 1+1?"))
```

## Pipeline at a glance

```
   HF model         usage logs (.jsonl)
       │                     │
       ▼                     ▼
   extract             ingest + split
       │                     │
       │            cal.txt | val.txt | eval.txt | holdout.jsonl
       │                     │
       └──── HF → F16 GGUF ◄─┘
                  │
                  │  calibrate (imatrix | AWQ | GPTQ)
                  ▼
       imatrix.gguf  /  awq.pt  /  hf_model_gptq/
                  │
                  │  llama-quantize  (--type IQ2_XS, Q2_K_S, Q4_K_M, …)
                  ▼
              model.gguf
                  │
                  │  bench (KLD vs FP16, PPL, BPW, top-p, prefill/decode tok/s)
                  ▼
              results.csv  →  LEADERBOARD.md
```

`pipeline.run_pipeline(RunConfig)` is the canonical end-to-end flow; every stage
is idempotent (existence-based via `experiments.step`), so re-running a populated
workspace just verifies state and re-renders.

## Layout

```
src/quant_tuner/
├── calibrate/        # imatrix | awq | gptq calibrators (+ generated IQ2 codebook grids)
├── quantize/         # HF → F16 GGUF, F16 → Q* GGUF
├── bench/            # bpw | kld | speed | runner (CSV row builder)
├── data/             # log ingest, stratified packing, train/test/holdout split
├── eval/             # task-level evals (toolcall, mmlu_pro) + generic N-rep runner
├── experiments/      # shared log/phase/step helpers for driver scripts
├── leaderboard/      # CSV → markdown aggregation with SQS scoring
├── models/           # HF extract, llama.cpp binary wrappers, HF→GGUF name map
├── recipes/          # YAML recipes consumed by `quant-tuner run --recipe ...`
├── cli.py            # typer CLI: run | bench | leaderboard
└── pipeline.py       # end-to-end: extract → calibrate → quantize → bench

vendor/llama.cpp      # pinned submodule, commit 32782998
scripts/              # corpus builders, experiment drivers, leaderboard reproducer
tests/unit/           # 100+ passing tests
```

## Development

```bash
uv run pytest                    # all unit tests
uv run ruff check src tests      # lint
uv run mypy src                  # types
```

## Status

Beta. End-to-end calibration runs via the CLI (`quant-tuner run --recipe …`) or
the Python API. The Gemma-4-31B sub-3-bpw release above is reproducible from the
shipped `iq2_xs_awq` / `iq2_m_awq` / `q2_k_awq` recipes plus
`scripts/build_corpora.py`.

## License & attribution

- Released quantizations inherit their base model's license (e.g. the
  [Gemma Terms of Use](https://ai.google.dev/gemma/terms) for the Gemma release).
- Quantization performed locally with **quant-tuner** + vendored
  [llama.cpp](https://github.com/ggerganov/llama.cpp) pinned to commit `32782998`.
- Usage-log calibration data scraped with [**LogMiner**](https://github.com/pearsonkyle/LogMiner).
