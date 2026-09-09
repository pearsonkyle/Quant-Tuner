# quant-tuner

**Calibrate your own GGUF quantizations and vLLM W4A16 checkpoints from a
corpus derived from real agentic-coding usage logs — and push working models
below 3 bits per weight.**

`quant-tuner` takes a HuggingFace model plus a corpus derived from your own
prompt/response logs and produces a GGUF (or a compressed-tensors W4A16
checkpoint for vLLM) tuned to the distribution your model actually sees. It
benchmarks the result against FP16 (KL-divergence, perplexity, top-token
agreement, tok/s) and against task-level metrics (tool-call accuracy,
MMLU-Pro, **agentic SWE-rebench pass rate**). Every GGUF output is a
**standard GGUF** that runs on unmodified `llama.cpp` — calibration changes
*which* weights get the quantizer's budget, never the inference path.

---

## Capabilities

Released quants, all calibrated on the
[llmtk SFT corpus](https://huggingface.co/datasets/pearsonkyle/llmtk-sft-corpus-v2)
(15M tokens, 65 536 ctx, the 32K-ctx slice) and the agentic SWE-rebench
holdout:

| Model | GGUF | vLLM W4A16 | Notes |
|---|---|---|---|
| Qwen3.8-27B | [imatrix + MTP](https://huggingface.co/pearsonkyle/Qwen3.8-27B-imatrix-MTP-GGUF) | [GPTQ W4A16](https://huggingface.co/pearsonkyle/Qwen3.8-27B-GPTQ-W4A16) | IQ2_M / IQ3_M / IQ4_XS / Q5_K_M rungs; W4A16 deploys on an RTX 3090 |
| Qwopus3.6-27B-Coder-2bit-MTP | [GGUF](https://huggingface.co/pearsonkyle/Qwopus3.6-27B-Coder-2bit-MTP-GGUF) | — | IQ2_M **63% SWE-rebench pass — matches the 5-bit Q5_K_M (57%)** |
| tmax-27b-imatrix-MTP | [GGUF](https://huggingface.co/pearsonkyle/tmax-27b-imatrix-MTP-GGUF) | — | IQ2_XS / Q2_K_S / IQ3_M / IQ4_XS all **resolve 7/10** SWE-rebench |
| gemma-4-31b-it | [GGUF](https://huggingface.co/pearsonkyle/gemma-4-31b-it-imatrix-GGUF) | — | IQ2_M (2.85 bpw) **solves 40% of SWE-rebench** |

Browse the full family at the [Qwen collection](https://huggingface.co/collections/pearsonkyle/qwen).

> These are aggressive sub-5-bpw quants of 27–31B models. They beat the
> alternatives *at the same size*, but they are **not** a substitute for FP16 /
> Q5_K_M when you have the VRAM. Reach for them when memory is the binding
> constraint.

## The methods

Three calibration methods, all producing a standard GGUF with **zero inference
cost**:

| Method | What it does | Best for | Key metric |
|---|---|---|---|
| **imatrix** | Per-tensor importance (`E[a²]`) steers the quantizer's bit budget | the default; 2–5 bpw | Qwopus IQ4_XS near-lossless (KLD **0.004**); IQ2_M at 2 bits matches a 5-bit quant on agentic coding |
| **AWQ** | Per-channel weight rescale folded into RMSNorm, flattening outliers | the hardest sub-3 bpw, where a few outlier channels wreck the codebook | Gemma QAT Q2_K_S: KLD **1.22 → 1.09**, PPL 125 → 120 (vanilla source: PPL **1436 → 73**) |
| **GPTQ** | Hessian-based rounding with error compensation | when you want an explicit PPL guardrail, or a W4A16 vLLM checkpoint | auto-relaxed bits-aware PPL/logit bounds at 2–3 bits; W4A16 deploys on an RTX 3090 |

Plus **MTP**: bundle the model's own multi-token-prediction draft head at Q8 for
built-in speculative decoding — **1.26× decode** on Metal (Qwopus, 79.9% accept),
up to **95.6% draft acceptance** at n=1.

Plus **QAT for natively-ternary models** (`qat/`): for the Qwen-Bonsai family
and the `prism-ml/Ternary-Bonsai-8B` reference, post-hoc calibration is a
structural no-op (the "F16" is a lossless container of `w = s·c`, `c ∈
{−1,0,+1}`). The only lever is more training with the ternarization in the loop
(TWN straight-through estimator). See [`docs/ternary_qat.md`](ternary_qat.md).

### Why AWQ helps at 2 bits

At 2-bit each weight has only ~4 codebook values, so a few outlier channels can
wreck quantization error for an entire layer. For every linear `y = W · a`, AWQ
picks a per-channel scale `s` and rewrites:

```
y  =  W · a  =  (W · diag(s))  ·  (diag(1/s) · a)
              └──────┬──────┘    └──────┬──────┘
              quantize this       fold into prev RMSNorm gain
```

Math-equivalent to the original layer, but the rescaled weight matrix has a
flatter per-channel range, so the 2-bit codebook fits it with far less error —
**no runtime overhead, the GGUF stays standard.** Three refinements make it work
below 3 bpw: **per-tensor α refinement**, a **held-out gate** (a per-tensor α is
accepted only if it also lowers loss on a disjoint validation slice — without it,
sub-3-bpw α over-fits and PPL collapses on unseen text), and **codebook-faithful
proxy quantizers** (bit-exact E8-lattice proxies matching llama.cpp's grids).

> Caveat from the Gemma study: AWQ wins on static PPL/KLD at every sub-3-bpw
> budget, but on **agentic** SWE-rebench the plain hybrid-imatrix IQ2_M solved
> more issues than its AWQ variant — so imatrix shipped. Static gains don't
> always translate to agentic gains; benchmark the metric you care about.

### The calibration corpora

| Slice | Source | Used for |
|---|---|---|
| **Calibration** | llmtk SFT 32K slice + usage logs + SWE trajectories + broad supplement | imatrix collection + AWQ α search + GPTQ Hessians |
| **Validation** | out-of-distribution supplement | held-out gate for per-tensor α (accept/reject only) |
| **Eval** | external code+math+tools ([`eaddario/imatrix-calibration`](https://huggingface.co/datasets/eaddario/imatrix-calibration)) | all PPL/KLD numbers; neither cal nor val appears here |

The agentic SWE-rebench holdout is a wholly separate dataset whose issues never
touch any calibration corpus. See
[`docs/calibration_datasets.md`](calibration_datasets.md) for the full
source list, the 32K-ctx packing invariants, and how to point the pipeline at a
pre-built corpus.

---

## Quick start

```bash
# 1. Clone + fetch the pinned llama.cpp submodule
git clone <this repo> quant-tuner && cd quant-tuner
git submodule update --init --recursive

# 2. Build llama.cpp once (Metal on macOS; -DGGML_CUDA=ON on Linux+NVIDIA)
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build -DGGML_METAL=ON
cmake --build vendor/llama.cpp/build -j

# 3. Install the Python package
uv sync
```

**Requirements:** Python ≥ 3.11 · [`uv`](https://github.com/astral-sh/uv) · a C++
toolchain for `llama.cpp`. Override the build location with
`LLAMA_CPP_DIR=/path/to/your/build` to use a system install.

## Running a calibration

The CLI is recipe-driven; each recipe under `src/quant_tuner/recipes/` declares
one method × quant type. The 32K-ctx recipes are the default for new models.

```bash
# GGUF: hybrid imatrix at IQ2_M, 32K ctx
uv run quant-tuner run --recipe iq2m_imatrix_32k \
    --model Qwen/Qwen3.5-8B --logs logs.jsonl --workspace ./out/run

# GGUF: GPTQ at IQ3_M, 32K ctx (Hessian calibration + PPL guardrail)
uv run quant-tuner run --recipe iq3m_gptq_32k \
    --model Qwen/Qwen3.5-8B --logs logs.jsonl --workspace ./out/run

# vLLM: W4A16 compressed-tensors, 32K ctx (no llama.cpp in the loop)
uv sync --extra vllm-ptq
uv run python examples/gptq_w4a16.py --model Qwen/Qwen3.5-8B \
    --corpus out/corpora/corpus.cal.txt --out out/qwen3.5-w4a16

# Validate-only (resolves recipe + overrides, prints merged config)
uv run quant-tuner run --recipe iq2m_imatrix_32k --model X --logs Y --workspace W --dry-run

# Bench any GGUF against FP16 → CSV row, then aggregate to markdown
uv run quant-tuner bench --quant out/run/gguf/IQ2_M-imatrix-hybrid_custom.gguf \
    --reference out/run/gguf/model-f16.gguf \
    --eval out/run/corpus/eval.txt --out out/run/results.csv
uv run quant-tuner leaderboard --results out/run/results.csv --out LEADERBOARD.md
```

**Shipped recipes:**
- **32K-ctx (new models):** `iq2m_imatrix_32k`, `iq2m_gptq_32k`,
  `iq3m_imatrix_32k`, `iq3m_gptq_32k`, `iq4xs_imatrix_32k`, `iq4xs_gptq_32k`,
  `q5km_imatrix_32k`, `q5km_gptq_32k`
- **4-bit baselines:** `q4_k_m_{imatrix,awq,gptq,none}`
- **Low-bit (2–3 bpw):** `q2_k_awq`, `iq2_xs_awq`, `iq2_m_awq` (AWQ + hybrid
  imatrix, codebook proxies auto-selected); `q2_k_gptq`, `iq3_s_gptq`
- **GPTQ ladder (2–4.5 bpw):** `iq2_xs_gptq`, `iq2_m_gptq`, `iq3_m_gptq`,
  `iq4_xs_gptq` (+ the two above and `q4_k_m_gptq`) — per-tensor grid mix
  matching llama-quantize's tensor bumps, imatrix collected on the rounded F16
- **Model-specific:** `q4_k_m_qwen3_5_4b`, `{q4_k_m,q5_k_s}_qwen3_6_mtp{,_awq,_none}`,
  `iq3_s_9b_mtp` (MTP heads retained)

A recipe is just YAML — copy any of them and pass the path to `--recipe` to
override the variant, token budget, or sampling. See
[`docs/recipes.md`](recipes.md) for the full recipe reference, including
the AWQ codebook proxies and the GPTQ grid mix.

### Building the calibration corpora

`scripts/build_universal_corpus.py` is the canonical builder for new models:
it writes the calibration / validation / eval corpora, asserts the slices are
disjoint, and includes the llmtk SFT 32K slice by default.

```bash
uv run python scripts/build_universal_corpus.py \
    --out out/corpora --model out/run/model_extracted --ctx 32768
```

`scripts/build_corpora.py` remains for reproducing the published two-source
runs.

## Task-level benchmarks

Two task evals report **mean ± stdev across N reps** (default 10), each spawning
one `llama-server` per model:

```bash
# Tool-calling on a held-out session corpus
uv run python scripts/run_toolcall_reps.py --models a.gguf b.gguf --reps 10

# MMLU-Pro CS + math (25 questions/subject, 2-shot)
uv run python scripts/build_mmlu_pro_holdout.py    # one-time, downloads dataset
uv run python scripts/run_mmlu_pro_reps.py --models a.gguf b.gguf --reps 10
```

### Agentic benchmark (SWE-rebench)

KLD and tool-call accuracy tell you a quant *emits valid tokens*; they don't tell
you it can **read a repo, run commands, edit code, and land a working patch**.
This benchmark drives [`mini-swe-agent`](https://github.com/SWE-agent/mini-swe-agent)
or the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) against
real issues from [`nebius/SWE-rebench`](https://huggingface.co/datasets/nebius/SWE-rebench),
each in a **clean per-instance Docker container**, and grades a true **pass rate**
by running the gold `FAIL_TO_PASS` / `PASS_TO_PASS` tests.

```bash
uv sync --extra swebench           # installs mini-swe-agent; needs a running Docker daemon
PYTHONPATH=src .venv/bin/python scripts/build_swebench_holdout.py --n 10
PYTHONPATH=src .venv/bin/python scripts/run_swebench_eval.py --progress \
    --models out/run/gguf/IQ2_M.gguf
```

## Serving a quant

**GGUF** — load in `llama-server`, LM Studio, Ollama, or anything that reads GGUF.
Add `--spec-type draft-mtp --spec-draft-n-max 1` for MTP releases.

```bash
./llama-server --model qwen3.5-8B-IQ2_M.gguf \
    --ctx-size 32768 --n-gpu-layers 999 --flash-attn on \
    --cache-type-k q8_0 --cache-type-v q8_0 --host 0.0.0.0 --port 1234
```

**vLLM** — the W4A16 checkpoint is a plain safetensors dir; vLLM auto-detects the
compressed-tensors config.

```bash
vllm serve out/qwen3.5-w4a16 --max-model-len 32768 --tensor-parallel-size 1
```

## vLLM instead of GGUF (W4A16 + fp8 KV cache)

The GGUF ladder's sibling: a **compressed-tensors INT4 checkpoint vLLM serves
directly**, calibrated on the same corpora, with **fp8 KV-cache scales baked in**.

Quantizing weights saves memory once. Quantizing the KV cache saves it *per token
per sequence* — which is what actually sets how long a context fits and how many
requests run concurrently. On a 27B model at 262k context that is ~17 GB of bf16
KV versus ~8.5 GB of fp8.

```bash
uv sync --extra vllm-ptq

# Always dry-run a new model first: prints the resolved recipe, how many live
# modules each ignore pattern matches, and the tensors that would vanish.
uv run python scripts/run_vllm_ptq.py --model <hf-dir> --dry-run-ignore

uv run python scripts/run_vllm_ptq.py \
    --model <hf-dir> --corpus corpus.cal.txt --out out/w4a16-fp8kv \
    --ctx 32768 --budget-tokens 4194304 \
    --group-size 32 --asymmetric --observer imatrix-mse --actorder static \
    --kv-cache-dtype fp8_e4m3

vllm serve out/w4a16-fp8kv --kv-cache-dtype fp8_e4m3 --max-model-len 262144
```

`imatrix-mse` is the direct analogue of the GGUF ladder's imatrix — it weights
the MSE clipping search by per-input-channel activation importance rather than
taking raw range endpoints. The export is verified before it is written: a
missing `quantization_config` (vLLM would serve bf16 at full size) and a
requested KV scheme that produced **no** `k_scale`/`v_scale` tensors both fail
loudly, because each otherwise yields a checkpoint that loads and serves
perfectly while being wrong.

See [`docs/vllm_w4a16_fp8kv.md`](docs/vllm_w4a16_fp8kv.md) for the full guide.

## Inspecting a quant (Jacobian lens)

Beyond the leaderboard numbers, `quant-tuner lens` opens up the *inside* of a
quant with the [Jacobian lens](https://transformer-circuits.pub/2026/workspace/index.html):
layer-by-layer readouts, where tool-call decisions form, why a 2-bit model
loops, what heavy quantization erased versus merely suppressed, and an A/B diff
between two quants. If a runtime steer/ablate edit fixes a problem, it can be
baked back into the GGUF.

```bash
uv run quant-tuner lens build-server
uv run quant-tuner lens fit --model model-f16.gguf --corpus corpus.cal.txt -o lens.gguf
uv run quant-tuner lens serve --model quant.gguf --lens lens.gguf \
    --runs-dir out/lens/runs --model-b model-f16.gguf   # → http://127.0.0.1:8090
```

See [`docs/lens.md`](lens.md) for the full guide.

## Pipeline at a glance

```
HF model + usage logs (.jsonl)
   → extract            HF → F16 GGUF
   → prepare corpora    cal | val | eval | holdout   (llmtk SFT 32K slice included)
   → calibrate          imatrix | AWQ | GPTQ
   → quantize           F16 → Q* GGUF (llama-quantize)
   → bench              KLD vs FP16, PPL, BPW, top-p, tok/s → results.csv
   → leaderboard        results.csv → LEADERBOARD.md
```

`pipeline.run_pipeline(RunConfig)` is the canonical end-to-end flow; every stage
is idempotent (existence-based via `experiments.step`), so re-running a populated
workspace just verifies state and re-renders.

## Layout

```
src/quant_tuner/
├── calibrate/   # imatrix | awq | gptq calibrators (+ generated IQ2 codebook grids)
├── quantize/    # HF → F16 GGUF, F16 → Q* GGUF
├── bench/       # bpw | kld | speed | runner (CSV row builder)
├── data/        # log ingest, stratified packing, train/test/holdout split, universal corpus
├── eval/        # task-level evals (toolcall, mmlu_pro, swebench, redteam) + N-rep runner
├── leaderboard/ # CSV → markdown aggregation with SQS scoring
├── models/      # HF extract, llama.cpp wrappers, HF→GGUF name map
├── lens/        # Jacobian-lens interpretability (fit, capture, diff, probes, bake) + D3 UI
├── qat/         # continued QAT for natively-ternary models (TWN STE, stop probe, KD table)
├── vllm_export/ # W4A16 compressed-tensors PTQ for vLLM serving
├── drafter/     # MTP draft-head fine-tuning on usage logs
├── datasets/    # publishable dataset specs (swe-agentic-trajectories, redteam disclosures)
├── recipes/     # YAML recipes consumed by `quant-tuner run --recipe ...`
├── cli.py       # typer CLI: run | bench | leaderboard | lens
└── pipeline.py  # end-to-end: extract → calibrate → quantize → bench

examples/        # three one-file entry points (imatrix_gguf, gptq_w4a16, ternary_qat)
scripts/         # corpus builders, experiment drivers, leaderboard reproducer
tests/unit/      # 100+ passing tests
```

## Development

```bash
uv run pytest                 # all unit tests
uv run ruff check src tests   # lint
uv run mypy src               # types
```

The docs site (this README + `docs/`) is built with mkdocs-material and published
to GitHub Pages on push to `main` (see `.github/workflows/ci.yml`).

## License & attribution

- Released quantizations inherit their base model's license.
- Quantization performed locally with **quant-tuner** + vendored
  [llama.cpp](https://github.com/ggerganov/llama.cpp).
- Usage-log calibration data scraped with [**LogMiner**](https://github.com/pearsonkyle/LogMiner).
- The Jacobian-lens tooling (`src/quant_tuner/lens/`, `native/jlens_server/`) is
  adapted from [jlens-gguf](https://github.com/igorbarshteyn/jlens-gguf) and
  [jacobian-lens](https://github.com/anthropics/jacobian-lens) (both Apache-2.0);
  see the root [`NOTICE`](NOTICE) for details.
