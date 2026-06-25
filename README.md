# quant-tuner

**Calibrate your own GGUF quantizations from real usage logs — and push working
models below 3 bits per weight.**

`quant-tuner` takes a HuggingFace model plus a corpus derived from your own
prompt/response logs and produces a GGUF tuned to the distribution your model
actually sees. It benchmarks the result against FP16 (KL-divergence, perplexity,
top-token agreement, tok/s) and against task-level metrics (tool-call accuracy,
MMLU-Pro, **agentic SWE-rebench pass rate**). Every output is a **standard GGUF**
that runs on unmodified `llama.cpp` — calibration changes *which* weights get the
quantizer's budget, never the inference path.

---

## Released quantizations

Three sub-3-bpw release families calibrated with this tool, all on real
agentic-coding logs (English + Python):

| Release | Method | Headline result |
|---|---|---|
| [`gemma-4-31b-it-imatrix-GGUF`](https://huggingface.co/pearsonkyle/gemma-4-31b-it-imatrix-GGUF) | hybrid imatrix (AWQ studied) | IQ2_M (2.85 bpw) **solves 40% of SWE-rebench**, IQ4_XS 47% @ 100% patch rate |
| [`Qwopus3.6-27B-Coder-2bit-MTP-GGUF`](https://huggingface.co/pearsonkyle/Qwopus3.6-27B-Coder-2bit-MTP-GGUF) | hybrid imatrix + bundled MTP | IQ2_M **63% SWE-rebench pass — matches the 5-bit Q5_K_M (57%)**; IQ4_XS KLD 0.004 / top_p 94% |
| [`tmax-27b-imatrix-MTP-GGUF`](https://huggingface.co/pearsonkyle/tmax-27b-imatrix-MTP-GGUF) | hybrid imatrix + grafted MTP | IQ2_XS / Q2_K_S / IQ3_M / IQ4_XS all **resolve 7/10** SWE-rebench; Q5_K_M KLD 0.0014 |

> These are aggressive sub-5-bpw quants of 27–31B models. They beat the
> alternatives *at the same size*, but they are **not** a substitute for FP16 /
> Q5_K_M when you have the VRAM. Reach for them when memory is the binding
> constraint.

---

## The methods

Three calibration methods, all producing a standard GGUF with **zero inference cost**:

| Method | What it does | Best for | Key metric |
|---|---|---|---|
| **imatrix** | Per-tensor importance (`E[a²]`) steers the quantizer's bit budget | the default; 2–5 bpw | Qwopus IQ4_XS near-lossless (KLD **0.004**); IQ2_M at 2 bits matches a 5-bit quant on agentic coding |
| **AWQ** | Per-channel weight rescale folded into RMSNorm, flattening outliers | the hardest sub-3 bpw, where a few outlier channels wreck the codebook | Gemma QAT Q2_K_S: KLD **1.22 → 1.09**, PPL 125 → 120 (vanilla source: PPL **1436 → 73**) |
| **GPTQ** | Hessian-based rounding with error compensation | when you want an explicit PPL guardrail | auto-relaxed bits-aware PPL/logit bounds at 2–3 bits |

Plus **MTP**: bundle the model's own multi-token-prediction draft head at Q8 for
built-in speculative decoding — **1.26× decode** on Metal (Qwopus, 79.9% accept),
up to **95.6% draft acceptance** at n=1.

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

### Three disjoint corpora

| Slice | Source | Used for |
|---|---|---|
| **Calibration** | ~500k tokens usage-log + all of `wiki.test.raw` | imatrix collection + AWQ α search |
| **Validation** | out-of-distribution supplement | held-out gate for per-tensor α (accept/reject only) |
| **Eval** | external code+math+tools ([`eaddario/imatrix-calibration`](https://huggingface.co/datasets/eaddario/imatrix-calibration)) | all PPL/KLD numbers; neither cal nor val appears here |

Eval is drawn from a third distribution, so a winning α genuinely generalizes
rather than re-fitting the calibration mix. The agentic SWE-rebench holdout is a
wholly separate dataset whose issues never touch any calibration corpus.

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
one method × quant type.

```bash
# AWQ + hybrid imatrix at IQ2_XS
uv run quant-tuner run --recipe iq2_xs_awq \
    --model google/gemma-4-31B-it --logs logtrain.jsonl --workspace ./out/run

# Validate-only (resolves recipe + overrides, prints merged config)
uv run quant-tuner run --recipe iq2_xs_awq --model X --logs Y --workspace W --dry-run

# Bench any GGUF against FP16 → CSV row, then aggregate to markdown
uv run quant-tuner bench --quant out/run/gguf/IQ2_XS-awq.gguf \
    --reference out/run/gguf/model-f16.gguf \
    --eval out/run/corpus/corpus.eval.txt --out out/run/results.csv
uv run quant-tuner leaderboard --results out/run/results.csv --out LEADERBOARD.md
```

**Shipped recipes:**
- **4-bit baselines:** `q4_k_m_{imatrix,awq,gptq,none}`
- **Low-bit (2–3 bpw):** `q2_k_awq`, `iq2_xs_awq`, `iq2_m_awq` (AWQ + hybrid imatrix, codebook proxies auto-selected); `q2_k_gptq`, `iq3_s_gptq`
- **Model-specific:** `q4_k_m_qwen3_5_4b`, `{q4_k_m,q5_k_s}_qwen3_6_mtp{,_awq,_none}`, `iq3_s_9b_mtp` (MTP heads retained)

A recipe is just YAML — copy any of them and pass the path to `--recipe` to
override the variant, token budget, or sampling.

### Building the calibration corpora

`scripts/build_corpora.py` is the canonical one-pass builder: it writes the
calibration / validation / eval corpora and asserts the three slices are disjoint
before returning.

```bash
uv run python scripts/build_corpora.py --logs logtrain.jsonl --out out/corpora
```

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

Plug in a new benchmark by writing a `dict[str, float]`-returning adapter on top
of `quant_tuner.eval.reps.run_reps_for_models`.

### Agentic benchmark (SWE-rebench)

KLD and tool-call accuracy tell you a quant *emits valid tokens*; they don't tell
you it can **read a repo, run commands, edit code, and land a working patch**.
This benchmark drives [`mini-swe-agent`](https://github.com/SWE-agent/mini-swe-agent)
or the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) against
real issues from [`nebius/SWE-rebench`](https://huggingface.co/datasets/nebius/SWE-rebench),
each in a **clean per-instance Docker container**, and grades a true **pass rate**
by running the gold `FAIL_TO_PASS` / `PASS_TO_PASS` tests. Every trajectory is
saved for inspection and future training data.

```bash
uv sync --extra swebench           # installs mini-swe-agent; needs a running Docker daemon

# Build a holdout (default 10 lite instances, seed 42 — pages HF's API, no multi-GB download)
PYTHONPATH=src .venv/bin/python scripts/build_swebench_holdout.py --n 10

# Run it (fails fast if Docker is down)
PYTHONPATH=src .venv/bin/python scripts/run_swebench_eval.py --progress \
    --models out/run/gguf/IQ2_M.gguf
```

Outputs land under `--workspace`: `results.csv` (per-instance), `aggregated.csv`
(per-model), `summary.json`, and `trajectories/<model>/<id>.traj.json`.

> SWE-rebench ships linux/amd64 images; on Apple Silicon they run under emulation
> (correct but slow). The `swebench` deps live in the uv-managed `.venv`, so these
> scripts run via `PYTHONPATH=src .venv/bin/python …`, not `uv run`.

## Serving a quant

Plain GGUFs — load them in `llama-server`, LM Studio, Ollama, or anything that
reads GGUF. Add `--spec-type draft-mtp --spec-draft-n-max 1` for MTP releases.

```bash
./llama-server --model gemma-4-31B-it-IQ2_M.gguf \
    --ctx-size 16384 --n-gpu-layers 999 --flash-attn on \
    --cache-type-k q8_0 --cache-type-v q8_0 --host 0.0.0.0 --port 1234
```

```python
import json, urllib.request

def ask(content, max_tokens=256):
    body = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        # These are thinking models — set False (or raise max_tokens), else the
        # reply lands in reasoning_content and "content" is empty.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request("http://127.0.0.1:1234/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())["choices"][0]["message"]["content"]

print(ask("What is 1+1?"))
```

## Pipeline at a glance

```
HF model + usage logs (.jsonl)
   → extract            HF → F16 GGUF
   → prepare corpora    cal | val | eval | holdout
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
├── data/        # log ingest, stratified packing, train/test/holdout split
├── eval/        # task-level evals (toolcall, mmlu_pro, swebench) + N-rep runner
├── leaderboard/ # CSV → markdown aggregation with SQS scoring
├── models/      # HF extract, llama.cpp wrappers, HF→GGUF name map
├── recipes/     # YAML recipes consumed by `quant-tuner run --recipe ...`
├── cli.py       # typer CLI: run | bench | leaderboard
└── pipeline.py  # end-to-end: extract → calibrate → quantize → bench

vendor/llama.cpp # pinned submodule
scripts/         # corpus builders, experiment drivers, leaderboard reproducer
tests/unit/      # 100+ passing tests
```

## Development

```bash
uv run pytest                 # all unit tests
uv run ruff check src tests   # lint
uv run mypy src               # types
```

## License & attribution

- Released quantizations inherit their base model's license.
- Quantization performed locally with **quant-tuner** + vendored
  [llama.cpp](https://github.com/ggerganov/llama.cpp).
- Usage-log calibration data scraped with [**LogMiner**](https://github.com/pearsonkyle/LogMiner).
