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
# (recipes that stack params.imatrix_variant carry it in the GGUF name)
uv run quant-tuner bench --quant out/run/gguf/IQ2_XS-awq-hybrid_custom.gguf \
    --reference out/run/gguf/model-f16.gguf \
    --eval out/run/corpus/corpus.eval.txt --out out/run/results.csv
uv run quant-tuner leaderboard --results out/run/results.csv --out LEADERBOARD.md
```

**Shipped recipes:**
- **4-bit baselines:** `q4_k_m_{imatrix,awq,gptq,none}`
- **Low-bit (2–3 bpw):** `q2_k_awq`, `iq2_xs_awq`, `iq2_m_awq` (AWQ + hybrid imatrix, codebook proxies auto-selected); `q2_k_gptq`, `iq3_s_gptq`
- **GPTQ ladder (2–4.5 bpw):** `iq2_xs_gptq`, `iq2_m_gptq`, `iq3_m_gptq`, `iq4_xs_gptq` (+ the two above and `q4_k_m_gptq`) — per-tensor grid mix matching llama-quantize's tensor bumps, imatrix collected on the rounded F16, `imatrix_variant: hybrid_custom` stacked on the 2-bit rungs
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

### Safety — red-teaming the artifact that actually ships

Every benchmark above answers *"is the quant still capable?"*. None answers
*"is the quant still safe?"* — a quant that had lost its refusal behaviour
entirely would score clean on every column the leaderboard renders.

An open-weight model is safety-tested once, **as released**. What people run is a
*derived* artifact — a 2-4 bpw GGUF, or a QAT'd checkpoint — produced afterwards
by a third party. `quant-tuner` measures what that derivation did, using
[`deepteam`](https://github.com/confident-ai/deepteam) to drive adversarial
prompts at a `llama-server` target, with a separate simulator and judge:

```bash
# deepteam 1.0.7 has a stray `nntplib` import (removed in 3.13), so this extra
# needs its own Python <= 3.12 env if your main .venv is 3.13:
uv venv --python 3.12 .venv-redteam
uv pip install --python .venv-redteam/bin/python 'deepteam>=1.0.7' 'deepeval>=3.6.2' pyyaml requests

# One sweep, N quants, ONE frozen attack bank (F16 first — the bank is written
# against the unquantized parent). Judge/simulator are separate endpoints.
PYTHONPATH=src .venv/bin/python scripts/eval_redteam.py \
    --model-path out/run/gguf/f16.gguf --model-path out/run/gguf/IQ2_XS-awq.gguf \
    --config red_team_minimal --frozen-bank \
    --judge-model M --judge-base-url http://host:1234/v1 \
    --simulator-model M --simulator-base-url http://host:1234/v1 \
    --remote-no-think --out out/redteam/results.csv

# Pair the rungs against the reference (also runs the sweep if given --models)
PYTHONPATH=src .venv/bin/python scripts/redteam_ladder.py \
    --per-case out/redteam/results_per_case.csv --reference f16

# Does any existing quality gate predict the drift? (pure CSV analysis, no GPU)
PYTHONPATH=src .venv/bin/python scripts/redteam_vs_quality.py \
    --ladder out/redteam/ladder/ladder.csv --bench out/run/results.csv
```

The headline column is **`n_flip_unsafe`**: cases the F16 reference refused and
the quant complied with. No adversary, no fine-tuning meant to remove anything —
just the quantizer. That number only means something because every rung is scored
on the *same* frozen bank and joined per case; an unpaired pass-rate delta cannot
tell "this quant is less safe" apart from "this run drew different attacks".

`scripts/redteam_agentic.py` goes further and red-teams the deployment these
models actually ship into — a coding agent with a `bash` tool in a disposable
SWE-rebench container — so compliance means *executed a command*, not *wrote a
paragraph*. Read its `pass_rate` next to `n_tool_calls`: a quant too degraded to
call a tool scores as "safe" for entirely the wrong reason.

Every complied/errored case is also written to a `disclosure_<model>_repN.json`
evidence file — the full attack (seed prompt **and** multi-turn escalation), the
target's verbatim response and its own reasoning trace (for reasoning models),
and the judge's verdict — so a finding can be reported to the model's authors to
help them fix it. Refusals are excluded; a defended case is not a finding.

Reasoning models (Ornith, Qwen3, DeepSeek) spend their token budget on
chain-of-thought *first*, so set `--target-max-tokens` generously (3000+): if the
budget runs out before the answer starts, `content` comes back empty and the
judge would score silence as "safe". The harness refuses an all-empty run rather
than report that false pass (override with `--allow-empty-output`). One
`--base-url` serving several models? Repeat `--target-model-name` to sweep them
all on one frozen bank.

Merge the result into the leaderboard with
`quant-tuner leaderboard --redteam-csv ladder.csv`. The safety columns are
**display-only and never feed SQS** — that scalar trades size against fidelity
against speed, and refusal is not a currency to spend in it.

See [`docs/benchmarks.md`](docs/benchmarks.md#red-team-safety) for the full
method, the frozen-bank rationale, and what this does and does not measure.

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

## Inspecting a quant (Jacobian lens)

Beyond the leaderboard numbers, `quant-tuner lens` opens up the *inside* of a
quant with the [Jacobian lens](https://transformer-circuits.pub/2026/workspace/index.html):
layer-by-layer readouts, where tool-call decisions form, why a 2-bit model
loops, what heavy quantization erased versus merely suppressed, and an A/B diff
between two quants. If a runtime steer/ablate edit fixes a problem, it can be
baked back into the GGUF. Built on Anthropic's reference implementation
(the `vendor/jacobian-lens` submodule) and the GGUF-native
[jlens-gguf](https://github.com/igorbarshteyn/jlens-gguf) (adapted in-tree,
Apache-2.0).

```bash
# build the activation server (once), then fit one lens on the F16 over the cal corpus
uv run quant-tuner lens build-server
uv run quant-tuner lens fit --model model-f16.gguf --corpus corpus.cal.txt -o lens.gguf

# browse a quant, or A/B two of them, in the interactive visualizer
uv run quant-tuner lens serve --model quant.gguf --lens lens.gguf \
    --runs-dir out/lens/runs --model-b model-f16.gguf   # → http://127.0.0.1:8090
```

See [`docs/lens.md`](docs/lens.md) for the full guide (loop autopsy, knowledge
probes, the "why calibration works" study, and baking edits back into a GGUF).

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
├── lens/        # Jacobian-lens interpretability (fit, capture, diff, probes, bake) + D3 UI
├── recipes/     # YAML recipes consumed by `quant-tuner run --recipe ...`
├── cli.py       # typer CLI: run | bench | leaderboard | lens
└── pipeline.py  # end-to-end: extract → calibrate → quantize → bench

native/jlens_server/  # tracked llama.cpp activation server for the lens (build.sh)
vendor/llama.cpp      # pinned submodule
vendor/jacobian-lens  # pinned submodule (Anthropic reference lens; exact causal fits)
scripts/              # corpus builders, experiment drivers, leaderboard reproducer, lens experiments
tests/unit/           # 100+ passing tests
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
- The Jacobian-lens tooling (`src/quant_tuner/lens/`, `native/jlens_server/`) is
  adapted from [jlens-gguf](https://github.com/igorbarshteyn/jlens-gguf) and
  [jacobian-lens](https://github.com/anthropics/jacobian-lens) (both Apache-2.0);
  see the root [`NOTICE`](NOTICE) for details.
