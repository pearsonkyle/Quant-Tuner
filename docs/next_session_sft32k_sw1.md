# Session prompt — the `sft32k_sw1` stop-weight ablation

Copy everything below the line into a fresh session. It is self-contained.

---

## What you are picking up

A continued-QAT run of `prism-ml/Ternary-Bonsai-8B` (natively ternary, 36 layers, 6.95B
trainable) is **already training** on this box. Do not start it; monitor it, then evaluate it.

```
out/exp-058/trained_sft32k_sw1     613 steps, ~11 h, fp32, started ~03:5x UTC
  --train-layers 36 --optim adafactor --dtype fp32 --compute-dtype fp32
  --grad-accum 1 --epochs 1.0 --lr 5e-4 --warmup-frac 0.05
  --stop-weight 1.0 --grad-spike-factor 0
  --val-every 25 --val-windows 4 --ckpt-every 50 --ckpt-keep 3
```

Watch it with `python scripts/qat_progress_report.py out/exp-058/trained_sft32k_sw1 --watch 1800`
or `bash scripts/watch_qat_run_cuda.sh out/exp-058/trained_sft32k_sw1 600 48`. **Read the
report, don't poll the process.** Read `gnorm` and the code flips, not the loss.

## The one question this run answers

The previous run (`sft32k`, identical except `--stop-weight 6.0`) **broke the model's ability
to start**. Measured directly via llama.cpp `/completion` with `n_probs`, P(`<|im_end|>`) at
identical prompts:

| probe point | vanilla | sft32k (weight 6.0) |
|---|---|---|
| at generation start | 0.0000005 (rank 39) | **0.196** (rank 3) |
| mid-sentence | ~0 (rank >60) | 0.00007 (rank 22) |
| **after a sentence + period** | 0.0030 (rank 13) | **0.975 (rank 1)** |
| after sentence + newline | 0.0000004 | **0.879 (rank 1)** |
| after a tool-call structure | 0.00002 | 0.032 (rank 3) |

Every sentence boundary became an absorbing stop state. The model writes one correct
sentence ("Let me explore the repository structure and understand the bug.") and emits
`<|im_end|>` instead of the tool call. Grammar is intact — mid-sentence P(stop) is ~1e-4 —
so this is *not* degradation, it is a learned stopping policy that fires too early.

`--stop-weight 6.0` raised the terminating `<|im_end|>` from 0.58% to 3.40% of loss mass.
**This run sets it back to 1.0 (0.58%, the natural rate), making the 32768 window the only
intervention.** The question: does seeing whole task→completion arcs fix termination on its
own, with no loss reweighting?

## Comparability rule — do not compare against `sft8k-full`

`sft8k-full` (8064 window) is **context, not a control.** At 8064 only 27% of SWE
trajectories fit whole, so it never saw complete tool chains; at 32768 it is 97%. A
termination difference between them confounds "longer window" with "saw whole arcs at all",
and its validation corpus is packed at a different window so the CE values are not
comparable either.

**The valid controls all share this exact 32K corpus:**

| model | GGUF | what it is |
|---|---|---|
| vanilla | `out/exp-057/Ternary-Bonsai-8B-vanilla-Q2_0.gguf` | untrained, identical export path |
| sft32k | `out/exp-057/Ternary-Bonsai-8B-sft32k-Q2_0.gguf` | stop-weight 6.0, over-stops |
| **sft32k_sw1** | export it when training ends | stop-weight 1.0, this run |

The vanilla export prints `sign changes vs shipped: 0` — ternarization is a bit-exact no-op
on the shipped weights, so these GGUFs differ *only* in trained latents.

## When it finishes

**1. Export to Q2_0** (~20 min, CPU; ftype 41 is fork-only):

```bash
LLAMA_CPP_DIR=vendor/llama.cpp-prism PYTHONPATH=src .venv/bin/python \
  scripts/exp057_qat_export.py --latents out/exp-058/trained_sft32k_sw1/trained_latents.pt \
  --tag sft32k_sw1
```

**2. Re-measure P(im_end)** — the primary endpoint. The probe script and method are in this
repo's history; it posts a raw templated prompt to `/completion` with `n_predict=1,
n_probs=60, temperature=0` and reads `completion_probabilities[0].top_logprobs` for id
**151645**. Expect vanilla-like numbers if the window alone is enough.

**3. Tool-call eval** (no Docker needed):

```bash
LLAMA_CPP_DIR=vendor/llama.cpp-prism PYTHONPATH=src .venv/bin/python scripts/eval_toolcall.py \
  --model out/exp-057/Ternary-Bonsai-8B-sft32k_sw1-Q2_0.gguf \
  --holdout out/exp-060-32k/eval/toolcall_holdout.jsonl \
  --out out/exp-058/eval/toolcall_sft32k_sw1.csv --temperature 0 --seed 1234 --ctx 8192 --ngl 99
```
Verified 0 overlap with the SFT train split. Prior results (28–31 scored turns, so n is tiny
and nothing here is significant on its own):

| | vanilla | sft32k (6.0) |
|---|---|---|
| tool selection | 3/28 (0.107) | 7/31 (0.226) |
| param accuracy | 0.000 | 0.054 |
| schema-valid | 0.607 | 0.516 |

**4. Agentic run** — `/workspace/swe-mimic`, Docker-free, real dask repo:

```bash
cd /workspace/swe-mimic
/workspace/Quant-Tuner/vendor/llama.cpp-prism/build/bin/llama-server \
  --model <gguf> --ctx-size 32768 --n-gpu-layers 999 --jinja --flash-attn on \
  --host 127.0.0.1 --port 18080 &
.venv/bin/python run_agent.py --base-url http://127.0.0.1:18080/v1 --model-name local \
  --label SFT32K_SW1 --out swe_mimic_ternary.csv
```
Results so far on `dask__dask-11393` (n=1 each, and step counts vary 10↔19 between identical
reruns at temperature 0.25 — do not over-read):

| | resolved | patch | steps | out tok |
|---|---|---|---|---|
| vanilla | 0 | 0 | 19 | 1,781 |
| sft32k (6.0) | 0 | 0 | **0** | **1** |

## Environment traps (all cost time already)

- **No Docker.** Unprivileged container, no daemon, no `cap_sys_admin`. Real SWE-rebench
  cannot run; `swe-mimic` is the substitute.
- **Q2_0 needs the prism fork**, `vendor/llama.cpp-prism` (`llama-quantize` + `llama-server`
  are built). `LLAMA_CPP_DIR` must point there.
- **`--jinja` is mandatory** on `llama-server` or tool calls are never parsed and every model
  scores zero.
- **Do not use `--compute-dtype bf16`.** It is 5.15× faster and it **diverged twice**, both
  times non-reproducibly, with all corpus sources spiking at once and no anomalous gradient
  beforehand. fp32 ran the identical steps with max `gnorm` 1.88 vs bf16's 129.21. Full
  account in `docs/qat_32k_handoff.md` §10.6 and `docs/ternary_qat_sft32k_study.md` Obs. 8.
- **`GradSpikeGuard` cannot catch that failure** and is miscalibrated at both ends of a
  cosine schedule (not warmup-aware; threshold is a fixed multiple of a *falling* median, so
  it starts skipping ordinary norms late in a run). Leave it off in fp32.
- **`pkill -f "<pattern>"` matches your own shell** when the pattern appears in your command
  line. It has killed this session's shell three times. Kill by PID or `pgrep -x`.
- **The swe-mimic dask venv** is CPython 3.10 at `/.uv/python_install/`. If it disappears,
  `uv python install 3.10` restores it. A dangling interpreter makes every test fail to start
  and the harness records `p2p 0/34`, which reads as "the model broke 34 tests" — pure
  artifact. `run_agent.py` now has a golden gate that aborts with exit 2 instead.

## Open questions worth the GPU time

1. **Does the window alone fix termination?** — what this run answers.
2. **If it does not:** stop-weight 2.0 (1.16% of loss mass) is the next point. 6.0 is far too
   high; the useful range is narrow.
3. **lr 2.5e-4 control** — the `sft8k-full` study's #1 open question, still unrun.
4. **The val split cannot see what these runs change.** It is drawn from `logs` and
   `logs-agents` only, no SWE trajectories, no broad-instruct — so it went flat for 225 steps
   while code flips went 0.009% → 1.15%. Rebuild it to contain the sources under test.
5. **Freeze the low-movers.** Two independent runs show a ~9–30× depth spread in flip rate
   (`L0.q_proj` 1.80% vs `L30.up_proj` 0.21% at the end). Every tensor costs the same memory
   and LR. Untested.
