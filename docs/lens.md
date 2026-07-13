# Jacobian-lens interpretability for GGUF quants

`quant-tuner lens` opens up the *inside* of a quant: what each layer is disposed
to predict, where a tool-call decision forms, why a 2-bit model loops, what
heavy quantization erased versus merely suppressed, and — when a runtime edit
proves out — how to bake it back into the weights. It is built on two upstreams
(Apache-2.0; see the repo-root `NOTICE`):

- **[anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)** —
  the reference PyTorch implementation of the Jacobian lens
  (`lens_l(h) = unembed(J_l·h)`, `J_l = E[∂h_final/∂h_l]`), vendored as the
  `vendor/jacobian-lens` submodule. Used only by `lens fit-causal`.
- **[igorbarshteyn/jlens-gguf](https://github.com/igorbarshteyn/jlens-gguf)** —
  a GGUF-native reimplementation (a llama.cpp activation server + numpy lens
  math + a D3 visualizer), adapted in-tree under `native/jlens_server/` and
  `src/quant_tuner/lens/`.

## How it works

Three layers:

1. **`native/jlens_server`** — a llama-server-compatible binary that hooks the
   ggml scheduler eval callback to capture each layer's residual output
   (`l_out-<il>`) and apply runtime interventions (steer / ablate / swap)
   mid-graph. Built against the repo's vendored llama.cpp — it links only the
   public API, so a submodule bump is just a rebuild.
2. **`src/quant_tuner/lens`** — the Python bridge: the lens container
   (`lens.gguf`), the numpy readout math (dequantizes each model's own head, so
   it matches llama.cpp logits to >0.9999 correlation even at 2-bit), a capture
   run store, a diff engine, and the analysis modules.
3. **the visualizer** (`lens serve`) — the D3 heatmap plus a quant-tuner A/B
   panel that diffs two capture runs.

The lens strategy for comparability: fit **one regression lens per base model**
on the **F16 GGUF** over the **calibration corpus** (the tool-use logs), and
reuse it across every quant of that model. Because the decoder is fixed, an A/B
diff then isolates exactly how quantization moved the activations. (`fit-causal`
produces the same `lens.gguf` from the exact backprop Jacobian, as an option.)

## Setup

```bash
git submodule update --init --recursive          # brings in vendor/jacobian-lens
# build llama.cpp as usual (Metal/CUDA/CPU), then:
uv run quant-tuner lens build-server             # or: bash native/jlens_server/build.sh
```

The server binary lands at `native/jlens_server/jlens-server`; override its
location with `$JLENS_SERVER_BIN`, and the llama.cpp checkout with
`$LLAMA_CPP_DIR` (same variable `paths.llama_bin` honors).

## Fit a lens

```bash
uv run quant-tuner lens fit \
    --model out/exp-044/vanilla/model-f16.gguf \
    --corpus out/exp-044/corpora/corpus.cal.txt \
    -o out/lens/gemma-4-31B.lens.gguf
```

Forward-only, works on any quant, no torch. On wide models (gemma d≈5120, 60
layers) the `d_model²` Gram accumulators dominate memory; `--band-size` fits
layers in groups and merges (`--gram-float32` halves it further, `--layers
lo-hi` fits a subset). The exact causal alternative:

```bash
uv run quant-tuner lens fit-causal --hf-model <ws>/model_extracted \
    --model model-f16.gguf --corpus corpus.cal.txt -o lens.gguf
# or convert a pre-fit reference lens:  lens convert-pt lens.pt lens.gguf
```

## Inspect one quant

```bash
uv run quant-tuner lens serve --model quant.gguf --lens lens.gguf
# → http://127.0.0.1:8090  (layer × position heatmap, top-k panels,
#    pinned-token rank charts, activation decomposition, steer/ablate/swap)
```

## A/B compare two quants

Capture a run per model (the on-disk unit of comparison), then diff:

```bash
A=$(uv run quant-tuner lens capture --model quant.gguf --lens lens.gguf \
      --runs-dir out/lens/runs --prompt-file p.txt --tail 32 | awk '{print $2}')
B=$(uv run quant-tuner lens capture --model model-f16.gguf --lens lens.gguf \
      --runs-dir out/lens/runs --prompt-file p.txt --tail 32 | awk '{print $2}')
uv run quant-tuner lens diff "$A" "$B" --runs-dir out/lens/runs \
    --lens lens.gguf --out out/lens/diff
```

The diff reports per-(layer, position) top-1 disagreement, the rank the
reference's top-1 token fell to in the candidate, per-layer activation
divergence, and per-position final KLD (exact at flagged decision positions,
top-k-approximated elsewhere, marked as such). The visualizer renders all of
this interactively — launch with a run store (and optionally a second live
backend) and use the **A/B** panel:

```bash
uv run quant-tuner lens serve --model quant.gguf --lens lens.gguf \
    --runs-dir out/lens/runs --model-b model-f16.gguf
```

## Diagnose agentic failures

- **Tool-call representations** — replay the tool-call holdout through the lens
  and see where the gold tool token emerges across layers:
  ```bash
  uv run quant-tuner lens replay-toolcalls --model quant.gguf --lens lens.gguf \
      --holdout toolcall_holdout.jsonl --runs-dir out/lens/runs \
      --csv out/lens/lens_toolcall.csv
  uv run quant-tuner leaderboard --results results.csv --lens-csv out/lens/lens_toolcall.csv
  ```
  The sidecar joins into the leaderboard (Gold Rank / Emerge L / Decision KLD
  columns) by `basename(quant_path)`.

- **Infinite loops** — replay a looping generation (e.g. a swebench
  `.traj.json`), measure how early the quant's layers commit to the loop token
  versus FP16, and sweep ablations to break it:
  ```bash
  uv run quant-tuner lens loop --model quant.gguf --lens lens.gguf \
      --traj <ws>/trajectories/<model>/<id>.traj.json \
      --reference model-f16.gguf --runs-dir out/lens/runs --out out/lens/loop --sweep
  ```
  A winning ablation is saved as `direction.npz`.

- **Knowledge loss** — classify factual probes as `correct` / `suppressed`
  (present in the residual stream, readout degraded) / `absent` (erased), and
  compare a quant to FP16:
  ```bash
  uv run python scripts/build_probe_set.py --holdout toolcall_holdout.jsonl --out artifacts/probes.jsonl
  uv run quant-tuner lens probe --model quant.gguf --lens lens.gguf \
      --probes artifacts/probes.jsonl --reference model-f16.gguf --out out/lens/probe
  ```

## Why does calibration work?

`lens study` overlays per-layer divergence-vs-FP16 profiles for several
calibration variants at the same bit-width, alongside per-tensor weight RMSE and
imatrix concentration:

```bash
uv run python scripts/lens_exp104_why_calibration.py --emit-example > study.yaml
uv run quant-tuner lens study --config study.yaml --out out/lens/exp104
```

## Bake an edit back into a GGUF

If an ablation direction proves out at runtime, make it permanent by
orthogonalizing it out of the FP16 residual-writing tensors
(`attn_output.weight`, `ffn_down.weight`; per-expert for MoE — the skip
connection is untouched) and requantizing with the existing imatrix:

```bash
uv run quant-tuner lens bake --f16 model-f16.gguf --direction out/lens/loop/direction.npz \
    --quant-type IQ2_M --imatrix imatrix.gguf --out-dir out/lens/baked \
    --verify-against quant.gguf --reference model-f16.gguf --eval corpus.eval.txt
```

Edits are only ever applied to the FP16 and then requantized fresh — never
block-edited in place (which would be lossy). Additive **steering** does not
bake cleanly (most architectures have no per-layer bias tensor to absorb it), so
it stays a runtime-only capability of the OpenAI-compatible jlens-server, which
you can point an agent framework at to serve a steered model live.

### Combining directions (multi-direction bake)

`orthogonalize_subspace(f16, [dir1, dir2, ...], out_f16)` removes the *span* of
several directions at once. It builds an SVD orthonormal basis Q (dropping
near-duplicate directions) and applies `W <- W - Q(QᵀW)` — the correct joint
edit for non-orthogonal directions; removing them one at a time would over- or
under-project. `scripts/lens_exp107_multidir_bake.py` folds a loop direction
together with an **explore-vs-act** direction (a contrastive mean-difference
vector from the *real* tool-call holdout: residual at "about to call an explore
tool" minus "about to call an act tool") and bakes the combined subspace.

**Verify against a matched control and a real task holdout — a mechanistic shift
is not a win.** On Ornith-1.0-9B IQ2_M the multi-direction bake did exactly what
the directions describe at the readout level — it suppressed explore tokens
(` inspect` rank 350→7892) and promoted action verbs (` modify` 167418→4817) at
the commit decision — yet the real 25-session tool-call holdout showed it *broke*
the model (param accuracy 0.193→0.013, tool-selection 0.40→0.20) and slightly
worsened looping. Ablating a behavioral subspace from a 2-bit model's weights
removes capability along with the behavior; the axis is not cleanly separable
from competence at that bit-width. So for **loops specifically, prefer the
repetition penalty** (a sampling fix — no weight edit, no capability loss); treat
weight-space bakes as an experimental capability that must clear a real-task
eval before it is called a correction. This is exactly why the bake pipeline
ships with `verify_bake` and the exp-106/107 scripts run a matched control.

## Repetition penalty (breaking loops without stopping exploration)

Low-bit quants loop during long agent rollouts. The **serving path**
(`/v1/chat/completions`, `/v1/completions`) applies a repetition penalty by
default — `repeat_penalty=1.1`, `repeat_last_n=256` — mirroring
`eval/server.py`'s llama-server flags. It only damps *verbatim* repetition, so
the agent still explores/plans freely; it just can't get stuck emitting the same
tokens. Any OpenAI request field (`repeat_penalty`/`repetition_penalty`,
`repeat_last_n`, `frequency_penalty`, `presence_penalty`) overrides the default.

The **capture path** (`/jlens/forward`) defaults the penalty **off** so lens
readouts stay faithful to the model's own distribution; pass
`sampling={"repeat_penalty": 1.2, ...}` when you want penalized generation (e.g.
to check whether a penalty alone breaks a loop before reaching for a bake).

## Experiment scripts (gemma-4-31B + Ornith-1.0-9B anchored)

- `scripts/lens_exp101_toolcall_lens.py` — tool-call representations under quantization
- `scripts/lens_exp102_loop_autopsy.py` — loop diagnosis + intervention
- `scripts/lens_exp103_knowledge_probes.py` — knowledge loss at 2 bpw
- `scripts/lens_exp104_why_calibration.py` — the calibration divergence study
- `scripts/lens_exp105_ornith_action_bias.py` — why Ornith-9B IQ2_M patches but resolves 0 (explore≫act)
- `scripts/lens_exp106_bake_deloop.py` — bake a jlens loop direction out of a GGUF (corrected quant)
- `scripts/build_probe_set.py` — mine the probe set
- `scripts/lens_smoke.sh` — CPU end-to-end smoke on a tiny GGUF (the acceptance gate)

## Testing

Unit tests (`tests/unit/test_lens_*.py`, `test_gguf_edit.py`,
`test_hf_gguf_map_inverse.py`) need no model files. Integration tests
(`tests/integration/`) are gated on `QT_LENS_IT=1` + `QT_TINY_GGUF=<path>` and
exercise the built server, including a numpy-readout-vs-server-logits parity
check (corr ≥ 0.9999) that is the canary for llama.cpp submodule drift.

## Caveats

- The lens is fit on FP16 and applied to quant activations; each capture run
  records observed residual norms and warns if they drift far from the lens's
  fit distribution. A/B conclusions are always stated relative to the shared lens.
- CPU grid latency is worst on gemma's ~262k vocab — capture only the decision
  tail (`--tail`), and readouts are cached per run.
- The pinned llama.cpp (`f3e182816`) differs from the commit jlens-gguf was
  tested against; the public-API linkage and the startup `l_out_ok` self-check
  guard this, and the serving commit is stamped into every capture-run manifest.
