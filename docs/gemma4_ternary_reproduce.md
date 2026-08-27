# Reproducing the gemma-4-E4B ternarization study

Every command behind `docs/gemma4_ternary_feasibility.md`, in order, with the check that
must pass at each step. Companion to that doc (findings) and to
`docs/ternary_qat_reproduce.md` (the Bonsai pipeline these tools came from).

**Nothing here needs a GPU** except the round-trip generation in §4, which needs ~5 GB for
a few seconds. Every measurement was produced on CPU while the card was busy.

Paths assume the repo root. `PY=.venv/bin/python`.

The measurement outputs are tracked under `docs/gemma4_ternary/` — the `out/` tree is
gitignored, so those files are the record. Re-running should reproduce them; a diff is a
finding, not a nuisance.

---

## 0. Prerequisites

```bash
uv sync
# Q2_0 is ftype 41 and gemma-4 conversion both live ONLY in the prism fork. Mainline
# llama.cpp can neither quantize nor read these GGUFs.
git clone https://github.com/PrismML-Eng/llama.cpp vendor/llama.cpp-prism
git -C vendor/llama.cpp-prism checkout 9ca265a   # the commit these numbers were made on
cmake -S vendor/llama.cpp-prism -B vendor/llama.cpp-prism/build -DGGML_CUDA=ON
cmake --build vendor/llama.cpp-prism/build -j
```

**Base model: `google/gemma-4-E4B-it-qat-q4_0-unquantized`** — the QAT weights stored
densely in bf16 (15.9 GB, one safetensors file). Not the stock `gemma-4-E4B-it`: that repo
ships 54 tensors `from_pretrained` silently drops (k/v/k_norm for the 18 KV-sharing layers
24–41). Adopt the QAT repo for that reason more than for its ~3% smaller distance to the
ternary grid.

Check before going further — 2076 tensors in, 0 dropped:

```bash
$PY - <<'EOF'
import glob, torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, Gemma4ForConditionalGeneration
R = "google/gemma-4-E4B-it-qat-q4_0-unquantized"
with torch.device("meta"):
    m = Gemma4ForConditionalGeneration._from_config(AutoConfig.from_pretrained(R))
live = {n for n, _ in m.named_parameters()} | {n for n, _ in m.named_buffers()}
ck = set()
for f in glob.glob(f"{snapshot_download(R)}/*.safetensors"):
    with safe_open(f, framework="pt") as h:
        ck |= set(h.keys())
print(f"checkpoint {len(ck)} tensors, dropped on load: {len(ck - live)}")
EOF
```

---

## 1. Weight-space damage — the probe that returns a NULL result

```bash
$PY scripts/gemma4_ternary_damage.py --model google/gemma-4-E4B-it-qat-q4_0-unquantized \
    --out out/gemma4-ternary/damage_g128.json
$PY scripts/gemma4_ternary_damage.py --model google/gemma-4-E4B-it \
    --out out/gemma4-ternary/damage_stock_g128.json
```

~70 s each on 384 cores. **Expect every tensor kind at 0.42–0.49 relative Frobenius error
with frac_zero ≈ 0.42** — which is what i.i.d. Gaussian scores under the same quantizer
(0.4350 / 0.4224). That is the point: there is no ordering to extract here, so do not build
a schedule on it. Tracked as `docs/gemma4_ternary/weight_damage_{qat,stock}.json`.

If a tensor ever lands near 0, the model is already on the grid and this is the wrong study.

---

## 2. Output-space damage — where the schedule actually comes from

Needs an SFT corpus to read eval text from (`out/corpora/qwen3-universal-v2/sft.jsonl.gz`,
built by `scripts/build_universal_corpus.py`). Uses the `test` split, so the number is not
read on data a later fine-tune trains on.

```bash
$PY scripts/gemma4_layer_damage.py --split test --window 2048 --windows 3 \
    --mode both --cumulative --threads 256 \
    --out out/gemma4-ternary/layer_damage.json
```

~2.5 h on CPU with the card busy. Three results, all in
`docs/gemma4_ternary/layer_damage.json`:

* **by kind** — `mlp.down_proj` 1.199 (PPL 12.91), 3.4× the next-worst; everything else
  0.009–0.43 and PPL in the 4.2–5.1 band.
* **by layer** — layers 22, 23 worst (0.626, 0.271); layers 0–3 best (~0.01). Layers 22–23
  are the last KV donors, and 24–41 consume what they produce.
* **cumulative** — 6/12/18/24/30/36/42 layers → KLD 0.105 / 0.269 / 0.610 / 1.222 / 2.171 /
  5.288 / 10.666. **Each step ≈ ×2.** If your run does not reproduce that doubling, the
  eval windows differ — the ordering is stable but the absolute KLD is not comparable
  across eval sets.

One window per conversation, deliberately: our sessions are long enough that a concatenated
stream would draw every window from the first conversation, and a damage number read off a
single session is not a distribution.

---

## 3. Termination baseline — before any training

```bash
$PY scripts/measure_stop_baseline.py \
    --model google/gemma-4-E4B-it-qat-q4_0-unquantized --device cpu --threads 64 \
    --out out/gemma4-ternary/stop_baseline.json
```

Expect diagnostic `sentence_period` **0.00274** and control `answer_after_tool` **0.07032**
(tracked as `docs/gemma4_ternary/stop_baseline.json`). These are already in
`PROBE_SPECS['gemma4'].vanilla`; re-measure only if the checkpoint changes.

**Do not reuse Qwen's probe points.** `after_tool_call` reads 0.99995 on Qwen and 0.00004
here, because gemma's template hands over to the harness at that point. Using it as the
control inverts the test.

---

## 4. Round-trip: convert → quantize → serve, with no training

```bash
SNAP=$($PY -c "from huggingface_hub import snapshot_download as d; \
    print(d('google/gemma-4-E4B-it-qat-q4_0-unquantized'))")
PYTHONPATH=vendor/llama.cpp-prism/gguf-py $PY \
    vendor/llama.cpp-prism/convert_hf_to_gguf.py "$SNAP" \
    --outfile out/gemma4-ternary/E4B-qat-F16.gguf --outtype f16          # 666 tensors, 14,236 MiB

Q=vendor/llama.cpp-prism/build/bin/llama-quantize
# ternary trunk, both embedding tables at Q4_0
$Q --token-embedding-type q4_0 --tensor-type per_layer_token_embd=q4_0 \
   out/gemma4-ternary/E4B-qat-F16.gguf \
   out/gemma4-ternary/E4B-qat-trunkQ2_0-embQ4_0.gguf Q2_0 64             # 2,926 MiB, 3.29 BPW
# THE CONTROL — identical except the trunk type
$Q --token-embedding-type q4_0 --tensor-type per_layer_token_embd=q4_0 \
   out/gemma4-ternary/E4B-qat-F16.gguf \
   out/gemma4-ternary/E4B-qat-trunkQ4_0-embQ4_0.gguf Q4_0 24             # 4,043 MiB, 4.54 BPW
```

Verify the overrides landed — `per_layer_token_embd` 5376 → 1512 MiB, `token_embd`
1280 → 360 MiB, `blk.*` at 2.125 bpw (`attn_q` 10.00 → 1.33 MiB):

```bash
grep -E "per_layer_token_embd|token_embd\.weight|blk\.0\.attn_q" out/gemma4-ternary/quantize.log
```

Then generate from each. **Run the control.** Without it, token soup is equally consistent
with a broken converter, and the whole point of the round-trip is to tell those apart:

```bash
for m in trunkQ2_0-embQ4_0 trunkQ4_0-embQ4_0; do
  vendor/llama.cpp-prism/build/bin/llama-cli -m out/gemma4-ternary/E4B-qat-$m.gguf \
    -ngl 99 -st -n 40 --temp 0 --no-warmup -p "What is the capital of France?"
done
```

Expected — and recorded verbatim in `docs/gemma4_ternary/roundtrip_generation_ab.txt`:
Q4_0 says *"The capital of France is Paris."*, Q2_0 emits multilingual token soup. That is
the correct starting point, not a failure.

Two traps. `-st` (single-turn) is required or llama-cli sits at an interactive prompt and
looks like a hang. `--jinja` makes it parse the model's output as a chat message, and the
chat parser **aborts** on the ternary model's garbage — a harness crash, not a model one.

---

## 5. Corpus — 32k windows, gemma masking

```bash
for SPLIT in train test; do
  OUT=out/gemma4-ternary/corpus_sft_gemma4_${SPLIT/test/val}_32768.pt
  $PY scripts/build_sft_qat_corpus.py \
      --sft out/corpora/qwen3-universal-v2/sft.jsonl.gz --split $SPLIT \
      --model google/gemma-4-E4B-it-qat-q4_0-unquantized \
      --window 32768 --max-tool-tokens 8192 --min-density 0.05 --out $OUT
done
```

Fingerprints to match: train `0c70d992882d29a7` (651 windows, 21.3 M tokens, 28.7%
supervised, 6,300 labeled `<turn|>`), val `16177b9a361cbdd7` (86 windows). Build logs are
tracked as `docs/gemma4_ternary/corpus_build_{train,val}.log`.

> **Known-stale lines in the tracked train log.** It was produced before bfc30c2 fixed the
> per-source audit counters (`count_rendered()` was matching Qwen's markers on a gemma
> render), so its "tool-calls 57/10,014 kept · reasoning 0/343 kept" lines are the counter
> bug, not the corpus — 25,772 of 26,389 calls are present and every one supervised. A
> rebuild at current HEAD reproduces the *fingerprint* but prints corrected audit lines
> (the val log, built 34 s after the fix landed, shows the true shape: 3,190/3,190 kept).
> Only an audit-line diff **plus a fingerprint diff** is a finding.

**Then audit it.** This is the step that catches a silently wrong mask:

```bash
$PY scripts/inspect_corpus_window.py out/gemma4-ternary/corpus_sft_gemma4_32768.pt \
    --audit --model google/gemma-4-E4B-it-qat-q4_0-unquantized
```

The line that matters is `supervised tokens inside a TOOL RESPONSE  0`. gemma renders a
whole tool exchange inside one `model` turn, so a mask that supervises the turn trains the
model on environment output while every aggregate looks healthy. The audit exits non-zero
if it finds any, or if no stop token is supervised.

Expect a large `carry-over` figure (3.2 M, 52% of supervised tokens) — gemma's model turns
run to tens of thousands of tokens, so a 32k window routinely opens inside one. The
tool-response check covers that region too.

---

## 6. Pre-flight the training path (CPU, ~1 min)

Before spending GPU time, confirm the real checkpoint loads, wraps and backprops:

```bash
$PY - <<'EOF'
import torch
from transformers import Gemma4ForConditionalGeneration
from quant_tuner.qat.train import wrap_model, decoder_layers
torch.set_num_threads(64)
R = "google/gemma-4-E4B-it-qat-q4_0-unquantized"
m = Gemma4ForConditionalGeneration.from_pretrained(R, dtype=torch.float32, device_map="cpu")
wrap_model(m, n_train=0, layer_spec="0-3,7,8", ternary_spec="0-3,7,8",
           dense_kinds=("down_proj",))
blob = torch.load("out/gemma4-ternary/corpus_sft_gemma4_32768.pt", weights_only=False)
out = m(blob["ids"][0:1, :512], labels=blob["labels"][0:1, :512])
out.loss.backward()
n = sum(1 for p in m.parameters() if p.grad is not None)
print(f"loss {out.loss.item():.4f}  tensors with grad {n}")
print("trainable outside language_model:",
      sum(1 for k, p in m.named_parameters() if p.requires_grad and "language_model" not in k))
EOF
```

**`trainable outside language_model` must be 0.** gemma's vision and audio towers have
their own `layers.N` and submodules literally called `linear`, which the old name-based
selection matched — 167.8 M params of tower were handed to the optimizer, silently, because
a text-only forward never gives them a gradient.

The number of tensors with a gradient should equal the linears in the scheduled layers:
**48 ternary + 6 dense `down_proj` = 54** for the stage above. It is layer-set dependent —
layers 24–41 have no `k_proj`/`v_proj`, so a stage drawn from there contributes 6 linears
per layer, not 8.

---

## 7. The first training stage (needs the GPU)

The damage profile says ~6 layers per stage, least-damaging first, `down_proj` kept dense
— and the loss is the full Bonsai anchor-ladder stack, not bare CE (see the feasibility
doc's loss-stack section: CE alone at a flip-capable lr diverges even a dense model, and a
from-scratch stage has dense tensors inside every trainable layer).

```bash
# 7a. Forced-stop KD table from a dense gemma teacher (forward-only). Stop id is 106.
#     ~6.6 h for 651 windows: 36.8 s/window, of which top-K is 0.07 s -- it is entirely
#     the 31B trunk at a 32k window. There is no faster kernel to reach for on this
#     architecture: gemma-4's full-attention layers use global_head_dim 512 and
#     FlashAttention-2 caps at 256, FA3 ships no kernel for this compute capability, and
#     flex_attention wants 201 KB of shared memory against the card's 101 KB. SDPA it is.
$PY scripts/kd_precompute.py --teacher google/gemma-4-31B-it \
    --corpus out/gemma4-ternary/corpus_sft_gemma4_32768.pt \
    --out out/gemma4-ternary/kd/gemma31b_topk64_fs106.pt --topk 64 --dtype bf16 \
    --student-model google/gemma-4-E4B-it-qat-q4_0-unquantized \
    --include-ids 106
# Read `coverage` at startup; below 0.8 the KL is far weaker than it looks.

# 7b. The teacher's own probe values — the report's dotted asymptotes.
$PY scripts/teacher_stop_probe.py --teacher google/gemma-4-31B-it \
    --student-model google/gemma-4-E4B-it-qat-q4_0-unquantized \
    --out out/gemma4-ternary/kd/teacher_probe_31b.json

# 7c. Stage 1. Anchor margins and abort thresholds are SCALED FROM GEMMA'S BASELINE
#     (stop_baseline.json: diagnostic 0.00274, control 0.070), not copied from Bonsai;
#     lr/clip are Bonsai's only as the first guess — re-measure with a 60-step A/B.
$PY -m quant_tuner.qat.train \
    --model-dir google/gemma-4-E4B-it-qat-q4_0-unquantized \
    --corpus out/gemma4-ternary/corpus_sft_gemma4_32768.pt \
    --val-corpus out/gemma4-ternary/corpus_sft_gemma4_val_32768.pt \
    --layers 0,1,2,3,7,8 --ternary-layers 0,1,2,3,7,8 --dense-kind down_proj \
    --kd-table out/gemma4-ternary/kd/gemma31b_topk64_fs106.pt --kd-alpha 0.5 --kd-temp 1.0 \
    --stop-anchor 0.2 --stop-anchor-margin 1.0 --stop-anchor-margin-hi 0.1 \
    --steer-weight 0 --clip-norm 0.25 --lr-scale group-scale \
    --lr 5e-4 --optim adafactor --dtype fp32 --compute-dtype fp32 \
    --val-every 50 --probe-every 25 \
    --probe-abort 0.03 --probe-abort-control 0.01 --probe-abort-patience 2 \
    --out out/gemma4-ternary/stage1
# Report watcher, same as Bonsai: bash scripts/qat_report_watch.sh out/gemma4-ternary/stage1
```

**`--model-dir` is not optional.** It defaults to `out/exp-057/model` — the Bonsai
checkpoint — and every other flag above is dialect-agnostic, so omitting it launches a
run against the wrong model entirely. It dies on the embedding lookup (the gemma corpus
carries ids up to 262,143 against Bonsai's 151,669 vocab) rather than training something
plausible-but-wrong, but only by luck of the vocab sizes.

```bash
# 7d. Damage, before and after, from the same probe in the same process. This is the
#     go/no-go number. Run it with no --ckpt first: the untrained baseline under the
#     stage's OWN configuration is not the matching row of layer_damage.json, because a
#     stage holds --dense-kind tensors dense and down_proj is the most damaging kind
#     there is. CPU, ~15 min, runs while the card is busy.
$PY scripts/gemma4_stage_damage.py --ternary-layers 0,1,2,3,7,8 --dense-kind down_proj \
    --out out/gemma4-ternary/stage1/stage_damage_untrained.json
$PY scripts/gemma4_stage_damage.py --ternary-layers 0,1,2,3,7,8 --dense-kind down_proj \
    --ckpt out/gemma4-ternary/stage1/trained_latents.pt \
    --out out/gemma4-ternary/stage1/stage_damage_trained.json
```

The control-side abort deserves a caveat: gemma's control baseline is 0.070 with ~25× of
headroom over the diagnostic (Qwen had ~10⁴), so `--probe-abort-control` here is a
last-ditch floor, not a health band — a stage that moves the control should be checked
against a generated trajectory before the probe is believed in either direction.
`--steer-weight` stays **0** until the steering batches are ported: their control class
teaches "stop after a tool call", which is correct Qwen and *inverted* gemma (gemma's
stop-is-right position is `answer_after_tool` — the same trap the probe section documents),
and their bodies hardcode Qwen markers. The anchor and the KD KL are dialect-clean and
carry the termination defense meanwhile.

Stage order from `layer_damage.json["layer_order"]` — stage 2 is `05,06,36,37,38,39`; the
last six (`10,11,17,21,22,23`) are the full-attention and late KV-donor layers and may be
worth leaving dense permanently.

**Read the stop probe from step 1, against §3's baseline, not Bonsai's.** And read it next
to the code-flip telemetry: a run can hold termination and learn nothing, or learn and
collapse, and neither number alone distinguishes them.

**The open question this stage answers:** whether QAT recovers a stage's damage before the
next stage compounds on it. The cumulative curve doubles every 6 layers with no training;
if training cannot pull a stage back down, the schedule buys nothing and the honest answer
is that fully-ternary E4B is out of reach.
