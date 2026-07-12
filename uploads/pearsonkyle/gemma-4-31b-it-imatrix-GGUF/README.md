---
library_name: gguf
base_model:
- google/gemma-4-31B-it
tags:
- gguf
- llama.cpp
- text-generation
- text-generation-inference
- transformers
- quantization
- quantized
- imatrix
- awq
- low-bit
- 2-bit
- iq2_m
- 5-bit
- q5_k_s
- gemma
- gemma-4
- 31b
- coder
- tool-use
- function-calling
- agentic
- swe-bench
- long-context
- multimodal
- vision
- image-text-to-text
- vlm
- mmproj
- siglip
license: gemma
language:
- en
pipeline_tag: image-text-to-text
---

<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; border: 1px solid #99f6e4; border-radius: 16px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05), 0 4px 6px -2px rgba(0,0,0,0.05); overflow: hidden; background: #ffffff; margin-bottom: 30px;">
  <div style="background: linear-gradient(135deg, #0d9488 0%, #134e4a 100%); padding: 24px; color: white;">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
      <h1 style="margin: 0; font-size: 26px; font-weight: 800; display: flex; align-items: center; gap: 12px; color: white; border: none;">🧊 Google/Gemma-4-31B-it · imatrix · GGUF</h1>
      <span style="background: #f59e0b; color: #1c1917; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; text-transform: uppercase; letter-spacing: 0.5px;">imatrix (hybrid)</span>
    </div>
  </div>
  <div style="display: flex; gap: 8px; flex-wrap: wrap; padding: 12px 24px; background: #f8fafc; border-bottom: 1px solid #e2e8f0;">
    <span style="background: #dbeafe; color: #1e40af; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #bfdbfe;">📦 10.2 · 13.4 · 15.6 · 19.8 GiB</span>
    <span style="background: #d1fae5; color: #065f46; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #a7f3d0;">IQ2_M · IQ3_M · IQ4_XS · Q5_K_S</span>
    <span style="background: #fef3c7; color: #92400e; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #fde68a;">🧪 2-bit AWQ · +54% tool-arg accuracy</span>
    <span style="background: #fce7f3; color: #9d174d; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #fbcfe8;">🏗️ llama.cpp f3e1828</span>
    <span style="background: #fef3c7; color: #92400e; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #fde68a;">🏅 Agent · 100% patch</span>
    <span style="background: #ede9fe; color: #5b21b6; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #ddd6fe;">👁️ Text + Image · mmproj 772 MB</span>
    <span style="background: #cffafe; color: #155e75; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #a5f3fc;">⚡ MTP drafter · 88% accept @ n=1</span>
    <span style="background: #dcfce7; color: #166534; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #bbf7d0;">🎯 Q5_K_S · text+vision+MTP in 24 GB</span>
  </div>
  <div style="padding: 24px; display: flex; flex-direction: column; gap: 20px;">
    <div style="background: #f0fdfa; border-left: 5px solid #0d9488; padding: 16px; border-radius: 0 8px 8px 0;">
      <h3 style="margin: 0 0 8px 0; font-size: 15px; color: #115e59; font-weight: 700; display: flex; align-items: center; gap: 6px;"><span>🧊</span> What this is</h3>
      <p style="margin: 0; font-size: 13px; color: #334155; line-height: 1.7;">An aggressively compressed (under 3 bpw) <b>IQ2_M</b> quantization of <a href="https://huggingface.co/google/gemma-4-31B-it"><b>google/gemma-4-31B-it</b></a>, calibrated from real coding/tool-use logs. The 2-bit ships an <b>AWQ build</b> (activation-aware scaling) — chosen over plain imatrix by a direct tool-call test: <b>+54% tool-argument accuracy and far more consistent</b> at the same 2.85 bpw, because at 2 bits static KLD/PPL doesn't predict whether the model still fills correct tool arguments (see <a href="#-why-the-2-bit-is-awq">Why the 2-bit is AWQ</a>). Runs in vanilla <code>llama.cpp</code> / Ollama / LM Studio — <b>no custom runtime, no extra inference cost</b>. Higher-bit <b>IQ3_M</b> (3.76 bpw), <b>IQ4_XS</b> (4.36 bpw), and <b>Q5_K_S</b> (5.55 bpw — highest fidelity, KLD 0.025) builds ship plain <b>hybrid imatrix</b> for users with more VRAM.</p>
    </div>
    <div style="background: #faf5ff; border-left: 5px solid #7c3aed; padding: 16px; border-radius: 0 8px 8px 0;">
      <h3 style="margin: 0 0 8px 0; font-size: 15px; color: #5b21b6; font-weight: 700; display: flex; align-items: center; gap: 6px;"><span>👁️</span> Now with vision (text + image input)</h3>
      <p style="margin: 0; font-size: 13px; color: #334155; line-height: 1.7;">Gemma 4 is natively multimodal. This repo ships the model's <b>vision tower as a separate <code>mmproj-gemma-4-31B-it-Q8_0.gguf</code></b> (772 MB, SigLIP-style 27-layer encoder, Q8_0 — visually lossless vs F16 at ⅔ the size). Pair it with <b>any</b> of the four quant files via <code>--mmproj</code> and the model can <b>see images</b> — describe screenshots, read diagrams, answer questions about a UI, and so on. The text quant is unchanged; vision adds only the small mmproj. See <b>Usage → Vision</b> below.</p>
    </div>
    <div style="background: #f0fdf4; border-left: 5px solid #16a34a; padding: 16px; border-radius: 0 8px 8px 0;">
      <h3 style="margin: 0 0 8px 0; font-size: 15px; color: #166534; font-weight: 700; display: flex; align-items: center; gap: 6px;"><span>🎯</span> The 24 GB build (Q5_K_S)</h3>
      <p style="margin: 0; font-size: 13px; color: #334155; line-height: 1.7;">The new <b>Q5_K_S</b> build (5.55 bpw, <b>19.85 GiB</b>) is sized so a single <b>24 GB</b> GPU can host the <b>full stack at once</b>: the 5-bit text trunk (19.85) + the <b>Q4_K_M MTP drafter</b> (0.36) + the <b>Q8 vision mmproj</b> (0.75) = <b>~21.0 GiB</b>, leaving <b>~3.0 GiB</b> for a real KV cache (more with <code>--cache-type-k q8_0 --cache-type-v q8_0</code>). Near-FP16 quality (KLD 0.025), images, <b>and</b> speculative decoding — all on one consumer card, no offload.</p>
    </div>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 10px;">
      <div style="border: 1px solid #e2e8f0; padding: 14px; border-radius: 8px; background: #fafafa; box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);"><span style="font-weight: 700; color: #115e59; font-size: 12px; display: block; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">📉 ~5.6× smaller</span><span style="font-size: 13px; color: #4b5563; line-height: 1.5;">10.17 GiB on disk vs 57.2 GiB FP16, at ~2.85 bits/weight.</span></div>
      <div style="border: 1px solid #e2e8f0; padding: 14px; border-radius: 8px; background: #fafafa; box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);"><span style="font-weight: 700; color: #115e59; font-size: 12px; display: block; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">🤖 Actually agentic</span><span style="font-size: 13px; color: #4b5563; line-height: 1.5;">47% pass / 100% patch on a 10-instance agentic SWE-rebench holdout (IQ4_XS). IQ2_M still resolved 40% — best of every sub-3-bpw arm tested.</span></div>
      <div style="border: 1px solid #e2e8f0; padding: 14px; border-radius: 8px; background: #fafafa; box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);"><span style="font-weight: 700; color: #115e59; font-size: 12px; display: block; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">🛠️ Standard GGUF</span><span style="font-size: 13px; color: #4b5563; line-height: 1.5;">Loads anywhere llama.cpp runs. No patches, kernels, or forks.</span></div>
    </div>
  </div>
</div>

## 📊 Unified benchmark & quality table

Agentic metrics from a [SWE-rebench](https://huggingface.co/datasets/nebius/SWE-rebench) holdout run through the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) (10 instances × 3 reps). Static metrics (PPL / KLD / top-p) measured against FP16 on a held-out eval corpus at `ctx=4096`. KLD column is **median** for robustness to per-token tails.

| Metric | FP16 *(ref)* | Q5_K_S | **IQ4_XS**  | IQ3_M  | IQ2_M  |
|:---|---:|---:|---:|---:|---:|
| File | — | [Q5_K_S.gguf](https://huggingface.co/pearsonkyle/gemma4-31b-imatrix-mtp-GGUF/resolve/main/gemma-4-31B-it-Q5_K_S.gguf) | [IQ4_XS.gguf](https://huggingface.co/pearsonkyle/gemma4-31b-imatrix-mtp-GGUF/resolve/main/gemma-4-31B-it-IQ4_XS.gguf) | [IQ3_M.gguf](https://huggingface.co/pearsonkyle/gemma4-31b-imatrix-mtp-GGUF/resolve/main/gemma-4-31B-it-IQ3_M.gguf) | [IQ2_M.gguf](https://huggingface.co/pearsonkyle/gemma4-31b-imatrix-mtp-GGUF/resolve/main/gemma-4-31B-it-IQ2_M.gguf) |
| Method | — | imatrix | imatrix | imatrix | **AWQ + imatrix** |
| Quality| - | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| BPW | 16.0 | 5.55 | 4.36 | 3.76 | 2.85 |
| Size (GiB) | 57.20 | 19.85 | 15.59 | 13.43 | 10.17 |
| 🤖 Pass Rate | — | 40±8% | **47±5%** | 33±12% | 40±8% † |
| 🤖 Patch Rate | — | 100% | 100% | 100% | 100% † |
| 🤖 Tool Errors | — | 11±2% | 10±3% | 16±2% | 16±1% † |
| 🤖 Mean Tokens | — | 663K±111K | 575K±70K | 483K±75K | 558K±94K † |
| 📐 PPL | **215.5** | 256.5 | 319.4 | 734.1 | 1040.0 |
| 📐 KLD (med) | 0.000 | 0.025 | 0.073 | 0.435 | 1.804 |
| 📐 same_top_p | 100.0% | 85.5% | 78.8% | 63.1% | 43.9% |

> **† IQ2_M now ships an AWQ build** (see [§ Why the 2-bit is AWQ](#-why-the-2-bit-is-awq) below).
> The SWE-rebench figures (†) were measured on the earlier *imatrix* 2-bit; the AWQ swap is
> justified by a direct tool-call fidelity comparison (param-acc **+54%**, far tighter run-to-run
> variance) at the same 2.85 bpw. AWQ also cuts IQ2_M perplexity nearly in half (1959 → 1040) —
> it trades a hair of median-KLD for a much more *usable* 2-bit.
>
> Q5_K_S resolves **40%** of the holdout (tying the old IQ2_M, ahead of IQ3_M) at **100%** patch and a
> low **11%** tool-error rate; IQ4_XS remains the agentic leader at 47% (gap within run-to-run noise).

<details>
<summary><b>📌 Sampling & methodology details</b></summary>

> Sampling: `temperature=0.25, top_p=0.95, top_k=20, max_tokens=32768, ctx=131072, thinking=false`. Run on Apple Silicon (Metal); SWE-rebench linux/amd64 images under emulation, so wall-clock is relative, not absolute.
>
> **Pass Rate** = gold tests pass after agent's patch (real resolution). **Patch Rate** = non-empty diff produced.

</details>


---

## 🧪 Why the 2-bit is AWQ

Static KLD/PPL measures how closely a quant's logits track FP16 — it does **not** measure whether
the model still fills **correct tool-call arguments**, which is what agentic use actually needs. At
2 bits the plain-imatrix build keeps decent tool *selection* but loses argument precision. We replay
25 held-out real tool-use sessions (disjoint from calibration) through `llama-server`, scoring per
assistant turn whether the model picks the right tool (**tool-sel**) and fills the right arguments
(**param-acc**). 3 reps, mean ± σ, same sampling as above.

Both builds are measured on the **same** held-out eval + FP16 baseline (static) and the same 25-session
replay (agentic). Static: ↓ PPL / ↓ KLD / ↑ top_p is better. Agentic: ↑ is better.

| IQ2_M (2.85 bpw) | PPL | KLD (med) | top_p | tool-sel | param-acc | schema-valid |
|---|---:|---:|---:|---:|---:|---:|
| imatrix | 1958.7 | **1.571** | **46.6%** | 0.454 ± .071 | 0.171 ± .082 | 0.805 |
| **AWQ (shipped)** | **1040.0** | 1.804 | 43.9% | **0.492 ± .013** | **0.263 ± .009** | **0.823** |

Read the split: on the metric the static table headlines (**median-KLD**) imatrix looks *better*
(1.571 vs 1.804), and it edges top_p (46.6% vs 43.9%) — yet the AWQ build is the one that actually
tool-calls. **AWQ [activation-aware scaling](https://arxiv.org/abs/2306.00978) wins where it counts**:
+54% tool-argument accuracy (0.17 → 0.26) and — just as important for agents — it **collapses
run-to-run variance** (param-acc ±.009 vs ±.082; the imatrix build swings 0.51/0.48/0.38 on tool-sel
across seeds, AWQ holds 0.50/0.50/0.48). (AWQ also nearly halves PPL, 1959 → 1040 — PPL and KLD
disagree here because the fold shifts the whole distribution, lowering average surprise while nudging
the very top token slightly.) The upshot: the static table alone would have picked the wrong build. All 60 gemma layers are standard attention, so AWQ
scales the full network (120 groups); the per-channel scales fold into RMSNorm with **no inference
cost**. The higher-bit builds (IQ3_M+) don't need it — at 3-4 bits the imatrix build already
preserves arguments and AWQ's fold slightly perturbs general text, so those ship plain imatrix.

> Harness: [quant-tuner](https://github.com/pearsonkyle/quant-tuner) `run_toolcall_reps.py`. This
> lighter replay is the proxy used to rank the 2-bit builds; a full SWE-rebench re-run on the AWQ
> build is pending (the table's IQ2_M SWE-rebench figures are the prior imatrix build's).

---

## 🔬 How it was made

- **Hybrid imatrix** — activation energy `E[a²]` mixed with weight-column energy `‖W[:,i]‖²·E[a²]` per tensor, collected over real coding/tool-use logs + `wiki.test.raw` via [quant-tuner](https://github.com/pearsonkyle/quant-tuner). Used for the IQ3_M / IQ4_XS / Q5_K_S builds.
- **AWQ for the 2-bit** — the IQ2_M build additionally applies [activation-aware weight scaling](https://arxiv.org/abs/2306.00978): each input channel is rescaled by `mean(|x|)^α` (α grid-searched on the tool-use logs, sampled across the whole corpus) and the inverse folds into the preceding RMSNorm — mathematically identity in FP16, but it moves the channels that carry tool-argument signal onto finer quant levels. All 60 layers are standard attention, so the fold covers the full network; the imatrix is then collected on the folded weights. Net: **+54% tool-argument accuracy** vs plain 2-bit imatrix, at no inference cost. See [§ Why the 2-bit is AWQ](#-why-the-2-bit-is-awq).
- **IQ2_M codebook** — 2-bit E8-lattice non-uniform codes with per-tensor tier bumps (attention output, early `ffn_down` get more bits). `llama-quantize` decides the mix.
- **Vision mmproj** — the model's SigLIP-style vision tower (27 layers, 280 soft tokens/image) exported separately at **Q8_0** with `convert_hf_to_gguf.py --mmproj` (visually lossless, 772 MB), so the encoder stays high-precision while the text path runs at 2 bits. No audio encoder is shipped (the source has none).
- **Disjoint splits** — calibration (imatrix), validation (per-tensor α gate), and eval (PPL/KLD) come from different corpora; the SWE-rebench holdout never appears in any calibration set.
- Toolchain: [quant-tuner](https://github.com/pearsonkyle/quant-tuner) for imatrix calibration, [llama.cpp](https://github.com/ggml-org/llama.cpp) `@ f3e1828` for final quantization. Calibration logs mined with [LogMiner](https://github.com/pearsonkyle/LogMiner).

---

## 🚀 Usage

### Ollama

```bash
ollama run hf.co/pearsonkyle/gemma4-31b-imatrix-mtp-GGUF:IQ2_M
```

### llama.cpp (GPU)

```bash
# Build with CUDA (-DGGML_CUDA=OFF for CPU/Metal)
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON
cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-server
cp llama.cpp/build/bin/llama-* llama.cpp/

# Run the server
./llama-server \
    --model gemma-4-31B-it-IQ2_M.gguf \
    --ctx-size 16384 --n-gpu-layers 999 --split-mode layer \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
    --parallel 1 --batch-size 2048 --ubatch-size 512 \
    --host 0.0.0.0 --port 1234
```

### OpenAI-compatible API (Python)

```python
import json, urllib.request

def ask(content, max_tokens=256):
    body = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        # Gemma 4 is a thinking model — disable or raise max_tokens
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        "http://127.0.0.1:1234/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req).read())["choices"][0]["message"]["content"]

print(ask("What is 1+1?"))
```

### 🖼️ Vision (text + image)

Gemma 4 is natively multimodal. The vision tower ships **separately** as
`mmproj-gemma-4-31B-it-Q8_0.gguf` (772 MB) so you only download it if you need
images. It pairs with **any** of the four quant files (IQ2_M / IQ3_M / IQ4_XS / Q5_K_S) —
the text weights are identical; the mmproj just adds the SigLIP encoder + projector.

**One-shot from the CLI** (`llama-mtmd-cli`):

```bash
./llama-mtmd-cli \
    --model gemma-4-31B-it-IQ4_XS.gguf \
    --mmproj mmproj-gemma-4-31B-it-Q8_0.gguf \
    --image screenshot.png \
    --jinja -ngl 999 --temp 0.2 -n 256 \
    -p "Describe this image. What's in it?"
```

> `--jinja` is **required** — Gemma 4's chat template is Jinja-based and the CLI
> aborts without it. `--image` can be repeated for multi-image prompts; URLs work too.
>
> ⚠️ **Thinking + the CLI.** Gemma 4 is a reasoning model. From `llama-mtmd-cli`,
> leave thinking **on** and give it enough budget (`-n 800+`) so the answer survives
> the reasoning preamble — the `--chat-template-kwargs '{"enable_thinking":false}'`
> flag currently returns an empty completion on the CLI path. To get a clean,
> reasoning-free answer, disable thinking over the **HTTP server** instead (below).

**Vision server** — host the quant with the mmproj attached (this is exactly how the
worked example above was generated). `--jinja` is required; the vision tower is loaded
via `--mmproj`:

```bash
./llama-server \
    -m gemma-4-31B-it-IQ4_XS.gguf \
    --mmproj mmproj-gemma-4-31B-it-Q8_0.gguf \
    --jinja --ctx-size 8192 --n-gpu-layers 999 \
    --host 127.0.0.1 --port 1234
```

Vision is purely additive — drop the `--mmproj` flag and you're back to the identical text-only model. 

The OpenAI-compatible `/v1/chat/completions` endpoint then accepts `image_url` content
parts. With `chat_template_kwargs.enable_thinking=false` the server returns just the
answer (no reasoning preamble). This is the exact call used to generate the mecha
prompt above:

```python
import base64, json, urllib.request

with open("mecha.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

body = {
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": (
            "Look at this image and write a single, detailed text-to-image "
            "generation prompt that would recreate it. Cover the subject, colors, "
            "pose, lighting, style, and background. Respond with only the prompt."
        )},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]}],
    "max_tokens": 400,
    "temperature": 0.3,
    "chat_template_kwargs": {"enable_thinking": False},
}
req = urllib.request.Request(
    "http://127.0.0.1:1234/v1/chat/completions",
    json.dumps(body).encode(),
    {"Content-Type": "application/json"},
)
print(json.loads(urllib.request.urlopen(req).read())["choices"][0]["message"]["content"])
```

#### Worked example — same image, all three quants
Each shipped quant was handed the mecha on the right and asked to *"write a single,
detailed text-to-image generation prompt that would recreate it"*.

<img src="mecha.png" alt="anime mecha — white/lavender armor, neon-green energy sword, purple railgun" width="420" />


| Quant | BPW | Generated text-to-image prompt |
|:--|:--|:--|
| **IQ4_XS** | 4.36 | A high-detail anime-style illustration of a heavy armored mecha robot standing in a powerful, wide-legged stance against a solid black background. The mecha features bulky, angular white and light-grey plating with deep purple accents and glowing neon green energy vents and eyes. In its right hand, it holds a massive, oversized purple railgun, and in its left hand, a glowing neon green energy blade that casts a vibrant light across the armor. The lighting is dramatic and cinematic, with strong rim lighting and high contrast, emphasizing the metallic textures and the bioluminescent glow of the green components. The art style is clean, modern mecha anime with sharp line work and smooth digital shading. |
| **IQ3_M** | 3.76 | Full-body shot of a heavily armored futuristic mecha robot in a powerful standing pose, designed in a detailed 90s retro anime style. The mecha features a bulky, angular white and light-grey armor plating with deep purple accents and glowing neon green energy vents and sensors integrated into the chest, legs, and eyes. It is armed with a massive, oversized purple sniper rifle held in its right hand and a glowing neon green energy sword held in its left. The lighting is dramatic and moody, with a strong rim light highlighting the silhouette against a solid black background. High contrast, clean cel-shaded lines, sharp mechanical detailing, cinematic composition. |
| **IQ2_M** | 2.85 | Full-body shot of a futuristic mecha robot in a stylized anime aesthetic, featuring a heavy armored chassis in a palette of white, grey, and deep purple. The mecha is posed in a powerful stance, holding a large purple futuristic firearm in its right hand and a glowing neon-green energy blade in its left hand. The design includes glowing mint-green accents and circuitry lines across the chest, legs, and head. The lighting is dramatic and moody, with a strong rim lighting and a dark, atmospheric background with subtle purple gradients and a slight digital scanline texture. High-contrast cel-shaded style with clean lines and sharp metallic reflections. |

<br clear="right"/>

### ⚡ Speculative decoding (MTP drafter)

This repo also bundles a **multi-token-prediction (MTP) drafter** at the repo root,
[`mtp-gemma-4-31B-it.gguf`](https://huggingface.co/pearsonkyle/gemma4-31b-imatrix-mtp-GGUF/resolve/main/mtp-gemma-4-31B-it.gguf)
(358 MB, Q4_K_M) — a self-quantized conversion of
[`google/gemma-4-31B-it-assistant`](https://huggingface.co/google/gemma-4-31B-it-assistant)
(arch `gemma4-assistant`, `nextn_predict_layers = 4`). The acceptance rates where the model's drafted tokens were accepted by the trunk were measured using three reps over 5 mixed coding/reasoning prompts × 200 tokens, `temperature=0.3`, thinking off; distinct seeds per rep. 

| Quant | n=1 | n=2 | n=3 | n=4 |
|:--|--:|--:|--:|--:|
| **Q5_K_S** | 87.9% | 81.8% | 73.0% | 66.0% |
| **IQ4_XS** | 86.5% | 80.2% | 68.6% | 64.0% |
| **IQ3_M** | 87.2% | 79.1% | 70.8% | 64.6% |
| **IQ2_M** | 83.1% | 77.1% | 70.6% | 61.4% |

Acceptance holds up across all four trunks — the highest-fidelity **Q5_K_S** leads at every
draft depth (87.9% at `n=1`, still 66.0% at `n=4`), and even the 2-bit IQ2_M accepts 83% of
single-token drafts. 

**How small can the *drafter* be?** Fixing the trunk at **IQ2_M** and quantizing the drafter
itself shows acceptance barely moves down to Q4_K_M, then erodes at 2-bit — and the erosion
grows with draft depth (`n`):

| Drafter | size | n=1 | n=2 | n=3 | n=4 |
|:--|--:|--:|--:|--:|--:|
| **Q8_0** | 514 MB | 84.5 ±0.5% | 76.7 ±2.5% | 69.3 ±1.5% | 61.2 ±2.0% |
| **Q6_K** | 401 MB | 84.8 ±0.7% | 76.6 ±2.4% | 69.2 ±1.4% | 61.4 ±1.7% |
| **Q4_K_M** (shipped) | 358 MB | 85.0 ±0.4% | 76.8 ±1.9% | 69.0 ±0.4% | 60.8 ±0.9% |
| **IQ3_M** | 328 MB | 83.9 ±0.9% | 75.1 ±1.4% | 67.6 ±0.3% | 60.6 ±0.7% |
| **IQ2_M** | 269 MB | 83.7 ±0.7% | 74.1 ±2.0% | 65.9 ±2.3% | 57.9 ±2.4% |

![Acceptance vs drafter quantization (IQ2_M trunk)](drafter-quant-acceptance.png)

**Q6_K and Q4_K_M are statistically free** — indistinguishable from the Q8_0 drafter at every
depth (within ±1σ); the drafter only has to be *directionally* right for the trunk to accept.

**Usage** — add `--model-draft` + `--spec-type draft-mtp` to the server command:

```bash
./llama-server \
    -m gemma-4-31B-it-IQ4_XS.gguf \
    --model-draft mtp-gemma-4-31B-it.gguf \
    --spec-type draft-mtp --spec-draft-n-max 4 \
    --jinja -ngl 999 -fa on \
    --host 127.0.0.1 --port 1234
```

> The drafter lives at the **repo root** so `--spec-type draft-mtp` auto-discovers it
> when you load the trunk with `-hf` (no manual `--model-draft` needed):
> `llama-server -hf pearsonkyle/gemma4-31b-imatrix-mtp-GGUF:IQ4_XS --spec-type draft-mtp --spec-draft-n-max 4`.

---

## 🪪 License & attribution

- Inherits the [**Gemma Terms of Use**](https://ai.google.dev/gemma/terms) from the base model.
- Base weights: [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it).
- MTP drafter converted from [`google/gemma-4-31B-it-assistant`](https://huggingface.co/google/gemma-4-31B-it-assistant) (same Gemma Terms of Use).
- Calibration + quantization: [**Quant-Tuner**](https://github.com/pearsonkyle/Quant-Tuner) with vendored llama.cpp `@ f3e1828`.
- Calibration logs mined with [**LogMiner**](https://github.com/pearsonkyle/LogMiner).
