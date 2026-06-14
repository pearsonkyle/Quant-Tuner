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
- awq
- imatrix
- activation-aware
- cross-validation
- cv-gate
- held-out-validation
- low-bit
- 2-bit
- iq2_xs
- iq2_m
- q2_k_s
- gemma
- gemma-4
- 31b
- coder
- tool-use
- function-calling
- long-context
license: gemma
language:
- en
pipeline_tag: text-generation
---

<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; border: 1px solid #99f6e4; border-radius: 16px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05), 0 4px 6px -2px rgba(0,0,0,0.05); overflow: hidden; background: #ffffff; margin-bottom: 30px;">
  <div style="background: linear-gradient(135deg, #0d9488 0%, #134e4a 100%); padding: 24px; color: white;">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
      <h1 style="margin: 0; font-size: 26px; font-weight: 800; display: flex; align-items: center; gap: 12px; color: white; border: none;">🧊 Gemma-4-31B-it · AWQ · 2-bit · GGUF</h1>
      <span style="background: #f59e0b; color: #1c1917; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; text-transform: uppercase; letter-spacing: 0.5px;">AWQ + imatrix (hybrid)</span>
    </div>
  </div>
  <div style="display: flex; gap: 8px; flex-wrap: wrap; padding: 12px 24px; background: #f8fafc; border-bottom: 1px solid #e2e8f0;">
    <span style="background: #dbeafe; color: #1e40af; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #bfdbfe;">📦 8.88 / 10.17 / 10.22 GiB</span>
    <span style="background: #d1fae5; color: #065f46; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #a7f3d0;"> IQ2_XS / IQ2_M / Q2_K_S</span>
    <span style="background: #ede9fe; color: #5b21b6; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #ddd6fe;">+ QAT-sourced IQ2_XS / Q2_K_S</span>
    <span style="background: #fce7f3; color: #9d174d; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #fbcfe8;">🏗️ llama.cpp 32782998</span>
    <span style="background: #fef3c7; color: #92400e; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #fde68a;">🏅 medKLD 1.08 · top_p 51.1% · PPL 73</span>
    
  </div>
  <div style="padding: 24px; display: flex; flex-direction: column; gap: 20px;">
    <div style="background: #f0fdfa; border-left: 5px solid #0d9488; padding: 16px; border-radius: 0 8px 8px 0;">
      <h3 style="margin: 0 0 8px 0; font-size: 15px; color: #115e59; font-weight: 700; display: flex; align-items: center; gap: 6px;"><span>🧊</span> What this is</h3>
      <p style="margin: 0; font-size: 13px; color: #334155; line-height: 1.7;">Three aggressively compressed (under 3 bits per weight) quantizations of <b>google/gemma-4-31B-it</b>. Before quantizing, each linear layer is rescaled by a per-channel factor (the <b>AWQ</b> trick: Activation-aware Weight Quantization) so that outlier channels don't blow up the 2-bit codebook. We search for the best rescale strength on a calibration text, but only keep an aggressive per-tensor choice if it <b>also</b> improves loss on a separate <b>held-out</b> text it never saw during the search; otherwise we fall back to a safer default. The rescale is absorbed into the preceding RMSNorm layer, so the file is a plain GGUF with <b>no custom runtime and no extra inference cost</b>.</p>
    </div>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 10px;">
      <div style="border: 1px solid #e2e8f0; padding: 14px; border-radius: 8px; background: #fafafa; box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);"><span style="font-weight: 700; color: #115e59; font-size: 12px; display: block; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">📉 ~5-6.5x smaller size (Gb)</span><span style="font-size: 13px; color: #4b5563; line-height: 1.5;">At ~2 bits per weight, these quants are under 11 GiB on disk vs 57.2 GiB for FP16.</span></div>
      <div style="border: 1px solid #e2e8f0; padding: 14px; border-radius: 8px; background: #fafafa; box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);"><span style="font-weight: 700; color: #115e59; font-size: 12px; display: block; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">🎯 up to 51.1% top-p at 2-bit</span><span style="font-size: 13px; color: #4b5563; line-height: 1.5;">Top-token agreement with FP16: <b>51.1%</b> on vanilla Q2_K_S-AWQ, <b>49.4%</b> on QAT Q2_K_S-AWQ, <b>48.9%</b> on QAT IQ2_XS-AWQ — best-in-class at this bit budget and ~2× the top-p of plain Q2_K.</span></div>
      <div style="border: 1px solid #e2e8f0; padding: 14px; border-radius: 8px; background: #fafafa; box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);"><span style="font-weight: 700; color: #115e59; font-size: 12px; display: block; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">🛠️ Standard GGUF</span><span style="font-size: 13px; color: #4b5563; line-height: 1.5;">Loads in vanilla llama.cpp / llama-server / LM Studio with no custom runtime, kernels, or patches.</span></div>
    </div>
  </div>
</div>

## 🧰 1. Files & comparison

Three **vanilla-source** quants and two **QAT-source** quants (from `google/gemma-4-31B-it-qat-q4_0-unquantized` — Google's quantization-aware-trained checkpoint) at the AWQ cv-gate recipe, each paired with its imatrix-only baseline. Plain Q2_K is shipped as the no-calibration anchor.

FP16 reference: 57.20 GiB (not included; fetch from [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it)).

Corpora under `calibration_data/`:

| File | Role |
|---|---|
| `corpus.cal.txt` | imatrix collection + AWQ α search (wiki.test.raw + logtrain TRAIN slice) |
| `corpus.val.txt` | held-out gate for per-tensor α (MMMU disciplines) |
| `corpus.eval.txt` | PPL / KLD eval (external code+math+tools, ~90k tokens) |

All rows benched on the same `corpus.eval.txt`, same llama.cpp build, **ctx=4096**. AWQ calibrate uses **proxy_tokens=1024, ctx=4096**; the Q2_K_S AWQ rows use the new `q2k_super` codebook proxy and the IQ2_M AWQ row uses `q2k_b16` base + `iq3_s` mix for the IQ3_S-bumped tensors. KLD and top_p are measured against `google/gemma-4-31B-it` FP16 (so QAT and vanilla rows are directly comparable). KLD column is **median** for robustness to per-token tails.

#### Vanilla `google/gemma-4-31B-it` source

| File | Quant | Technique | Size (GiB) | BPW | PPL | KLD (median) | same_top_p |
|---|---|---|---:|---:|---:|---:|---:|
| n/a | FP16 | none (reference) | 57.20 | 16.005 | **277.89** | 0.00000 | 100.00% |
| [`gemma-4-31B-it-Q2_K-plain.gguf`](https://huggingface.co/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/resolve/main/gemma-4-31B-it-Q2_K-plain.gguf) | Q2_K | plain (no imatrix, no AWQ) | 11.10 | 3.105 | 3370.57 | 5.147 | 25.83% |
|||||||||
| [`gemma-4-31B-it-IQ2_XS-imatrix.gguf`](https://huggingface.co/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/resolve/main/gemma-4-31B-it-IQ2_XS-imatrix.gguf) | IQ2_XS | imatrix only (baseline) | 8.88 | 2.484 | 12116.47 | 3.327 | 33.23% |
| [`gemma-4-31B-it-IQ2_XS-awq-cv-gate.gguf`](https://huggingface.co/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/resolve/main/gemma-4-31B-it-IQ2_XS-awq-cv-gate.gguf) | **IQ2_XS** | **AWQ cv-gate + imatrix** | **8.88** | **2.484** | **327.28** | **1.817** | **46.29%** |
| [`gemma-4-31B-it-IQ2_M-imatrix.gguf`](https://huggingface.co/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/resolve/main/gemma-4-31B-it-IQ2_M-imatrix.gguf) | IQ2_M | imatrix only (baseline) | 10.17 | 2.845 | 2060.73 | 1.496 | 47.61% |
| [`gemma-4-31B-it-IQ2_M-awq-cv-gate.gguf`](https://huggingface.co/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/resolve/main/gemma-4-31B-it-IQ2_M-awq-cv-gate.gguf) | **IQ2_M** | **AWQ cv-gate + imatrix (q2k_b16 + iq3_s mix)** | **10.17** | **2.845** | **652.81** | **1.548** | **48.79%** |
| [`gemma-4-31B-it-Q2_K_S-imatrix.gguf`](https://huggingface.co/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/resolve/main/gemma-4-31B-it-Q2_K_S-imatrix.gguf) | Q2_K_S | imatrix only (baseline) | 10.22 | 2.861 | 1436.40 | 2.138 | 42.60% |
| [`gemma-4-31B-it-Q2_K_S-awq-cv-gate.gguf`](https://huggingface.co/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/resolve/main/gemma-4-31B-it-Q2_K_S-awq-cv-gate.gguf) | **Q2_K_S** | **AWQ cv-gate + imatrix (q2k_super)** | **10.22** | **2.861** | **73.09** | **1.632** | **51.14%** |

#### QAT `google/gemma-4-31B-it-qat-q4_0-unquantized` source

| File | Quant | Technique | Size (GiB) | BPW | PPL | KLD (median) | same_top_p |
|---|---|---|---:|---:|---:|---:|---:|
| [`google/gemma-4-31B-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized) | Q4_0 | QAT (Google official, reference only) | 16.44 | 4.600 | 78.19 | 0.913¹ | 64.39% |
|||||||||
| [`gemma-4-31B-it-qat-IQ2_XS-imatrix.gguf`](https://huggingface.co/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/resolve/main/gemma-4-31B-it-qat-IQ2_XS-imatrix.gguf) | IQ2_XS | imatrix only (from QAT) | 8.88 | 2.484 | 209.00 | 1.270 | 47.09% |
| [`gemma-4-31B-it-qat-IQ2_XS-awq-cv-gate.gguf`](https://huggingface.co/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/resolve/main/gemma-4-31B-it-qat-IQ2_XS-awq-cv-gate.gguf) | **IQ2_XS** | **AWQ cv-gate + imatrix (from QAT)** | **8.88** | **2.484** | **108.65** | **1.151** | **48.94%** |
| [`gemma-4-31B-it-qat-Q2_K_S-imatrix.gguf`](https://huggingface.co/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/resolve/main/gemma-4-31B-it-qat-Q2_K_S-imatrix.gguf) | Q2_K_S | imatrix only (from QAT) | 10.22 | 2.861 | 110.71 | 1.332 | 47.65% |
| [`gemma-4-31B-it-qat-Q2_K_S-awq-cv-gate.gguf`](https://huggingface.co/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/resolve/main/gemma-4-31B-it-qat-Q2_K_S-awq-cv-gate.gguf) | **Q2_K_S** | **AWQ cv-gate + imatrix (from QAT, q2k_super)** | **10.22** | **2.861** | **88.18** | **1.081** | **49.40%** |

> ¹ Q4_0 reference row carries the mean KLD from Google's measurement; not re-benched at median.

> **No IQ2_M-from-QAT row.** IQ2_M quantization of the QAT weights collapses to garbage across every arm tested (PPL ≈ 2×10¹⁰, top_p = 0%) — the IQ2_M codebook and QAT-shaped weights are geometrically incompatible. Use the vanilla-source `IQ2_M-awq-cv-gate`, or step to the QAT-sourced Q2_K_S / IQ2_XS.

<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; border: 1px solid #99f6e4; border-radius: 12px; background: #f0fdfa; padding: 16px; margin: 16px 0; color: #115e59; font-size: 13px; line-height: 1.7;">
  <b>Headline.</b> <b>qat-Q2_K_S-awq-cv-gate</b> wins on KLD/PPL (median KLD 1.081, PPL 88.2). <b>Q2_K_S-awq-cv-gate</b> (vanilla source) leads on top_p at <b>51.1%</b> with PPL 73.1. <b>qat-IQ2_XS-awq-cv-gate</b> is the best size/quality pick at 8.88 GiB (KLD 1.151, top_p 48.9%). AWQ cv-gate beats its imatrix-only baseline on PPL by <b>4–20×</b> at every working bit budget; plain Q2_K (no calibration) loses ~25 absolute top-p points despite a larger file.
</div>

![Comparison: AWQ cv-gate release vs imatrix-only and FP16](./awq_cv_gate_release.png)

<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; border: 1px solid #fde68a; border-radius: 12px; background: #fffbeb; padding: 16px; margin: 16px 0; color: #92400e; font-size: 13px; line-height: 1.7;">
  <b>⚠️ Caveat.</b> These are sub-3-bpw quants of a 31B reasoning model. They are meaningfully better than the alternatives at the same size, but they are <b>not</b> a substitute for FP16 / Q4_K_M / Q5_K_M when you have the VRAM. Use them when memory is the binding constraint.
</div>

---

## 🔬 2. How they were made

<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; display: flex; flex-direction: column; gap: 16px; margin-bottom: 24px;">
  <div style="border: 1px solid #99f6e4; border-radius: 12px; overflow: hidden; background: #ffffff;">
    <div style="background: linear-gradient(135deg, #0d9488 0%, #115e59 100%); padding: 12px 16px; color: white; font-weight: 700; font-size: 14px;">⚖️ 2.1 AWQ: rescale before quantizing</div>
    <div style="padding: 16px; font-size: 13px; color: #334155; line-height: 1.7;">
      <p style="margin: 0 0 10px 0;">At 2-bit each weight has only 4 possible codebook values, so a handful of outlier channels can wreck quantization error for an entire layer. <b>AWQ</b> (activation-aware weight quantization) sidesteps this: for every linear <code>y = W · a</code>, it picks a per-channel scale <code>s</code> and rewrites</p>
      <pre style="margin:0 0 10px 0; padding:10px 12px; background:#f0fdfa; border:1px solid #99f6e4; border-radius:8px; font-size:12px; color:#115e59; overflow-x:auto;"><code>y  =  W · a  =  (W · diag(s))  ·  (diag(1/s) · a)
              └──────┬──────┘    └──────┬──────┘
              quantize this       fold into prev RMSNorm gain</code></pre>
      <p style="margin: 0 0 10px 0;">Math-equivalent to the original layer, but the rescaled weight matrix has a flatter per-channel range so the 2-bit codebook fits it with less error. The inverse scale gets absorbed into the preceding RMSNorm, so there is <b>no runtime overhead</b> and the GGUF stays standard.</p>
      <p style="margin: 0 0 10px 0;">Baseline AWQ picks one shared <b>α</b> per group of layers that share an input (e.g. q/k/v). This release adds three refinements:</p>
      <ol style="margin: 0; padding-left: 20px;">
        <li><b>Per-tensor α refinement.</b> Each member of a group (q, k, v individually) gets to nudge its α within a small local grid around the group choice, lowering its own reconstruction error.</li>
        <li><b>Binary held-out gate.</b> The per-tensor α is only accepted if it doesn't worsen the proxy loss on a <i>disjoint validation slice</i>. If it would, the gate rejects it and the tensor falls back to the safer group α. Without the gate, per-tensor refinement over-fits the calibration corpus at sub-3 bpw and PPL collapses on unseen text.</li>
        <li><b>Codebook-faithful proxy quantizers.</b> The α search scores candidates against a proxy quantizer; an inaccurate proxy drifts the optimum. IQ2_* targets use bit-exact E8-lattice codebook proxies (matching llama.cpp's <code>iq2xxs/xs/s_grid</code>); Q2_K_S uses the new <code>q2k_super</code> proxy mirroring Q2_K's real 256-weight super-block; IQ2_M uses <code>q2k_b16</code> for the iq2_s tensors and routes the IQ3_S-bumped tensors through a faithful <code>iq3_s</code> codebook proxy.</li>
      </ol>
    </div>
  </div>
  <div style="border: 1px solid #fde68a; border-radius: 12px; overflow: hidden; background: #ffffff;">
    <div style="background: linear-gradient(135deg, #f59e0b 0%, #b45309 100%); padding: 12px 16px; color: white; font-weight: 700; font-size: 14px;">🧮 2.2 Why AWQ beats imatrix alone</div>
    <div style="padding: 16px; font-size: 13px; color: #334155; line-height: 1.7;">
      <p style="margin: 0 0 10px 0;">Both techniques look at the same calibration activations, but they spend the signal differently:</p>
      <ul style="margin: 0 0 10px 0; padding-left: 20px;">
        <li><b>Imatrix only.</b> Tells the quantizer <i>which</i> channels are important via <code>E[a²]ᵢ</code> so the codebook spends more precision on them. The weight numerics themselves don't change, so outlier weights still exist and still cause large per-bin errors.</li>
        <li><b>AWQ.</b> Actually rewrites the weight matrix to be easier to quantize. Channels that drive the output get scaled down in the weight domain (their range shrinks), so a 4-value codebook covers them with less error. The rescaling is folded into the preceding norm, so the math the layer computes is unchanged.</li>
      </ul>
      <p style="margin: 0;">They are complementary, and this release uses both: <b>AWQ scales the weights</b>, then <b>a hybrid imatrix</b> (<code>E[a²]</code> mixed with weight-column energy <code>‖W[:, i]‖² · E[a²]</code>) guides the final <code>llama-quantize</code> pass. Pure imatrix can't fix outlier weights; AWQ alone under-uses the per-channel sensitivity signal during the quantizer's bin assignment.</p>
    </div>
  </div>
  <div style="border: 1px solid #bfdbfe; border-radius: 12px; overflow: hidden; background: #ffffff;">
    <div style="background: linear-gradient(135deg, #2563eb 0%, #1e3a8a 100%); padding: 12px 16px; color: white; font-weight: 700; font-size: 14px;">📚 2.3 Data slices</div>
    <div style="padding: 16px; font-size: 13px; color: #334155; line-height: 1.7;">
      <p style="margin: 0 0 10px 0;">Three disjoint corpora with distinct roles. The gate uses validation text only to make a binary accept/reject decision per tensor (a very low-capacity signal), and validation never feeds the eval numbers in §1.</p>
      <table style="width:100%; border-collapse:collapse; font-size:12px; margin:0;">
        <thead><tr style="background:#eff6ff;"><th style="padding:8px 10px; border:1px solid #bfdbfe; text-align:left; color:#1e3a8a;">Slice</th><th style="padding:8px 10px; border:1px solid #bfdbfe; text-align:left; color:#1e3a8a;">Source</th><th style="padding:8px 10px; border:1px solid #bfdbfe; text-align:left; color:#1e3a8a;">Used for</th></tr></thead>
        <tbody>
          <tr><td style="padding:8px 10px; border:1px solid #bfdbfe;"><b>Calibration</b></td><td style="padding:8px 10px; border:1px solid #bfdbfe;">~500k tokens of usage-log + all of <code>wiki.test.raw</code></td><td style="padding:8px 10px; border:1px solid #bfdbfe;">imatrix collection + AWQ α search</td></tr>
          <tr><td style="padding:8px 10px; border:1px solid #bfdbfe;"><b>Validation</b></td><td style="padding:8px 10px; border:1px solid #bfdbfe;">MMMU disciplines corpus (~100–200k tokens) drawn from <code>calibration_supplements/mmmu/combined.txt</code>, out-of-distribution relative to the calibration mix</td><td style="padding:8px 10px; border:1px solid #bfdbfe;">held-out gate for per-tensor α</td></tr>
          <tr><td style="padding:8px 10px; border:1px solid #bfdbfe;"><b>Eval</b></td><td style="padding:8px 10px; border:1px solid #bfdbfe;">~90k tokens from <a href="https://huggingface.co/datasets/eaddario/imatrix-calibration"><code>eaddario/imatrix-calibration</code></a> (code+math+tools)</td><td style="padding:8px 10px; border:1px solid #bfdbfe;">all numbers in §1; neither calibration nor validation appears here</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>

Toolchain: AWQ + imatrix orchestrated by [`quant-tuner`](https://github.com/pearsonkyle/quant-tuner); final quantization with `llama-quantize` from [llama.cpp](https://github.com/ggerganov/llama.cpp) pinned to commit `32782998`.

---
## 🚀 3. Usage

### Building llama.cpp from source (GPU)

To run these models on the GPU, build llama.cpp with CUDA support:

```bash
# Install dependencies and clone repo
apt-get update && apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y
git clone https://github.com/ggml-org/llama.cpp

# Build with CUDA (set -DGGML_CUDA=OFF for CPU/Metal)
cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON
cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-server

# Move binaries
cp llama.cpp/build/bin/llama-* llama.cpp/
```

### Running the server 
```bash
# Start the server with your chosen model
 ./llama-server \
    --model gemma-4-31B-it-IQ2_M-imatrix.gguf \
    --ctx-size 16384 \
    --n-gpu-layers 999 \
    --split-mode layer \
    --flash-attn on \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --parallel 1 \
    --batch-size 2048 \
    --ubatch-size 512 \
    --host 0.0.0.0 \
    --port 1234
```

### Querying via the OpenAI compatible API

```python
import json, base64, urllib.request

def ask(content, max_tokens=256):
    body = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        # Gemma 4 is a thinking model. Set this to False (or raise max_tokens),
        # otherwise the reply lands in reasoning_content and "content" is empty.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request("http://127.0.0.1:1234/v1/chat/completions",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())["choices"][0]["message"]["content"]

b64 = lambda p: base64.b64encode(open(p, "rb").read()).decode()

# Text
print(ask("What is 1+1?"))
```

**Which file to pick:**
- **`qat-Q2_K_S-awq-cv-gate`** (10.22 GiB) — best KLD/PPL overall (1.081 / 88.2). Closest to FP16 at this bit budget.
- **`qat-IQ2_XS-awq-cv-gate`** (8.88 GiB) — best size/quality (1.151 / 108.6, top_p 48.9%). Pick when memory is binding.
- **`Q2_K_S-awq-cv-gate`** (10.22 GiB, vanilla) — best top_p (51.1%) and PPL (73.1); higher KLD (1.632) reflects heavier per-token tail.
- **`qat-Q2_K_S-imatrix`** / **`qat-IQ2_XS-imatrix`** — imatrix-only QAT baselines; pick if AWQ is disallowed.
- **`IQ2_M-awq-cv-gate`** (10.17 GiB, vanilla) — the only working IQ2_M at this bit budget. QAT IQ2_M is broken (§1).
- **`IQ2_XS-awq-cv-gate`** (vanilla) — pick over its QAT sibling only if licensing requires vanilla provenance.
- **`IQ2_XS-imatrix`** / **`IQ2_M-imatrix`** / **`Q2_K_S-imatrix`** (vanilla) — imatrix-only baselines, shipped for AWQ-vs-imatrix comparison.
- **`Q2_K-plain`** (11.10 GiB) — no-calibration anchor; not recommended for use.

---

## 🪪 4. License & attribution

* Inherits the [**Gemma Terms of Use**](https://ai.google.dev/gemma/terms) from the base model.
* Base weights — vanilla-source files: [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it).
* Base weights — QAT-source files (`qat-*`): [`google/gemma-4-31B-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized), Google's quantization-aware-trained FP16 checkpoint. The Q4_0 reference row in §1 links to the matching official GGUF release.
* Calibration + AWQ scaling + quantization performed locally with [**Quant-Tuner**](https://github.com/pearsonkyle/Quant-Tuner); vendored llama.cpp at commit `32782998`.
* Calibration data (usage logs) scraped using [**LogMiner**](https://github.com/pearsonkyle/LogMiner).