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
    <span style="background: #fce7f3; color: #9d174d; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #fbcfe8;">🏗️ llama.cpp 45b455e6</span>
    
  </div>
  <div style="padding: 24px; display: flex; flex-direction: column; gap: 20px;">
    <div style="background: #f0fdfa; border-left: 5px solid #0d9488; padding: 16px; border-radius: 0 8px 8px 0;">
      <h3 style="margin: 0 0 8px 0; font-size: 15px; color: #115e59; font-weight: 700; display: flex; align-items: center; gap: 6px;"><span>🧊</span> What this is</h3>
      <p style="margin: 0; font-size: 13px; color: #334155; line-height: 1.7;">Three aggressively compressed (under 3 bits per weight) quantizations of <b>google/gemma-4-31B-it</b>. Before quantizing, each linear layer is rescaled by a per-channel factor (the <b>AWQ</b> trick: Activation-aware Weight Quantization) so that outlier channels don't blow up the 2-bit codebook. We search for the best rescale strength on a calibration text, but only keep an aggressive per-tensor choice if it <b>also</b> improves loss on a separate <b>held-out</b> text it never saw during the search; otherwise we fall back to a safer default. The rescale is absorbed into the preceding RMSNorm layer, so the file is a plain GGUF with <b>no custom runtime and no extra inference cost</b>.</p>
    </div>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 10px;">
      <div style="border: 1px solid #e2e8f0; padding: 14px; border-radius: 8px; background: #fafafa; box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);"><span style="font-weight: 700; color: #115e59; font-size: 12px; display: block; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">📉 ~5-6.5x smaller size (Gb)</span><span style="font-size: 13px; color: #4b5563; line-height: 1.5;">At ~2 bits per weight, these quants are under 11 GiB on disk vs 57.2 GiB for FP16.</span></div>
      <div style="border: 1px solid #e2e8f0; padding: 14px; border-radius: 8px; background: #fafafa; box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);"><span style="font-weight: 700; color: #115e59; font-size: 12px; display: block; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">🎯 up to 51% top-p at 2-bit</span><span style="font-size: 13px; color: #4b5563; line-height: 1.5;">Top-token agreement with FP16: <b>50.9%</b> on the QAT-sourced IQ2_XS and <b>50.4%</b> on the QAT-sourced Q2_K_S — the best numbers at this bit budget and ~2× the top-p of plain Q2_K.</span></div>
      <div style="border: 1px solid #e2e8f0; padding: 14px; border-radius: 8px; background: #fafafa; box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);"><span style="font-weight: 700; color: #115e59; font-size: 12px; display: block; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">🛠️ Standard GGUF</span><span style="font-size: 13px; color: #4b5563; line-height: 1.5;">Loads in vanilla llama.cpp / llama-server / LM Studio with no custom runtime, kernels, or patches.</span></div>
    </div>
  </div>
</div>

## 🧰 1. Files & comparison

The three **vanilla-source AWQ cv-gate** quants are the original headline release. Two **QAT-sourced AWQ cv-gate** quants (built from `google/gemma-4-31B-it-qat-q4_0-unquantized` — Google's quantization-aware-trained checkpoint, released in their FP16 form before the official Q4_0 quantization) are added in the second block of the table. The matched **imatrix-only** baselines (identical bit budget, no AWQ) and the **plain Q2_K** anchor (no calibration of any kind) are also shipped so the comparison below is fully reproducible.

FP16 reference: 57.20 GiB, 16.005 BPW (not included; fetch from [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it)).

The exact text corpora used to produce these quants ship under `calibration_data/`:

| File | Role |
|---|---|
| `calibration_data/corpus.cal.txt` | imatrix collection + AWQ α search (wiki.test.raw + logtrain TRAIN slice) |
| `calibration_data/corpus.val.txt` | held-out gate for per-tensor α (MMMU disciplines) |
| `calibration_data/corpus.eval.txt` | PPL / KLD eval (external code+math+tools, ~90k tokens) |
| `calibration_data/corpora_audit.json` | source provenance + token counts + seed |

### Comparison

All rows benched on the same eval corpus (~90k tokens from [`eaddario/imatrix-calibration`](https://huggingface.co/datasets/eaddario/imatrix-calibration): code + math + tools), same llama.cpp build, **ctx=4096** for both imatrix collection and PPL/KLD eval. AWQ calibrate uses **proxy_tokens=512, ctx=4096** for every shipped AWQ row (the recipe selected by the exp-022 → exp-026 proxy/ctx sweep, written up in `out/exp-022..027/`). Neither the calibration nor the validation slice appears in the eval corpus. KLD and same_top_p are measured against the original `google/gemma-4-31B-it` FP16, so QAT-sourced rows and vanilla-sourced rows are directly comparable.

#### Vanilla `google/gemma-4-31B-it` source

| File | Quant | Technique | Size (GiB) | BPW | PPL | KLD (mean) | same_top_p |
|---|---|---|---:|---:|---:|---:|---:|
| n/a | FP16 | none (reference) | 57.20 | 16.005 | **277.89** | 0.00000 | 100.00% |
| [`gemma-4-31B-it-Q2_K-plain.gguf`](https://huggingface.co/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/resolve/main/gemma-4-31B-it-Q2_K-plain.gguf) | Q2_K | plain (no imatrix, no AWQ) | 11.10 | 3.105 | 3370.57 | 6.119 | 25.83% |
|||||||||
| [`gemma-4-31B-it-IQ2_XS-imatrix.gguf`](https://huggingface.co/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/resolve/main/gemma-4-31B-it-IQ2_XS-imatrix.gguf) | IQ2_XS | imatrix only (baseline) | 8.88 | 2.484 | 12116.47 | 4.584 | 33.23% |
| [`gemma-4-31B-it-IQ2_XS-awq-cv-gate.gguf`](https://huggingface.co/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/resolve/main/gemma-4-31B-it-IQ2_XS-awq-cv-gate.gguf) | **IQ2_XS** | **AWQ cv-gate + imatrix** | **8.88** | **2.484** | **857.40** | **3.464** | **41.83%** |
| [`gemma-4-31B-it-IQ2_M-imatrix.gguf`](https://huggingface.co/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/resolve/main/gemma-4-31B-it-IQ2_M-imatrix.gguf) | IQ2_M | imatrix only (baseline) | 10.17 | 2.845 | 2060.73 | 2.685 | 47.61% |
| [`gemma-4-31B-it-IQ2_M-awq-cv-gate.gguf`](https://huggingface.co/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/resolve/main/gemma-4-31B-it-IQ2_M-awq-cv-gate.gguf) | **IQ2_M** | **AWQ cv-gate + imatrix** | **10.17** | **2.845** | **1224.40** | **3.133** | **45.13%** |
| [`gemma-4-31B-it-Q2_K_S-imatrix.gguf`](https://huggingface.co/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/resolve/main/gemma-4-31B-it-Q2_K_S-imatrix.gguf) | Q2_K_S | imatrix only (baseline) | 10.22 | 2.861 | 1436.40 | 3.867 | 42.60% |
| [`gemma-4-31B-it-Q2_K_S-awq-cv-gate.gguf`](https://huggingface.co/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/resolve/main/gemma-4-31B-it-Q2_K_S-awq-cv-gate.gguf) | **Q2_K_S** | **AWQ cv-gate + imatrix** | **10.22** | **2.861** | **124.32** | **3.418** | **48.92%** |

#### QAT `google/gemma-4-31B-it-qat-q4_0-unquantized` source

| File | Quant | Technique | Size (GiB) | BPW | PPL | KLD (mean) | same_top_p |
|---|---|---|---:|---:|---:|---:|---:|
| [`google/gemma-4-31B-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized) | Q4_0 | QAT (Google official, reference only) | 16.44 | 4.600 | 78.19 | 0.913 | 64.39% |
|||||||||
| [`gemma-4-31B-it-qat-IQ2_XS-imatrix.gguf`](https://huggingface.co/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/resolve/main/gemma-4-31B-it-qat-IQ2_XS-imatrix.gguf) | IQ2_XS | imatrix only (from QAT) | 8.88 | 2.484 | 209.00 | 1.859 | 47.09% |
| [`gemma-4-31B-it-qat-IQ2_XS-awq-cv-gate.gguf`](https://huggingface.co/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/resolve/main/gemma-4-31B-it-qat-IQ2_XS-awq-cv-gate.gguf) | **IQ2_XS** | **AWQ cv-gate + imatrix (from QAT)** | **8.88** | **2.484** | **127.40** | **1.769** | **50.87%** |
| [`gemma-4-31B-it-qat-Q2_K_S-imatrix.gguf`](https://huggingface.co/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/resolve/main/gemma-4-31B-it-qat-Q2_K_S-imatrix.gguf) | Q2_K_S | imatrix only (from QAT) | 10.22 | 2.861 | 110.71 | 1.820 | 47.65% |
| [`gemma-4-31B-it-qat-Q2_K_S-awq-cv-gate.gguf`](https://huggingface.co/pearsonkyle/gemma-4-31B-it-awq-2bit-GGUF/resolve/main/gemma-4-31B-it-qat-Q2_K_S-awq-cv-gate.gguf) | **Q2_K_S** | **AWQ cv-gate + imatrix (from QAT)** | **10.22** | **2.861** | **77.66** | **1.833** | **50.42%** |

> **No IQ2_M-from-QAT row.** IQ2_M quantization of the QAT-unquantized weights collapses to garbage in both imatrix-only and AWQ arms (PPL ≈ 2×10¹⁰, mean KLD ≈ 23, same_top_p = 0%). The failure reproduces with no calibration involved — it appears to be a hard geometric incompatibility between the IQ2_M codebook and the QAT-shaped weight distribution. We chose not to ship a broken file. Use the vanilla-source `IQ2_M-awq-cv-gate` instead at that bit budget, or step up/down to the QAT-sourced Q2_K_S / IQ2_XS.

<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; border: 1px solid #99f6e4; border-radius: 12px; background: #f0fdfa; padding: 16px; margin: 16px 0; color: #115e59; font-size: 13px; line-height: 1.7;">
  <b>Reading the table.</b> Three independent signals stack at the same 2-bit budget. (1) <b>Calibration vs none</b>: AWQ cv-gate beats imatrix-only on PPL by <b>2–12×</b> at each bit budget; plain Q2_K (no calibration, ~3.1 bpw) loses ~22 absolute top-p points despite a larger file. (2) <b>QAT source vs vanilla source</b>: starting from Google's QAT-unquantized checkpoint cuts mean KLD by <b>~2×</b> across every working bit budget — even plain Q2_K jumps from 25.8% → 42.6% top_p with no calibration at all, just from the change in source weights. QAT shapes the weight distribution to be inherently codebook-friendlier. (3) <b>AWQ on top of QAT — does both jobs</b>: at the new <code>proxy=512, ctx=4096</code> recipe (see exp-026 in the toolchain notes), QAT-IQ2_XS-AWQ reaches <b>KLD 1.769</b> and <b>top_p 50.87%</b>, beating its imatrix-only QAT baseline on both metrics. QAT-Q2_K_S-AWQ reaches <b>KLD 1.833</b> and <b>top_p 50.42%</b> — small KLD regression versus QAT-Q2_K_S-imatrix (1.820) but a +2.8pt top_p win and a 30% PPL drop. The headline row is now <b>QAT IQ2_XS AWQ cv-gate</b>: same KLD/top_p as Q2_K_S at <b>1.3 GiB less</b> on disk.
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
      <p style="margin: 0 0 10px 0;">Baseline AWQ picks one shared <b>α</b> per group of layers that share an input (e.g. q/k/v). This release adds two refinements:</p>
      <ol style="margin: 0; padding-left: 20px;">
        <li><b>Per-tensor α refinement.</b> Each member of a group (q, k, v individually) gets to nudge its α within a small local grid around the group choice, lowering its own reconstruction error.</li>
        <li><b>Binary held-out gate.</b> The per-tensor α is only accepted if it doesn't worsen the proxy loss on a <i>disjoint validation slice</i>. If it would, the gate rejects it and the tensor falls back to the safer group α. Without the gate, per-tensor refinement over-fits the calibration corpus at sub-3 bpw and PPL collapses on unseen text.</li>
      </ol>
      <p style="margin: 10px 0 0 0;"><b>Why not just merge the held-out text into the search?</b> Then it's no longer held-out. At 2-bit the α grid is expressive enough that some candidate will lower loss on any text by chance, so the search needs a second distribution it can't peek at to tell "generalizes" apart from "memorized the corpus." The gate spends that signal sparingly (one bit per tensor: accept the aggressive α, or fall back to the group α), which is too low-capacity to overfit even across thousands of tensors. Using val to pick α directly would re-introduce the same overfitting problem and require a third corpus to guard against it.</p>
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

Toolchain: AWQ + imatrix orchestrated by [`quant-tuner`](https://github.com/pearsonkyle/quant-tuner); final quantization with `llama-quantize` from [llama.cpp](https://github.com/ggerganov/llama.cpp) pinned to commit `45b455e6`.

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
- **`qat-IQ2_XS-awq-cv-gate`** (8.88 GiB) — **recommended default**. KLD 1.769, top_p 50.87% — the best numbers in the table at the smallest size in the AWQ class. Built from Google's QAT-unquantized checkpoint with the proxy=512+ctx=4096 recipe (exp-026).
- **`qat-Q2_K_S-awq-cv-gate`** (10.22 GiB) — same top_p (~50.4%) but slightly worse KLD (1.833) at 1.3 GiB more on disk. Pick over IQ2_XS only when llama.cpp performance on Q2_K_S is meaningfully better on your hardware (rare; profile to be sure).
- **`qat-Q2_K_S-imatrix`** (10.22 GiB) — best **KLD** in the 10.22 GiB tier (1.820 — slightly better than the AWQ variant's 1.833) but loses ~2.8 top_p. Pick this when you're optimizing pure FP16-distribution faithfulness over top-token agreement.
- **`qat-IQ2_XS-imatrix`** (8.88 GiB) — only beats its AWQ sibling on neither metric; ships as the AWQ-comparison baseline.
- **`IQ2_M-awq-cv-gate`** (10.17 GiB, vanilla source) — only IQ2_M file shipped. The QAT-source IQ2_M is broken (see callout in §1) so the vanilla-source build is the only working option at this exact bit budget. Note: the prior release (proxy=256, ctx=1024) had stronger top_p numbers here than the current recipe (49.4% vs 45.1%); the new recipe was kept for consistency with the rest of the table — see exp-027 in the experiment notes.
- **`Q2_K_S-awq-cv-gate`** / **`IQ2_XS-awq-cv-gate`** (vanilla source) — pick these over the QAT counterparts if you need provenance traceable to `google/gemma-4-31B-it` only (e.g. licensing or audit reasons that disqualify the QAT-distilled checkpoint).
- **`Q2_K-plain`** (11.10 GiB) — for reproducing the no-calibration baseline only. Not recommended for use.

---

## 🪪 4. License & attribution

* Inherits the [**Gemma Terms of Use**](https://ai.google.dev/gemma/terms) from the base model.
* Base weights — vanilla-source files: [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it).
* Base weights — QAT-source files (`qat-*`): [`google/gemma-4-31B-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized), Google's quantization-aware-trained FP16 checkpoint. The Q4_0 reference row in §1 links to the matching official GGUF release.
* Calibration + AWQ scaling + quantization performed locally with [**Quant-Tuner**](https://github.com/pearsonkyle/Quant-Tuner); vendored llama.cpp at commit `45b455e6`.
* Calibration data (usage logs) scraped using [**LogMiner**](https://github.com/pearsonkyle/LogMiner).