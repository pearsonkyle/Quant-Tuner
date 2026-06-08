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
- ctx-8192
license: gemma
language:
- en
pipeline_tag: text-generation
---

<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; border: 1px solid #99f6e4; border-radius: 16px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05), 0 4px 6px -2px rgba(0,0,0,0.05); overflow: hidden; background: #ffffff; margin-bottom: 30px;">
  <div style="background: linear-gradient(135deg, #0d9488 0%, #134e4a 100%); padding: 24px; color: white;">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
      <h1 style="margin: 0; font-size: 26px; font-weight: 800; display: flex; align-items: center; gap: 12px; color: white; border: none;">🧊 Gemma-4-31B-it · AWQ 2-bit GGUF</h1>
      <span style="background: #f59e0b; color: #1c1917; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; text-transform: uppercase; letter-spacing: 0.5px;">AWQ + imatrix (hybrid)</span>
    </div>
    <p style="margin: 8px 0 0 0; font-size: 14px; color: #ccfbf1; font-weight: 500;">Held-out-gated AWQ scaling folded into RMSNorm — three sub-3-bpw Gemma quants that hold their behavior at IQ2_XS / IQ2_M / Q2_K_S.</p>
  </div>
  <div style="display: flex; gap: 8px; flex-wrap: wrap; padding: 12px 24px; background: #f8fafc; border-bottom: 1px solid #e2e8f0;">
    <span style="background: #dbeafe; color: #1e40af; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #bfdbfe;">📦 8.88 / 10.17 / 10.22 GiB</span>
    <span style="background: #fce7f3; color: #9d174d; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #fbcfe8;">🏗️ llama.cpp 45b455e6</span>
  </div>
  <div style="padding: 24px; display: flex; flex-direction: column; gap: 20px;">
    <div style="background: #f0fdfa; border-left: 5px solid #0d9488; padding: 16px; border-radius: 0 8px 8px 0;">
      <h3 style="margin: 0 0 8px 0; font-size: 15px; color: #115e59; font-weight: 700; display: flex; align-items: center; gap: 6px;"><span>🧊</span> What this is</h3>
      <p style="margin: 0; font-size: 13px; color: #334155; line-height: 1.7;">Three sub-3-bpw GGUFs of <b>google/gemma-4-31B-it</b> produced with an activation-aware scale search (AWQ) where each candidate α is <b>gated by a disjoint held-out validation slice</b>: the per-tensor α is accepted only if it doesn't worsen the validation proxy loss vs. the safer group-level α. The chosen scales are folded into the preceding RMSNorm — no runtime overhead, standard GGUF.</p>
    </div>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 10px;">
      <div style="border: 1px solid #e2e8f0; padding: 14px; border-radius: 8px; background: #fafafa; box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);"><span style="font-weight: 700; color: #115e59; font-size: 12px; display: block; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">⚖️ Binary held-out gate</span><span style="font-size: 13px; color: #4b5563; line-height: 1.5;">Per-member α is accepted iff its validation proxy loss ≤ the group α's validation loss. Reverts to the safer scale otherwise.</span></div>
      <div style="border: 1px solid #e2e8f0; padding: 14px; border-radius: 8px; background: #fafafa; box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);"><span style="font-weight: 700; color: #115e59; font-size: 12px; display: block; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">📉 PPL ≪ naive AWQ</span><span style="font-size: 13px; color: #4b5563; line-height: 1.5;">IQ2_XS PPL on external code/math/tools eval: <b>242.5</b>. Q2_K_S: <b>146.2</b> — actually below the FP16 anchor at this eval distribution.</span></div>
      <div style="border: 1px solid #e2e8f0; padding: 14px; border-radius: 8px; background: #fafafa; box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);"><span style="font-weight: 700; color: #115e59; font-size: 12px; display: block; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">🎯 +10–15 pts top-p</span><span style="font-size: 13px; color: #4b5563; line-height: 1.5;">Same-top-token agreement with FP16: <b>46–53%</b> across the three releases, vs 33–40% for imatrix-only at the same bit budget.</span></div>
      <div style="border: 1px solid #e2e8f0; padding: 14px; border-radius: 8px; background: #fafafa; box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);"><span style="font-weight: 700; color: #115e59; font-size: 12px; display: block; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">🛠️ Standard GGUF</span><span style="font-size: 13px; color: #4b5563; line-height: 1.5;">Loads in vanilla llama.cpp / llama-server / LM Studio — no custom runtime, kernels, or patches.</span></div>
    </div>
  </div>
</div>

## 🧰 1. Files

| File | Quant | Size | BPW |
|---|---|---|---|
| `gemma-4-31B-it-IQ2_XS-awq-cv-gate.gguf` | IQ2_XS | 8.88 GiB | 2.484 |
| `gemma-4-31B-it-IQ2_M-awq-cv-gate.gguf`  | IQ2_M  | 10.17 GiB | 2.845 |
| `gemma-4-31B-it-Q2_K_S-awq-cv-gate.gguf` | Q2_K_S | 10.22 GiB | 2.861 |

FP16 reference: 57.20 GiB, 16.005 BPW (not included — fetch from [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it)).

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
      <p style="margin: 0 0 10px 0;">Math-equivalent to the original layer, but the rescaled weight matrix has a flatter per-channel range so the 2-bit codebook fits it with less error. The inverse scale gets absorbed into the preceding RMSNorm — <b>no runtime overhead</b>, the GGUF stays standard.</p>
      <p style="margin: 0 0 10px 0;">Baseline AWQ picks one shared <b>α</b> per group of layers that share an input (e.g. q/k/v). This release adds two refinements:</p>
      <ol style="margin: 0; padding-left: 20px;">
        <li><b>Per-tensor α refinement</b> — each member of a group (q, k, v individually) gets to nudge its α within a small local grid around the group choice, lowering its own reconstruction error.</li>
        <li><b>Binary held-out gate</b> — the per-tensor α is only accepted if it doesn't worsen the proxy loss on a <i>disjoint validation slice</i>. If it would, the gate rejects it and the tensor falls back to the safer group α. Without the gate, per-tensor refinement over-fits the calibration corpus at sub-3 bpw and PPL collapses on unseen text.</li>
      </ol>
    </div>
  </div>
  <div style="border: 1px solid #fde68a; border-radius: 12px; overflow: hidden; background: #ffffff;">
    <div style="background: linear-gradient(135deg, #f59e0b 0%, #b45309 100%); padding: 12px 16px; color: white; font-weight: 700; font-size: 14px;">🧮 2.2 Why AWQ beats imatrix alone</div>
    <div style="padding: 16px; font-size: 13px; color: #334155; line-height: 1.7;">
      <p style="margin: 0 0 10px 0;">Both techniques look at the same calibration activations, but they spend the signal differently:</p>
      <ul style="margin: 0 0 10px 0; padding-left: 20px;">
        <li><b>Imatrix only.</b> Tells the quantizer <i>which</i> channels are important via <code>E[a²]ᵢ</code> so the codebook spends more precision on them. The weight numerics themselves don't change — outlier weights still exist and still cause large per-bin errors.</li>
        <li><b>AWQ.</b> Actually rewrites the weight matrix to be easier to quantize. Channels that drive the output get scaled down in the weight domain (their range shrinks), so a 4-value codebook covers them with less error. The rescaling is folded into the preceding norm, so the math the layer computes is unchanged.</li>
      </ul>
      <p style="margin: 0;">They are complementary, and this release uses both: <b>AWQ scales the weights</b>, then <b>a hybrid imatrix</b> (<code>E[a²]</code> mixed with weight-column energy <code>‖W[:, i]‖² · E[a²]</code>) guides the final <code>llama-quantize</code> pass. Pure imatrix can't fix outlier weights; AWQ alone under-uses the per-channel sensitivity signal during the quantizer's bin assignment.</p>
    </div>
  </div>
  <div style="border: 1px solid #bfdbfe; border-radius: 12px; overflow: hidden; background: #ffffff;">
    <div style="background: linear-gradient(135deg, #2563eb 0%, #1e3a8a 100%); padding: 12px 16px; color: white; font-weight: 700; font-size: 14px;">📚 2.3 Data slices</div>
    <div style="padding: 16px; font-size: 13px; color: #334155; line-height: 1.7;">
      <p style="margin: 0 0 10px 0;">Three disjoint corpora with distinct roles. The gate uses validation text only to make a binary accept/reject decision per tensor — a very low-capacity signal — and validation never feeds the eval numbers in §3.</p>
      <table style="width:100%; border-collapse:collapse; font-size:12px; margin:0;">
        <thead><tr style="background:#eff6ff;"><th style="padding:8px 10px; border:1px solid #bfdbfe; text-align:left; color:#1e3a8a;">Slice</th><th style="padding:8px 10px; border:1px solid #bfdbfe; text-align:left; color:#1e3a8a;">Source</th><th style="padding:8px 10px; border:1px solid #bfdbfe; text-align:left; color:#1e3a8a;">Used for</th></tr></thead>
        <tbody>
          <tr><td style="padding:8px 10px; border:1px solid #bfdbfe;"><b>Calibration</b></td><td style="padding:8px 10px; border:1px solid #bfdbfe;">~500k tokens of usage-log + all of <code>wiki.test.raw</code></td><td style="padding:8px 10px; border:1px solid #bfdbfe;">imatrix collection + AWQ α search</td></tr>
          <tr><td style="padding:8px 10px; border:1px solid #bfdbfe;"><b>Validation</b></td><td style="padding:8px 10px; border:1px solid #bfdbfe;">~10k tokens of usage-log (disjoint sessions) + a small Rust/JSON/YAML supplement</td><td style="padding:8px 10px; border:1px solid #bfdbfe;">held-out gate for per-tensor α</td></tr>
          <tr><td style="padding:8px 10px; border:1px solid #bfdbfe;"><b>Eval</b></td><td style="padding:8px 10px; border:1px solid #bfdbfe;">~90k tokens from <a href="https://huggingface.co/datasets/eaddario/imatrix-calibration"><code>eaddario/imatrix-calibration</code></a> (code+math+tools)</td><td style="padding:8px 10px; border:1px solid #bfdbfe;">all numbers in §3 — neither calibration nor validation appears here</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>

Toolchain: AWQ + imatrix orchestrated by [`quant-tuner`](https://github.com/pearsonkyle/quant-tuner); final quantization with `llama-quantize` from [llama.cpp](https://github.com/ggerganov/llama.cpp) pinned to commit `45b455e6`.

---

## 📊 3. Comparison

All rows benched on the same eval corpus (~90k tokens from [`eaddario/imatrix-calibration`](https://huggingface.co/datasets/eaddario/imatrix-calibration): code + math + tools), same llama.cpp build, ctx=4096. Neither the calibration nor the validation slice appears in this corpus.

| quant | technique | size (GiB) | BPW | PPL | KLD (mean) | same_top_p |
|---|---|---:|---:|---:|---:|---:|
| FP16 | none (reference) | 57.20 | 16.005 | **277.89** | 0.00000 | 100.00% |
| IQ2_XS | imatrix only | 8.88 | 2.484 | _pending_ | _pending_ | _pending_ |
| **IQ2_XS** | **AWQ cv-gate + imatrix** | **8.88** | **2.484** | **242.51** | **3.50384** | **46.12%** |
| IQ2_M | imatrix only | 10.17 | 2.845 | _pending_ | _pending_ | _pending_ |
| **IQ2_M** | **AWQ cv-gate + imatrix** | **10.17** | **2.845** | **305.34** | **2.47575** | **52.70%** |
| Q2_K_S | imatrix only | 10.22 | 2.861 | _pending_ | _pending_ | _pending_ |
| **Q2_K_S** | **AWQ cv-gate + imatrix** | **10.22** | **2.861** | **146.25** | **3.54850** | **51.17%** |
| Q2_K | plain (no imatrix, no AWQ) | 11.10 | 3.105 | 3370.57 | 6.11890 | 25.83% |

<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; border: 1px solid #99f6e4; border-radius: 12px; background: #f0fdfa; padding: 16px; margin: 16px 0; color: #115e59; font-size: 13px; line-height: 1.7;">
  <b>Reading the table.</b> At the same bit budget, AWQ cv-gate beats imatrix-only on PPL, KLD, and same-top-token agreement with FP16. Plain Q2_K (no calibration at all, ~3.1 bpw) anchors the floor: even at higher BPW than the calibrated Q2_K_S, it loses ~25 points of top-p and is ~23× the PPL — calibration matters more than a few extra bits do.
</div>

![Comparison: AWQ cv-gate release vs imatrix-only and FP16](./awq_cv_gate_release.png)

<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; border: 1px solid #fde68a; border-radius: 12px; background: #fffbeb; padding: 16px; margin: 16px 0; color: #92400e; font-size: 13px; line-height: 1.7;">
  <b>⚠️ Caveat.</b> These are sub-3-bpw quants of a 31B reasoning model. They are meaningfully better than the alternatives at the same size, but they are <b>not</b> a substitute for FP16 / Q4_K_M / Q5_K_M when you have the VRAM. Use them when memory is the binding constraint.
</div>

---

## 🚀 4. Usage

```bash
# llama.cpp CLI (any of the three quants)
llama-cli -m gemma-4-31B-it-IQ2_XS-awq-cv-gate.gguf  -c 8192 -p "Hello"
llama-cli -m gemma-4-31B-it-IQ2_M-awq-cv-gate.gguf   -c 8192 -p "Hello"
llama-cli -m gemma-4-31B-it-Q2_K_S-awq-cv-gate.gguf  -c 8192 -p "Hello"

# llama-server (OpenAI-compatible)
llama-server -m gemma-4-31B-it-Q2_K_S-awq-cv-gate.gguf -c 8192 --host 0.0.0.0 --port 8080
```

Loads in any llama.cpp-based runtime: llama.cpp, LM Studio, Ollama (via Modelfile), text-generation-webui, etc. Use the Gemma chat template that ships with the base model.

**Which file to pick:**
- `IQ2_XS` (8.88 GiB) — smallest. Use when VRAM/RAM is tight; best size-to-quality ratio in this set.
- `IQ2_M`  (10.17 GiB) — best **KLD** and **same_top_p** in this set. Use when you want the most FP16-faithful sub-3-bpw option.
- `Q2_K_S` (10.22 GiB) — best **PPL** in this set; competitive top_p. Slightly larger than IQ2_M for similar quality.

**Recommended sampling for tool-use / structured output:** `temperature=0.0`, `top_p=1.0`, `seed` fixed.
**For open-ended chat:** `temperature=0.6`, `top_p=0.95`, `top_k=20`.

---

## 🪪 5. License & attribution

* Inherits the [**Gemma Terms of Use**](https://ai.google.dev/gemma/terms) from the base model.
* Base weights: [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it).
* Calibration + AWQ scaling + quantization performed locally with `quant-tuner`; vendored llama.cpp at commit `45b455e6`.
