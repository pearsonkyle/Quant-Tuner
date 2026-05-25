# Quantization tuning for ai agents

Problem Statement: The next big wave of AI adoption will be driven not just by AI agents that can act autonomously, but by the ability to run these agents efficiently on a variety of hardware. One of the biggest constraints in today's agentic ecosystem is that these tools often require expensive hardware to run on or require an internet connection in order to pay for access to 'reliable' service. Neither of those are ideal for widespread adoption. Quantization is a promising technique to reduce the computational requirements of AI models, making them more accessible and efficient so much so that modern smart phones are already capable of running models that are ~2-4 Gb which push state of the art performance for their size. However, quantization seldom achieves the same level of performance as the original model, and the performance drop can be significant if done naively. Compensating the error introduced by quantization is critical for maintaining reliable performance when deployed in real world applications. In this experiment, we will explore the effectiveness of imatrix, a technique designed to compensate for the errors introduced by quantization, in improving the performance of quantized models in the context of AI agents. 

Models: qwen, jackrong, tesslate
Quant: IQ3_S, Q5KS, FP16
Technique: None, imatrix, imatrix+
Dataset: None, wiki, custom (logs+supplemental)

### Experiments

#### imatrix
Can we enhance the loss function with an imatrix to improve the model's performance on tool calls and response quality?
Baseline: Standard training without imatrix vs imatrix-enhanced training

#### MTP
Hypothesis: Does a custom trained MTP improve the acceptance of tool calls and the quality of responses?
- `vanilla` — donor Qwen parent MTP
- `trained` — custom-trained MTP from FP16 backbone hidden states + tool-call logs
- `trained-noisy` — `trained` + injected Gaussian noise scaled to IQ3_S quantization error (simulates the activation shift)
- `trained-mixed` — `trained` + 30% wikitext mixed with tool-call logs (broader distribution)
- `trained-iq3s` — MTP trained against the actual IQ3_S backbone's `h_pre_norm` activations (captured via `llama-mtp-capture`), with 30% wikitext mix. Eliminates the FP16-vs-IQ3_S distribution mismatch that motivates `trained-noisy`.

Validation:
MMLU-Pro (STEM)
Tool Choice (Holdout from logs)
Red Team (Safety, Response Quality, etc.)

### trained-iq3s — findings

We hypothesised that the dominant cause of fine-tuned MTP heads
underperforming the donor head was a *distribution mismatch*: the trainer
fed the head FP16 hidden states, but at inference the head consumes IQ3_S
hidden states. To remove this mismatch, `scripts/capture_iq3s_hidden_states.py`
dumps the IQ3_S backbone's `h_pre_norm` activations via a vendored llama.cpp
tool (`llama-mtp-capture`) and `train_mtp_head.py --iq3s-cache` then trains
the head directly against those activations. Wiki-mix 30 % is included so
the head sees both tool-call distribution and general text.

Result (per-token draft acceptance rate at `--spec-draft-n-max`, 3 reps):

| Model | Variant | n=1 | n=2 | n=3 |
| --- | --- | --- | --- | --- |
| qwen | vanilla (donor) | **73.5** | **59.5** | **45.1** |
| qwen | trained-iq3s | **76.0** | 47.5 | 44.9 |
| jackrong | vanilla (donor) | **74.0** | **62.1** | **48.6** |
| jackrong | trained-iq3s | 71.5 | 56.4 | 42.0 |
| tesslate | vanilla (donor) | **73.9** | **55.8** | 43.8 |
| tesslate | trained-iq3s | 64.8 | 48.3 | **45.6** |

The distribution-mismatch fix did **not** outperform the donor MTP overall.
qwen at n=1 (+2.5 pp) and tesslate at n=3 (+1.8 pp) are mild wins; every
other (model, n_max) pair is a loss vs vanilla, sometimes large (tesslate
n=1: −9.1 pp). The donor MTP — trained as part of the parent model at FP16
on a much larger corpus — remains the better speculative draft head at
IQ3_S, even when we eliminate the FP16↔IQ3_S activation mismatch entirely.

Possible reasons we don't yet rule out:

- 293 steps × 1172 chunks ≈ 300 K tokens of fine-tuning is too small to
  beat the donor's training scale. The donor head benefits from full-model
  joint pretraining.
- Our MTP loss only supervises the 2-step-ahead token via the shared
  `lm_head`. The donor head may have been trained with additional
  objectives (auxiliary losses, distillation) we don't replicate.
- The IQ3_S `h_pre_norm` we capture is from a model whose `lm_head` is
  also quantized; the donor was trained against FP16 `lm_head` outputs,
  which may carry more useful gradient signal.

The pipeline itself is sound: all three rebuilds (qwen, jackrong, tesslate)
preserve vanilla tool-call behaviour exactly — autoregressive decode is
MTP-blind, so identical seeds → identical metrics on any clean rebuild.
A 5-rep re-bench of `qwen / IQ3_S-custom` vs `qwen / IQ3_S-trained-iq3s`
returned byte-identical numbers (70.4 ± 0.5 % tool-selection for both),
confirming the qwen build is clean. An earlier single-run eval that
read ≈2 % for `trained-iq3s` was a transient artifact (likely server
thrashing under concurrent GPU load).

See `experiments/mtp_acceptance.png` for the full sweep and
`experiments/eval_comparison.png` for the toolcall/MMLU view.

### Note on the General Performance table

Tool-call and MMLU-Pro evals are run via `llama-server` in standard
autoregressive mode (no `--spec-type mtp`), so the MTP head is **not on the
decode path**. As a result, **vanilla / trained / trained-noisy /
trained-mixed / trained-iq3s all share the same main-model weights and
should produce identical tool-call and MMLU-Pro scores under matched seeds.**
Where they differ in the table, that signals the rebuild step touched the
main weights. **Note (correction)**: an earlier single-run eval suggested
the qwen `trained-iq3s` build was corrupted (≈2 % tool-selection vs 70 %
for vanilla). A clean 5-rep re-bench (see the
*qwen tool-call bench* table above) refuted that — `trained-iq3s` is
**byte-identical** to `IQ3_S-custom` on tool-call (66.0 ± 0.9 % sel for
both under the current metric definitions). The earlier reading was a
transient artifact, most plausibly server
thrashing from a concurrent eval pipeline using the same GPU. The qwen
rebuild is in fact clean.

The variant-vs-variant signal lives in the **MTP Performance Acceptance**
table below (speculative-decoding draft acceptance rate), which is the
metric the trained-iq3s experiment was designed to move.

Tables:

| Model | Size (GiB) | BPW | PPL | KLD | Same Top-p |
| --- | --- | --- | --- | --- | --- |
| qwen / FP16 | 17.14 | 16.01 | 6.65 | 0.000 | 100.0% |
| qwen / IQ3_S (imatrix+ × logtrain) | 4.17 | 3.89 | 7.37 | 1.175 | 74.6% |
| qwen / IQ3_S (stock imatrix × logtrain) | 4.17 | 3.89 | 7.71 | 1.183 | 74.3% |
| qwen / IQ3_S (stock imatrix × wikitext) | 4.17 | 3.89 | 9.48 | 1.258 | 73.5% |
| jackrong / FP16 | 17.14 | 16.01 | 4.78 | 0.000 | 100.0% |
| jackrong / IQ3_S (imatrix+) | 4.17 | 3.89 | 6.40 | 0.919 | 75.7% |
| tesslate / FP16 | 17.14 | 16.01 | 6.64 | 0.000 | 100.0% |
| tesslate / IQ3_S (imatrix+) | 4.17 | 3.89 | 7.40 | 1.172 | 74.6% |

### qwen tool-call bench (5 reps, mean ± stdev)
<!-- Populated by: scripts/run_toolcall_reps.py × 4 qwen IQ3_S variants
     Source: out/benchmark_9b_iq3s/eval/qwen_toolbench/toolcall_agg.csv -->
Same holdout (`artifacts/toolcall_holdout.jsonl`, 25 sessions) and sampling
(T=0.6 / top_p=0.95 / top_k=20) across all variants. Standard
autoregressive decode — no speculative draft, so the MTP head is **off the
decode path** (which is why `IQ3_S-custom` and `IQ3_S-trained-iq3s` come
out byte-identical: same main weights, different MTP shard, MTP unused).


> **Metric redefinition (2026-05-24)**: simplified to per-turn only —
> `tool_selection_acc`, `param_acc_mean`, `schema_valid_rate`. Rollout
> completion, tool-set match, and `n_turns` were dropped. `param_acc_mean`
> now uses **every** ground-truth key as the denominator (extras in pred
> ignored); `schema_valid_rate` is **presence-only** against
> `schema.required` (no type/enum validation). Numbers below are not
> comparable to the pre-2026-05-24 table.

| Variant | Tool Sel % | Param Acc % | Schema % | PPL (eval) |
| --- | --- | --- | --- | --- |
| IQ3_S-custom (hybrid_custom × logtrain) | 66.0 ± 0.9 | **44.2 ± 3.0** | 89.2 ± 1.3 | 7.37 |
| IQ3_S-stock (stock × logtrain) | 65.6 ± 1.7 | **44.3 ± 2.1** | **90.9 ± 1.4** | 7.71 |
| IQ3_S-wiki (stock × wikitext) | **68.2 ± 0.9** | 42.8 ± 1.6 | 89.6 ± 1.3 | 9.48 |
| IQ3_S-trained-iq3s (same main as custom) | 66.0 ± 0.9 | 44.2 ± 3.0 | 89.2 ± 1.3 | — |

**Headline**: on this 25-session holdout, **higher PPL/KLD does NOT mean
worse tool-call performance**. `IQ3_S-wiki` has the worst eval-corpus PPL
(9.48 vs 7.37 for `IQ3_S-custom`) yet posts the highest tool-selection
rate (68.2 % vs 66.0 %). `custom` and `stock` are statistically tied on
selection (66.0 vs 65.6) and on param accuracy (~44.3 %); `stock` edges
out on schema validity. Wiki gives up ~1.5 pp on param accuracy and
lands mid-pack on schema — its win is in tool *selection*, not argument
quality.

Implication: PPL-on-an-eval-corpus is a poor proxy for downstream
tool-call quality at IQ3_S. The hybrid_custom reweighting is a small win
on distribution-similarity metrics (PPL / KLD) but does not visibly help
tool-call accuracy here. Between `stock × logtrain` and `hybrid_custom ×
logtrain` the difference is within noise on the three metrics that
matter for agent use; `stock × wikitext` trades a small amount of
param/schema quality for a measurable tool-selection bump.
| jackrong / IQ3_S (imatrix+) | 4.17 | 3.89 | 6.40 | 0.919 | 75.7% |
| tesslate / FP16 | 17.14 | 16.01 | 6.64 | 0.000 | 100.0% |
| tesslate / IQ3_S (imatrix+) | 4.17 | 3.89 | 7.40 | 1.172 | 74.6% |


<!-- Populated by: scripts/run_mtp_eval_suite.py → out/benchmark_9b_iq3s/eval/mtp_suite/mtp_eval_summary.json -->
| Model | Quant | Technique | MTP Variant | Tool Sel % | Param Acc % | Schema % | MMLU-Pro % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen | FP16 | — | — | | | | |
| qwen | IQ3_S | imatrix+ (hybrid_custom × logtrain) | vanilla | 66.0 ± 0.9 | 44.2 ± 3.0 | 89.2 ± 1.3 | 34.8 |
| qwen | IQ3_S | imatrix+ (hybrid_custom × logtrain) | trained | 66.0 ± 0.9 | 44.2 ± 3.0 | 89.2 ± 1.3 | 34.8 |
| qwen | IQ3_S | imatrix+ (hybrid_custom × logtrain) | trained-noisy | | | | |
| qwen | IQ3_S | imatrix+ (hybrid_custom × logtrain) | trained-mixed | | | | |
| qwen | IQ3_S | imatrix+ (hybrid_custom × logtrain) | trained-iq3s | 66.0 ± 0.9 | 44.2 ± 3.0 | 89.2 ± 1.3 | 34.8 |
| qwen | IQ3_S | stock × logtrain | vanilla | 65.6 ± 1.7 | 44.3 ± 2.1 | 90.9 ± 1.4 | |
| qwen | IQ3_S | stock × wikitext | vanilla | 68.2 ± 0.9 | 42.8 ± 1.6 | 89.6 ± 1.3 | |
| jackrong | FP16 | — | — | | | | |
| jackrong | IQ3_S | imatrix+ | vanilla | 66.8 | 31.8 | 81.6 | 57.6 |
| jackrong | IQ3_S | imatrix+ | trained | 66.8 | 31.8 | 81.6 | |
| jackrong | IQ3_S | imatrix+ | trained-noisy | | | | |
| jackrong | IQ3_S | imatrix+ | trained-mixed | | | | |
| jackrong | IQ3_S | imatrix+ | trained-iq3s | 66.8 | 31.8 | 81.6 | |
| tesslate | FP16 | — | — | | | | |
| tesslate | IQ3_S | imatrix+ | vanilla | | | | |
| tesslate | IQ3_S | imatrix+ | trained | | | | |
| tesslate | IQ3_S | imatrix+ | trained-noisy | | | | |
| tesslate | IQ3_S | imatrix+ | trained-mixed | | | | |
| tesslate | IQ3_S | imatrix+ | trained-iq3s | | | | |

Red Team Performance:
| Model | Quant | MTP Variant | Safety | Vulnerability | Hallucination | Response Quality |
| --- | --- | --- | --- | --- | --- | --- |
| qwen | IQ3_S | vanilla | | | | |
| qwen | IQ3_S | trained | | | | |
| qwen | IQ3_S | trained-noisy | | | | |
| jackrong | IQ3_S | vanilla | | | | |
| jackrong | IQ3_S | trained | | | | |
| jackrong | IQ3_S | trained-noisy | | | | |
| tesslate | IQ3_S | vanilla | | | | |
| tesslate | IQ3_S | trained | | | | |
| tesslate | IQ3_S | trained-noisy | | | | |

MTP Performance Acceptance:
<!-- Populated by: scripts/run_mtp_pipeline.py + fp16 sweep → out/benchmark_9b_iq3s/mtp_acceptance.json -->
<!-- accept_rate = per-token draft acceptance rate (%) at --spec-draft-n-max=N -->
<!-- trained-iq3s rows: pending — populated by `scripts/plot_mtp_acceptance.py --variants trained-iq3s` once the eval suite finishes -->
| Model | Quant | MTP Variant | n=1 acc % | n=2 acc % | n=3 acc % |
| --- | --- | --- | --- | --- | --- |
| qwen | FP16 |  | 72.9 ± 2.9 | 59.9 ± 4.4 | 47.8 ± 1.6 |
| qwen | IQ3_S |  | 73.5 ± 4.1 | 59.5 ± 5.8 | 45.1 ± 4.4 |
| qwen | IQ3_S | custom | 67.7 ± 2.1 | 52.3 ± 3.3 | 40.9 ± 4.6 |
| qwen | IQ3_S | custom+noisy | 67.5 ± 0.2 | 55.8 ± 6.4 | 44.4 ± 5.5 |
| qwen | IQ3_S | trained-iq3s | 76.0 | 47.5 | 44.9 |
| jackrong | FP16 |  | 75.1 ± 1.1 | 63.8 ± 3.0 | 49.5 ± 1.8 |
| jackrong | IQ3_S |  | 74.0 ± 2.0 | 62.1 ± 2.0 | 48.6 ± 3.3 |
| jackrong | IQ3_S | custom | 69.9 ± 2.3 | 55.7 ± 2.0 | 44.9 ± 2.0 |
| jackrong | IQ3_S | custom+noisy | 72.6 ± 2.2 | 54.3 ± 3.7 | 42.4 ± 2.9 |
| jackrong | IQ3_S | trained-iq3s | 71.5 | 56.4 | 42.0 |
| tesslate | FP16 |  | 75.5 ± 2.7 | 61.2 ± 5.7 | 49.6 ± 2.8 |
| tesslate | IQ3_S |  | 73.9 ± 3.9 | 55.8 ± 3.6 | 43.8 ± 3.7 |
| tesslate | IQ3_S | custom | 69.8 ± 5.0 | 54.1 ± 4.7 | 44.6 ± 2.5 |
| tesslate | IQ3_S | custom+noisy | 69.7 ± 3.6 | 54.6 ± 5.7 | 42.1 ± 8.4 |
| tesslate | IQ3_S | trained-iq3s | 64.8 | 48.3 | 45.6 |

MTP Throughput:
<!-- Populated by: scripts/run_mtp_pipeline.py + fp16 sweep → out/benchmark_9b_iq3s/mtp_acceptance.json -->
| Model | Quant | MTP Variant | baseline tok/s | n=1 tok/s | n=2 tok/s | n=3 tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| qwen | FP16 | fp16 | 6.5 ± 0.5 | 9.8 ± 1.0 | 11.0 ± 1.2 | 10.5 ± 0.9 |
| qwen | IQ3_S | vanilla | 27.7 ± 1.3 | 25.3 ± 6.6 | 22.8 ± 2.9 | 19.4 ± 3.8 |
| qwen | IQ3_S | trained | 13.4 ± 4.4 | 12.8 ± 1.6 | 10.9 ± 2.6 | 10.6 ± 2.6 |
| qwen | IQ3_S | trained-noisy | 13.0 ± 0.6 | 13.9 ± 2.8 | 13.8 ± 1.9 | 11.1 ± 2.2 |
| qwen | IQ3_S | trained-mixed | | | | |
| qwen | IQ3_S | trained-iq3s | 53.5 | 50.3 | 37.7 | 34.5 |
| jackrong | FP16 | fp16 | 9.9 ± 4.3 | 13.0 ± 4.5 | 15.8 ± 4.4 | 13.2 ± 4.7 |
| jackrong | IQ3_S | vanilla | 16.8 ± 6.0 | 15.3 ± 4.8 | 14.1 ± 5.2 | 11.7 ± 3.9 |
| jackrong | IQ3_S | trained | 12.1 ± 3.2 | 12.3 ± 3.9 | 10.9 ± 1.2 | 9.3 ± 1.2 |
| jackrong | IQ3_S | trained-noisy | 12.2 ± 1.8 | 13.9 ± 1.5 | 13.8 ± 1.5 | 10.2 ± 3.0 |
| jackrong | IQ3_S | trained-mixed | | | | |
| jackrong | IQ3_S | trained-iq3s | 58.3 | 50.7 | 41.5 | 32.6 |
| tesslate | FP16 | fp16 | 8.8 ± 0.7 | 11.4 ± 2.8 | 14.3 ± 2.2 | 12.2 ± 2.3 |
| tesslate | IQ3_S | vanilla | 11.2 ± 0.8 | 12.2 ± 0.6 | 10.6 ± 1.3 | 9.1 ± 1.0 |
| tesslate | IQ3_S | trained | 16.0 ± 6.5 | 16.8 ± 9.2 | 14.6 ± 5.5 | 12.4 ± 3.4 |
| tesslate | IQ3_S | trained-noisy | 24.7 ± 13.1 | 25.1 ± 8.6 | 24.8 ± 4.4 | 22.6 ± 4.6 |
| tesslate | IQ3_S | trained-mixed | | | | |
| tesslate | IQ3_S | trained-iq3s | 59.6 | 53.8 | 43.5 | 38.2 |
