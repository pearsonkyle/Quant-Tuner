# Quantization tuning for ai agents

Problem Statement: The next big wave of AI adoption will be driven not just by AI agents that can act autonomously, but by the ability to run these agents efficiently on a variety of hardware. One of the biggest constraints in today's agentic ecosystem is that these tools often require expensive hardware to run on or require an internet connection in order to pay for access to 'reliable' service. Neither of those are ideal for widespread adoption. Quantization is a promising technique to reduce the computational requirements of AI models, making them more accessible and efficient so much so that modern smart phones are already capable of running models that are ~2-4 Gb which push state of the art performance for their size. However, quantization seldom achieves the same level of performance as the original model, and the performance drop can be significant if done naively. Compensating the error introduced by quantization is critical for maintaining reliable performance when deployed in real world applications. In this experiment, we will explore the effectiveness of imatrix, a technique designed to compensate for the errors introduced by quantization, in improving the performance of quantized models in the context of AI agents. 

Models: qwen, jackrong, tesslate
Quant: IQ3_S, Q5KS, FP16
Technique: None, imatrix, imatrix+
Dataset: None, wiki, custom (logs+supplemental)

imatrix Experiment:
Can we enhance the loss function with an imatrix to improve the model's performance on tool calls and response quality?
Baseline: Standard training without imatrix vs imatrix-enhanced training

MTP Experiments:
Hypothesis: Does a custom trained MTP improve the acceptance of tool calls and the quality of responses?
- MTP from Qwen parent model 
- Custom trained MTP using logs+supplementary data 
- Custom trained MTP with injected noise to simulate quantization errors (interesting to explore how noise changes performance afterwards)

Validation:
MMLU-Pro (STEM)
Tool Choice (Holdout from logs)
Red Team (Safety, Response Quality, etc.)

Tables:

First look into quantization performance:
| Model | Size (GiB) | BPW | PPL | KLD | Same Top-p | 
| --- | --- | --- | --- | --- | --- |


General Performance:
<!-- Populated by: scripts/run_mtp_eval_suite.py → out/benchmark_9b_iq3s/eval/mtp_suite/mtp_eval_summary.json -->
| Model | Quant | Technique | MTP Variant | Tool Sel % | Param Acc % | Schema % | Rollout % | MMLU-Pro % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen | FP16 | — | — | | | | | |
| qwen | IQ3_S | imatrix+ | vanilla | | | | | |
| qwen | IQ3_S | imatrix+ | trained | | | | | |
| qwen | IQ3_S | imatrix+ | trained-noisy | | | | | |
| qwen | IQ3_S | imatrix+ | trained-mixed | | | | | |
| jackrong | FP16 | — | — | | | | | |
| jackrong | IQ3_S | imatrix+ | vanilla | | | | | |
| jackrong | IQ3_S | imatrix+ | trained | | | | | |
| jackrong | IQ3_S | imatrix+ | trained-noisy | | | | | |
| jackrong | IQ3_S | imatrix+ | trained-mixed | | | | | |
| tesslate | FP16 | — | — | | | | | |
| tesslate | IQ3_S | imatrix+ | vanilla | | | | | |
| tesslate | IQ3_S | imatrix+ | trained | | | | | |
| tesslate | IQ3_S | imatrix+ | trained-noisy | | | | | |
| tesslate | IQ3_S | imatrix+ | trained-mixed | | | | | |

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
| Model | Quant | MTP Variant | n=1 acc % | n=2 acc % | n=3 acc % |
| --- | --- | --- | --- | --- | --- |
| qwen | FP16 |  | 72.9 ± 2.9 | 59.9 ± 4.4 | 47.8 ± 1.6 |
| qwen | IQ3_S |  | 73.5 ± 4.1 | 59.5 ± 5.8 | 45.1 ± 4.4 |
| qwen | IQ3_S | custom | 67.7 ± 2.1 | 52.3 ± 3.3 | 40.9 ± 4.6 |
| qwen | IQ3_S | custom+noisy | 67.5 ± 0.2 | 55.8 ± 6.4 | 44.4 ± 5.5 |
| jackrong | FP16 |  | 75.1 ± 1.1 | 63.8 ± 3.0 | 49.5 ± 1.8 |
| jackrong | IQ3_S |  | 74.0 ± 2.0 | 62.1 ± 2.0 | 48.6 ± 3.3 |
| jackrong | IQ3_S | custom | 69.9 ± 2.3 | 55.7 ± 2.0 | 44.9 ± 2.0 |
| jackrong | IQ3_S | custom+noisy | 72.6 ± 2.2 | 54.3 ± 3.7 | 42.4 ± 2.9 |
| tesslate | FP16 |  | 75.5 ± 2.7 | 61.2 ± 5.7 | 49.6 ± 2.8 |
| tesslate | IQ3_S |  | 73.9 ± 3.9 | 55.8 ± 3.6 | 43.8 ± 3.7 |
| tesslate | IQ3_S | custom | 69.8 ± 5.0 | 54.1 ± 4.7 | 44.6 ± 2.5 |
| tesslate | IQ3_S | custom+noisy | 69.7 ± 3.6 | 54.6 ± 5.7 | 42.1 ± 8.4 |

MTP Throughput:
<!-- Populated by: scripts/run_mtp_pipeline.py + fp16 sweep → out/benchmark_9b_iq3s/mtp_acceptance.json -->
| Model | Quant | MTP Variant | baseline tok/s | n=1 tok/s | n=2 tok/s | n=3 tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| qwen | FP16 | fp16 | 6.5 ± 0.5 | 9.8 ± 1.0 | 11.0 ± 1.2 | 10.5 ± 0.9 |
| qwen | IQ3_S | vanilla | 27.7 ± 1.3 | 25.3 ± 6.6 | 22.8 ± 2.9 | 19.4 ± 3.8 |
| qwen | IQ3_S | trained | 13.4 ± 4.4 | 12.8 ± 1.6 | 10.9 ± 2.6 | 10.6 ± 2.6 |
| qwen | IQ3_S | trained-noisy | 13.0 ± 0.6 | 13.9 ± 2.8 | 13.8 ± 1.9 | 11.1 ± 2.2 |
| qwen | IQ3_S | trained-mixed | | | | |
| jackrong | FP16 | fp16 | 9.9 ± 4.3 | 13.0 ± 4.5 | 15.8 ± 4.4 | 13.2 ± 4.7 |
| jackrong | IQ3_S | vanilla | 16.8 ± 6.0 | 15.3 ± 4.8 | 14.1 ± 5.2 | 11.7 ± 3.9 |
| jackrong | IQ3_S | trained | 12.1 ± 3.2 | 12.3 ± 3.9 | 10.9 ± 1.2 | 9.3 ± 1.2 |
| jackrong | IQ3_S | trained-noisy | 12.2 ± 1.8 | 13.9 ± 1.5 | 13.8 ± 1.5 | 10.2 ± 3.0 |
| jackrong | IQ3_S | trained-mixed | | | | |
| tesslate | FP16 | fp16 | 8.8 ± 0.7 | 11.4 ± 2.8 | 14.3 ± 2.2 | 12.2 ± 2.3 |
| tesslate | IQ3_S | vanilla | 11.2 ± 0.8 | 12.2 ± 0.6 | 10.6 ± 1.3 | 9.1 ± 1.0 |
| tesslate | IQ3_S | trained | 16.0 ± 6.5 | 16.8 ± 9.2 | 14.6 ± 5.5 | 12.4 ± 3.4 |
| tesslate | IQ3_S | trained-noisy | 24.7 ± 13.1 | 25.1 ± 8.6 | 24.8 ± 4.4 | 22.6 ± 4.6 |
| tesslate | IQ3_S | trained-mixed | | | | |
