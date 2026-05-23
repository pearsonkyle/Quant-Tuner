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

General Performance:
<!-- Populated by: scripts/run_mtp_eval_suite.py → out/benchmark_9b_iq3s/eval/mtp_suite/mtp_eval_summary.json -->
| Model | Quant | Technique | MTP Variant | Tool Sel % | Param Acc % | Schema % | Rollout % | MMLU-Pro % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen | FP16 | — | — | | | | | |
| qwen | IQ3_S | imatrix+ | vanilla | | | | | |
| qwen | IQ3_S | imatrix+ | trained | | | | | |
| qwen | IQ3_S | imatrix+ | trained-noisy | | | | | |
| jackrong | FP16 | — | — | | | | | |
| jackrong | IQ3_S | imatrix+ | vanilla | | | | | |
| jackrong | IQ3_S | imatrix+ | trained | | | | | |
| jackrong | IQ3_S | imatrix+ | trained-noisy | | | | | |
| tesslate | FP16 | — | — | | | | | |
| tesslate | IQ3_S | imatrix+ | vanilla | | | | | |
| tesslate | IQ3_S | imatrix+ | trained | | | | | |
| tesslate | IQ3_S | imatrix+ | trained-noisy | | | | | |

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
| qwen | FP16 | fp16 | 75.6 | 54.5 | 49.7 |
| qwen | IQ3_S | vanilla | 73.0 | 53.3 | 39.2 |
| qwen | IQ3_S | trained | 75.4 | 54.2 | 45.1 |
| qwen | IQ3_S | trained-noisy | 74.6 | 55.4 | 40.8 |
| jackrong | FP16 | fp16 | 73.2 | 61.7 | 48.4 |
| jackrong | IQ3_S | vanilla | 72.6 | 60.3 | 47.5 |
| jackrong | IQ3_S | trained | 76.0 | 59.5 | 49.2 |
| jackrong | IQ3_S | trained-noisy | 69.6 | 53.3 | 39.4 |
| tesslate | FP16 | fp16 | 78.9 | 56.0 | 50.8 |
| tesslate | IQ3_S | vanilla | 75.0 | 64.0 | 42.4 |
| tesslate | IQ3_S | trained | 70.3 | 51.2 | 40.2 |
| tesslate | IQ3_S | trained-noisy | 68.3 | 45.1 | 40.7 |

MTP Throughput:
<!-- Populated by: scripts/run_mtp_pipeline.py + fp16 sweep → out/benchmark_9b_iq3s/mtp_acceptance.json -->
| Model | Quant | MTP Variant | baseline tok/s | n=1 tok/s | n=2 tok/s | n=3 tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| qwen | FP16 | fp16 | 15.9 | 23.6 | 16.4 | 18.9 |
| qwen | IQ3_S | vanilla | 55.2 | 49.9 | 40.0 | 31.9 |
| qwen | IQ3_S | trained | 55.4 | 51.1 | 40.8 | 37.1 |
| qwen | IQ3_S | trained-noisy | 55.6 | 51.0 | 42.8 | 32.7 |
| jackrong | FP16 | fp16 | 13.5 | 18.3 | 39.7 | 14.9 |
| jackrong | IQ3_S | vanilla | 55.0 | 49.9 | 43.3 | 35.9 |
| jackrong | IQ3_S | trained | 55.3 | 50.8 | 42.9 | 36.4 |
| jackrong | IQ3_S | trained-noisy | 54.9 | 48.6 | 40.6 | 31.9 |
| tesslate | FP16 | fp16 | 12.4 | 13.0 | 22.4 | 29.0 |
| tesslate | IQ3_S | vanilla | 64.3 | 60.9 | 53.2 | 38.1 |
| tesslate | IQ3_S | trained | 60.2 | 50.5 | 38.8 | 30.6 |
| tesslate | IQ3_S | trained-noisy | 52.4 | 47.2 | 36.5 | 32.3 |
