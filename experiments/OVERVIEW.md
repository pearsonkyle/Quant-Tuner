
Model Overview
  ┌─────────────────┬───────────────┬───────────────┬───────────────┐
  │      model      │  Parameters   │  Supported    │ Sampling      │
  │      model      │   (active)    │  Modalities   │ Parameters    │
  ├─────────────────┼───────────────┼───────────────┼───────────────┤
  │ Google/Gemma-4-E4B-it │ 8B (4.5B).    │ Text, Image Audio │ temperature=1.0, top_p=0.95, top_k=64 | 8192 ctx only
  ├─────────────────┼───────────────┼───────────────┼───────────────┤
  | Jackrong/Qwopus3.5-9B-Coder | 9B | Text | temperature=0.5, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0 | 8192 ctx only
  ├─────────────────┼───────────────┼───────────────┼───────────────┤
  | [Google/Gemma-4-31b-it](https://huggingface.co/google/gemma-4-31B-it) | 31 B | Text, Image | temperature=1.0, top_p=0.95, top_k=64 | 4096 ctx only
  ├─────────────────┼───────────────┼───────────────┼───────────────┤
  | [Jackrong/Qwopus3.6-27B-v2](https://huggingface.co/Jackrong/Qwopus3.6-27B-v2) | 27 B | Text, Image | temperature=0.2, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0 | 8192 ctx only (smaller vocab than gemma so we can get away with it)
  ├─────────────────┼───────────────┼───────────────┼───────────────┤

Categories
- Size (heavy/medium/light)
- Modalities (text, image, audio)
- Capability (coder vs generalist)

Quantization Tests Across Models

Heavy (< 24Gb)
- unsloth/Gemma-4-31B-it-GGUF/UD-Q5_K_XL (21.9 Gb)
- Jackrong/Qwopus3.6-27B-v2-MTP-GGUF/Q6_K (22.4 Gb)
- custom/500k-custom+wiki;8192/Q5_K_M (19.5 Gb)

Medium (< 16Gb)
- unsloth/Gemma-4-31B-it-GGUF/UD-Q3_K_S (13.2 Gb)
- Jackrong/Qwopus3.6-27B-v2-MTP-GGUF/Q3_K_S (12.3 Gb)
- custom/500k-custom+wiki;8192/IQ3_S (11.3 Gb)

Light (< 8Gb)
- Jackrong/Qwopus3.5-9B-Coder-GGUF/Q5_K_M (6.47 Gb)
- unsloth/Gemma-4-E4B-it-GGUF/UD-Q5_K_XL (6.66 Gb)
- custom/500k-custom+wiki;8192/IQ4_NL (5.67 Gb)


Quantization Test with calibration data using google/gemma-4-E4B-it

| quant | technique | dataset | size (GiB) | BPW | PPL | KLD (mean) | same_top_p |
|---|---|---|---|---|---|---|---|
| FP16   | none    | —             | 14.02 | 16.018 | 4.7841 | 0.00000 | 100.0000 |
| Q4_K_M | imatrix | wiki.test.raw | 4.97 | 5.677 | 4.8114 | 0.03959 | 94.2930 |
| Q4_K_M | imatrix | custom | 4.97 | 5.677 | 4.8479 | 0.03777 | 94.5020 |
| Q4_K_M | imatrix | 500k-custom+wiki (ctx=512) | 4.97 | 5.677 | 4.8315 | 0.03891 | 94.2710 |
| Q4_K_M | imatrix | 500k-custom+wiki (ctx=2048) | 4.97 | 5.677 | 4.8223 | 0.03786 | 94.3730 |
| Q4_K_M | imatrix | 500k-custom+wiki (ctx=8192) | 4.97 | 5.677 | 4.8342 | 0.03710 | 94.5510 |
| IQ4_NL | imatrix | wiki.test.raw | 4.84 | 5.527 | 4.7534 | 0.04689 | 93.8760 |
| IQ4_NL | imatrix | custom | 4.84 | 5.527 | 4.7658 | 0.04464 | 93.9870 |
| IQ4_NL | imatrix | 500k-custom+wiki (ctx=512) | 4.84 | 5.527 | 4.7953 | 0.04469 | 93.9820 |
| IQ4_NL | imatrix | 500k-custom+wiki (ctx=2048) | 4.84 | 5.527 | 4.7685 | 0.04521 | 93.9070 |
| IQ4_NL | imatrix | 500k-custom+wiki (ctx=8192) | 4.84 | 5.527 | 4.7527 | 0.04556 | 93.8490 |

| quant | technique | dataset | size (GiB) | BPW | PPL | KLD (mean) | same_top_p |
|---|---|---|---|---|---|---|---|
| FP16   | none    | —             | 16.69 | 16.012 | 3.8035 | 0.00000 | 100.0000 |
| Q4_K_M | none    | —             | 5.24 | 5.029 | 2.5144 | 0.95961 | 87.6340 |
| Q4_K_M | imatrix | wiki.test.raw | 5.24 | 5.029 | 2.9200 | 0.53350 | 90.4180 |
| Q4_K_M | imatrix | custom | 5.24 | 5.029 | 3.3549 | 0.51300 | 90.4960 |
| Q4_K_M | imatrix | 500k-custom+wiki (ctx=512) | 5.24 | 5.029 | 3.0949 | 0.50676 | 90.5450 |
| Q4_K_M | imatrix | 500k-custom+wiki (ctx=2048) | 5.24 | 5.029 | 3.1488 | 0.52010 | 90.4080 |
| Q4_K_M | imatrix | 500k-custom+wiki (ctx=8192) | 5.24 | 5.029 | 3.1578 | 0.51996 | 90.3440 |
| IQ4_NL | imatrix | wiki.test.raw | 5.05 | 4.841 | 2.6990 | 0.75214 | 89.2890 |
| IQ4_NL | imatrix | custom | 5.05 | 4.841 | 2.6736 | 0.72520 | 89.6310 |
| IQ4_NL | imatrix | 500k-custom+wiki (ctx=512) | 5.05 | 4.841 | 2.7907 | 0.70275 | 89.5630 |
| IQ4_NL | imatrix | 500k-custom+wiki (ctx=2048) | 5.05 | 4.841 | 2.7028 | 0.71220 | 89.5580 |
| IQ4_NL | imatrix | 500k-custom+wiki (ctx=8192) | 5.05 | 4.841 | 2.6742 | 0.71079 | 89.5820 |


Quick look at MMLU Pro (CS, Math, Eng)

| quant | technique | dataset | size (GiB) | Comp Sci. | Eng. | Math | Average |
|---|---|---|---|---|---|---|---|
| FP16       | none    | —             | 4.97 |  |  |  |  |
| Q4_K_M     | imatrix | wiki.test.raw | 4.97 |  |  |  |  |
| Q4_K_M     | imatrix | custom        | 4.97 |  |  |  |  |
| Q4_K_M     | imatrix | 500k-custom+wiki (ctx=512) | 4.97 |  |  |  |  |
| Q4_K_M     | imatrix | 500k-custom+wiki (ctx=2048) | 4.97 |  |  |  | |
| Q4_K_M     | imatrix | 500k-custom+wiki (ctx=8192) | 4.97 |  |  |  |  |
| IQ4_NL     | imatrix | wiki.test.raw | 4.97 |  |  |  |  |
| IQ4_NL     | imatrix | custom        | 4.97 |  |  |  |  |
| IQ4_NL     | imatrix | 500k-custom+wiki (ctx=512) | 4.97 |  |  |  |  |
| IQ4_NL     | imatrix | 500k-custom+wiki (ctx=2048) | 4.97 |  |  |  | |
| IQ4_NL     | imatrix | 500k-custom+wiki (ctx=8192) | 4.97 |  |  |  |  |

Figure (To Do)
Radial MMLU based on FP16, Q4KM None, Q4KM Custom (best)

Tool Performance

| quant | technique | dataset | size (GiB) | Comp Sci. | Eng. | Math | Average |
|---|---|---|---|---|---|---|---|
| FP16       | none    | —             | 4.97 |  |  |  |  |
| Q4_K_M     | imatrix | wiki.test.raw | 4.97 |  |  |  |  |
| Q4_K_M     | imatrix | custom        | 4.97 |  |  |  |  |
| Q4_K_M     | imatrix | 500k-custom+wiki (ctx=512) | 4.97 |  |  |  |  |
| Q4_K_M     | imatrix | 500k-custom+wiki (ctx=2048) | 4.97 |  |  |  | |
| Q4_K_M     | imatrix | 500k-custom+wiki (ctx=8192) | 4.97 |  |  |  |  |
| IQ4_NL     | imatrix | wiki.test.raw | 4.97 |  |  |  |  |
| IQ4_NL     | imatrix | custom        | 4.97 |  |  |  |  |
| IQ4_NL     | imatrix | 500k-custom+wiki (ctx=512) | 4.97 |  |  |  |  |
| IQ4_NL     | imatrix | 500k-custom+wiki (ctx=2048) | 4.97 |  |  |  | |
| IQ4_NL     | imatrix | 500k-custom+wiki (ctx=8192) | 4.97 |  |  |  |  |
