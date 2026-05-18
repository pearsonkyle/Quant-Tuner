# quant-tuner

Calibrate your own GGUF quantizations from real usage logs.

`quant-tuner` takes a HuggingFace model and a corpus derived from your own
prompt/response logs, then produces a GGUF quantization tuned to the
distribution your model actually sees in production. It also benchmarks
the result against an FP16 reference using KL-divergence, perplexity, tool-call
accuracy, and prefill/decode speed.

## Methods

| Method   | What it does                                              | Pre-quant cost |
| -------- | --------------------------------------------------------- | -------------- |
| imatrix  | Per-tensor importance from llama.cpp forward passes       | Low            |
| AWQ      | Activation-aware weight scaling before quantize           | Medium         |
| GPTQ     | Hessian-based weight update before quantize               | High           |

## Quick start

```bash
uv sync
git submodule update --init --recursive
# Build llama.cpp once (Metal on macOS)
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build -DGGML_METAL=ON
cmake --build vendor/llama.cpp/build -j

uv run quant-tuner run \
    --model meta-llama/Llama-3.2-1B \
    --logs ~/usage.jsonl \
    --method awq \
    --quant Q4_K_M \
    --out ./out
```

## Status

Pre-alpha. See `docs/methods.md` and `docs/benchmarks.md` for details.

## Pinned llama.cpp

This repo vendors `llama.cpp` at commit `45b455e66fc09abed65b7d52d42a4a29ba0d45d6`
as a git submodule under `vendor/llama.cpp`.
