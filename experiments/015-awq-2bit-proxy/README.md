# Experiment 012 — AWQ with a 2-bit K-quant-shaped proxy

## Hypothesis

The default AWQ proxy quantizer is symmetric INT4 group-128 RTN. When the
final target is IQ2_XS, IQ2_M, or Q2_K_S, the α search is choosing scales
that minimize *INT4* reconstruction error — which is a poor stand-in for
K-quant's 16-element sub-blocks with shared 4-bit (min, scale). The proxy
mismatch likely explains part of the "AWQ helps i-quants more than Q2_K"
asymmetry in exp-010.

This experiment swaps in `fake_quant_q2k_block16`: asymmetric per-16-element
2-bit RTN with nested ~4-bit per-row quantization of the per-block (min,
scale). Not bit-exact to llama.cpp, but matches the *error shape*.

## Setup

- Model: `google/gemma-4-31B-it`
- Quants: `IQ2_XS`, `IQ2_M`, `Q2_K_S`
- Bundle records `proxy="q2k_b16"` for traceability
- Reuses exp-009 artifacts; compares against exp-009 and exp-010

## Run

```bash
PYTHONPATH=src .venv/bin/python scripts/run_exp015_awq_2bit_proxy.py
```

Read results:

```bash
cat out/exp-015/google__gemma-4-31B-it/table.md
```

## Success criteria

Either: IQ2_XS / IQ2_M PPL improves vs exp-010 by ≥5%, OR the α histogram
shifts noticeably (more weight on higher α) — indicating the proxy is in
fact recommending different scales for K-quant targets.
