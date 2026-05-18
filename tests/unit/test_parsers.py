"""Unit tests for log-parsing helpers (no llama.cpp build required)."""

from quant_tuner.bench.kld import parse_kld_log
from quant_tuner.bench.speed import parse_bench_log


def test_parse_kld_log_extracts_all_metrics():
    sample = """
Mean PPL(Q)                   :    9.123456 ±   0.012345
Mean PPL(base)                :    9.000000 ±   0.012000
Mean PPL(Q)/PPL(base)         :    1.013717 ±   0.000456
Mean KLD:    0.852188   ±   0.001000
Median KLD:    0.001567
Same top p:   87.033 %
RMS Δp     :   12.868
""".strip()
    m = parse_kld_log(sample)
    assert m.ppl == 9.123456
    assert m.ppl_ratio == 1.013717
    assert m.mean_kld == 0.852188
    assert m.median_kld == 0.001567
    assert m.same_top_p == 87.033
    assert m.rms_dp == 12.868


def test_parse_kld_log_missing_returns_none():
    m = parse_kld_log("no metrics here")
    assert all(v is None for v in (m.ppl, m.mean_kld, m.same_top_p))


def test_parse_bench_log_extracts_prefill_and_decode():
    sample = """
some preamble
[
  {"n_prompt": 2048, "n_gen": 0, "avg_ts": 664.18},
  {"n_prompt": 0, "n_gen": 128, "avg_ts": 43.14}
]
trailing
""".strip()
    s = parse_bench_log(sample)
    assert s.prefill_tok_s == 664.18
    assert s.decode_tok_s == 43.14
    assert s.ttft_2k_ms is not None
    assert abs(s.ttft_2k_ms - (2048 / 664.18 * 1000.0)) < 1e-6


def test_parse_bench_log_no_json_returns_empty():
    s = parse_bench_log("garbage")
    assert s.prefill_tok_s is None and s.decode_tok_s is None and s.ttft_2k_ms is None
