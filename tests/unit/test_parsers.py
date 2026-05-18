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
    assert s.prefill_stdev is None and s.decode_stdev is None


def test_parse_bench_log_extracts_stdev_from_json():
    """llama-bench -r >=2 emits stddev_ts and samples_ts; we surface both."""
    sample = """
[
  {"n_prompt": 2048, "n_gen": 0, "avg_ts": 664.18, "stddev_ts": 5.43,
   "samples_ts": [660.1, 663.2, 668.0, 665.4, 663.18]},
  {"n_prompt": 0, "n_gen": 128, "avg_ts": 43.14, "stddev_ts": 0.21,
   "samples_ts": [42.9, 43.1, 43.3, 43.2, 43.0]}
]
""".strip()
    s = parse_bench_log(sample)
    assert s.prefill_tok_s == 664.18 and s.prefill_stdev == 5.43
    assert s.decode_tok_s == 43.14 and s.decode_stdev == 0.21
    assert s.n_repetitions == 5
    # TTFT stdev is propagated from prefill stdev via the reciprocal relation.
    assert s.ttft_stdev_ms is not None
    expected_rel = 5.43 / 664.18
    assert abs(s.ttft_stdev_ms / s.ttft_2k_ms - expected_rel) < 1e-9


def test_parse_bench_log_falls_back_to_computing_stdev_from_samples():
    """Older llama-bench builds emit samples_ts without stddev_ts."""
    sample = """
[
  {"n_prompt": 2048, "n_gen": 0, "avg_ts": 100.0, "samples_ts": [95, 100, 105]},
  {"n_prompt": 0, "n_gen": 128, "avg_ts": 50.0, "samples_ts": [49, 50, 51]}
]
""".strip()
    s = parse_bench_log(sample)
    # Sample stdev of [95, 100, 105] = 5.0 (n-1 denominator)
    assert s.prefill_stdev is not None and abs(s.prefill_stdev - 5.0) < 1e-9
    assert s.decode_stdev is not None and abs(s.decode_stdev - 1.0) < 1e-9
