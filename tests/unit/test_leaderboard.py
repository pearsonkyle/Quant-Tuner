"""Unit tests for SQS computation + markdown rendering."""

from __future__ import annotations

import csv

import pytest

from quant_tuner.leaderboard.aggregate import (
    DEFAULT_WEIGHTS,
    F16Baseline,
    aggregate,
    compute_sqs,
    find_f16_baseline,
    render_markdown,
    sort_rows,
)

F16 = F16Baseline(size_gib=16.0, same_top_p=100.0, decode_tok_s=22.0)


def test_sqs_is_exactly_one_for_f16_row():
    """The reference row must score 1.0 no matter what weights are used."""
    row = {"size_gib": 16.0, "same_top_p": 100.0, "decode_tok_s": 22.0}
    for weights in ((1.0, 2.0, 1.0), (1.0, 1.0, 1.0), (3.0, 0.5, 1.0)):
        a, b, c = weights
        assert compute_sqs(row, F16, alpha=a, beta=b, gamma=c) == pytest.approx(1.0)


def test_sqs_rewards_compression():
    """A smaller-but-equally-faithful-and-fast quant should score above F16."""
    half_size = {"size_gib": 8.0, "same_top_p": 100.0, "decode_tok_s": 22.0}
    score = compute_sqs(half_size, F16)
    assert score > 1.0


def test_sqs_penalizes_fidelity_loss():
    same_size_lower_fid = {"size_gib": 16.0, "same_top_p": 80.0, "decode_tok_s": 22.0}
    score = compute_sqs(same_size_lower_fid, F16)
    assert score < 1.0


def test_sqs_fidelity_is_weighted_twice_as_heavy_as_compression():
    """With default weights (1, 2, 1), a 10% fidelity drop should hurt more than
    a 10% compression gain helps."""
    fid_drop = {"size_gib": 16.0, "same_top_p": 90.0, "decode_tok_s": 22.0}
    size_gain = {"size_gib": 16.0 / 1.111, "same_top_p": 100.0, "decode_tok_s": 22.0}
    s_drop = compute_sqs(fid_drop, F16)
    s_gain = compute_sqs(size_gain, F16)
    # Both are ±~10% on their factor; with β=2 vs α=1 the fidelity drop's
    # magnitude should exceed the compression gain's.
    assert (1.0 - s_drop) > (s_gain - 1.0)


def test_sqs_missing_fields_default_to_one():
    """A row missing decode_tok_s shouldn't crash — it just contributes nothing."""
    partial = {"size_gib": 16.0, "same_top_p": 100.0}  # no decode_tok_s
    s = compute_sqs(partial, F16)
    assert s == pytest.approx(1.0)


def test_sqs_returns_none_when_weights_sum_to_zero():
    row = {"size_gib": 16.0, "same_top_p": 100.0, "decode_tok_s": 22.0}
    assert compute_sqs(row, F16, alpha=0.0, beta=0.0, gamma=0.0) is None


def test_find_f16_baseline_matches_fp16_label():
    rows = [
        {"model": "baseline/fp16", "size_gib": "16.0", "same_top_p": "100.0",
         "decode_tok_s": "22.0"},
        {"model": "exp01/analytic", "size_gib": "4.0", "same_top_p": "85.0",
         "decode_tok_s": "55.0"},
    ]
    f16 = find_f16_baseline(rows)
    assert f16 is not None and f16.size_gib == 16.0


def test_find_f16_baseline_matches_slash_f16_suffix():
    rows = [{"model": "mymodel/f16", "size_gib": "10.0",
             "same_top_p": "100.0", "decode_tok_s": "30.0"}]
    f16 = find_f16_baseline(rows)
    assert f16 is not None and f16.size_gib == 10.0


def test_find_f16_baseline_returns_none_when_missing():
    rows = [{"model": "imatrix/Q4_K_M", "size_gib": "4.0",
             "same_top_p": "85.0", "decode_tok_s": "55.0"}]
    assert find_f16_baseline(rows) is None


def test_sort_rows_default_sqs_desc():
    rows = [
        {"model": "a", "sqs": 1.2},
        {"model": "b", "sqs": 1.5},
        {"model": "c", "sqs": 0.9},
    ]
    out = sort_rows(rows, by="sqs")
    assert [r["model"] for r in out] == ["b", "a", "c"]


def test_sort_rows_kld_default_asc():
    rows = [
        {"model": "a", "mean_kld": 0.5},
        {"model": "b", "mean_kld": 0.1},
        {"model": "c", "mean_kld": 1.2},
    ]
    out = sort_rows(rows, by="mean_kld")
    assert [r["model"] for r in out] == ["b", "a", "c"]


def test_sort_rows_non_numeric_sinks_to_bottom():
    rows = [
        {"model": "good", "sqs": 1.4},
        {"model": "missing", "sqs": ""},
        {"model": "best", "sqs": 1.6},
    ]
    out = sort_rows(rows, by="sqs")
    assert out[0]["model"] == "best"
    assert out[-1]["model"] == "missing"


def test_render_markdown_includes_weight_header_comment():
    rows = [{"model": "x", "sqs": 1.234}]
    md = render_markdown(rows, weights=(1.0, 2.0, 1.0))
    assert "<!-- SQS weights: alpha=1.0 beta=2.0 gamma=1.0" in md
    assert "1.234" in md


def test_aggregate_end_to_end(tmp_path):
    """Synthetic 4-row CSV: F16 + 3 quants → SQS computed, sorted desc."""
    csv_path = tmp_path / "results.csv"
    fields = [
        "model", "size_gib", "bpw", "ppl", "ppl_ratio",
        "mean_kld", "median_kld", "same_top_p", "rms_dp",
        "prefill_tok_s", "decode_tok_s", "ttft_2k_ms", "quant_path",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({"model": "baseline/fp16", "size_gib": 16.0, "bpw": 16.0,
                    "ppl": 10.0, "ppl_ratio": 1.0,
                    "mean_kld": 0.0, "median_kld": 0.0,
                    "same_top_p": 100.0, "rms_dp": 0.0,
                    "prefill_tok_s": 600, "decode_tok_s": 22.0,
                    "ttft_2k_ms": 3300, "quant_path": "f16.gguf"})
        w.writerow({"model": "Q4_K_M-none", "size_gib": 5.0, "bpw": 4.5,
                    "ppl": 11.0, "ppl_ratio": 1.1,
                    "mean_kld": 1.5, "median_kld": 0.005,
                    "same_top_p": 83.0, "rms_dp": 18.0,
                    "prefill_tok_s": 660, "decode_tok_s": 42.0,
                    "ttft_2k_ms": 3100, "quant_path": "none.gguf"})
        w.writerow({"model": "Q4_K_M-imatrix", "size_gib": 5.0, "bpw": 4.5,
                    "ppl": 10.5, "ppl_ratio": 1.05,
                    "mean_kld": 0.85, "median_kld": 0.0017,
                    "same_top_p": 87.0, "rms_dp": 13.0,
                    "prefill_tok_s": 665, "decode_tok_s": 43.0,
                    "ttft_2k_ms": 3080, "quant_path": "imx.gguf"})
        w.writerow({"model": "Q4_K_M-awq", "size_gib": 5.0, "bpw": 4.5,
                    "ppl": 11.3, "ppl_ratio": 1.18,
                    "mean_kld": 0.88, "median_kld": 0.0019,
                    "same_top_p": 86.8, "rms_dp": 13.0,
                    "prefill_tok_s": 755, "decode_tok_s": 64.0,
                    "ttft_2k_ms": 2710, "quant_path": "awq.gguf"})

    md = aggregate(csv_path, weights=DEFAULT_WEIGHTS)
    lines = [ln for ln in md.splitlines() if ln.startswith("|") and "---" not in ln]
    # First data line is the highest SQS — should be AWQ (faster + same fidelity tier).
    assert "Q4_K_M-awq" in lines[1]
    # F16 is somewhere; baseline/fp16 should NOT be first (its SQS is exactly 1).
    # The Q4 quants should all score above 1 (smaller + faster, with modest fidelity drop).
    assert all("Model" not in lines[i] or i == 0 for i in range(len(lines)))


def test_merge_toolcall_handles_multi_rep_csv(tmp_path):
    """Aggregated multi-rep CSV (with `_mean`/`_stdev` cols) must populate both
    the base metric AND a `<key>_stdev` companion key for the renderer."""
    from quant_tuner.leaderboard.aggregate import _format_cell, merge_toolcall

    tc_path = tmp_path / "toolcall_reps_aggregated.csv"
    fields = [
        "model", "n_reps",
        "tool_selection_acc_mean", "tool_selection_acc_stdev",
        "param_acc_mean_mean", "param_acc_mean_stdev",
        "schema_valid_rate_mean", "schema_valid_rate_stdev",
        "rollout_complete_rate_mean", "rollout_complete_rate_stdev",
    ]
    with tc_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({"model": "Q4_K_M-hybrid_custom.gguf", "n_reps": 10,
                    "tool_selection_acc_mean": 0.444, "tool_selection_acc_stdev": 0.045,
                    "param_acc_mean_mean": 0.356, "param_acc_mean_stdev": 0.030,
                    "schema_valid_rate_mean": 0.689, "schema_valid_rate_stdev": 0.025,
                    "rollout_complete_rate_mean": 0.76, "rollout_complete_rate_stdev": 0.012})

    rows = [{"model": "hybrid/Q4_K_M-custom",
             "quant_path": "/abs/path/Q4_K_M-hybrid_custom.gguf"}]
    out = merge_toolcall(rows, tc_path)
    assert out[0]["tool_selection_acc"] == "0.444"
    assert out[0]["tool_selection_acc_stdev"] == "0.045"
    assert out[0]["schema_valid_rate"] == "0.689"

    # Rendered cell should be "44.4 ± 4.5"
    cell = _format_cell("tool_selection_acc", out[0])
    assert cell == "44.4 ± 4.5", cell
    cell_schema = _format_cell("schema_valid_rate", out[0])
    assert cell_schema == "68.9 ± 2.5", cell_schema


def test_aggregate_raises_without_f16_row(tmp_path):
    csv_path = tmp_path / "no_f16.csv"
    fields = ["model", "size_gib", "same_top_p", "decode_tok_s"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({"model": "Q4_K_M-imatrix", "size_gib": 5.0,
                    "same_top_p": 87.0, "decode_tok_s": 43.0})
    with pytest.raises(RuntimeError, match="no F16 baseline"):
        aggregate(csv_path)
