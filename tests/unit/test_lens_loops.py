"""Repetition-detection tests (no model load required)."""

from __future__ import annotations

from quant_tuner.lens.loops import detect_repetition


def test_period_1_loop():
    tokens = [5, 6, 7] + [9] * 30
    loop = detect_repetition(tokens)
    assert loop is not None
    assert loop.period == 1
    assert loop.tokens == [9]
    assert loop.n_repeats >= 3


def test_kgram_loop():
    tokens = [1, 2, 3] + [7, 8, 9] * 10
    loop = detect_repetition(tokens)
    assert loop is not None
    assert loop.period == 3
    assert loop.tokens == [7, 8, 9]


def test_no_loop_returns_none():
    tokens = list(range(50))
    assert detect_repetition(tokens) is None


def test_min_repeats_threshold():
    # only 2 repeats of a 4-gram -> below default min_repeats=3
    tokens = [0, 1, 2, 3, 0, 1, 2, 3]
    assert detect_repetition(tokens, min_span=8) is None


def test_min_span_filters_short_cycles():
    # period-1 repeated 3x is only span 3 (< default min_span 24)
    tokens = [1, 2, 3, 9, 9, 9]
    assert detect_repetition(tokens) is None


def test_earliest_onset_preferred():
    # a period-2 loop starting early should be picked over a later period-1 tail
    tokens = [0] + [4, 5] * 20
    loop = detect_repetition(tokens)
    assert loop is not None
    assert loop.start == 1
