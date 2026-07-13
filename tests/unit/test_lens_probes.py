"""Probe classification + comparison tests (no model load required)."""

from __future__ import annotations

from quant_tuner.lens.probes import ProbeResult, _classify, compare_probes


def test_classify_correct():
    assert _classify(0, [5, 3, 1, 0], k=40, late_frac=0.5) == "correct"


def test_classify_suppressed():
    # not top-1 finally, but in top-k at late layers -> suppressed
    assert _classify(7, [500, 100, 20, 15], k=40, late_frac=0.5) == "suppressed"


def test_classify_absent():
    # never within top-k at any layer
    assert _classify(9000, [8000, 7000, 6000, 5000], k=40, late_frac=0.5) == "absent"


def _pr(pid, cls, domain="general", rank=5):
    return ProbeResult(probe_id=pid, domain=domain, gold="x", rank_final=rank,
                       rank_by_layer=[rank], classification=cls)


def test_compare_transition_matrix():
    ref = [_pr("a", "correct"), _pr("b", "correct"), _pr("c", "correct")]
    quant = [_pr("a", "correct"), _pr("b", "suppressed"), _pr("c", "absent")]
    cmp = compare_probes(quant, ref)
    assert cmp.transitions["correct->correct"] == 1
    assert cmp.transitions["correct->suppressed"] == 1
    assert cmp.transitions["correct->absent"] == 1
    s = cmp.summary()
    assert s["probe_preserved"] == 1
    assert s["probe_suppressed"] == 1
    assert s["probe_erased"] == 1
    # of the 2 lost, 1 is suppressed
    assert abs(s["probe_suppressed_frac_of_lost"] - 0.5) < 1e-9


def test_regressed_list_only_captures_correct_to_worse():
    ref = [_pr("a", "correct"), _pr("b", "absent")]
    quant = [_pr("a", "absent"), _pr("b", "absent")]
    cmp = compare_probes(quant, ref)
    ids = {r["probe_id"] for r in cmp.regressed}
    assert ids == {"a"}  # b was already absent on ref
