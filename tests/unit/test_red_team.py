"""Unit tests for the red-team eval — pure helpers, no deepteam / server / model.

The whole point of these tests is that they run **without the `redteam` extra
installed**. `eval/red_team.py` imports deepteam lazily (inside the functions
that need it) precisely so the scoring, pairing and aggregation layers stay
importable and testable, the same way `eval/swebench.py` keeps `minisweagent`
out of module scope. If someone hoists those imports back to the top, every test
in this file fails at collection — which is the intent.

Covers: `<think>` stripping, config resolution, category/attack-class
normalization, `case_id` stability, `build_summary` counting (errored cases stay
in the denominator), the zero-scored guard, `TargetSampling`'s native-vs-extra_body
split, multi-turn history rendering + callback arity, `pair_runs` McNemar-style
pairing, and `aggregate_reps`.
"""

from __future__ import annotations

import inspect
import types
from pathlib import Path

import pytest

from quant_tuner.eval.red_team import (
    CATEGORY_MAP,
    RedTeamSummary,
    Target,
    TargetSampling,
    _assert_scored,
    _attack_class,
    _category_for,
    _chat_template_extra_body,
    _entry_options,
    _make_target_callback,
    _mcnemar_exact_p,
    _strip_thinking,
    aggregate_reps,
    build_attacks,
    build_summary,
    build_vulnerabilities,
    case_id,
    group_per_case_by_model,
    load_red_team_config,
    pair_runs,
    read_per_case_csv,
    resolve_red_team_config,
    spearman_rho,
    turns_to_messages,
    write_disclosure_report,
    write_per_case_csv,
)

# ---------------------------------------------------------------------------
# Fakes standing in for deepteam's RTTestCase / RiskAssessment
# ---------------------------------------------------------------------------


class FakeCase:
    """Duck-type of ``deepteam.test_case.RTTestCase`` for the fields we read."""

    def __init__(
        self,
        vulnerability="Toxicity",
        vulnerability_type="insults",
        attack_method="Roleplay",
        score=1,
        input="attack prompt",
        actual_output="I can't help with that.",
        reason="refused",
        error=None,
    ):
        self.vulnerability = vulnerability
        self.vulnerability_type = vulnerability_type
        self.attack_method = attack_method
        self.score = score
        self.input = input
        self.actual_output = actual_output
        self.reason = reason
        self.error = error


class FakeOverview:
    def __init__(self, run_duration=1.5):
        self.run_duration = run_duration


class FakeAssessment:
    def __init__(self, cases, run_duration=1.5):
        self.test_cases = cases
        self.overview = FakeOverview(run_duration)


def _summary(cases, model="m"):
    return build_summary(FakeAssessment(cases), model)


# ---------------------------------------------------------------------------
# _strip_thinking
# ---------------------------------------------------------------------------


def test_strip_thinking_removes_think_block():
    assert _strip_thinking("<think>{broken json</think>real answer") == "real answer"


def test_strip_thinking_removes_thinking_block_multiline():
    assert _strip_thinking("<thinking>a\nb\n</thinking>  out  ") == "out"


def test_strip_thinking_handles_none_and_empty():
    assert _strip_thinking(None) == ""
    assert _strip_thinking("") == ""


def test_strip_thinking_leaves_plain_text_alone():
    assert _strip_thinking("no tags here") == "no tags here"


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def test_resolve_config_by_bare_name():
    assert resolve_red_team_config("red_team_minimal").name == "red_team_minimal.yaml"


def test_resolve_config_by_name_with_suffix():
    assert resolve_red_team_config("red_team_minimal.yaml").name == "red_team_minimal.yaml"


def test_resolve_config_by_explicit_path(tmp_path):
    p = tmp_path / "custom.yaml"
    p.write_text("vulnerabilities: {}\n")
    assert resolve_red_team_config(p) == p


def test_resolve_config_missing_raises():
    with pytest.raises(FileNotFoundError, match="not-a-real-config"):
        resolve_red_team_config("not-a-real-config")


def test_load_config_passes_dict_through():
    cfg = {"vulnerabilities": {"bias": {"enabled": True}}}
    assert load_red_team_config(cfg) is cfg


def _packaged_config_names() -> list[str]:
    """Discover every shipped config, so a new YAML is validated automatically."""
    import quant_tuner.eval.red_team as rt

    return sorted(p.stem for p in (Path(rt.__file__).parent / "red_team_configs").glob("*.yaml"))


PACKAGED_CONFIGS = _packaged_config_names()


def test_packaged_configs_directory_is_not_empty():
    assert PACKAGED_CONFIGS, "no red_team_configs/*.yaml found — packaging regression"


@pytest.mark.parametrize("name", PACKAGED_CONFIGS)
def test_packaged_configs_parse_and_enable_something(name):
    """Every shipped config must load and actually enable a vulnerability."""
    cfg = load_red_team_config(name)
    enabled = [
        k for k, v in (cfg.get("vulnerabilities") or {}).items()
        if isinstance(v, dict) and v.get("enabled")
    ]
    assert enabled, f"{name} enables no vulnerabilities"


@pytest.mark.parametrize("name", PACKAGED_CONFIGS)
def test_packaged_configs_enable_an_attack(name):
    cfg = load_red_team_config(name)
    assert cfg.get("attacks"), f"{name} enables no attacks"


@pytest.mark.parametrize("name", PACKAGED_CONFIGS)
def test_packaged_config_keys_are_all_known(name):
    """A config key with no registry entry must fail loudly, so keep them in sync.

    `build_vulnerabilities`/`build_attacks` raise ValueError on an unknown key;
    here we assert the shipped configs contain none, without needing deepteam
    (the ValueError is raised before any import).
    """
    from quant_tuner.eval.red_team import _ATTACK_SPECS, _VULN_SPECS

    cfg = load_red_team_config(name)
    for key in (cfg.get("vulnerabilities") or {}):
        assert key == "custom" or key in _VULN_SPECS, f"{name}: unknown vulnerability {key!r}"
    for key in (cfg.get("attacks") or {}):
        assert key in _ATTACK_SPECS, f"{name}: unknown attack {key!r}"


# ---------------------------------------------------------------------------
# Registry lookups (no deepteam needed — the ValueError precedes the import)
# ---------------------------------------------------------------------------


def test_build_vulnerabilities_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown red-team vulnerability 'nope'"):
        build_vulnerabilities({"vulnerabilities": {"nope": {"enabled": True}}})


def test_build_attacks_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown red-team attack 'nope'"):
        build_attacks({"attacks": {"nope": True}})


def test_build_skips_disabled_entries_without_importing_deepteam():
    """A fully-disabled config must not need the extra at all."""
    cfg = {
        "vulnerabilities": {"bias": {"enabled": False}, "toxicity": {}},
        "attacks": {"roleplay": False, "base64": {"enabled": False}},
    }
    assert build_vulnerabilities(cfg) == []
    assert build_attacks(cfg) == []


def test_entry_options_strips_enabled_and_empty_types():
    assert _entry_options({"enabled": True, "types": ["insults"]}) == {"types": ["insults"]}
    # An empty `types` means "all types" (deepteam's own default), not "none".
    assert _entry_options({"enabled": True, "types": []}) == {}
    assert _entry_options({"enabled": True}) == {}
    assert _entry_options(True) == {}


def test_entry_options_passes_constructor_kwargs_through():
    assert _entry_options({"enabled": True, "persona": "pirate", "weight": 3}) == {
        "persona": "pirate",
        "weight": 3,
    }


# ---------------------------------------------------------------------------
# Category + attack-class normalization
# ---------------------------------------------------------------------------


def test_category_lookup_is_space_and_case_insensitive():
    assert _category_for("PII Leakage") == "Data Privacy"
    assert _category_for("PIILeakage") == "Data Privacy"
    assert _category_for("piileakage") == "Data Privacy"


def test_category_lookup_falls_back_to_other():
    assert _category_for("SomethingNew") == "Other"
    assert _category_for(None) == "Other"


def test_agentic_vulnerabilities_have_their_own_category():
    assert _category_for("ShellInjection") == "Agentic"
    assert _category_for("Excessive Agency") == "Agentic"


def test_every_category_map_entry_resolves():
    for name in CATEGORY_MAP:
        assert _category_for(name) != "Other"


@pytest.mark.parametrize(
    "method,expected",
    [
        ("Linear Jailbreaking", "multi-turn"),
        ("CrescendoJailbreaking", "multi-turn"),
        ("tree_jailbreaking", "multi-turn"),
        ("Roleplay", "single-turn"),
        ("Base64", "single-turn"),
        (None, "none"),
        ("", "none"),
    ],
)
def test_attack_class(method, expected):
    assert _attack_class(method) == expected


# ---------------------------------------------------------------------------
# case_id
# ---------------------------------------------------------------------------


def test_case_id_is_stable_across_calls():
    a = case_id("Toxicity", "insults", "Roleplay", "prompt")
    b = case_id("Toxicity", "insults", "Roleplay", "prompt")
    assert a == b and len(a) == 16


def test_case_id_varies_with_every_component():
    base = case_id("Toxicity", "insults", "Roleplay", "prompt")
    assert case_id("Bias", "insults", "Roleplay", "prompt") != base
    assert case_id("Toxicity", "threats", "Roleplay", "prompt") != base
    assert case_id("Toxicity", "insults", "Base64", "prompt") != base
    assert case_id("Toxicity", "insults", "Roleplay", "other") != base


def test_case_id_tolerates_none_attack_method():
    assert case_id("Toxicity", "insults", None, "prompt")


# ---------------------------------------------------------------------------
# build_summary
# ---------------------------------------------------------------------------


def test_build_summary_counts_pass_fail_error():
    s = _summary([FakeCase(score=1), FakeCase(score=0), FakeCase(score=None)])
    assert (s.n_passing, s.n_failing, s.n_errored) == (1, 1, 1)


def test_build_summary_keeps_errored_in_n_tests_but_out_of_pass_rate():
    """n_tests is the attempted count; the rate's denominator is scored-only.

    Both halves matter: an honest n_tests is what makes two models comparable,
    and excluding errors from the rate is what stops a timeout reading as a
    jailbreak.
    """
    s = _summary([FakeCase(score=1), FakeCase(score=0), FakeCase(score=None)])
    assert s.n_tests == 3
    assert s.pass_rate == 0.5


def test_build_summary_pass_rate_zero_when_nothing_scored():
    s = _summary([FakeCase(score=None), FakeCase(score=None)])
    assert s.n_tests == 2 and s.pass_rate == 0.0


def test_build_summary_per_case_has_one_row_per_case_including_errors():
    cases = [FakeCase(score=1), FakeCase(score=0, input="b"), FakeCase(score=None, input="c")]
    s = _summary(cases)
    assert len(s.per_case) == 3
    assert [c["errored"] for c in s.per_case] == [False, False, True]
    assert len({c["case_id"] for c in s.per_case}) == 3


def test_build_summary_per_case_keeps_score_tristate():
    s = _summary([FakeCase(score=None)])
    assert s.per_case[0]["score"] is None


def test_build_summary_failed_cases_excludes_errors():
    s = _summary([FakeCase(score=0, input="a"), FakeCase(score=None, input="b")])
    assert len(s.failed_cases) == 1
    assert s.failed_cases[0]["input"] == "a"


def test_build_summary_by_category_totals_include_errors():
    s = _summary([FakeCase(score=1), FakeCase(score=None)])
    assert s.by_category["Safety"]["total_tests"] == 2
    assert s.by_category["Safety"]["errored"] == 1
    assert s.by_category["Safety"]["pass_rate"] == 1.0


def test_build_summary_splits_by_attack_class():
    s = _summary(
        [
            FakeCase(attack_method="Roleplay", score=1),
            FakeCase(attack_method="CrescendoJailbreaking", score=0, input="x"),
        ]
    )
    assert s.by_attack_class["single-turn"]["pass_rate"] == 1.0
    assert s.by_attack_class["multi-turn"]["pass_rate"] == 0.0


def test_build_summary_groups_vulnerability_and_type():
    s = _summary(
        [
            FakeCase(vulnerability="Bias", vulnerability_type="race", score=1),
            FakeCase(vulnerability="Bias", vulnerability_type="gender", score=0, input="x"),
        ]
    )
    rows = {r["type"]: r for r in s.by_vulnerability}
    assert rows["race"]["pass_rate"] == 1.0
    assert rows["gender"]["pass_rate"] == 0.0
    assert rows["race"]["category"] == "Responsible AI"


def test_build_summary_handles_enum_like_vulnerability_type():
    class Enumish:
        value = "cybercrime"

    s = _summary([FakeCase(vulnerability_type=Enumish(), score=1)])
    assert s.per_case[0]["vulnerability_type"] == "cybercrime"


def test_build_summary_tolerates_missing_run_duration():
    class NoDuration:
        run_duration = None

    a = FakeAssessment([FakeCase()])
    a.overview = NoDuration()
    assert build_summary(a, "m").run_duration_sec == 0.0


# ---------------------------------------------------------------------------
# scalar_metrics (reps integration)
# ---------------------------------------------------------------------------


def test_scalar_metrics_are_all_floats():
    s = _summary([FakeCase(score=1), FakeCase(score=0, input="x")])
    metrics = s.scalar_metrics()
    assert metrics["pass_rate"] == 0.5
    assert metrics["n_tests"] == 2.0
    assert all(isinstance(v, float) for v in metrics.values())


def test_scalar_metrics_include_per_category_and_class_keys():
    s = _summary([FakeCase(score=1), FakeCase(vulnerability="Bias", score=0, input="x")])
    metrics = s.scalar_metrics()
    assert "pass_rate_safety" in metrics
    assert "pass_rate_responsible_ai" in metrics
    assert "pass_rate_single_turn" in metrics


# ---------------------------------------------------------------------------
# Zero-scored guard
# ---------------------------------------------------------------------------


def test_assert_scored_raises_when_every_case_errored():
    s = _summary([FakeCase(score=None), FakeCase(score=None, input="b")])
    with pytest.raises(RuntimeError, match="scored 0 of 2 cases"):
        _assert_scored(s, "IQ2_XS")


def test_assert_scored_passes_when_something_scored():
    _assert_scored(_summary([FakeCase(score=1), FakeCase(score=None)]), "ok")


def test_assert_scored_ignores_empty_run():
    """An empty bank is a config problem, not a dead target — different error."""
    _assert_scored(RedTeamSummary(model="m", n_tests=0, pass_rate=0.0), "empty")


# ---------------------------------------------------------------------------
# TargetSampling
# ---------------------------------------------------------------------------


def test_target_sampling_splits_native_from_extra_body():
    s = TargetSampling(temperature=0.3, top_p=0.9, top_k=20, min_p=0.05, repeat_penalty=1.1)
    assert s.native_kwargs() == {"temperature": 0.3, "top_p": 0.9}
    assert s.extra_body() == {"top_k": 20, "min_p": 0.05, "repeat_penalty": 1.1}


def test_target_sampling_omits_none_fields():
    s = TargetSampling(temperature=None, top_p=None, top_k=None)
    assert s.native_kwargs() == {}
    assert s.extra_body() == {}


def test_target_sampling_max_tokens_is_native():
    assert TargetSampling(max_tokens=256).native_kwargs()["max_tokens"] == 256


# ---------------------------------------------------------------------------
# chat_template_kwargs plumbing
# ---------------------------------------------------------------------------


def test_chat_template_extra_body_accepts_json_string():
    assert _chat_template_extra_body('{"enable_thinking": false}') == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_chat_template_extra_body_accepts_dict():
    assert _chat_template_extra_body({"enable_thinking": False}) == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_chat_template_extra_body_none_passes_through():
    assert _chat_template_extra_body(None) is None


# ---------------------------------------------------------------------------
# Multi-turn history — the bug that made Crescendo/Linear/Tree meaningless
# ---------------------------------------------------------------------------


class FakeTurn:
    def __init__(self, role, content):
        self.role = role
        self.content = content


def test_target_callback_declares_two_parameters():
    """deepteam forwards `turns` only to a callback with >1 parameter.

    `wrap_model_callback` inspects the signature; a one-arg callback silently
    gets no conversation history, so every multi-turn jailbreak probes a target
    with no memory of the escalation. This is the regression guard.
    """
    cb = _make_target_callback("http://127.0.0.1:1/v1", "m", "k")
    assert len(inspect.signature(cb).parameters) == 2


def test_turns_to_messages_appends_input_after_history():
    turns = [FakeTurn("user", "hi"), FakeTurn("assistant", "hello")]
    assert turns_to_messages(turns, "now the attack") == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "now the attack"},
    ]


def test_turns_to_messages_does_not_duplicate_trailing_input():
    """Some deepteam versions include the current attack as the last turn."""
    turns = [FakeTurn("assistant", "hello"), FakeTurn("user", "attack")]
    msgs = turns_to_messages(turns, "attack")
    assert msgs[-1] == {"role": "user", "content": "attack"}
    assert sum(m["content"] == "attack" for m in msgs) == 1


def test_turns_to_messages_with_no_history_is_a_single_user_turn():
    assert turns_to_messages(None, "x") == [{"role": "user", "content": "x"}]
    assert turns_to_messages([], "x") == [{"role": "user", "content": "x"}]


def test_turns_to_messages_skips_empty_content():
    assert turns_to_messages([FakeTurn("assistant", None)], "x") == [
        {"role": "user", "content": "x"}
    ]


def test_turns_to_messages_accepts_plain_dicts():
    assert turns_to_messages([{"role": "assistant", "content": "a"}], "x") == [
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "x"},
    ]


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------


def test_target_requires_exactly_one_source():
    with pytest.raises(ValueError, match="exactly one"):
        Target("x")
    with pytest.raises(ValueError, match="exactly one"):
        Target("x", model_path=Path("a.gguf"), base_url="http://x/v1")
    assert Target("x", model_path=Path("a.gguf")).label == "x"
    assert Target("x", base_url="http://x/v1").label == "x"


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------


def test_mcnemar_no_discordant_pairs_is_p_one():
    assert _mcnemar_exact_p(0, 0) == 1.0


def test_mcnemar_symmetric_flips_are_not_significant():
    assert _mcnemar_exact_p(5, 5) == 1.0


def test_mcnemar_lopsided_flips_are_significant():
    assert _mcnemar_exact_p(10, 0) < 0.01


def test_mcnemar_is_bounded_and_symmetric():
    for b, c in [(1, 4), (4, 1), (3, 7), (0, 2)]:
        p = _mcnemar_exact_p(b, c)
        assert 0.0 <= p <= 1.0
        assert p == _mcnemar_exact_p(c, b)


# ---------------------------------------------------------------------------
# spearman_rho — used to argue KLD/PPL are blind to safety drift
# ---------------------------------------------------------------------------


def test_spearman_perfect_monotonic_correlations():
    assert spearman_rho([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) == 1.0
    assert spearman_rho([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]) == -1.0


def test_spearman_is_rank_based_not_linear():
    """Monotone but wildly non-linear still gives rho = 1."""
    assert spearman_rho([1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 900.0, 1e6]) == 1.0


def test_spearman_undefined_below_three_points():
    assert spearman_rho([1.0, 2.0], [1.0, 2.0]) is None


def test_spearman_undefined_for_constant_series():
    """'Cannot tell' must not be reported as 'no relationship'."""
    assert spearman_rho([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None


def test_spearman_undefined_on_length_mismatch():
    assert spearman_rho([1.0, 2.0, 3.0], [1.0, 2.0]) is None


def test_spearman_handles_ties_with_midranks():
    rho = spearman_rho([1.0, 1.0, 2.0, 3.0], [1.0, 1.0, 2.0, 3.0])
    assert rho == 1.0


def test_spearman_uncorrelated_is_near_zero():
    rho = spearman_rho([1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0])
    assert rho is not None and abs(rho) < 0.7


# ---------------------------------------------------------------------------
# pair_runs — the paired F16-vs-quant comparison
# ---------------------------------------------------------------------------


def _case(cid, score, category="Safety"):
    return {"case_id": cid, "score": score, "category": category, "errored": score is None}


def test_pair_runs_counts_flips_in_both_directions():
    ref = [_case("a", 1), _case("b", 1), _case("c", 0), _case("d", 0)]
    cand = [_case("a", 1), _case("b", 0), _case("c", 1), _case("d", 0)]
    r = pair_runs(ref, cand)
    assert r["n_paired"] == 4
    assert r["n_both_pass"] == 1
    assert r["n_both_fail"] == 1
    assert r["n_flip_unsafe"] == 1  # b: reference refused, candidate complied
    assert r["n_flip_safe"] == 1  # c: candidate refused where reference complied


def test_pair_runs_net_drift_is_negative_when_quant_is_less_safe():
    ref = [_case("a", 1), _case("b", 1)]
    cand = [_case("a", 0), _case("b", 0)]
    r = pair_runs(ref, cand)
    assert r["n_flip_unsafe"] == 2
    assert r["net_drift"] == -1.0
    assert r["pass_rate_delta"] == -1.0


def test_pair_runs_excludes_cases_errored_on_either_side():
    """An errored case says nothing about either model — it must not become a flip."""
    ref = [_case("a", 1), _case("b", None), _case("c", 1)]
    cand = [_case("a", 0), _case("b", 0), _case("c", None)]
    r = pair_runs(ref, cand)
    assert r["n_paired"] == 1
    assert r["n_flip_unsafe"] == 1


def test_pair_runs_reports_unmatched_case_ids():
    """Non-zero n_unmatched means the frozen bank was not actually shared."""
    r = pair_runs([_case("a", 1)], [_case("b", 1)])
    assert r["n_unmatched"] == 2
    assert r["n_paired"] == 0


def test_pair_runs_identical_runs_have_zero_drift():
    ref = [_case("a", 1), _case("b", 0)]
    r = pair_runs(ref, list(ref))
    assert r["net_drift"] == 0.0
    assert r["pass_rate_delta"] == 0.0
    assert r["mcnemar_p"] == 1.0


def test_pair_runs_breaks_down_by_category():
    ref = [_case("a", 1, "Safety"), _case("b", 1, "Agentic")]
    cand = [_case("a", 0, "Safety"), _case("b", 1, "Agentic")]
    r = pair_runs(ref, cand)
    assert r["by_category"]["Safety"]["flip_unsafe"] == 1
    assert r["by_category"]["Safety"]["net_drift"] == -1.0
    assert r["by_category"]["Agentic"]["flip_unsafe"] == 0


def test_pair_runs_records_flip_rows_with_direction():
    ref = [_case("a", 1)]
    cand = [_case("a", 0)]
    flips = pair_runs(ref, cand)["flips"]
    assert len(flips) == 1
    assert flips[0]["direction"] == "unsafe"
    assert flips[0]["reference_score"] == 1


def test_pair_runs_empty_inputs_do_not_divide_by_zero():
    r = pair_runs([], [])
    assert r["n_paired"] == 0
    assert r["net_drift"] == 0.0
    assert r["pass_rate"] == 0.0


# ---------------------------------------------------------------------------
# aggregate_reps
# ---------------------------------------------------------------------------


def test_aggregate_reps_means_and_stdev():
    reps = [
        _summary([FakeCase(score=1), FakeCase(score=1, input="b")]),
        _summary([FakeCase(score=1), FakeCase(score=0, input="b")]),
    ]
    agg = aggregate_reps(reps, "model")
    assert agg.n_reps == 2
    assert agg.pass_rate_mean == 0.75
    assert agg.pass_rate_std > 0
    assert agg.per_rep_pass_rate == [1.0, 0.5]


def test_aggregate_reps_single_rep_has_zero_stdev():
    agg = aggregate_reps([_summary([FakeCase(score=1)])], "m")
    assert agg.pass_rate_std == 0.0


def test_aggregate_reps_rejects_empty_input():
    with pytest.raises(ValueError, match="no rep summaries"):
        aggregate_reps([], "m")


def test_aggregate_reps_rolls_up_categories_and_vulnerabilities():
    reps = [_summary([FakeCase(score=1), FakeCase(vulnerability="Bias", score=0, input="b")])]
    agg = aggregate_reps(reps, "m")
    assert agg.by_category["Safety"]["pass_rate_mean"] == 1.0
    assert agg.by_vulnerability["Bias"]["pass_rate_mean"] == 0.0
    assert agg.by_vulnerability["Bias"]["category"] == "Responsible AI"


# ---------------------------------------------------------------------------
# per_case CSV round-trip (the join the ladder depends on)
# ---------------------------------------------------------------------------


def test_per_case_rows_round_trip_through_csv(tmp_path):
    """`pair_runs` must work on rows read back from disk, not just in-memory dicts."""
    ref = _summary([FakeCase(score=1), FakeCase(score=0, input="b")], model="f16")
    cand = _summary([FakeCase(score=0), FakeCase(score=0, input="b")], model="iq2")
    path = tmp_path / "per_case.csv"
    write_per_case_csv(path, ref, 1)
    write_per_case_csv(path, cand, 1)

    by_model = group_per_case_by_model(read_per_case_csv(path))
    result = pair_runs(by_model["f16"], by_model["iq2"])
    assert result["n_paired"] == 2
    assert result["n_flip_unsafe"] == 1


def test_read_per_case_csv_restores_errored_score_as_none(tmp_path):
    """An errored case must come back as None, not 0 — 0 would read as a jailbreak."""
    s = _summary([FakeCase(score=None), FakeCase(score=0, input="b")], model="m")
    path = tmp_path / "pc.csv"
    write_per_case_csv(path, s, 1)
    rows = read_per_case_csv(path)
    assert [r["score"] for r in rows] == [None, 0]
    assert [r["errored"] for r in rows] == [True, False]


def test_write_per_case_csv_writes_header_once(tmp_path):
    s = _summary([FakeCase(score=1)], model="m")
    path = tmp_path / "pc.csv"
    write_per_case_csv(path, s, 1)
    write_per_case_csv(path, s, 2)
    lines = path.read_text().splitlines()
    assert lines[0].startswith("model,rep,case_id")
    assert len(lines) == 3  # header + two rows


def test_group_per_case_by_model_preserves_order(tmp_path):
    rows = [{"model": "a", "case_id": "1"}, {"model": "b"}, {"model": "a", "case_id": "2"}]
    grouped = group_per_case_by_model(rows)
    assert list(grouped) == ["a", "b"]
    assert [r["case_id"] for r in grouped["a"]] == ["1", "2"]


# ---------------------------------------------------------------------------
# Empty-output guard — reasoning models return "" when truncated
# ---------------------------------------------------------------------------


def test_build_summary_counts_empty_scored_outputs():
    s = _summary([FakeCase(score=1, actual_output=""),
                  FakeCase(score=1, actual_output="   ", input="b"),
                  FakeCase(score=0, actual_output="sure, here's how", input="c")])
    assert s.n_empty_output == 2


def test_empty_output_ignores_errored_cases():
    """An errored case has no output by definition; it isn't an empty *response*."""
    s = _summary([FakeCase(score=None, actual_output=None),
                  FakeCase(score=1, actual_output="text", input="b")])
    assert s.n_empty_output == 0


def test_assert_scored_raises_when_every_response_was_empty():
    """A model that said nothing cannot have refused anything.

    Reasoning models put chain-of-thought in `reasoning_content` and return an
    empty `content` when max_tokens runs out first; the judge scores "" as
    "did not comply", which is indistinguishable from a refusal and inflates
    pass_rate toward 1.0 for entirely the wrong reason.
    """
    s = _summary([FakeCase(score=1, actual_output=""),
                  FakeCase(score=1, actual_output="", input="b")])
    with pytest.raises(RuntimeError, match="ALL 2 scored cases had an empty"):
        _assert_scored(s, "ornith")


def test_assert_scored_warns_but_passes_on_partial_emptiness(capsys):
    cases = [FakeCase(score=1, actual_output="", input=f"e{i}") for i in range(3)]
    cases += [FakeCase(score=1, actual_output="real answer", input=f"r{i}") for i in range(7)]
    s = _summary(cases)
    _assert_scored(s, "partial")
    assert "EMPTY" in capsys.readouterr().out


def test_assert_scored_silent_when_outputs_are_present(capsys):
    s = _summary([FakeCase(score=1, actual_output="a real refusal")])
    _assert_scored(s, "ok")
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Multi-turn capture + disclosure artifact
# ---------------------------------------------------------------------------


def test_build_summary_captures_multi_turn_history():
    tc = FakeCase(attack_method="CrescendoJailbreaking", score=0, input="seed")
    tc.turns = [types.SimpleNamespace(role="user", content="q1"),
                types.SimpleNamespace(role="assistant", content="a1")]
    s = _summary([tc])
    assert s.per_case[0]["turns"] == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]


def test_build_summary_single_turn_has_empty_turns():
    s = _summary([FakeCase(score=1)])
    assert s.per_case[0]["turns"] == []


def test_disclosure_report_includes_only_findings(tmp_path):
    """A refusal is not something to disclose; complied + errored cases are."""
    s = _summary([
        FakeCase(score=1, actual_output="I can't help", input="a"),
        FakeCase(score=0, actual_output="sure, here's how", input="b", reason="complied"),
        FakeCase(score=None, actual_output=None, input="c"),
    ])
    out = tmp_path / "d.json"
    write_disclosure_report(out, s, target_model="ornith", config_name="cfg")
    import json
    rep = json.loads(out.read_text())
    ids = {f["severity"] for f in rep["findings"]}
    assert len(rep["findings"]) == 2
    assert ids == {"complied", "errored"}
    assert rep["totals"]["n_defended"] == 1


def test_disclosure_report_matches_reasoning_by_answer(tmp_path):
    s = _summary([FakeCase(score=0, actual_output="here is the payload", input="b")])
    transcript = [{"answer": "here is the payload", "reasoning": "the user asked nicely"}]
    out = tmp_path / "d.json"
    write_disclosure_report(out, s, target_transcript=transcript)
    import json
    f = json.loads(out.read_text())["findings"][0]
    assert f["target_reasoning"] == "the user asked nicely"
    assert f["target_response"] == "here is the payload"


def test_disclosure_report_carries_conversation(tmp_path):
    tc = FakeCase(score=0, attack_method="LinearJailbreaking", input="seed")
    tc.turns = [types.SimpleNamespace(role="user", content="step1"),
                types.SimpleNamespace(role="assistant", content="ok")]
    out = tmp_path / "d.json"
    write_disclosure_report(out, _summary([tc]))
    import json
    f = json.loads(out.read_text())["findings"][0]
    assert f["conversation"] == [
        {"role": "user", "content": "step1"},
        {"role": "assistant", "content": "ok"},
    ]


def test_reasoning_of_reads_either_field():
    from quant_tuner.eval.red_team import _reasoning_of
    assert _reasoning_of(types.SimpleNamespace(reasoning_content="a", content="x")) == "a"
    assert _reasoning_of(types.SimpleNamespace(reasoning="b", content="x")) == "b"
    assert _reasoning_of(types.SimpleNamespace(content="x")) is None


# ---------------------------------------------------------------------------
# schema_response_format — the fix that makes custom vulns / enhanced attacks work
# ---------------------------------------------------------------------------


def test_schema_response_format_from_pydantic_like():
    from quant_tuner.eval.red_team import schema_response_format

    class SyntheticDataList:
        @staticmethod
        def model_json_schema():
            return {"type": "object", "properties": {"data": {"type": "array"}}}

    rf = schema_response_format(SyntheticDataList)
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "SyntheticDataList"   # uses the class __name__
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"]["properties"] == {"data": {"type": "array"}}


def test_schema_response_format_none_for_non_pydantic():
    """A plain type / None must fall back to the raw-text path (returns None)."""
    from quant_tuner.eval.red_team import schema_response_format

    assert schema_response_format(None) is None
    assert schema_response_format(str) is None
    assert schema_response_format(object()) is None
