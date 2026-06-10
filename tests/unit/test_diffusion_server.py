"""Unit tests for the DiffusionGemma eval backend's pure pieces (no model,
no subprocess): Gemma-4 tool-call markup parsing, canvas trimming, and the
entropy-bound acceptance rule."""

from __future__ import annotations

import json

import numpy as np

from quant_tuner.eval.diffusion_server import (
    EbParams,
    _parse_gemma4_args,
    parse_gemma4_response,
    trim_canvas,
)


def test_parse_gemma4_single_tool_call():
    text = '<|tool_call>call:read_file{"path": "/etc/hosts"}<tool_call|>'
    content, calls = parse_gemma4_response(text)
    assert content == ""
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "read_file"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "/etc/hosts"}


def test_parse_gemma4_bare_keys_and_parallel_calls():
    text = (
        "Let me check.\n"
        "<|tool_call>call:search{query: \"llama\", limit: 3}<tool_call|>"
        "<|tool_call>call:fetch{url: \"http://x\"}<tool_call|>"
    )
    content, calls = parse_gemma4_response(text)
    assert content == "Let me check."
    assert [c["function"]["name"] for c in calls] == ["search", "fetch"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": "llama", "limit": 3}


def test_parse_gemma4_strips_reasoning_channel():
    text = (
        "<|channel>thinking hard about it<channel|>The answer is B."
        "<end_of_turn>"
    )
    content, calls = parse_gemma4_response(text)
    assert content == "The answer is B."
    assert calls == []


def test_parse_gemma4_unparseable_args_kept_raw():
    text = "<|tool_call>call:run{not json at all<tool_call|>"
    content, calls = parse_gemma4_response(text)
    # no closing brace before the end marker: regex requires {...}, so this is
    # treated as content (the eval scores it as a malformed/absent call)
    assert calls == [] or "_raw" in json.loads(calls[0]["function"]["arguments"])


def test_trim_canvas_cuts_at_eog():
    eog = frozenset({7})
    assert trim_canvas([1, 2, 3, 7, 4, 5], eog) == 3
    assert trim_canvas([1, 2, 3], eog) == 3  # no EOG -> full length


def test_trim_canvas_cuts_repetition_loop():
    # >= 6 consecutive stride-1 repeats
    toks = [1, 2] + [9] * 9 + [3]
    cut = trim_canvas(toks, frozenset())
    assert cut == 2


def test_entropy_bound_acceptance_rule():
    """Mirror of the C++ rule: accept positions in ascending-entropy order
    while the sum of strictly-earlier entropies stays within the bound."""
    entropy = np.array([0.5, 0.01, 0.02, 9.0])
    bound = 0.04
    order = np.argsort(entropy, kind="stable")
    cum_e = np.concatenate([[0.0], entropy[order].cumsum()[:-1]])
    accepted = np.zeros(4, dtype=bool)
    accepted[order[cum_e <= bound]] = True
    # 0.01 (cum 0), 0.02 (cum 0.01), 0.5 (cum 0.03 <= 0.04!), 9.0 (cum 0.53 > bound)
    assert accepted.tolist() == [True, True, True, False]


def test_eb_params_defaults_match_llama_cpp_reference():
    eb = EbParams()
    assert (eb.max_steps, eb.t_min, eb.t_max) == (48, 0.4, 0.8)
    assert (eb.entropy_bound, eb.stability_threshold, eb.confidence_threshold) == (
        0.1, 1, 0.005,
    )


def test_parse_args_fallback_chain():
    assert _parse_gemma4_args('{"a": 1}') == {"a": 1}
    assert _parse_gemma4_args('{a: 1, b-c: "x"}') == {"a": 1, "b-c": "x"}
    assert "_raw" in _parse_gemma4_args("{a: }")
