"""External SFT adapters — the conversions whose failures are silent.

Each test here pins a failure that produces a corpus that builds, trains and exports
without error while teaching the model something wrong.
"""

from __future__ import annotations

import collections

from quant_tuner.data import external_sft as ext


def test_normalize_tool_parses_parameters_json_string():
    """The distillation rows nest the schema as a STRING under ``parameters_json``.

    Left alone, the chat template's ``tool | tojson`` renders it as one escaped blob and
    the model learns a doubly-encoded schema no real caller emits.
    """
    raw = {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute Python.",
            "parameters_json": '{"properties":{"code":{"type":"string"}},'
                               '"required":["code"],"type":"object"}',
        },
    }
    out = ext.normalize_tool(raw)
    params = out["function"]["parameters"]
    assert isinstance(params, dict), "parameters must be an object, not a string"
    assert params["properties"]["code"]["type"] == "string"
    assert "parameters_json" not in out["function"]


def test_normalize_tool_passes_through_already_correct_schema():
    good = {"type": "function", "function": {
        "name": "Bash", "description": "run", "parameters": {"type": "object"}}}
    assert ext.normalize_tool(good)["function"]["parameters"] == {"type": "object"}


def test_normalize_tool_unparseable_schema_does_not_become_a_string():
    """A schema we cannot parse must degrade to an empty object, never to the raw string —
    a string here is exactly the corruption this function exists to prevent."""
    bad = {"function": {"name": "x", "parameters_json": "{not json"}}
    params = ext.normalize_tool(bad)["function"]["parameters"]
    assert isinstance(params, dict)
    assert params == {"type": "object", "properties": {}}


def test_normalize_tool_supplies_missing_parameters():
    out = ext.normalize_tool({"function": {"name": "noargs"}})
    assert out["function"]["parameters"] == {"type": "object", "properties": {}}
    assert out["function"]["description"]


def test_assign_split_is_deterministic_and_roughly_proportional():
    """Hashed, not positional: re-sharding upstream must not move a row across the
    train/holdout boundary, or the eval holdout stops meaning anything."""
    a = [ext.assign_split(f"id{i}", salt="s") for i in range(4000)]
    b = [ext.assign_split(f"id{i}", salt="s") for i in range(4000)]
    assert a == b
    c = collections.Counter(a)
    assert 0.75 < c["train"] / 4000 < 0.85
    assert set(c) <= {"train", "test", "holdout"}


def test_assign_split_salt_separates_sources():
    same = [ext.assign_split(f"id{i}") == ext.assign_split(f"id{i}", salt="other")
            for i in range(200)]
    assert not all(same), "salt must change the assignment"


def test_upstream_split_map_keeps_graded_rows_out_of_training():
    """Their validation becomes our test (the trainer's val corpus) and their test becomes
    our holdout. Nothing upstream maps onto train except train."""
    assert ext.UPSTREAM_SPLIT_MAP["train"] == "train"
    assert ext.UPSTREAM_SPLIT_MAP["validation"] == "test"
    assert ext.UPSTREAM_SPLIT_MAP["test"] == "holdout"
    assert [k for k, v in ext.UPSTREAM_SPLIT_MAP.items() if v == "train"] == ["train"]


def test_sft_agent_is_recorded_as_an_alias_of_sft_tools():
    """They are byte-identical parquet. Taking both silently double-weights the agentic
    half of the curriculum."""
    assert ext.DISTILL_ALIASES["sft_agent"] == "sft_tools"


def test_convert_distill_rows_uses_the_file_split_not_the_row_field():
    """Every row inside the upstream train file is tagged ``split="train"``, including the
    ones in the validation and test files — so trusting the field puts graded rows in
    training."""
    rows = [{"id": "r1", "split": "train",
             "messages": [{"role": "user", "content": "hi"},
                          {"role": "assistant", "content": "yo"}],
             "tools": [], "source": "x", "domain": "d"}]
    out = list(ext.convert_distill_rows(rows, source="s", split="validation"))
    assert out[0]["split"] == "test"


def test_convert_distill_rows_normalizes_tools_and_counts():
    rows = [{
        "id": "r2", "split": "train", "source": "u", "domain": "agent_tool",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": "<think>plan</think>",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "content": "ok", "tool_call_id": "c1"},
            {"role": "assistant", "content": "done"},
        ],
        "tools": [{"type": "function",
                   "function": {"name": "f", "parameters_json": '{"type":"object"}'}}],
    }]
    r = list(ext.convert_distill_rows(rows, source="distill-tools"))[0]
    assert isinstance(r["tools"][0]["function"]["parameters"], dict)
    assert r["n_tool_calls"] == 1
    assert r["n_tool_results"] == 1
    assert r["n_messages"] == 5
    assert r["source"] == "distill-tools"


def test_convert_distill_rows_drops_benchmark_sources_when_asked():
    """sft_science is drawn from ARC/SciQ/OpenBookQA test material; training on it
    contaminates any multiple-choice eval."""
    rows = [{"id": f"r{i}", "split": "train", "source": s, "domain": "reasoning",
             "messages": [{"role": "user", "content": "q"},
                          {"role": "assistant", "content": "a"}], "tools": []}
            for i, s in enumerate(["SciQ", "ARC-Easy", "k3_science_logic_data"])]
    kept = list(ext.convert_distill_rows(rows, source="s", drop_benchmarks=True))
    assert [r["meta"]["upstream_source"] for r in kept] == ["k3_science_logic_data"]
    assert len(list(ext.convert_distill_rows(rows, source="s"))) == 3


def test_convert_ultrachat_rows_shape():
    rows = [{"prompt_id": "p1", "messages": [
        {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}]}]
    r = list(ext.convert_ultrachat_rows(rows))[0]
    assert r["source"] == "ultrachat"
    assert r["tools"] is None
    assert r["n_messages"] == 4
    assert r["n_tool_calls"] == 0
    assert r["split"] in {"train", "test", "holdout"}


def test_short_conversations_are_dropped():
    """A single message cannot carry a supervised target."""
    assert list(ext.convert_ultrachat_rows([{"prompt_id": "x", "messages": [
        {"role": "user", "content": "only"}]}])) == []


def test_dedup_rows_keeps_first_occurrence():
    """canonical ships 98,455 rows for 56,773 distinct ids — packing the duplicates trains
    the same conversation ~1.7x."""
    rows = [{"id": "a", "v": 1}, {"id": "b", "v": 2}, {"id": "a", "v": 3}]
    out = ext.dedup_rows(rows)
    assert [r["id"] for r in out] == ["a", "b"]
    assert out[0]["v"] == 1


def test_convert_reads_the_messages_json_schema():
    """The per-config files carry `messages`/`tools` as lists; canonical carries
    `messages_json`/`tools_json` as STRINGS. Reading only one silently drops every row of
    the other — a 56,773-row build reported zero without erroring."""
    import json as _j
    rows = [{"id": "c1", "source": "u", "domain": "agent_tool",
             "messages_json": _j.dumps([{"role": "user", "content": "q"},
                                        {"role": "assistant", "content": "a"}]),
             "tools_json": _j.dumps([{"type": "function", "function": {
                 "name": "f", "parameters_json": '{"type":"object"}'}}])}]
    out = list(ext.convert_distill_rows(rows, source="distill"))
    assert len(out) == 1
    assert out[0]["n_messages"] == 2
    assert isinstance(out[0]["tools"][0]["function"]["parameters"], dict)


def test_benchmark_sources_include_the_code_benchmarks():
    """The `code` domain carries HumanEval and MBPP, not just synthetic Evol-Code."""
    assert {"HumanEval", "MBPP"} <= ext.BENCHMARK_SOURCES
    assert {"SciQ", "ARC-Easy"} <= ext.BENCHMARK_SOURCES


def test_select_by_domain_budgets_in_tokens_not_rows():
    """Tokens-per-conversation varies ~20x across these domains (348 vs 8,034), so a row
    quota would make the mix mean something different from what it says."""
    recs = [{"id": f"a{i}", "meta": {"domain": "big"}} for i in range(10)]
    recs += [{"id": f"b{i}", "meta": {"domain": "small"}} for i in range(10)]
    sizes = {"big": 1000, "small": 100}
    kept, audit = ext.select_by_domain(
        recs, {"big": 3000, "small": None},
        token_of=lambda r: sizes[r["meta"]["domain"]])
    assert audit["big"]["tokens"] <= 3000
    assert audit["big"]["taken"] == 3
    assert audit["small"]["taken"] == 10          # None = take whole
    assert len(kept) == 13


def test_select_by_domain_is_deterministic():
    recs = [{"id": f"x{i}", "meta": {"domain": "d"}} for i in range(50)]
    f = lambda r: 10  # noqa: E731
    a, _ = ext.select_by_domain(recs, {"d": 200}, token_of=f)
    b, _ = ext.select_by_domain(recs, {"d": 200}, token_of=f)
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_round2_mix_is_agentic_first_and_benchmark_free():
    b = ext.ROUND2_DOMAIN_BUDGETS
    assert set(b) >= ext.AGENTIC_DOMAINS
    assert all(b[d] is None for d in ext.AGENTIC_DOMAINS), "agentic domains taken whole"
    # the 100%-benchmark domain must not be in the mix
    assert "reasoning" not in b
