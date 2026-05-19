"""Unit tests for quant_tuner.eval.scoring — type-aware tool-call value comparison."""

from __future__ import annotations

import pytest

from quant_tuner.eval.scoring import (
    compare_value,
    is_schema_valid,
    param_score,
    parse_arguments,
    schema_for,
)


class TestParseArguments:
    def test_dict_passes_through(self):
        assert parse_arguments({"a": 1}) == {"a": 1}

    def test_json_string_decoded(self):
        assert parse_arguments('{"a": 1}') == {"a": 1}

    def test_malformed_json_returns_none(self):
        assert parse_arguments("{not json") is None

    def test_non_object_json_returns_none(self):
        # JSON array is not a dict → reject.
        assert parse_arguments("[1, 2, 3]") is None

    def test_other_types_return_none(self):
        assert parse_arguments(42) is None
        assert parse_arguments(None) is None


class TestCompareValue:
    def test_path_normalization_exact(self):
        e, s, m = compare_value("file_path", "/tmp/foo/", "/tmp/foo", None)
        assert (e, s, m) == (True, True, "path")

    def test_path_same_basename_only_similar(self):
        e, s, m = compare_value("file_path", "/a/b/x.py", "/c/d/x.py", None)
        assert e is False and s is True and m == "path"

    def test_path_completely_different_not_similar(self):
        e, s, m = compare_value("file_path", "/a/foo.py", "/b/bar.py", None)
        assert (e, s) == (False, False)

    def test_number_within_tolerance(self):
        # 10% tolerance — 100 vs 105 is within 5%.
        e, s, _ = compare_value("count", 105, 100, {"type": "number"})
        assert e is False and s is True

    def test_number_exact_match(self):
        e, s, _ = compare_value("count", 100, 100, {"type": "integer"})
        assert (e, s) == (True, True)

    def test_number_outside_tolerance(self):
        e, s, _ = compare_value("count", 200, 100, {"type": "number"})
        assert (e, s) == (False, False)

    def test_boolean_exact_only(self):
        e, s, m = compare_value("verbose", True, True, {"type": "boolean"})
        assert (e, s, m) == (True, True, "boolean")
        e, s, _ = compare_value("verbose", False, True, {"type": "boolean"})
        assert (e, s) == (False, False)

    def test_command_same_program_high_jaccard_similar(self):
        e, s, m = compare_value("command", "ls -la /tmp", "ls -l /tmp", None)
        assert m == "command"
        assert e is False and s is True  # shared argv[0]=ls, ≥30% Jaccard

    def test_command_different_program_not_similar(self):
        e, s, _ = compare_value("command", "ls /tmp", "rm /tmp", None)
        assert (e, s) == (False, False)

    def test_command_exact_after_strip(self):
        e, s, _ = compare_value("command", "  ls  ", "ls", None)
        assert (e, s) == (True, True)

    def test_structural_list_canonical_json(self):
        e, s, m = compare_value("items", [1, 2, 3], [1, 2, 3], None)
        assert (e, s, m) == (True, True, "structural")

    def test_structural_dict_key_order_irrelevant(self):
        e, s, _ = compare_value("opts", {"a": 1, "b": 2}, {"b": 2, "a": 1}, None)
        assert (e, s) == (True, True)

    def test_generic_string_jaccard_similar(self):
        e, s, m = compare_value(
            "label", "the quick brown fox", "quick brown fox jumps", None
        )
        assert m == "string-jaccard"
        # 3/5 word overlap = 0.6 ≥ 0.5
        assert e is False and s is True

    def test_generic_string_jaccard_below_threshold(self):
        e, s, _ = compare_value("label", "alpha beta", "gamma delta", None)
        assert (e, s) == (False, False)

    def test_path_keyword_in_key_name_routes_to_path_comparator(self):
        # "filepath" not in _PATH_ARG_KEYS but contains "path" substring.
        e, s, m = compare_value("output_path", "/x/y", "/x/y", None)
        assert m == "path"
        assert (e, s) == (True, True)


class TestSchemaFor:
    def test_finds_by_function_name(self):
        tools = [{"function": {"name": "read", "parameters": {"x": 1}}}]
        assert schema_for("read", tools) == {"x": 1}

    def test_returns_none_for_missing(self):
        assert schema_for("write", [{"function": {"name": "read"}}]) is None

    def test_handles_flat_tool_format(self):
        tools = [{"name": "ping", "parameters": {"y": 2}}]
        assert schema_for("ping", tools) == {"y": 2}


class TestIsSchemaValid:
    def test_tool_not_in_list_fails(self):
        ok, msg = is_schema_valid("ghost", {}, [{"function": {"name": "real"}}])
        assert ok is False and "not in tools list" in msg

    def test_unparseable_arguments_fails(self):
        tools = [{"function": {"name": "f", "parameters": {"type": "object"}}}]
        ok, msg = is_schema_valid("f", None, tools)
        assert ok is False and "JSON object" in msg

    def test_no_schema_falls_back_to_presence(self):
        ok, _ = is_schema_valid("f", {"any": "thing"}, [{"function": {"name": "f"}}])
        assert ok is True

    def test_valid_args_pass(self):
        tools = [{
            "function": {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "required": ["x"],
                    "properties": {"x": {"type": "integer"}},
                },
            }
        }]
        ok, _ = is_schema_valid("f", {"x": 5}, tools)
        assert ok is True

    def test_missing_required_fails(self):
        tools = [{
            "function": {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "required": ["x"],
                    "properties": {"x": {"type": "integer"}},
                },
            }
        }]
        ok, msg = is_schema_valid("f", {"y": 5}, tools)
        assert ok is False
        # Either jsonschema or fallback reports the missing key.
        assert "x" in msg


class TestParamScore:
    def test_all_required_present_and_equal(self):
        truth = {"path": "/a", "n": 5}
        schema = {"required": ["path", "n"], "properties": {"n": {"type": "integer"}}}
        score, details = param_score({"path": "/a", "n": 5}, truth, schema)
        assert score == 1.0
        assert "exact" in details["path"]

    def test_missing_keys_dock_score(self):
        truth = {"a": 1, "b": 2}
        schema = {"required": ["a", "b"], "properties": {}}
        score, _ = param_score({"a": 1}, truth, schema)
        assert score == 0.5

    def test_free_text_key_counts_as_presence_only(self):
        # `description` is a free-text key — present counts as hit, even if value differs wildly.
        truth = {"description": "explain X carefully"}
        schema = {"required": ["description"], "properties": {}}
        score, details = param_score(
            {"description": "totally unrelated text"}, truth, schema
        )
        assert score == 1.0
        assert "free-text" in details["description"]

    def test_similar_path_counts_as_hit(self):
        truth = {"file_path": "/a/b/x.py"}
        schema = {"required": ["file_path"], "properties": {}}
        score, details = param_score(
            {"file_path": "/c/d/x.py"}, truth, schema  # same basename only
        )
        assert score == 1.0
        assert "similar" in details["file_path"]

    def test_none_args_returns_zero(self):
        score, details = param_score(None, {"a": 1}, {"required": ["a"]})
        assert score == 0.0
        assert details == {"missing_all": True}

    def test_no_required_keys_returns_one(self):
        score, _ = param_score({}, {}, {"required": []})
        assert score == 1.0

    def test_falls_back_to_truth_keys_when_schema_missing(self):
        # No schema → use truth_args.keys() as the required set.
        truth = {"a": 1, "b": 2}
        score, _ = param_score({"a": 1, "b": 2}, truth, None)
        assert score == 1.0
        score, _ = param_score({"a": 1}, truth, None)
        assert score == 0.5


@pytest.mark.parametrize("key,pred,truth,want_similar", [
    ("FILE_PATH", "/a/b", "/A/B", True),  # case-insensitive path
    ("cmd", "git status", "git status -s", True),  # shared argv[0]
])
def test_compare_value_parametrized(key, pred, truth, want_similar):
    _, similar, _ = compare_value(key, pred, truth, None)
    assert similar is want_similar
