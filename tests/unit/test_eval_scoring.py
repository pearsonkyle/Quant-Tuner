"""Unit tests for quant_tuner.eval.scoring — type-aware tool-call value comparison."""

from __future__ import annotations

import pytest

from quant_tuner.eval.scoring import (
    canonical_tool,
    compare_value,
    is_schema_valid,
    param_score,
    param_score_strict,
    parse_arguments,
    schema_for,
    score_turn,
)
from quant_tuner.eval.toolcall import tool_result_is_error


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
        # 4/5 word overlap = 0.8, above the 0.7 threshold.
        e, s, m = compare_value(
            "label",
            "the quick brown fox",
            "the quick brown dog fox",
            None,
        )
        assert m == "string-jaccard"
        assert e is False and s is True

    def test_generic_string_jaccard_below_threshold(self):
        # 3/5 = 0.6 — was "similar" under the old 0.5 threshold, no longer is.
        e, s, _ = compare_value(
            "label", "the quick brown fox", "quick brown fox jumps", None
        )
        assert (e, s) == (False, False)

    def test_generic_string_empty_vs_empty_not_exact(self):
        # Empty-vs-empty no longer counted as exact — guards against
        # placeholder new_string/old_string inflation in Edit calls.
        e, s, _ = compare_value("new_string", "", "", None)
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
        assert "x" in msg

    def test_wrong_value_type_fails(self):
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
        ok, msg = is_schema_valid("f", {"x": "five"}, tools)
        assert ok is False
        assert "integer" in msg

    def test_enum_violation_fails(self):
        tools = [{
            "function": {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "required": ["mode"],
                    "properties": {"mode": {"type": "string", "enum": ["a", "b"]}},
                },
            }
        }]
        ok, msg = is_schema_valid("f", {"mode": "c"}, tools)
        assert ok is False
        assert "enum" in msg

    def test_numeric_range_violation_fails(self):
        tools = [{
            "function": {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "required": ["n"],
                    "properties": {"n": {"type": "integer", "minimum": 1, "maximum": 100}},
                },
            }
        }]
        ok, msg = is_schema_valid("f", {"n": 999}, tools)
        assert ok is False
        assert "maximum" in msg

    def test_boolean_rejected_when_integer_expected(self):
        tools = [{
            "function": {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "required": ["n"],
                    "properties": {"n": {"type": "integer"}},
                },
            }
        }]
        ok, _ = is_schema_valid("f", {"n": True}, tools)
        assert ok is False


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

    def test_free_text_key_no_longer_presence_only(self):
        # `description` is now scored via generic string comparison — unrelated
        # text fails. Previously presence-only inflated this to 1.0.
        truth = {"description": "explain X carefully"}
        schema = {"required": ["description"], "properties": {}}
        score, details = param_score(
            {"description": "totally unrelated text"}, truth, schema
        )
        assert score == 0.0
        assert "mismatch" in details["description"]["result"]

    def test_free_text_command_mismatch_scores_zero(self):
        # `command` routes to the shell comparator: different argv[0] → no credit.
        truth = {"command": "git push origin main"}
        schema = {"required": ["command"], "properties": {}}
        score, _ = param_score({"command": "ls"}, truth, schema)
        assert score == 0.0

    def test_pred_unparseable_sentinel(self):
        score, details = param_score(None, {"a": 1}, {"required": ["a"]})
        assert score == 0.0
        assert details == {"pred_unparseable": True}

    def test_similar_path_counts_as_hit(self):
        truth = {"file_path": "/a/b/x.py"}
        schema = {"required": ["file_path"], "properties": {}}
        score, details = param_score(
            {"file_path": "/c/d/x.py"}, truth, schema  # same basename only
        )
        assert score == 1.0
        assert "similar" in details["file_path"]

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

    def test_extra_pred_keys_ignored(self):
        # Extras in pred_args don't reduce the score.
        truth = {"a": 1}
        score, _ = param_score({"a": 1, "extra": "junk", "another": 99}, truth, None)
        assert score == 1.0

    def test_truth_keys_used_even_when_schema_required_is_narrower(self):
        # Schema says only "path" is required, but truth carries more — score
        # against all truth keys so partial edits don't get full credit.
        truth = {"path": "/a", "new_value": "X"}
        schema = {"required": ["path"], "properties": {}}
        score, _ = param_score({"path": "/a"}, truth, schema)
        assert score == 0.5


class TestScoreTurn:
    @staticmethod
    def _tools(*names):
        return [{"function": {"name": n, "parameters": {"type": "object"}}} for n in names]

    def test_single_call_exact_match(self):
        tools = self._tools("read")
        out = score_turn(
            pred_calls=[("read", {"path": "/a"})],
            truth_calls=[("read", {"path": "/a"})],
            tools=tools,
        )
        assert out["selection"] is True
        assert out["param_acc"] == 1.0
        assert out["schema_valid"] is True

    def test_parallel_set_match_order_invariant(self):
        tools = self._tools("read", "edit")
        # Pred emits in reverse order — set match still passes.
        out = score_turn(
            pred_calls=[("edit", {"path": "/b"}), ("read", {"path": "/a"})],
            truth_calls=[("read", {"path": "/a"}), ("edit", {"path": "/b"})],
            tools=tools,
        )
        assert out["selection"] is True
        assert out["param_acc"] == 1.0

    def test_parallel_missing_one_tool_fails_selection(self):
        tools = self._tools("read", "edit")
        out = score_turn(
            pred_calls=[("read", {"path": "/a"})],
            truth_calls=[("read", {"path": "/a"}), ("edit", {"path": "/b"})],
            tools=tools,
        )
        assert out["selection"] is False
        assert out["param_acc"] == 0.0

    def test_parallel_extra_tool_fails_selection(self):
        tools = self._tools("read", "edit", "shell")
        out = score_turn(
            pred_calls=[
                ("read", {"path": "/a"}),
                ("edit", {"path": "/b"}),
                ("shell", {"command": "ls"}),
            ],
            truth_calls=[("read", {"path": "/a"}), ("edit", {"path": "/b"})],
            tools=tools,
        )
        assert out["selection"] is False

    def test_parallel_param_acc_is_per_call_mean(self):
        # Two parallel reads, one correct path, one mismatched.
        tools = self._tools("read")
        out = score_turn(
            pred_calls=[
                ("read", {"path": "/a"}),
                ("read", {"path": "/wrong"}),
            ],
            truth_calls=[
                ("read", {"path": "/a"}),
                ("read", {"path": "/b"}),
            ],
            tools=tools,
        )
        assert out["selection"] is True
        # First truth /a matches pred[0] /a → 1.0; second truth /b matches
        # pred[1] /wrong (different basename → no credit) → 0.0. Mean = 0.5.
        assert out["param_acc"] == 0.5

    def test_no_pred_calls_emitted(self):
        tools = self._tools("read")
        out = score_turn(
            pred_calls=[],
            truth_calls=[("read", {"path": "/a"})],
            tools=tools,
        )
        assert out["selection"] is False
        assert out["param_acc"] == 0.0
        assert out["schema_valid"] is False

    def test_schema_valid_requires_all_pred_calls_pass(self):
        # One emitted call references a tool not in the list — schema fails.
        tools = self._tools("read", "edit")
        out = score_turn(
            pred_calls=[("read", {"path": "/a"}), ("ghost", {})],
            truth_calls=[("read", {"path": "/a"}), ("edit", {"path": "/b"})],
            tools=tools,
        )
        # Selection fails (ghost ≠ edit), but schema_valid specifically should
        # also be False since "ghost" isn't a declared tool.
        assert out["schema_valid"] is False


@pytest.mark.parametrize("key,pred,truth,want_similar", [
    ("FILE_PATH", "/a/b", "/A/B", True),  # case-insensitive path
    ("cmd", "git status", "git status -s", True),  # shared argv[0]
])
def test_compare_value_parametrized(key, pred, truth, want_similar):
    _, similar, _ = compare_value(key, pred, truth, None)
    assert similar is want_similar


class TestCanonicalTool:
    @pytest.mark.parametrize("name,canonical", [
        # read-a-file family across the three agents
        ("Read", "read"), ("read_file", "read"), ("read", "read"),
        # shell family
        ("Bash", "shell"), ("run_shell_command", "shell"), ("bash", "shell"),
        # edit / write
        ("Edit", "edit"), ("write_file", "write"), ("Write", "write"),
        # search
        ("Grep", "grep"), ("grep_search", "grep"), ("Glob", "glob"),
        # misc capabilities
        ("WebFetch", "web_fetch"), ("list_directory", "list"),
        ("TaskCreate", "todo"), ("AskUserQuestion", "ask"),
        ("ExitPlanMode", "plan"),
    ])
    def test_known_families_collapse(self, name, canonical):
        assert canonical_tool(name) == canonical

    def test_cross_agent_read_names_share_one_label(self):
        # The whole point: claude/qwen/opencode spellings unify.
        assert canonical_tool("Read") == canonical_tool("read_file") == canonical_tool("read")

    def test_unknown_name_passes_through_lowercased(self):
        assert canonical_tool("SomeCustomTool") == "somecustomtool"

    def test_empty_name(self):
        assert canonical_tool("") == ""


class TestParamScoreStrict:
    def test_exact_match_full_credit(self):
        truth = {"path": "/a", "n": 5}
        schema = {"required": ["path", "n"], "properties": {"n": {"type": "integer"}}}
        assert param_score_strict({"path": "/a", "n": 5}, truth, schema) == 1.0

    def test_similar_but_not_exact_scores_zero(self):
        # Same basename, different dir → lenient param_score credits it, strict does not.
        truth = {"file_path": "/a/b/x.py"}
        schema = {"required": ["file_path"], "properties": {}}
        pred = {"file_path": "/c/d/x.py"}
        assert param_score(pred, truth, schema)[0] == 1.0   # lenient: similar = hit
        assert param_score_strict(pred, truth, schema) == 0.0  # strict: exact-only

    def test_extra_keys_penalized(self):
        # 1 truth key matched exactly, but pred has 2 extra keys → denom = 3.
        truth = {"a": 1}
        pred = {"a": 1, "extra": "junk", "more": 2}
        assert param_score_strict(pred, truth, None) == pytest.approx(1 / 3)

    def test_missing_key_penalized(self):
        truth = {"a": 1, "b": 2}
        pred = {"a": 1}
        # union = {a, b}, hits = 1 → 0.5
        assert param_score_strict(pred, truth, None) == 0.5

    def test_pred_unparseable_scores_zero(self):
        assert param_score_strict(None, {"a": 1}, None) == 0.0

    def test_both_empty_full_credit(self):
        assert param_score_strict({}, {}, None) == 1.0


class TestScoreTurnStrict:
    @staticmethod
    def _tools(*names):
        return [{"function": {"name": n, "parameters": {"type": "object"}}} for n in names]

    def test_strict_present_and_lower_on_similar_path(self):
        tools = self._tools("read")
        out = score_turn(
            pred_calls=[("read", {"file_path": "/c/d/x.py"})],
            truth_calls=[("read", {"file_path": "/a/b/x.py"})],
            tools=tools,
        )
        assert out["selection"] is True
        assert out["param_acc"] == 1.0           # lenient: same basename = hit
        assert out["param_acc_strict"] == 0.0    # strict: not exact
        assert out["param_acc_strict"] <= out["param_acc"]

    def test_strict_equals_lenient_on_exact(self):
        tools = self._tools("read")
        out = score_turn(
            pred_calls=[("read", {"path": "/a"})],
            truth_calls=[("read", {"path": "/a"})],
            tools=tools,
        )
        assert out["param_acc"] == out["param_acc_strict"] == 1.0

    def test_strict_zero_when_selection_fails(self):
        tools = self._tools("read", "edit")
        out = score_turn(
            pred_calls=[("read", {"path": "/a"})],
            truth_calls=[("read", {"path": "/a"}), ("edit", {"path": "/b"})],
            tools=tools,
        )
        assert out["selection"] is False
        assert out["param_acc_strict"] == 0.0


class TestToolResultIsError:
    def test_is_error_flag_true_wins(self):
        assert tool_result_is_error({"is_error": True, "content": "all good"}) is True

    def test_is_error_flag_false_wins_over_content(self):
        # Flag is authoritative: even error-looking content is not an error
        # when the source explicitly recorded is_error=False.
        assert tool_result_is_error(
            {"is_error": False, "content": "Error: boom\nTraceback"}
        ) is False

    def test_file_content_with_error_word_does_not_trigger(self):
        # The whole reason for tightening: a file dump that mentions "error"
        # somewhere mid-line must NOT register as a tool error.
        content = (
            "import logging\n"
            "class CustomError(Exception):\n"
            "    pass\n"
            "# this module raises errors when misused\n"
        )
        assert tool_result_is_error({"content": content}) is False

    @pytest.mark.parametrize("content", [
        "Error: file not found",
        "  error: something broke",
        "Traceback (most recent call last):",
        "process exited with exit code 1",
        "Exit code: 2",
        "bash: foo: command not found",
        "cat: /x: No such file or directory",
        "Permission denied",
    ])
    def test_anchored_signals_trigger(self, content):
        assert tool_result_is_error({"content": content}) is True

    def test_empty_content_not_error(self):
        assert tool_result_is_error({"content": ""}) is False
        assert tool_result_is_error({}) is False
