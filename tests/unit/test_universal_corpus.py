"""Universal-corpus primitives + the MTP pin resolution.

The full :func:`quant_tuner.data.universal.build` needs a real tokenizer and the published
datasets, so it is exercised by ``scripts/build_universal_corpus.py``; what is unit-tested
here is everything that can silently produce a *plausible but wrong* corpus.
"""

from __future__ import annotations

from quant_tuner.data import split, universal
from quant_tuner.models import mtp


class WSTokenizer:
    """Whitespace tokenizer: one token per word, so budgets are readable in tests.

    Accepts a list like a real HF fast tokenizer does — sft_record batches its inputs.
    """

    def __call__(self, text, add_special_tokens=False):
        if isinstance(text, list):
            return {"input_ids": [list(range(len(t.split()))) for t in text]}
        return {"input_ids": list(range(len(text.split())))}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"w{i}" for i in ids)


# ----------------------------------------------------------------- interleaving
def test_interleave_many_matches_interleave_for_two_lists():
    a = [f"a{i}" for i in range(7)]
    b = [f"b{i}" for i in range(3)]
    assert split.interleave_many([a, b]) == split.interleave(a, b)


def test_interleave_many_spreads_every_source_through_the_file():
    """The property AWQ/GPTQ depend on: no source is confined to one region."""
    lists = [[f"{tag}{i}" for i in range(n)]
             for tag, n in (("log", 20), ("swe", 10), ("broad", 5), ("wiki", 40))]
    out = split.interleave_many(lists)
    assert len(out) == 75
    for tag in ("log", "swe", "broad", "wiki"):
        first = next(i for i, c in enumerate(out) if c.startswith(tag))
        last = max(i for i, c in enumerate(out) if c.startswith(tag))
        # each source appears in the first third and the last third
        assert first < len(out) / 3 and last > 2 * len(out) / 3, tag


def test_interleave_many_preserves_order_and_drops_nothing():
    lists = [["x1", "x2", "x3"], [], ["y1", "y2"]]
    out = split.interleave_many(lists)
    assert [c for c in out if c.startswith("x")] == ["x1", "x2", "x3"]
    assert [c for c in out if c.startswith("y")] == ["y1", "y2"]


def test_interleave_many_empty():
    assert split.interleave_many([[], []]) == []


# ------------------------------------------------------------- tool-output clipping
def test_clip_tool_messages_keeps_head_and_tail():
    tok = WSTokenizer()
    big = " ".join(f"w{i}" for i in range(1000))
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "tool", "content": big},
        {"role": "assistant", "content": "done"},
    ]
    out, n = universal.clip_tool_messages(msgs, tok, max_tokens=20)
    assert n == 1
    assert "truncated" in out[1]["content"]
    assert len(out[1]["content"].split()) < 40
    # non-tool messages are untouched, and the originals are not mutated
    assert out[0] is msgs[0] and out[2] is msgs[2]
    assert msgs[1]["content"] == big


def test_clip_tool_messages_leaves_short_output_alone():
    tok = WSTokenizer()
    msgs = [{"role": "tool", "content": "ok done"}]
    out, n = universal.clip_tool_messages(msgs, tok, max_tokens=512)
    assert n == 0 and out[0]["content"] == "ok done"


def test_clip_tool_messages_disabled():
    tok = WSTokenizer()
    long = " ".join("w" for _ in range(100))
    out, n = universal.clip_tool_messages([{"role": "tool", "content": long}], tok, 0)
    assert n == 0 and out[0]["content"] == long


# ------------------------------------------------------------------ raw packing
def test_pack_raw_samples_respects_budget_and_window():
    tok = WSTokenizer()
    texts = [" ".join(["w"] * 30) for _ in range(100)]
    chunks, total = universal.pack_raw_samples(texts, tok, target_tokens=300,
                                               window_tokens=100, seed=42)
    assert 300 <= total <= 330            # stops at the first sample past the target
    assert all(len(c.split()) <= 120 for c in chunks)


def test_pack_raw_samples_is_deterministic():
    tok = WSTokenizer()
    texts = [f"sample {i} " + " ".join(["w"] * 10) for i in range(50)]
    a, _ = universal.pack_raw_samples(texts, tok, 200, 60, seed=7)
    b, _ = universal.pack_raw_samples(texts, tok, 200, 60, seed=7)
    assert a == b
    c, _ = universal.pack_raw_samples(texts, tok, 200, 60, seed=8)
    assert c != a


# --------------------------------------------------------------- source splitting
def test_stable_fraction_is_row_stable():
    """Adding rows must never re-split the ones already calibrated on."""
    keys = [f"instance-{i}" for i in range(200)]
    before = {k: universal._stable_fraction(k, 42) < 0.1 for k in keys}
    keys += [f"new-{i}" for i in range(50)]
    after = {k: universal._stable_fraction(k, 42) < 0.1 for k in keys}
    assert all(before[k] == after[k] for k in before)
    n_eval = sum(after[k] for k in after)
    assert 5 <= n_eval <= 45          # roughly 10% of 250, not degenerate


def test_swe_sessions_carry_tools_and_source():
    rows = [
        {"instance_id": "a__b-1", "messages": [{"role": "system", "content": "s"},
                                               {"role": "user", "content": "u"}],
         "tools": [{"type": "function", "function": {"name": "bash"}}]},
        {"instance_id": "empty", "messages": []},
    ]
    sess = universal.swe_sessions(rows)
    assert len(sess) == 1
    assert sess[0]["source"] == universal.SOURCE_SWE
    assert sess[0]["tools"][0]["function"]["name"] == "bash"
    # session_tools must find them, or the windows render with no schema context
    assert split.session_tools(sess[0], sess[0]["messages"]) == sess[0]["tools"]


def test_tool_call_marker_counts():
    text = "<tool_call>a</tool_call>\n<tool_call>b</tool_call>"
    counts = universal.tool_call_marker_counts(text, ("<tool_call>", "[TOOL_CALLS]"))
    assert counts == {"<tool_call>": 2}


# ------------------------------------------------------------------------- MTP pin
def _tensors(n_trunk: int, n_draft: int) -> list[str]:
    names = [f"blk.{i}.attn_q.weight" for i in range(n_trunk)]
    names += [f"blk.{n_trunk + j}.{t}"
              for j in range(n_draft)
              for t in ("attn_q.weight", "ffn_down.weight", "nextn.eh_proj.weight")]
    return ["token_embd.weight", *names, "output.weight"]


def test_mtp_layers_found_past_the_trunk_depth():
    names = _tensors(64, 1)
    assert mtp.mtp_layer_indices(names, block_count=64) == [64]
    assert mtp.pin_map([64]) == {"blk.64.": "q8_0"}
    assert len(mtp.mtp_tensor_names(names, 64)) == 3


def test_mtp_pin_index_follows_the_model_not_a_constant():
    """A 48-layer model's head is blk.48 — the hardcoded blk.64 would match nothing."""
    names = _tensors(48, 1)
    assert mtp.pin_map(mtp.mtp_layer_indices(names, 48)) == {"blk.48.": "q8_0"}


def test_block_count_that_includes_the_draft_layer_still_resolves():
    """Qwopus3.6's real GGUF: trunk 64, head at blk.64, but block_count=65.

    The index test alone matches nothing here — the `nextn` name hint is what finds it.
    """
    names = _tensors(64, 1)
    assert mtp.mtp_layer_indices(names, block_count=65) == [64]
    assert mtp.pin_map(mtp.mtp_layer_indices(names, 65)) == {"blk.64.": "q8_0"}


def test_no_mtp_head_yields_an_empty_pin():
    names = _tensors(32, 0)
    assert mtp.mtp_layer_indices(names, 32) == []
    assert mtp.pin_map([]) == {}


def test_mtp_found_by_name_hint_without_block_count():
    names = ["blk.0.attn_q.weight", "blk.9.nextn.eh_proj.weight", "mtp.norm.weight"]
    assert mtp.mtp_layer_indices(names, None) == [9]
    assert "mtp.norm.weight" in mtp.mtp_tensor_names(names, None)


def test_pin_map_uses_a_trailing_dot_so_blk_640_is_not_swallowed():
    assert mtp.pin_map([64]) == {"blk.64.": "q8_0"}
    names = ["blk.640.attn_q.weight", "blk.64.attn_q.weight"]
    assert [n for n in names if "blk.64." in n] == ["blk.64.attn_q.weight"]


def test_config_declares_mtp_reads_every_known_key_and_nests():
    assert mtp.config_declares_mtp({"mtp_num_hidden_layers": 1}) == 1
    assert mtp.config_declares_mtp({"num_nextn_predict_layers": 2}) == 2
    assert mtp.config_declares_mtp({"text_config": {"mtp_num_hidden_layers": 1}}) == 1
    assert mtp.config_declares_mtp({"mtp_num_hidden_layers": 0}) == 0
    assert mtp.config_declares_mtp({}) == 0


# ------------------------------------------------------------------------ SFT export
def test_sft_record_keeps_tool_calls_and_reasoning_as_separate_fields():
    """The training view, unlike the calibration view, must lose nothing."""
    session = {
        "id": "sess-1",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "fix it"},
            {"role": "assistant", "content": "<think>plan it out</think>on it",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "name": "bash", "content": "a.py b.py"},
            {"role": "assistant", "content": "done"},
        ],
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "meta": {"language": "python"},
    }
    rec = universal.sft_record(session, split_name="train", source="logs")

    assert rec["id"] == "sess-1" and rec["split"] == "train" and rec["source"] == "logs"
    assert rec["n_messages"] == 5
    assert rec["n_tool_calls"] == 1 and rec["n_tool_results"] == 1 and rec["n_reasoning"] == 1
    asst = rec["messages"][2]
    # reasoning is a FIELD, not inlined — a trainer can mask it independently
    assert asst["reasoning_content"] == "plan it out"
    assert asst["content"] == "on it"
    # tool-call arguments are coerced to a dict, not left as a JSON string
    assert asst["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}
    # the tool result keeps its linkage back to the call
    assert rec["messages"][3]["tool_call_id"] == "c1" and rec["messages"][3]["name"] == "bash"
    assert rec["tools"] and rec["meta"] == {"language": "python"}
    assert "n_tokens" not in rec           # only when a tokenizer is passed


def test_sft_record_does_not_truncate_anything():
    """A 40k-char tool dump is clipped for calibration and must NOT be for training."""
    dump = "x" * 40_000
    rec = universal.sft_record(
        {"id": "s", "messages": [{"role": "user", "content": "go"},
                                 {"role": "tool", "content": dump},
                                 {"role": "assistant", "content": "ok"}]},
        split_name="train", source="logs")
    assert rec["messages"][1]["content"] == dump
    assert rec["n_chars"] >= 40_000


def test_sft_record_counts_tokens_only_when_asked():
    rec = universal.sft_record(
        {"id": "s", "messages": [{"role": "user", "content": "a b c"}]},
        split_name="train", source="logs", tok=WSTokenizer())
    assert rec["n_tokens"] == 3


def test_write_jsonl_gz_roundtrips(tmp_path):
    import gzip
    import json

    path = tmp_path / "out.jsonl.gz"
    n, nbytes = universal.write_jsonl_gz(path, ({"i": i, "t": "café"} for i in range(3)))
    assert n == 3 and nbytes > 0
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        rows = [json.loads(ln) for ln in fh]
    assert rows == [{"i": i, "t": "café"} for i in range(3)]


# ------------------------------------------------------ strict templates / user anchor
class StrictTokenizer(WSTokenizer):
    """Qwen3.6's official template: refuses any render without a user turn."""

    chat_template = "strict"

    def apply_chat_template(self, messages, tools=None, tokenize=False):
        if not any(m.get("role") == "user" for m in messages):
            raise ValueError("No user query found in messages.")
        return " ".join(f"<{m['role']}>{m.get('content') or ''}" for m in messages)


def _agentic_session(n_turns: int = 12) -> list[dict]:
    """One user task followed by a long assistant/tool run — the agentic shape."""
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "fix the bug"}]
    for i in range(n_turns):
        msgs.append({"role": "assistant", "content": f"step {i}"})
        msgs.append({"role": "tool", "content": f"output {i}"})
    return msgs


def test_strict_template_drops_mid_trajectory_windows_without_an_anchor():
    """The measured failure: reasoning fell 1.00M -> 232k tokens on the official template."""
    tok = StrictTokenizer()
    wins = split.session_windows(tok, _agentic_session(), None, cap_tokens=12,
                                 max_windows=8, user_anchor=False)
    # everything after the first window is refused, because no later span has a user turn
    assert len(wins) == 1


def test_user_anchor_rescues_them_and_keeps_the_task_in_context():
    tok = StrictTokenizer()
    wins = split.session_windows(tok, _agentic_session(), None, cap_tokens=12,
                                 max_windows=8, user_anchor=True)
    assert len(wins) > 1
    # every window carries the original task statement, which is what inference gives it
    assert all("fix the bug" in text for text, _ in wins)
    # and the later windows really are later spans, not a repeat of the head
    assert any("step 6" in text or "step 7" in text for text, _ in wins)


def test_user_anchor_is_a_no_op_for_lenient_templates():
    """The published two-source builder must be byte-identical with the flag off or on."""
    class Lenient(WSTokenizer):
        chat_template = "lenient"

        def apply_chat_template(self, messages, tools=None, tokenize=False):
            return " ".join(f"<{m['role']}>{m.get('content') or ''}" for m in messages)

    tok = Lenient()
    session = _agentic_session()
    off = split.session_windows(tok, session, None, cap_tokens=12, max_windows=8)
    on = split.session_windows(tok, session, None, cap_tokens=12, max_windows=8,
                               user_anchor=True)
    assert off == on
