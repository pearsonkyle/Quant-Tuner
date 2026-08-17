"""Unit tests for the universal-SFT QAT corpus builder (no model/tokenizer files needed).

The masking core is exercised elsewhere; what's tested here is the part that decides
WHICH rows reach the trainer — split filtering, source selection, per-source token
budgets (uncapped by default), and the ``budget=0`` drop.
"""

from __future__ import annotations

import gzip
import json

import pytest

from quant_tuner.qat import corpus as qc


class FakeTok:
    """Character-per-token stand-in with the surface the builder uses.

    Renders ``<|im_start|>{role}\\n{content}<|im_end|>\\n`` per message so the real
    ``_ASST_RE`` span logic runs, and returns byte offsets so the label mapping is the
    production code path.
    """

    def apply_chat_template(self, msgs, tools=None, tokenize=False, add_generation_prompt=False):
        parts = []
        for m in msgs:
            body = m.get("content") or ""
            for tc in m.get("tool_calls") or []:
                body += json.dumps(tc.get("function", tc))
            parts.append(f"<|im_start|>{m['role']}\n{body}<|im_end|>\n")
        return "".join(parts)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids = [ord(c) % 256 for c in text]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def convert_tokens_to_ids(self, tokenizer):
        return None  # no <|im_end|> id -> _finalize skips the stop-token assert


def _row(idx, source, split, n_chars=4000):
    return {
        "id": f"{source}-{idx}", "source": source, "split": split,
        "messages": [
            {"role": "user", "content": "u" * n_chars},
            {"role": "assistant", "content": "a" * n_chars},
        ],
        "tools": [],
    }


@pytest.fixture
def sft_file(tmp_path):
    path = tmp_path / "sft.jsonl.gz"
    rows = (
        [_row(i, "logs", "train") for i in range(20)]
        + [_row(i, "swe-trajectories", "train") for i in range(5)]
        + [_row(i, "logs", "holdout") for i in range(10)]
    )
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def test_reads_gzip_and_plain(tmp_path, sft_file):
    assert len(qc.read_sft_jsonl(sft_file)) == 35
    plain = tmp_path / "sft.jsonl"
    plain.write_text(json.dumps(_row(0, "logs", "train")) + "\n")
    assert len(qc.read_sft_jsonl(plain)) == 1


def test_default_budgets_are_uncapped(sft_file):
    """QAT takes every sample; the calibration corpus is the budgeted one."""
    assert qc.SFT_DEFAULT_BUDGETS == {}
    blob = qc.build_sft_corpus(sft_path=sft_file, window=512, max_tool_tokens=0,
                               tok=FakeTok(), out=None)
    per = blob["per_source"]
    assert per["logs"]["conversations_used"] == 20
    assert per["swe-trajectories"]["conversations_used"] == 5


def test_split_filter_holds_out_the_eval_slice(sft_file):
    blob = qc.build_sft_corpus(sft_path=sft_file, data_split="train", window=512,
                               max_tool_tokens=0, tok=FakeTok(), out=None)
    # the 10 holdout rows must not appear in any source's used count
    assert sum(v["conversations_used"] for v in blob["per_source"].values()) == 25
    assert blob["split"] == "sft:train"

    held = qc.build_sft_corpus(sft_path=sft_file, data_split="holdout", window=512,
                               max_tool_tokens=0, tok=FakeTok(), out=None)
    assert held["per_source"]["logs"]["conversations_used"] == 10


def test_per_source_budget_stops_consuming(sft_file):
    blob = qc.build_sft_corpus(sft_path=sft_file, budgets={"logs": 20_000}, window=512,
                               max_tool_tokens=0, tok=FakeTok(), out=None)
    logs = blob["per_source"]["logs"]
    assert logs["conversations_used"] < logs["conversations_available"]
    assert logs["tokens"] >= 20_000  # stops on the first conversation that crosses
    assert blob["per_source"]["swe-trajectories"]["conversations_used"] == 5


def test_budget_zero_drops_the_source(sft_file):
    blob = qc.build_sft_corpus(sft_path=sft_file, budgets={"logs": 0}, window=512,
                               max_tool_tokens=0, tok=FakeTok(), out=None)
    assert set(blob["per_source"]) == {"swe-trajectories"}


def test_unknown_explicit_source_is_a_hard_error(sft_file):
    with pytest.raises(SystemExit):
        qc.build_sft_corpus(sft_path=sft_file, sources=["nope"], window=512,
                            max_tool_tokens=0, tok=FakeTok(), out=None)


def test_sources_are_packed_separately(sft_file):
    """A window never straddles two sources — per-source window counts must sum."""
    blob = qc.build_sft_corpus(sft_path=sft_file, window=512, max_tool_tokens=0,
                               tok=FakeTok(), out=None)
    assert sum(v["windows"] for v in blob["per_source"].values()) == blob["ids"].shape[0]


def test_window_source_labels_every_window(sft_file):
    """Per-source loss in the trainer keys off this; a wrong length silently misattributes."""
    blob = qc.build_sft_corpus(sft_path=sft_file, window=512, max_tool_tokens=0,
                               tok=FakeTok(), out=None)
    src = blob["window_source"]
    names = blob["source_names"]
    assert src.shape[0] == blob["ids"].shape[0]
    assert int(src.max()) < len(names)
    # contiguous runs: sources are packed separately, so the label never alternates back
    runs = [int(src[0])] + [int(b) for a, b in zip(src, src[1:], strict=False) if a != b]
    assert len(runs) == len(set(runs)), "a source's windows must be contiguous"
    # and the run lengths must match the per-source window counts
    for i, name in enumerate(names):
        assert int((src == i).sum()) == blob["per_source"][name]["windows"]


def test_build_is_deterministic_for_a_seed(sft_file):
    a = qc.build_sft_corpus(sft_path=sft_file, budgets={"logs": 20_000}, window=512,
                            max_tool_tokens=0, seed=7, tok=FakeTok(), out=None)
    b = qc.build_sft_corpus(sft_path=sft_file, budgets={"logs": 20_000}, window=512,
                            max_tool_tokens=0, seed=7, tok=FakeTok(), out=None)
    assert a["fingerprint"] == b["fingerprint"]


# ------------------------------------------------- split assistant turns (the real defect)
def test_merge_consecutive_assistant_joins_prose_and_its_tool_call():
    """Agent logs record one assistant turn as prose + a separate tool_calls message.
    Rendered verbatim each fragment gets its own <|im_end|>, which is what taught the
    model that a short preamble is followed by the STOP token."""
    from quant_tuner.qat.corpus import merge_consecutive_assistant
    msgs = [
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": "Let me check the current state:"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "bash"}}]},
        {"role": "tool", "content": "ok", "tool_call_id": "c1"},
    ]
    out, n = merge_consecutive_assistant(msgs)
    assert n == 1
    assert [m["role"] for m in out] == ["user", "assistant", "tool"]
    a = out[1]
    assert a["content"] == "Let me check the current state:"   # no trailing separator
    assert len(a["tool_calls"]) == 1


def test_merge_joins_two_prose_fragments_as_paragraphs():
    from quant_tuner.qat.corpus import merge_consecutive_assistant
    out, n = merge_consecutive_assistant([
        {"role": "assistant", "content": "First."},
        {"role": "assistant", "content": "Second."},
    ])
    assert n == 1
    assert out[0]["content"] == "First.\n\nSecond."


def test_merge_concatenates_tool_calls_from_both():
    from quant_tuner.qat.corpus import merge_consecutive_assistant
    out, _ = merge_consecutive_assistant([
        {"role": "assistant", "content": "a", "tool_calls": [{"id": "1"}]},
        {"role": "assistant", "tool_calls": [{"id": "2"}]},
    ])
    assert [c["id"] for c in out[0]["tool_calls"]] == ["1", "2"]


def test_merge_never_touches_user_or_tool_messages():
    """Tool results arrive as user-role messages under this template, so merging
    consecutive user messages would fuse a real user turn with a tool response."""
    from quant_tuner.qat.corpus import merge_consecutive_assistant
    msgs = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"},
            {"role": "tool", "content": "x"}, {"role": "tool", "content": "y"}]
    out, n = merge_consecutive_assistant(msgs)
    assert n == 0
    assert len(out) == 4


def test_merge_does_not_mutate_the_input():
    from quant_tuner.qat.corpus import merge_consecutive_assistant
    msgs = [{"role": "assistant", "content": "a"}, {"role": "assistant", "content": "b"}]
    merge_consecutive_assistant(msgs)
    assert msgs[0]["content"] == "a", "caller's messages must be untouched"


def test_merge_preserves_reasoning_from_both_fragments():
    from quant_tuner.qat.corpus import merge_consecutive_assistant
    out, _ = merge_consecutive_assistant([
        {"role": "assistant", "content": "a", "reasoning_content": "think one"},
        {"role": "assistant", "content": "b", "reasoning_content": "think two"},
    ])
    assert out[0]["reasoning_content"] == "think one\n\nthink two"


def test_inline_control_tokens_are_detected():
    """From our own sessions debugging chat templates: the assistant wrote
    rendered.find('<|im_end|>') in a code block, and special-token parsing makes that a
    real stop token inside supervised prose."""
    from quant_tuner.qat.corpus import has_inline_control_tokens
    assert has_inline_control_tokens(
        [{"role": "assistant", "content": "idx = rendered.find('<|im_end|>')"}])
    assert has_inline_control_tokens(
        [{"role": "assistant", "reasoning_content": "the <|im_start|> marker"}])
    assert not has_inline_control_tokens(
        [{"role": "assistant", "content": "a normal message about tools"}])
    assert not has_inline_control_tokens([{"role": "user", "content": None}])
