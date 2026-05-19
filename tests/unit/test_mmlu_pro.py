"""Unit tests for quant_tuner.eval.mmlu_pro — prompt + parsing + aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_tuner.eval.mmlu_pro import (
    DEFAULT_SYSTEM_PROMPT,
    MmluProHoldout,
    MmluProSample,
    _aggregate,
    build_messages,
    format_question_block,
    load_holdout,
    parse_answer,
    save_holdout,
)


def _sample(question="Q?", options=("A1", "B1", "C1", "D1"), answer="C",
            subject="math", qid=1) -> MmluProSample:
    return MmluProSample(
        question_id=qid, subject=subject, question=question,
        options=list(options), answer=answer,
        answer_index=ord(answer) - ord("A"),
    )


class TestFormatQuestionBlock:
    def test_lettered_options(self):
        s = _sample(options=["x", "y", "z"])
        block = format_question_block(s)
        assert "(A) x" in block and "(B) y" in block and "(C) z" in block

    def test_subject_and_question_present(self):
        s = _sample(question="What is 2+2?", subject="math")
        block = format_question_block(s)
        assert "Subject: math" in block
        assert "Question: What is 2+2?" in block

    def test_handles_up_to_ten_options(self):
        s = _sample(options=[f"opt{i}" for i in range(10)])
        block = format_question_block(s)
        # MMLU-Pro uses A..J for ten-option questions
        assert "(J) opt9" in block


class TestBuildMessages:
    def test_system_then_shots_then_target(self):
        shots = [_sample(qid=1, answer="A"), _sample(qid=2, answer="D")]
        target = _sample(qid=99, answer="B")
        msgs = build_messages(target, shots)
        # system + 2*(user, assistant) + user
        assert len(msgs) == 6
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        assert msgs[2]["content"] == "Answer: A"
        assert msgs[3]["role"] == "user"
        assert msgs[4]["role"] == "assistant"
        assert msgs[4]["content"] == "Answer: D"
        assert msgs[5]["role"] == "user"
        # Target's question text appears in the final turn, not in any shot turn.
        assert format_question_block(target) == msgs[5]["content"]

    def test_no_shots_yields_zero_shot_prompt(self):
        msgs = build_messages(_sample(), shots=[])
        assert len(msgs) == 2  # system + target user
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_system_prompt_can_be_disabled(self):
        msgs = build_messages(_sample(), shots=[], system_prompt=None)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_uses_provided_system_prompt(self):
        msgs = build_messages(_sample(), shots=[], system_prompt="be terse")
        assert msgs[0] == {"role": "system", "content": "be terse"}

    def test_default_system_prompt_is_used(self):
        msgs = build_messages(_sample(), shots=[])
        assert msgs[0]["content"] == DEFAULT_SYSTEM_PROMPT


class TestParseAnswer:
    @pytest.mark.parametrize("text,expected", [
        ("Answer: B", "B"),
        ("answer: c", "C"),
        ("Answer: (D)", "D"),
        ("(A)", "A"),
        ("The answer is B.", "B"),
        ("the answer is (E)", "E"),
        ("B", "B"),
        ("  B  ", "B"),
        ("After working through this, B is right.", "B"),
    ])
    def test_extracts_letter(self, text, expected):
        assert parse_answer(text, n_options=10) == expected

    def test_out_of_range_letter_rejected(self):
        # 4-option question, model says E → invalid
        assert parse_answer("Answer: E", n_options=4) is None

    def test_in_range_letter_accepted_at_boundary(self):
        assert parse_answer("Answer: D", n_options=4) == "D"

    def test_empty_returns_none(self):
        assert parse_answer("", n_options=4) is None
        assert parse_answer(None, n_options=4) is None  # type: ignore

    def test_no_letter_returns_none(self):
        assert parse_answer("the answer is not stated", n_options=4) is None

    def test_explicit_answer_marker_wins_over_other_letters(self):
        # The sentence mentions "A great...", but the answer marker says C.
        # The "Answer:" pattern should win.
        assert parse_answer("A great question. Answer: C", n_options=4) == "C"

    def test_lowercase_letter_normalized(self):
        assert parse_answer("answer: b", n_options=4) == "B"


class TestAggregate:
    def test_overall_and_per_subject(self):
        rows = [
            {"question_id": 1, "subject": "math", "truth": "A", "pred": "A", "correct": True},
            {"question_id": 2, "subject": "math", "truth": "B", "pred": "C", "correct": False},
            {"question_id": 3, "subject": "cs",   "truth": "D", "pred": "D", "correct": True},
            {"question_id": 4, "subject": "cs",   "truth": "E", "pred": "E", "correct": True},
        ]
        summ = _aggregate("model.gguf", rows)
        assert summ.n_total == 4
        assert summ.n_correct == 3
        assert summ.accuracy == 0.75
        assert summ.by_subject["math"]["accuracy"] == 0.5
        assert summ.by_subject["cs"]["accuracy"] == 1.0
        assert summ.by_subject["cs"]["n"] == 2

    def test_unparseable_counted(self):
        rows = [
            {"question_id": 1, "subject": "math", "truth": "A", "pred": None, "correct": False},
            {"question_id": 2, "subject": "math", "truth": "B", "pred": "B", "correct": True},
        ]
        summ = _aggregate("m", rows)
        assert summ.n_unparseable == 1
        assert summ.by_subject["math"]["unparseable"] == 1
        # Unparseable still counts as a wrong answer for the accuracy denominator.
        assert summ.accuracy == 0.5

    def test_empty_rows(self):
        summ = _aggregate("m", [])
        assert summ.n_total == 0
        assert summ.accuracy == 0.0
        assert summ.by_subject == {}


class TestHoldoutRoundTrip:
    def test_roundtrip_preserves_fields(self, tmp_path: Path):
        h = MmluProHoldout(
            subjects=["math", "computer science"],
            n_per_subject=2,
            n_shot=1,
            seed=42,
            shots={
                "math": [_sample(qid=10, answer="A", subject="math")],
                "computer science": [_sample(qid=20, answer="B", subject="computer science")],
            },
            samples=[
                _sample(qid=1, answer="C", subject="math"),
                _sample(qid=2, answer="D", subject="computer science"),
            ],
        )
        path = tmp_path / "h.json"
        save_holdout(h, path)
        loaded = load_holdout(path)
        assert loaded.subjects == h.subjects
        assert loaded.n_per_subject == 2 and loaded.n_shot == 1 and loaded.seed == 42
        assert len(loaded.samples) == 2
        assert loaded.samples[0].answer == "C"
        assert loaded.shots["math"][0].question_id == 10

    def test_file_is_valid_json(self, tmp_path: Path):
        h = MmluProHoldout(
            subjects=["math"], n_per_subject=1, n_shot=0, seed=1,
            shots={"math": []}, samples=[_sample(subject="math")],
        )
        path = tmp_path / "h.json"
        save_holdout(h, path)
        # Anyone can re-parse it without our dataclass machinery.
        d = json.loads(path.read_text())
        assert d["subjects"] == ["math"]
        assert d["samples"][0]["answer"] == "C"
