"""Unit tests for quant_tuner.eval.mmmu — schema, prompt building, parsing, aggregation."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pytest

from quant_tuner.eval.mmmu import (
    DEFAULT_SYSTEM_PROMPT,
    MmmuHoldout,
    MmmuSample,
    _aggregate,
    _build_content_blocks,
    _eval_open,
    _parse_open_response,
    build_messages,
    encode_image_file,
    load_holdout,
    parse_answer,
    parse_mc_answer,
    pil_to_base64,
    render_summary,
    save_holdout,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample(
    sid: str = "s1",
    subject: str = "Math",
    question_type: str = "multiple-choice",
    question: str = "What is 2+2?",
    options: list[str] | None = None,
    answer: str = "A",
    images: list[str | None] | None = None,
) -> MmmuSample:
    return MmmuSample(
        id=sid,
        subject=subject,
        question_type=question_type,
        question=question,
        options=options if options is not None else ["4", "3", "5", "2"],
        answer=answer,
        images=images if images is not None else [None] * 7,
    )


def _minimal_jpeg_bytes() -> bytes:
    """Return the smallest valid JPEG bytes (1×1 white pixel)."""
    try:
        from PIL import Image as PILImage
        buf = io.BytesIO()
        PILImage.new("RGB", (1, 1), (255, 255, 255)).save(buf, format="JPEG")
        return buf.getvalue()
    except ImportError:
        # Fallback: minimal JPEG magic bytes (just enough to trigger MIME detection)
        return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 20


# ---------------------------------------------------------------------------
# MmmuSample round-trip
# ---------------------------------------------------------------------------


class TestMmmuSampleRoundTrip:
    def test_to_from_dict_preserves_all_fields(self):
        s = _sample(sid="x1", subject="Physics", answer="C",
                    question="<image 1> What is shown?",
                    images=["x1_1.jpg", None, None, None, None, None, None])
        d = s.to_dict()
        s2 = MmmuSample.from_dict(d)
        assert s2.id == "x1"
        assert s2.subject == "Physics"
        assert s2.answer == "C"
        assert s2.question == "<image 1> What is shown?"
        assert s2.images[0] == "x1_1.jpg"
        assert s2.images[1] is None

    def test_images_defaults_to_seven_nones(self):
        d = {
            "id": "t", "subject": "Bio", "question_type": "multiple-choice",
            "question": "Q", "options": ["A"], "answer": "A",
        }
        s = MmmuSample.from_dict(d)
        assert len(s.images) == 7
        assert all(v is None for v in s.images)


class TestMmmuHoldoutRoundTrip:
    def test_full_roundtrip(self, tmp_path: Path):
        s1 = _sample(sid="1", subject="Math", answer="B")
        s2 = _sample(sid="2", subject="Physics",
                     question_type="open", options=[], answer="42")
        h = MmmuHoldout(
            subjects=["Math", "Physics"],
            n_per_subject=1,
            n_shot=0,
            seed=7,
            shots={},
            samples=[s1, s2],
        )
        p = tmp_path / "holdout.json"
        save_holdout(h, p)
        h2 = load_holdout(p)
        assert h2.subjects == ["Math", "Physics"]
        assert h2.n_per_subject == 1
        assert h2.seed == 7
        assert len(h2.samples) == 2
        assert h2.samples[1].answer == "42"

    def test_file_is_valid_json(self, tmp_path: Path):
        h = MmmuHoldout(
            subjects=["Math"], n_per_subject=1, n_shot=0, seed=1,
            shots={"Math": []}, samples=[_sample(subject="Math")],
        )
        p = tmp_path / "h.json"
        save_holdout(h, p)
        d = json.loads(p.read_text())
        assert d["subjects"] == ["Math"]
        assert d["samples"][0]["id"] == "s1"

    def test_shots_preserved(self, tmp_path: Path):
        shot = _sample(sid="shot1", subject="Math", answer="D")
        h = MmmuHoldout(
            subjects=["Math"], n_per_subject=5, n_shot=1, seed=42,
            shots={"Math": [shot]}, samples=[_sample(subject="Math")],
        )
        p = tmp_path / "h.json"
        save_holdout(h, p)
        loaded = load_holdout(p)
        assert loaded.n_shot == 1
        assert loaded.shots["Math"][0].id == "shot1"


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------


class TestPilToBase64:
    def test_returns_data_url(self):
        try:
            from PIL import Image as PILImage
        except ImportError:
            pytest.skip("Pillow not installed")
        img = PILImage.new("RGB", (4, 4), (100, 150, 200))
        url = pil_to_base64(img)
        assert url.startswith("data:image/jpeg;base64,")
        # The base64 payload must be valid
        payload = url.split(",", 1)[1]
        decoded = base64.b64decode(payload)
        assert decoded[:2] == b"\xff\xd8"  # JPEG magic bytes

    def test_rgba_converted_to_rgb(self):
        try:
            from PIL import Image as PILImage
        except ImportError:
            pytest.skip("Pillow not installed")
        img = PILImage.new("RGBA", (2, 2), (0, 0, 0, 128))
        url = pil_to_base64(img)
        assert "image/jpeg" in url


class TestEncodeImageFile:
    def test_jpeg_detected_from_magic(self, tmp_path: Path):
        data = _minimal_jpeg_bytes()
        p = tmp_path / "test.jpg"
        p.write_bytes(data)
        url = encode_image_file(p)
        assert url.startswith("data:image/jpeg;base64,")

    def test_png_detected_from_magic(self, tmp_path: Path):
        # Minimal PNG header
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
        p = tmp_path / "test.png"
        p.write_bytes(data)
        url = encode_image_file(p)
        assert url.startswith("data:image/png;base64,")

    def test_unknown_defaults_to_jpeg(self, tmp_path: Path):
        p = tmp_path / "test.bin"
        p.write_bytes(b"\x00\x01\x02\x03")
        url = encode_image_file(p)
        assert "image/jpeg" in url


# ---------------------------------------------------------------------------
# Content-block building
# ---------------------------------------------------------------------------


class TestBuildContentBlocks:
    def _make_url(self, tag: str = "data:image/jpeg;base64,abc") -> str:
        return tag

    def test_plain_text_no_images(self):
        s = _sample(question="What colour is the sky?")
        blocks = _build_content_blocks(s, {})
        text_blocks = [b for b in blocks if b["type"] == "text"]
        assert any("What colour is the sky?" in b["text"] for b in text_blocks)
        assert not any(b["type"] == "image_url" for b in blocks)

    def test_image_tag_replaced_by_block(self):
        s = _sample(question="Look at <image 1> and answer.")
        img_url = self._make_url()
        blocks = _build_content_blocks(s, {1: img_url})
        types = [b["type"] for b in blocks]
        assert "image_url" in types
        img_blocks = [b for b in blocks if b["type"] == "image_url"]
        assert img_blocks[0]["image_url"]["url"] == img_url

    def test_text_before_and_after_image_preserved(self):
        s = _sample(question="Before <image 1> After")
        blocks = _build_content_blocks(s, {1: self._make_url()})
        # Should have: text("Before "), image, text(" After"), text(options)
        text_parts = [b["text"] for b in blocks if b["type"] == "text"]
        combined = " ".join(text_parts)
        assert "Before" in combined
        assert "After" in combined

    def test_unreferenced_image_appended_after_text(self):
        s = _sample(question="No explicit image tag here.")
        blocks = _build_content_blocks(s, {1: self._make_url()})
        # Image should still appear somewhere
        assert any(b["type"] == "image_url" for b in blocks)

    def test_options_appended_for_mc(self):
        s = _sample(question="Q?", options=["yes", "no", "maybe"], answer="A")
        blocks = _build_content_blocks(s, {})
        text_content = " ".join(b["text"] for b in blocks if b["type"] == "text")
        assert "(A) yes" in text_content
        assert "(B) no" in text_content
        assert "(C) maybe" in text_content

    def test_options_not_appended_for_open(self):
        s = _sample(
            question_type="open", options=[], answer="Paris",
            question="What is the capital of France?",
        )
        blocks = _build_content_blocks(s, {})
        text_content = " ".join(b["text"] for b in blocks if b["type"] == "text")
        # No "(A)" labels for open questions
        assert "(A)" not in text_content

    def test_multiple_images_in_order(self):
        s = _sample(question="<image 2> then <image 1>")
        urls = {1: "url1", 2: "url2"}
        blocks = _build_content_blocks(s, urls)
        img_blocks = [b for b in blocks if b["type"] == "image_url"]
        # image 2 appears before image 1 because the question lists them that way
        assert img_blocks[0]["image_url"]["url"] == "url2"
        assert img_blocks[1]["image_url"]["url"] == "url1"


# ---------------------------------------------------------------------------
# build_messages
# ---------------------------------------------------------------------------


class TestBuildMessages:
    def test_zero_shot_structure(self):
        s = _sample()
        msgs = build_messages(s, shots=[], image_data_urls={})
        assert msgs[0]["role"] == "system"
        assert msgs[-1]["role"] == "user"

    def test_system_prompt_can_be_disabled(self):
        s = _sample()
        msgs = build_messages(s, shots=[], image_data_urls={}, system_prompt=None)
        assert msgs[0]["role"] == "user"

    def test_default_system_prompt(self):
        s = _sample()
        msgs = build_messages(s, shots=[], image_data_urls={})
        assert msgs[0]["content"] == DEFAULT_SYSTEM_PROMPT

    def test_one_shot_structure(self):
        shot = _sample(sid="shot", answer="B")
        target = _sample(sid="tgt", answer="C")
        msgs = build_messages(target, shots=[shot], image_data_urls={})
        # system, user(shot), assistant(shot), user(target)
        assert len(msgs) == 4
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        assert msgs[2]["content"] == "Answer: B"
        assert msgs[3]["role"] == "user"

    def test_shot_answer_letter_in_assistant_turn(self):
        shot = _sample(answer="D")
        msgs = build_messages(_sample(), shots=[shot], image_data_urls={})
        asst = [m for m in msgs if m["role"] == "assistant"]
        assert asst[0]["content"] == "Answer: D"

    def test_target_content_is_list_of_blocks(self):
        s = _sample()
        msgs = build_messages(s, shots=[], image_data_urls={})
        last = msgs[-1]
        assert isinstance(last["content"], list)


# ---------------------------------------------------------------------------
# parse_mc_answer
# ---------------------------------------------------------------------------


class TestParseMcAnswer:
    @pytest.mark.parametrize("text,expected", [
        ("Answer: B", "B"),
        ("answer: c", "C"),
        ("Answer: (D)", "D"),
        ("(A)", "A"),
        ("The answer is B.", "B"),
        ("the answer is (E)", "E"),
        ("B", "B"),
        ("After thinking, B is right.", "B"),
    ])
    def test_extracts_letter(self, text, expected):
        assert parse_mc_answer(text, n_options=10) == expected

    def test_out_of_range_letter_rejected(self):
        assert parse_mc_answer("Answer: E", n_options=4) is None

    def test_boundary_letter_accepted(self):
        assert parse_mc_answer("Answer: D", n_options=4) == "D"

    def test_empty_returns_none(self):
        assert parse_mc_answer("", n_options=4) is None
        assert parse_mc_answer(None, n_options=4) is None  # type: ignore[arg-type]

    def test_lowercase_normalised(self):
        assert parse_mc_answer("answer: b", n_options=4) == "B"

    def test_explicit_marker_wins(self):
        assert parse_mc_answer("A great result. Answer: C", n_options=4) == "C"


# ---------------------------------------------------------------------------
# parse_answer dispatch
# ---------------------------------------------------------------------------


class TestParseAnswer:
    def test_multiple_choice_delegates_to_mc_parser(self):
        assert parse_answer("Answer: B", "multiple-choice", 4) == "B"

    def test_open_strips_answer_prefix(self):
        result = parse_answer("Answer: Paris", "open")
        assert result == "Paris"

    def test_open_no_prefix(self):
        result = parse_answer("Paris", "open")
        assert result == "Paris"

    def test_open_empty_returns_none(self):
        assert parse_answer("", "open") is None

    def test_open_whitespace_returns_none(self):
        assert parse_answer("   ", "open") is None


# ---------------------------------------------------------------------------
# Open-answer scoring helpers
# ---------------------------------------------------------------------------


class TestOpenScoringHelpers:
    def test_parse_open_response_extracts_number(self):
        preds = _parse_open_response("The result is 42")
        assert 42.0 in preds or 42 in preds

    def test_parse_open_response_extracts_string(self):
        preds = _parse_open_response("Answer is Paris")
        assert any("paris" in str(p) for p in preds)

    def test_eval_open_correct_number(self):
        preds = _parse_open_response("the answer is 3.14")
        assert _eval_open("3.14", preds)

    def test_eval_open_correct_string(self):
        preds = _parse_open_response("answer: red")
        assert _eval_open("red", preds)

    def test_eval_open_wrong_answer(self):
        preds = _parse_open_response("the answer is blue")
        assert not _eval_open("red", preds)


# ---------------------------------------------------------------------------
# _aggregate
# ---------------------------------------------------------------------------


class TestAggregate:
    def _row(self, sid, subject, qtype, truth, pred, correct):
        return {
            "id": sid,
            "subject": subject,
            "question_type": qtype,
            "truth": truth,
            "pred": pred,
            "correct": correct,
        }

    def test_overall_accuracy(self):
        rows = [
            self._row("1", "Math", "multiple-choice", "A", "A", True),
            self._row("2", "Math", "multiple-choice", "B", "C", False),
            self._row("3", "Physics", "open", "42", "42", True),
            self._row("4", "Physics", "open", "3.14", "2.71", False),
        ]
        s = _aggregate("model", rows)
        assert s.n_total == 4
        assert s.n_correct == 2
        assert s.accuracy == 0.5

    def test_mc_vs_open_split(self):
        rows = [
            self._row("1", "Math", "multiple-choice", "A", "A", True),
            self._row("2", "Math", "multiple-choice", "B", "B", True),
            self._row("3", "Science", "open", "red", "blue", False),
        ]
        s = _aggregate("m", rows)
        assert s.n_mc == 2
        assert s.n_mc_correct == 2
        assert s.n_open == 1
        assert s.n_open_correct == 0

    def test_per_subject_accuracy(self):
        rows = [
            self._row("1", "Math", "multiple-choice", "A", "A", True),
            self._row("2", "Math", "multiple-choice", "B", "C", False),
            self._row("3", "CS", "multiple-choice", "D", "D", True),
        ]
        s = _aggregate("m", rows)
        assert s.by_subject["Math"]["accuracy"] == 0.5
        assert s.by_subject["CS"]["accuracy"] == 1.0

    def test_unparseable_counted(self):
        rows = [
            self._row("1", "Math", "multiple-choice", "A", None, False),
            self._row("2", "Math", "multiple-choice", "B", "B", True),
        ]
        s = _aggregate("m", rows)
        assert s.n_unparseable == 1
        assert s.accuracy == 0.5

    def test_empty_rows(self):
        s = _aggregate("m", [])
        assert s.n_total == 0
        assert s.accuracy == 0.0
        assert s.by_subject == {}


# ---------------------------------------------------------------------------
# render_summary
# ---------------------------------------------------------------------------


class TestRenderSummary:
    def test_contains_model_and_accuracy(self):
        from quant_tuner.eval.mmmu import MmmuSummary

        s = MmmuSummary(
            model="test.gguf",
            n_total=10,
            n_correct=7,
            accuracy=0.7,
            n_mc=8,
            n_mc_correct=6,
            n_open=2,
            n_open_correct=1,
        )
        text = render_summary(s)
        assert "test.gguf" in text
        assert "0.700" in text

    def test_mc_and_open_lines_present(self):
        from quant_tuner.eval.mmmu import MmmuSummary

        s = MmmuSummary(
            model="m", n_total=10, n_correct=5, accuracy=0.5,
            n_mc=6, n_mc_correct=3, n_open=4, n_open_correct=2,
        )
        text = render_summary(s)
        assert "Multiple-choice" in text
        assert "Open-ended" in text

    def test_per_subject_section(self):
        from quant_tuner.eval.mmmu import MmmuSummary

        s = MmmuSummary(
            model="m", n_total=2, n_correct=1, accuracy=0.5,
            by_subject={
                "Math": {"n": 1, "correct": 1, "accuracy": 1.0, "unparseable": 0},
                "Physics": {"n": 1, "correct": 0, "accuracy": 0.0, "unparseable": 0},
            },
        )
        text = render_summary(s)
        assert "Math" in text
        assert "Physics" in text
