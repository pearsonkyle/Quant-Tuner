"""Tests for the broad-domain supplement parser, splitter, and dataset records.

These run against the real `calibration_supplements/broad/` tree when it is present and
are skipped otherwise, so the suite stays green in a checkout without the corpus.
"""

from __future__ import annotations

import pytest

from quant_tuner.datasets.broad_supplement import (
    Sample,
    assign_halves,
    iter_corpus_records,
    iter_instruct_records,
    iter_samples,
    source_files,
    split_samples,
    to_messages,
)

HAS_CORPUS = bool(source_files())
needs_corpus = pytest.mark.skipif(not HAS_CORPUS, reason="broad/ corpus not present")


def _sample(body: str, **kw) -> Sample:
    defaults = dict(
        area="physics", subject="optics", area_title="Physics", subject_title="Optics",
        section="Refraction", text=body, body=body, half="calib",
        file="physics/optics.txt", index=0,
    )
    return Sample(**{**defaults, **kw})


# ------------------------------------------------------------------------------ parsing
def test_split_samples_drops_header_and_merges_headings():
    text = (
        "==========\n"
        "AREA: Physics\n"
        "==========\n"
        "\n"
        "## Refraction\n"
        "\n"
        "Light bends at an interface because its speed changes.\n"
        "\n"
        "A second paragraph that stands alone.\n"
    )
    samples = split_samples(text)
    assert len(samples) == 2
    # the bare heading travels with the block it introduces, not as its own sample
    assert samples[0].startswith("## Refraction\n\nLight bends")
    assert samples[1].startswith("A second paragraph")


def test_register_classification():
    assert _sample("Q: What is n?\nA. one\nReasoning: x\nAnswer: A.").register == "qa"
    assert _sample("[user] hi\n[assistant] hello").register == "transcript"
    assert _sample("Key terms:\n  index    the ratio\n  angle    the tilt").register == "table"
    assert _sample("Light bends at an interface.").register == "prose"


# ------------------------------------------------------------------------------ splitting
def test_assign_halves_is_deterministic_and_partitions():
    a = assign_halves(50, "physics/optics.txt")
    assert a == assign_halves(50, "physics/optics.txt")          # reproducible
    assert set(a) <= {"calib", "mtp"}
    assert a.count("calib") == 25                                # honours the fraction
    assert len(a) == 50                                          # every sample labelled


def test_assign_halves_is_independent_per_file():
    """Adding a file must not reshuffle an existing one — that is the whole point of
    seeding per path rather than globally."""
    assert assign_halves(40, "a.txt") != assign_halves(40, "b.txt")
    before = assign_halves(40, "a.txt")
    assign_halves(40, "z-new-file.txt")                          # a "new file" appears
    assert assign_halves(40, "a.txt") == before


# -------------------------------------------------------------------- instruction records
def test_qa_becomes_question_and_rationale():
    body = ("Q: Why does light bend?\n"
            "A. It does not\n"
            "B. Its speed changes across the interface\n"
            "Reasoning: The wavefront pivots. Answer: B.")
    messages, source = to_messages(_sample(body))
    assert source == "authored"
    assert messages[0]["role"] == "user" and messages[0]["content"].startswith("Why does")
    assert messages[1]["content"].startswith("Reasoning:")


def test_transcript_becomes_chat_roles():
    body = ('[user] What is the index?\n'
            '[assistant] tool_call optics.index({"medium": "glass"})\n'
            '[tool] {"n": 1.5}\n'
            '[assistant] About 1.5 for common glass.')
    messages, source = to_messages(_sample(body))
    assert source == "authored"
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]


def test_table_prompt_uses_lead_in_and_keeps_entry_indent():
    body = "Key terms:\n  index      the ratio of speeds\n  angle      the tilt from normal"
    messages, source = to_messages(_sample(body))
    assert source == "templated"
    assert "key terms" in messages[0]["content"]
    # the lead-in became the prompt; the first entry must keep its indent or it hangs
    # out of alignment with every row below it
    assert messages[1]["content"].startswith("  index")


def test_prose_prompt_is_templated_and_deterministic():
    s = _sample("Light bends at an interface because its speed changes there, and the "
                "wavefront pivots as a result.")
    messages, source = to_messages(s)
    assert source == "templated"
    assert "Refraction" in messages[0]["content"] and "Optics" in messages[0]["content"]
    assert to_messages(s)[0] == messages                          # stable across rebuilds


# ------------------------------------------------------------------ whole-corpus invariants
@needs_corpus
def test_corpus_and_instruct_ids_are_unique_and_aligned():
    corpus = list(iter_corpus_records())
    instruct = list(iter_instruct_records())
    assert len({r["id"] for r in corpus}) == len(corpus)
    assert {r["id"] for r in instruct} <= {r["id"] for r in corpus}


@needs_corpus
def test_halves_partition_the_corpus():
    halves = [s.half for s in iter_samples()]
    assert set(halves) == {"calib", "mtp"}
    assert len(halves) == sum(1 for _ in iter_corpus_records())


@needs_corpus
def test_no_raw_control_tokens_survive_into_records():
    """The corpus doubles as a PPL/KLD eval file, and llama-perplexity has no
    --parse-special: an embedded marker would tokenize differently on the two stacks."""
    markers = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<start_of_turn>")
    for rec in iter_corpus_records():
        assert not any(m in rec["text"] for m in markers), rec["source_file"]


@needs_corpus
def test_every_instruct_row_declares_its_prompt_provenance():
    """`templated` prompts are generated from headings and must stay distinguishable
    from prompts that were actually authored as questions."""
    for rec in iter_instruct_records():
        assert rec["prompt_source"] in {"authored", "templated"}
        assert rec["messages"][0]["role"] == "user"
        assert rec["messages"][-1]["role"] == "assistant"
