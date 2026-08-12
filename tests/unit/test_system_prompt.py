"""System-prompt boilerplate scrubbing for the SFT export.

The failure mode being guarded is asymmetric: dropping too little wastes training tokens
(cheap), dropping a block that carried the repo layout removes the context the trajectory
depends on (expensive, and invisible afterwards).
"""

from __future__ import annotations

from quant_tuner.data import system_prompt as sp

HARNESS = (
    "You are an interactive CLI agent for software engineering tasks.\n\n"
    "# Tone and Style\n- Be concise and direct. Avoid preamble.\n\n"
    "# Git Repository\n- The working directory is managed by a git repository.\n"
    "- Never commit unless asked.\n\n"
    "<example>\nuser: Write tests for someFile.ts\nmodel: I'll read someFile.ts first.\n"
    "</example>"
)
PROJECT = "# Project\nThe parser lives in src/quant_tuner/data/split.py and is fed by\nlogs-cli.jsonl.gz."


def _boilerplate(n: int = 12) -> set[str]:
    """The harness prompt as it looks after appearing in many sessions."""
    return sp.boilerplate_blocks([HARNESS] * n, min_sessions=4)


def test_boilerplate_is_blocks_repeated_across_sessions():
    bp = _boilerplate()
    assert "# Tone and Style\n- Be concise and direct. Avoid preamble." in bp
    # a block appearing in only one session is never boilerplate
    assert sp.boilerplate_blocks([HARNESS, PROJECT], min_sessions=4) == set()


def test_repeated_block_counted_once_per_session():
    """A block repeated ten times inside ONE prompt is not corpus-wide boilerplate."""
    prompt = "\n\n".join(["# Note\nsame block"] * 10)
    assert sp.boilerplate_blocks([prompt], min_sessions=4) == set()


def test_ungrounded_harness_blocks_are_dropped():
    out, stats = sp.scrub(HARNESS, boilerplate=_boilerplate(), grounding="unrelated chatter")
    assert "Tone and Style" not in out
    assert "someFile.ts" not in out          # an example filename this session never touches
    assert stats["dropped"] >= 3
    assert stats["chars_after"] < stats["chars_before"]


def test_the_identity_line_survives():
    """The persona the assistant is answering as must not silently change."""
    out, _ = sp.scrub(HARNESS, boilerplate=_boilerplate(), grounding="nothing relevant")
    assert out.startswith("You are an interactive CLI agent")


def test_a_block_naming_a_path_this_session_touches_is_kept():
    """'unless it has information about files and the repo' — the whole point."""
    bp = sp.boilerplate_blocks([HARNESS + "\n\n" + PROJECT] * 12, min_sessions=4)
    assert PROJECT in bp                                   # repeated => boilerplate by frequency
    grounding = "let me open src/quant_tuner/data/split.py and fix the packer"
    out, _ = sp.scrub(HARNESS + "\n\n" + PROJECT, boilerplate=bp, grounding=grounding)
    assert "src/quant_tuner/data/split.py" in out          # ...but kept: it is grounded
    assert "Tone and Style" not in out                     # while the harness still goes


def test_generic_filenames_do_not_ground_a_block():
    """`package.json` / `CLAUDE.md` appear in most repos; they kept ~170 harness blocks."""
    # not the first block: keep_lead would preserve the identity line regardless
    block = "# Conventions\nCheck CLAUDE.md and package.json before editing."
    prompt = "You are an agent.\n\n" + block
    bp = sp.boilerplate_blocks([prompt] * 12, min_sessions=4)
    generic = sp.generic_path_tokens(["touches CLAUDE.md and package.json"] * 40)
    assert {"CLAUDE.md", "package.json"} <= generic

    grounding = "I read CLAUDE.md then package.json"
    kept, _ = sp.scrub(prompt, boilerplate=bp, grounding=grounding)
    dropped, _ = sp.scrub(prompt, boilerplate=bp, grounding=grounding, generic=generic)
    assert "package.json" in kept          # without the DF filter it survives...
    assert "package.json" not in dropped   # ...and with it, correctly does not


def test_urls_do_not_ground_a_block():
    """A link to github.com/... otherwise looks exactly like an absolute path."""
    assert sp.path_tokens("see https://github.com/anthropics/claude-code/issues") == set()
    assert sp.path_tokens("edit /src/quant_tuner/data/split.py") == {
        "/src/quant_tuner/data/split.py"}


def test_library_names_are_not_treated_as_paths_when_common():
    generic = sp.generic_path_tokens(["built with Node.js and Next.js"] * 40)
    assert {"Node.js", "Next.js"} <= generic


def test_scrub_messages_leaves_non_system_turns_alone_and_does_not_mutate():
    msgs = [{"role": "system", "content": HARNESS},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"}]
    before = [dict(m) for m in msgs]
    out, stats = sp.scrub_messages(msgs, boilerplate=_boilerplate())
    assert msgs == before                       # input untouched
    assert out[1:] == msgs[1:]
    assert len(out[0]["content"]) < len(HARNESS)
    assert stats["dropped"] > 0


def test_a_prompt_scrubbed_to_nothing_drops_the_system_turn_entirely():
    only_boiler = "# Tone\n- Be brief."
    bp = sp.boilerplate_blocks([only_boiler] * 12, min_sessions=4)
    out, _ = sp.scrub_messages(
        [{"role": "system", "content": only_boiler}, {"role": "user", "content": "hi"}],
        boilerplate=bp, keep_lead=False)
    assert [m["role"] for m in out] == ["user"]     # never an empty system turn


def test_no_boilerplate_set_is_a_no_op():
    msgs = [{"role": "system", "content": HARNESS}, {"role": "user", "content": "hi"}]
    out, stats = sp.scrub_messages(msgs, boilerplate=set())
    assert out is msgs and stats["dropped"] == 0


def test_body_text_excludes_the_system_turn_and_includes_tool_calls():
    msgs = [{"role": "system", "content": "SYSTEM ONLY"},
            {"role": "assistant", "content": "on it",
             "tool_calls": [{"function": {"name": "read", "arguments": {"path": "a/b/c.py"}}}]}]
    body = sp.body_text(msgs)
    assert "SYSTEM ONLY" not in body
    assert "a/b/c.py" in body                   # a path used only inside a tool call grounds
