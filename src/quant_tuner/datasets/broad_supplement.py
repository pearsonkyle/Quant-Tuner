"""The hand-authored broad-domain supplement: parsing, splitting, and dataset records.

Source tree: ``calibration_supplements/broad/<area>/<subject>.txt`` — plain UTF-8,
blank-line-separated samples, ``## Section`` headings, no raw chat-control tokens.

This module is the **single source of truth** for how that tree is parsed and split.
``scripts/build_supplement.py`` (the stats/lint/build CLI) imports from here rather than
reimplementing it, for the same reason ``qat/corpus.py`` owns ``trajectory_to_messages``:
if the published dataset and the corpus the repo actually calibrates on could parse or
split the tree differently, they eventually would, and the drift would be invisible.

Three consumers, one source: **calibration** (imatrix/AWQ/GPTQ) and **MTP draft-head
training** take raw text from disjoint halves; **instruction tuning** takes the same
samples as chat pairs.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO / "calibration_supplements" / "broad"

TARGET_TOKENS = 1_000_000

# Mean chars/token measured on the mmmu/ tree with the Qwen3 tokenizer. Only used when no
# real tokenizer is supplied; every number derived from it is labelled an estimate so it
# never silently stands in for a real count.
CHARS_PER_TOKEN = 3.70

DEFAULT_SEED = "42"
DEFAULT_CALIB_FRACTION = 0.5

# Raw chat-control tokens must never appear: these files feed PPL/KLD eval as well as
# calibration, and llama-perplexity has no --parse-special, so an embedded marker
# tokenizes as plain BPE on one stack and as a control token on the other.
FORBIDDEN_PATTERNS = [
    re.compile(r"<\|im_start\|>"),
    re.compile(r"<\|im_end\|>"),
    re.compile(r"<\|endoftext\|>"),
    re.compile(r"<\|begin_of_text\|>"),
    re.compile(r"<\|eot_id\|>"),
    re.compile(r"<s>|</s>"),
    re.compile(r"<start_of_turn>|<end_of_turn>"),
]

_HEADING = re.compile(r"^#{1,6}\s+(.*)$")
_TURN = re.compile(r"^\[(user|assistant|tool)\]\s?", re.M)


# --------------------------------------------------------------------------- source parsing
def source_files() -> list[Path]:
    if not SRC_ROOT.exists():
        return []
    return sorted(p for p in SRC_ROOT.rglob("*.txt") if p.is_file())


def split_samples(text: str) -> list[str]:
    """Blank-line-delimited samples, ``====`` header dropped, bare headings merged forward.

    A lone ``## Section`` line is not a sample. Merging it into the block that follows keeps
    the heading with its content: the split assigns whole samples to one half, and an
    orphaned heading would put a naked topic label in one corpus and unlabelled prose in
    the other.
    """
    body = text
    if body.startswith("="):
        head, _, rest = body.partition("\n\n")
        if set(head.splitlines()[0].strip()) == {"="}:
            body = rest

    merged: list[str] = []
    pending = ""
    for s in (b.strip() for b in re.split(r"\n\s*\n", body)):
        if not s:
            continue
        if s.startswith("#") and "\n" not in s:
            pending = f"{pending}\n{s}" if pending else s
            continue
        merged.append(f"{pending}\n\n{s}" if pending else s)
        pending = ""
    return merged + ([pending] if pending else [])


def assign_halves(
    n: int, rel_path: str, seed: str = DEFAULT_SEED,
    calib_fraction: float = DEFAULT_CALIB_FRACTION,
) -> list[str]:
    """Per-sample ``"calib"``/``"mtp"`` labels for one file, in source order.

    The RNG is seeded **per file path**, not globally. A global shuffle would reassign
    samples across halves every time a new file is added, silently contaminating anything
    already calibrated or trained on; per-file seeding makes each file's split depend only
    on that file, so the tree grows without disturbing what came before.
    """
    idx = list(range(n))
    random.Random(f"{seed}:{rel_path}").shuffle(idx)
    calib = set(idx[:int(n * calib_fraction)])
    return ["calib" if i in calib else "mtp" for i in range(n)]


@dataclass
class Sample:
    """One blank-line-delimited block, with everything needed to emit a dataset row."""

    area: str
    subject: str
    area_title: str
    subject_title: str
    section: str
    text: str          # the sample as authored, section heading included
    body: str          # the same, with a leading `## Heading` line removed
    half: str
    file: str
    index: int

    @property
    def sample_id(self) -> str:
        return hashlib.sha256(f"{self.file}:{self.index}:{self.body}".encode()).hexdigest()[:16]

    @property
    def register(self) -> str:
        """Which of the four authored registers this is written in."""
        first = self.body.splitlines()[0]
        if first.startswith("Q:"):
            return "qa"
        if first.startswith("[user]"):
            return "transcript"
        if first.rstrip().endswith(":") and any(
            re.match(r"^ {2,}\S", ln) for ln in self.body.splitlines()[1:4]
        ):
            return "table"
        return "prose"


def parse_file(
    path: Path, seed: str = DEFAULT_SEED, calib_fraction: float = DEFAULT_CALIB_FRACTION,
) -> list[Sample]:
    text = path.read_text(encoding="utf-8")
    rel = path.relative_to(SRC_ROOT)
    header = text.split("\n\n", 1)[0] if text.startswith("=") else ""
    titles = {
        key: (m.group(1).strip() if (m := re.search(rf"^{key.upper()}:\s*(.+)$", header, re.M))
              else "")
        for key in ("area", "subject")
    }
    samples = split_samples(text)
    halves = assign_halves(len(samples), str(rel), seed, calib_fraction)

    out: list[Sample] = []
    section = ""
    for i, (s, half) in enumerate(zip(samples, halves, strict=True)):
        lines = s.splitlines()
        if m := _HEADING.match(lines[0]):
            section, body = m.group(1).strip(), "\n".join(lines[1:]).strip()
        else:
            body = s
        if not body:              # a heading with nothing under it: carry it, emit nothing
            continue
        out.append(Sample(
            area=rel.parts[0] if len(rel.parts) > 1 else "_root",
            subject=path.stem,
            area_title=titles["area"],
            subject_title=titles["subject"] or path.stem.replace("_", " ").title(),
            section=section,
            text=s,
            body=body,
            half=half,
            file=str(rel),
            index=i,
        ))
    return out


def iter_samples(
    seed: str = DEFAULT_SEED, calib_fraction: float = DEFAULT_CALIB_FRACTION,
) -> Iterator[Sample]:
    for path in source_files():
        yield from parse_file(path, seed, calib_fraction)


# ------------------------------------------------------------------- instruction synthesis
# Prompts for `qa` and `transcript` come out of the source: the question and the user turn
# were authored as prompts. `prose` and `table` samples were authored as continuous text, so
# their prompts are **templated** from the section heading and subject. Every row carries
# `prompt_source` so this is filterable rather than implicit — templated prompts are fine for
# light instruction tuning and must not be mistaken for authored ones.
_PROSE_TEMPLATES = [
    'Explain "{section}" in the context of {subject}.',
    'In {subject}, what should I understand about "{section}"?',
    'Give me a clear explanation of "{section}" as it applies to {subject}.',
    'Walk me through "{section}" within {subject}.',
    'I\'m studying {subject}. Explain "{section}".',
    'What are the key ideas behind "{section}" in {subject}?',
]

_PROSE_TEMPLATES_NOSECTION = [
    "Explain a key concept in {subject}.",
    "Tell me something important about {subject}.",
    "Give me a clear explanation of an aspect of {subject}.",
]

_TABLE_TEMPLATES = [
    "Give me a reference list of the key terms for {topic} in {subject}.",
    "Summarize {topic} in {subject} as a compact glossary of terms and meanings.",
    "List the important concepts for {topic} in {subject}, with a short gloss for each.",
    "I need a quick reference on {topic} in {subject}. Terms and what they mean.",
]


def _pick(options: list[str], key: str) -> str:
    """Deterministic template choice, so a rebuild produces byte-identical rows."""
    return options[int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(options)]


def to_messages(s: Sample) -> tuple[list[dict], str] | None:
    """``(messages, prompt_source)`` for a sample, or ``None`` if it yields no usable pair."""
    subject = s.subject_title
    lines = s.body.splitlines()

    if s.register == "qa":
        # `Q: ... A. ... B. ...` then `Reasoning: ... Answer: X.` — split at the rationale.
        m = re.search(r"^Reasoning:", s.body, re.M)
        if not m or not re.search(r"Answer:\s*[A-Z]\b", s.body):
            return None
        question = re.sub(r"^Q:\s*", "", s.body[:m.start()].strip())
        return ([{"role": "user", "content": question},
                 {"role": "assistant", "content": s.body[m.start():].strip()}], "authored")

    if s.register == "transcript":
        # The tool call stays as the assistant's literal text rather than being lifted into
        # a structured `tool_calls` field: these dialogues are illustrative and were never
        # executed, and inventing a schema would present authored text as a captured trace.
        parts = _TURN.split(s.body)
        msgs = [{"role": r, "content": c.strip()}
                for r, c in zip(parts[1::2], parts[2::2], strict=True) if c.strip()]
        if len(msgs) < 2 or msgs[0]["role"] != "user":
            return None
        return (msgs, "authored")

    if s.register == "table":
        topic = lines[0].rstrip().rstrip(":").strip() or s.section or subject
        prompt = _pick(_TABLE_TEMPLATES, s.sample_id).format(
            topic=topic[0].lower() + topic[1:], subject=subject)
        # The lead-in became the prompt, so the answer is the entries alone. Strip only
        # trailing whitespace: `.strip()` would eat the first entry's indent and leave it
        # hanging out of alignment with every row below it.
        answer = "\n".join(lines[1:]).rstrip() or s.body
    elif s.section:
        prompt = _pick(_PROSE_TEMPLATES, s.sample_id).format(section=s.section, subject=subject)
        answer = s.body
    else:
        prompt = _pick(_PROSE_TEMPLATES_NOSECTION, s.sample_id).format(subject=subject)
        answer = s.body

    if len(answer) < 40:
        return None
    return ([{"role": "user", "content": prompt},
             {"role": "assistant", "content": answer}], "templated")


# ------------------------------------------------------------------------- dataset records
def _common(s: Sample) -> dict:
    return {
        "id": s.sample_id,
        "area": s.area,
        "subject": s.subject,
        "area_title": s.area_title,
        "subject_title": s.subject_title,
        "section": s.section,
        "register": s.register,
        "half": s.half,
        "source_file": s.file,
    }


def _sized(rec: dict, chars: int) -> dict:
    return rec | {"n_chars": chars, "est_tokens": int(chars / CHARS_PER_TOKEN)}


def iter_corpus_records() -> Iterator[dict]:
    """One row per authored sample: raw text plus provenance. Calibration / MTP view."""
    for s in iter_samples():
        yield _sized(_common(s) | {"text": s.text}, len(s.text))


def iter_instruct_records() -> Iterator[dict]:
    """One row per sample that yields a usable prompt/response pair. Instruction view."""
    for s in iter_samples():
        if (made := to_messages(s)) is None:
            continue
        messages, prompt_source = made
        rec = _common(s) | {"messages": messages, "prompt_source": prompt_source,
                            "n_turns": len(messages)}
        yield _sized(rec, sum(len(m["content"]) for m in messages))
