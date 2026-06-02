"""MMMU multimodal benchmark evaluation.

MMMU (``MMMU/MMMU``) is a 30-subject multimodal understanding and reasoning
benchmark with questions drawn from college-level exams.  Questions may include
up to seven images and are either multiple-choice (with A–J answer options) or
open-ended.

This module covers three concerns:

  * The data schema and holdout serialisation (:class:`MmmuSample`,
    :class:`MmmuHoldout`, :func:`load_holdout`, :func:`save_holdout`).
  * Prompt building with inline image content blocks compatible with the
    llama.cpp OpenAI-compatible ``/v1/chat/completions`` endpoint.
  * The orchestrator :func:`run_mmmu_eval` that spawns a llama-server (or
    reuses one via ``--base-url``) and scores a holdout sample by sample.

The holdout is built by ``scripts/build_mmmu_holdout.py``, which samples N
questions per subject from the test split and saves their images to an
``images/`` sibling directory.  Images are referenced in the holdout JSON by
relative paths so the holdout is relocatable as a single directory.

Multimodal images are delivered to llama-server as ``image_url`` content
blocks containing base64-encoded JPEG data URLs — the format documented in
the llama.cpp multimodal README (``docs/multimodal.md``).  The server must be
started with ``--mmproj <projector.gguf>``; pass ``mmproj_path`` to
:func:`run_mmmu_eval` when spawning the server here, or point ``--base-url``
at an already-running multimodal server.
"""

from __future__ import annotations

import ast
import base64
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from openai import OpenAI

from quant_tuner.eval.server import running_server
from quant_tuner.eval.toolcall import Sampling, call_model

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LETTERS = "ABCDEFGHIJ"

# Pattern that matches <image 1>, <image 2>, … (case-insensitive) inside
# MMMU question text.
_IMAGE_TAG_RE = re.compile(r"<image\s*(\d+)>", re.IGNORECASE)

# Canonical list of 30 MMMU subjects (HuggingFace dataset config names).
MMMU_SUBJECTS: list[str] = [
    "Accounting",
    "Agriculture",
    "Architecture_and_Engineering",
    "Art",
    "Art_Theory",
    "Basic_Medical_Science",
    "Biology",
    "Chemistry",
    "Clinical_Medicine",
    "Computer_Science",
    "Design",
    "Diagnostics_and_Laboratory_Medicine",
    "Economics",
    "Electronics",
    "Energy_and_Power",
    "Finance",
    "Geography",
    "History",
    "Literature",
    "Manage",
    "Marketing",
    "Materials",
    "Math",
    "Mechanical_Engineering",
    "Music",
    "Pharmacy",
    "Physics",
    "Psychology",
    "Public_Health",
    "Sociology",
]

# ---------------------------------------------------------------------------
# Holdout schema
# ---------------------------------------------------------------------------


@dataclass
class MmmuSample:
    """One MMMU question, stripped to the fields the eval needs.

    ``images`` is a list of exactly seven slots (one per possible image column
    ``image_1`` … ``image_7``).  Each slot is either a relative path string
    pointing to a JPEG in the sibling ``images/`` directory, or ``None`` if
    that slot has no image.  The path is relative to the holdout JSON file.
    """

    id: str
    subject: str
    question_type: str          # "multiple-choice" | "open"
    question: str               # may contain <image N> placeholders
    options: list[str]          # plain-text option strings (no letter prefix); empty for open
    answer: str                 # letter ("A"…"J") for MC; text/number for open
    images: list[str | None]    # 7 slots: relative JPEG paths or None

    @classmethod
    def from_dict(cls, d: dict) -> MmmuSample:
        return cls(
            id=d["id"],
            subject=d["subject"],
            question_type=d["question_type"],
            question=d["question"],
            options=list(d.get("options") or []),
            answer=d["answer"],
            images=list(d.get("images") or [None] * 7),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "subject": self.subject,
            "question_type": self.question_type,
            "question": self.question,
            "options": self.options,
            "answer": self.answer,
            "images": self.images,
        }


@dataclass
class MmmuHoldout:
    """Holdout file: optional few-shot pool per subject + the actual test samples."""

    subjects: list[str]
    n_per_subject: int
    n_shot: int
    seed: int
    shots: dict[str, list[MmmuSample]]
    samples: list[MmmuSample]

    @classmethod
    def from_dict(cls, d: dict) -> MmmuHoldout:
        return cls(
            subjects=list(d["subjects"]),
            n_per_subject=int(d["n_per_subject"]),
            n_shot=int(d["n_shot"]),
            seed=int(d["seed"]),
            shots={
                subj: [MmmuSample.from_dict(s) for s in slist]
                for subj, slist in d.get("shots", {}).items()
            },
            samples=[MmmuSample.from_dict(s) for s in d["samples"]],
        )

    def to_dict(self) -> dict:
        return {
            "subjects": self.subjects,
            "n_per_subject": self.n_per_subject,
            "n_shot": self.n_shot,
            "seed": self.seed,
            "shots": {subj: [s.to_dict() for s in v] for subj, v in self.shots.items()},
            "samples": [s.to_dict() for s in self.samples],
        }


def load_holdout(path: Path) -> MmmuHoldout:
    """Load an MMMU holdout JSON file."""
    with path.open() as f:
        return MmmuHoldout.from_dict(json.load(f))


def save_holdout(holdout: MmmuHoldout, path: Path) -> None:
    path.write_text(json.dumps(holdout.to_dict(), indent=2))


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------


def pil_to_base64(image: PILImage, fmt: str = "JPEG") -> str:
    """Convert a PIL Image to a base64 data-URL for use in image_url blocks."""
    buf = io.BytesIO()
    rgb = image.convert("RGB") if fmt.upper() in ("JPEG", "JPG") else image
    rgb.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    mime = "image/jpeg" if fmt.upper() in ("JPEG", "JPG") else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"


def save_pil_image(image: PILImage, path: Path, fmt: str = "JPEG") -> None:
    """Save a PIL Image to *path*, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = image.convert("RGB") if fmt.upper() in ("JPEG", "JPG") else image
    rgb.save(str(path), format=fmt)


def encode_image_file(path: Path) -> str:
    """Read an image file and return a base64 JPEG data-URL."""
    with path.open("rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    # Sniff the format from the magic bytes; default to JPEG.
    if data[:4] == b"\x89PNG":
        mime = "image/png"
    elif data[:6] in (b"GIF87a", b"GIF89a"):
        mime = "image/gif"
    elif data[:4] in (b"RIFF", b"WEBP"):
        mime = "image/webp"
    else:
        mime = "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _load_image_data_urls(sample: MmmuSample, images_dir: Path) -> dict[int, str]:
    """Return ``{1-based index → data-URL}`` for all non-None images in *sample*."""
    result: dict[int, str] = {}
    for i, rel in enumerate(sample.images, 1):
        if rel is not None:
            p = images_dir / rel
            if p.exists():
                result[i] = encode_image_file(p)
    return result


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert taking a college-level exam. "
    "Answer the question based on the provided image(s) and text. "
    "For multiple-choice questions, respond with only the letter of the best "
    "answer (A, B, C, …). "
    "For open-ended questions, give a concise, direct answer."
)


def _image_url_block(data_url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": data_url}}


def _text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def _build_content_blocks(
    sample: MmmuSample,
    image_data_urls: dict[int, str],
) -> list[dict]:
    """Build an OpenAI content-block list for one MMMU sample.

    The question text may contain ``<image N>`` placeholders.  Each placeholder
    is replaced with the corresponding ``image_url`` block.  Any images that
    exist but are not referenced in the text are appended at the end (before
    the options list for multiple-choice questions).
    """
    # Walk the question text, emitting text and image blocks in order.
    content: list[dict] = []
    referenced: set[int] = set()
    last = 0
    for m in _IMAGE_TAG_RE.finditer(sample.question):
        text_before = sample.question[last : m.start()]
        if text_before:
            content.append(_text_block(text_before))
        idx = int(m.group(1))
        referenced.add(idx)
        if idx in image_data_urls:
            content.append(_image_url_block(image_data_urls[idx]))
        last = m.end()
    remaining = sample.question[last:]
    if remaining:
        content.append(_text_block(remaining))

    # Append images that have no explicit <image N> tag in the question text.
    for idx in sorted(image_data_urls):
        if idx not in referenced:
            content.append(_image_url_block(image_data_urls[idx]))

    # Append lettered option list for multiple-choice questions.
    if sample.question_type == "multiple-choice" and sample.options:
        opts_text = "\nOptions:\n" + "\n".join(
            f"({_LETTERS[i]}) {opt}" for i, opt in enumerate(sample.options)
        )
        content.append(_text_block(opts_text))

    return content


def build_messages(
    target: MmmuSample,
    shots: list[MmmuSample],
    image_data_urls: dict[int, str],
    *,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    shot_image_data_urls: list[dict[int, str]] | None = None,
) -> list[dict]:
    """Build the OpenAI chat-completions message list for one MMMU question.

    Each demonstration shot becomes a user turn (content blocks) followed by
    an assistant turn with just the answer.  The final user turn is the target
    question with its images interleaved.

    ``shot_image_data_urls`` is a parallel list of per-shot image dicts.  When
    omitted, shots are rendered without images (text-only fallback for 0-shot
    or when shot images are unavailable).
    """
    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})

    for i, shot in enumerate(shots):
        shot_urls = (shot_image_data_urls[i]
                     if shot_image_data_urls and i < len(shot_image_data_urls)
                     else {})
        shot_content = _build_content_blocks(shot, shot_urls)
        msgs.append({"role": "user", "content": shot_content})
        msgs.append({"role": "assistant", "content": f"Answer: {shot.answer}"})

    target_content = _build_content_blocks(target, image_data_urls)
    msgs.append({"role": "user", "content": target_content})
    return msgs


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

# Multiple-choice: match patterns in priority order.
_MC_PATTERNS = [
    re.compile(r"(?i)answer\s*[:=\-]\s*\(?([A-J])\)?(?=\W|$)"),
    re.compile(r"\(([A-J])\)"),
    re.compile(r"(?i)(?:the\s+)?answer\s+is\s+\(?([A-J])\)?(?=\W|$)"),
    re.compile(r"^([A-J])(?=\W|$)"),
    re.compile(r"\b([A-J])\b"),
]


def parse_mc_answer(text: str, n_options: int) -> str | None:
    """Extract the predicted letter from a multiple-choice completion.

    ``n_options`` caps which letters are valid (a 4-option question rejects
    "E"). Returns the predicted letter or ``None``.
    """
    if not text:
        return None
    valid = set(_LETTERS[:n_options])
    for pat in _MC_PATTERNS:
        m = pat.search(text)
        if m and m.group(1).upper() in valid:
            return m.group(1).upper()
    return None


# ---------------------------------------------------------------------------
# Open-answer scoring (ported from MMMU official eval_utils.py)
# ---------------------------------------------------------------------------


def _check_is_number(s: str) -> bool:
    try:
        float(s.replace(",", ""))
        return True
    except ValueError:
        return False


def _normalize_str(s: str) -> list:
    """Normalise a string to a float list element or lower-cased string(s)."""
    s = s.strip()
    if _check_is_number(s):
        return [round(float(s.replace(",", "")), 2)]
    s = s.lower()
    if len(s) == 1:
        return [" " + s, s + " "]
    return [s]


def _extract_numbers(s: str) -> list[str]:
    commas = re.findall(r"-?\b\d{1,3}(?:,\d{3})+\b", s)
    scientific = re.findall(r"-?\d+(?:\.\d+)?[eE][+-]?\d+", s)
    simple = re.findall(r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])", s)
    return commas + scientific + simple


def _parse_open_response(response: str) -> list:
    """Extract candidate answers from an open-ended response (MMMU official logic)."""
    def _key_subresponses(resp: str) -> list[str]:
        key_responses: list[str] = []
        resp = resp.strip().strip(".").lower()
        sub_responses = re.split(r"\.\s(?=[A-Z])|\n", resp)
        indicators = [
            "could be ", "so ", "is ", "thus ", "therefore ",
            "final ", "answer ", "result ",
        ]
        for idx, sub in enumerate(sub_responses):
            if idx == len(sub_responses) - 1:
                indicators = indicators + ["="]
            shortest: str | None = None
            for ind in indicators:
                if ind in sub:
                    candidate = sub.split(ind)[-1].strip()
                    if shortest is None or len(candidate) < len(shortest):
                        shortest = candidate
            if shortest and shortest.strip() not in {",", ".", "!", "?", ";", ":", "'"}:
                key_responses.append(shortest)
        return key_responses if key_responses else [resp]

    key = _key_subresponses(response)
    pred_list = list(key)
    for r in key:
        pred_list.extend(_extract_numbers(r))
    normalised: list = []
    for item in pred_list:
        normalised.extend(_normalize_str(str(item)))
    return list(set(normalised))


def _eval_open(gold: str, pred_list: list) -> bool:
    """Return True if any element of *pred_list* matches *gold* (MMMU official logic)."""
    norm_answers = _normalize_str(gold)
    for pred in pred_list:
        if isinstance(pred, str):
            for norm in norm_answers:
                if isinstance(norm, str) and norm in pred:
                    return True
        elif pred in norm_answers:
            return True
    return False


def parse_answer(text: str, question_type: str, n_options: int = 10) -> str | None:
    """Parse the model's answer for any MMMU question type."""
    if question_type == "multiple-choice":
        return parse_mc_answer(text, n_options)
    # Open: strip common prefixes and return the trimmed text.
    stripped = re.sub(r"(?i)^answer\s*[:=\-]\s*", "", (text or "").strip())
    return stripped.strip() or None


# ---------------------------------------------------------------------------
# Single-sample scoring
# ---------------------------------------------------------------------------


def score_one(
    client: OpenAI,
    target: MmmuSample,
    shots: list[MmmuSample],
    sampling: Sampling,
    images_dir: Path,
    *,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    shot_images_dir: Path | None = None,
) -> dict:
    """Run one MMMU sample through the model and return a scored result dict."""
    image_data_urls = _load_image_data_urls(target, images_dir)
    _shot_dir = shot_images_dir or images_dir
    shot_img_urls = [_load_image_data_urls(shot, _shot_dir) for shot in shots]

    messages = build_messages(
        target, shots, image_data_urls,
        system_prompt=system_prompt,
        shot_image_data_urls=shot_img_urls if shots else None,
    )
    try:
        resp = call_model(client, messages, tools=[], sampling=sampling)
        msg = resp.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or ""
        combined = (reasoning + "\n" + content) if reasoning else content
    except Exception as exc:
        return {
            "id": target.id,
            "subject": target.subject,
            "question_type": target.question_type,
            "truth": target.answer,
            "pred": None,
            "correct": False,
            "raw": "",
            "error": str(exc),
        }

    n_options = len(target.options)
    pred = parse_answer(combined, target.question_type, n_options or 10)

    if target.question_type == "multiple-choice":
        correct = (pred == target.answer) if pred is not None else False
    else:
        try:
            pred_list = _parse_open_response(pred or "")
            correct = _eval_open(target.answer, pred_list)
        except Exception:
            correct = (pred or "").lower().strip() == target.answer.lower().strip()

    return {
        "id": target.id,
        "subject": target.subject,
        "question_type": target.question_type,
        "truth": target.answer,
        "pred": pred,
        "correct": correct,
        "raw": combined[-400:],
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class MmmuSummary:
    """Aggregate metrics returned by :func:`run_mmmu_eval`."""

    model: str
    n_total: int
    n_correct: int
    accuracy: float
    n_mc: int = 0
    n_mc_correct: int = 0
    n_open: int = 0
    n_open_correct: int = 0
    n_unparseable: int = 0
    by_subject: dict[str, dict] = field(default_factory=dict)
    per_sample: list[dict] = field(default_factory=list)


def _aggregate(model_label: str, per_sample: list[dict]) -> MmmuSummary:
    by_subj: dict[str, list[dict]] = {}
    for r in per_sample:
        by_subj.setdefault(r["subject"], []).append(r)

    by_subject: dict[str, dict] = {
        subj: {
            "n": len(rs),
            "correct": sum(1 for r in rs if r["correct"]),
            "accuracy": (sum(1 for r in rs if r["correct"]) / len(rs)) if rs else 0.0,
            "unparseable": sum(1 for r in rs if r.get("pred") is None),
        }
        for subj, rs in by_subj.items()
    }

    n_total = len(per_sample)
    n_correct = sum(1 for r in per_sample if r["correct"])
    n_mc = sum(1 for r in per_sample if r["question_type"] == "multiple-choice")
    n_mc_correct = sum(
        1 for r in per_sample
        if r["question_type"] == "multiple-choice" and r["correct"]
    )
    n_unparseable = sum(1 for r in per_sample if r.get("pred") is None)

    return MmmuSummary(
        model=model_label,
        n_total=n_total,
        n_correct=n_correct,
        accuracy=n_correct / n_total if n_total else 0.0,
        n_mc=n_mc,
        n_mc_correct=n_mc_correct,
        n_open=n_total - n_mc,
        n_open_correct=n_correct - n_mc_correct,
        n_unparseable=n_unparseable,
        by_subject=by_subject,
        per_sample=per_sample,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_mmmu_eval(
    holdout_path: Path,
    *,
    model_path: Path | None = None,
    mmproj_path: Path | None = None,
    base_url: str | None = None,
    sampling: Sampling | None = None,
    model_label: str | None = None,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    ctx: int = 8192,
    ngl: int = 99,
    server_log_path: Path | None = None,
    server_startup_timeout: float = 180.0,
    chat_template_kwargs: str | None = None,
    per_sample_log: Path | None = None,
    progress: bool = False,
) -> MmmuSummary:
    """Score every sample in the holdout and return aggregate accuracy.

    Exactly one of ``model_path`` (spawn a server) or ``base_url`` (reuse an
    existing server) must be supplied.

    When ``model_path`` is given, ``mmproj_path`` should point to the
    multimodal projector GGUF (``--mmproj``).  For models loaded via ``-hf``
    (which auto-download the projector), pass ``None`` and handle the mmproj
    flag externally.

    Images are read from an ``images/`` directory that is a sibling of
    ``holdout_path`` (created by ``scripts/build_mmmu_holdout.py``).
    """
    if (model_path is None) == (base_url is None):
        raise ValueError("provide exactly one of model_path or base_url")

    sampling = sampling or Sampling()
    holdout = load_holdout(holdout_path)
    images_dir = holdout_path.parent / "images"
    label = model_label or (model_path.name if model_path else "remote")
    n_shot = holdout.n_shot

    log_fh = per_sample_log.open("w") if per_sample_log is not None else None

    def _run_against(url: str) -> MmmuSummary:
        client = OpenAI(base_url=url, api_key="sk-no-key")
        rows: list[dict] = []
        for i, target in enumerate(holdout.samples, 1):
            shots = holdout.shots.get(target.subject, [])[:n_shot]
            row = score_one(
                client, target, shots, sampling, images_dir,
                system_prompt=system_prompt,
            )
            rows.append(row)
            if log_fh is not None:
                log_fh.write(json.dumps(row) + "\n")
            if progress:
                mark = "✓" if row["correct"] else ("?" if row.get("pred") is None else "✗")
                print(
                    f"[{i:4d}/{len(holdout.samples)}]"
                    f" {target.subject:35s}"
                    f" {target.question_type:16s}"
                    f" truth={target.answer!r:4s}"
                    f" pred={str(row.get('pred'))!r:6s}"
                    f" {mark}",
                    flush=True,
                )
        return _aggregate(label, rows)

    try:
        if base_url is not None:
            return _run_against(base_url)
        with running_server(
            model_path,
            ctx=ctx,
            ngl=ngl,
            log_path=server_log_path,
            startup_timeout=server_startup_timeout,
            chat_template_kwargs=chat_template_kwargs,
            mmproj_path=mmproj_path,
        ) as url:
            return _run_against(url)
    finally:
        if log_fh is not None:
            log_fh.close()


def render_summary(summary: MmmuSummary) -> str:
    """Format an :class:`MmmuSummary` for human-readable stdout."""
    lines = [
        "=" * 60,
        f"Model: {summary.model}",
        f"  Overall accuracy:   {summary.n_correct}/{summary.n_total}"
        f"  ({summary.accuracy:.3f})",
    ]
    if summary.n_mc > 0:
        mc_acc = summary.n_mc_correct / summary.n_mc
        lines.append(
            f"  Multiple-choice:    {summary.n_mc_correct}/{summary.n_mc}"
            f"  ({mc_acc:.3f})"
        )
    if summary.n_open > 0:
        open_acc = (summary.n_open_correct / summary.n_open) if summary.n_open else 0.0
        lines.append(
            f"  Open-ended:         {summary.n_open_correct}/{summary.n_open}"
            f"  ({open_acc:.3f})"
        )
    if summary.n_unparseable:
        lines.append(f"  Unparseable:        {summary.n_unparseable}")
    if summary.by_subject:
        lines.append("\n  Per-subject:")
        for subj in sorted(summary.by_subject):
            s = summary.by_subject[subj]
            unp = f"  [{s['unparseable']} unparseable]" if s.get("unparseable") else ""
            lines.append(
                f"    {subj:35s}  {s['correct']}/{s['n']}  ({s['accuracy']:.3f}){unp}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HuggingFace dataset row → MmmuSample converter (used by build script)
# ---------------------------------------------------------------------------


def hf_row_to_sample(row: dict, sample_id: str, images_dir: Path) -> MmmuSample:
    """Convert one HuggingFace MMMU row to an :class:`MmmuSample`.

    Saves any non-None image fields as JPEG files under *images_dir* and
    stores their names (relative to *images_dir*) in the ``images`` list.

    Returns the sample with ``images`` populated with filenames (not full
    paths) so they are stored as relative references in the holdout JSON.
    """
    # ``options`` may be a raw Python list literal string in some HF versions.
    raw_options = row.get("options") or []
    if isinstance(raw_options, str):
        try:
            raw_options = ast.literal_eval(raw_options)
        except Exception:
            raw_options = []
    options: list[str] = [str(o) for o in raw_options]

    # Save images; record filename or None in each of the 7 slots.
    image_slots: list[str | None] = []
    for slot_idx in range(1, 8):
        key = f"image_{slot_idx}"
        img = row.get(key)
        if img is None:
            image_slots.append(None)
        else:
            # img is a PIL Image when loaded from HuggingFace datasets.
            filename = f"{sample_id}_{slot_idx}.jpg"
            save_pil_image(img, images_dir / filename)
            image_slots.append(filename)

    return MmmuSample(
        id=sample_id,
        subject=row.get("subject", ""),
        question_type=row.get("question_type", "multiple-choice"),
        question=row.get("question", ""),
        options=options,
        answer=str(row.get("answer", "")),
        images=image_slots,
    )
