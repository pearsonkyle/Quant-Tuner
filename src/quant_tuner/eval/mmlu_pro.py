"""MMLU-Pro few-shot evaluation.

MMLU-Pro (``TIGER-Lab/MMLU-Pro``) extends the original MMLU with up-to-ten
answer choices and explicitly reasoning-heavy questions. We use it as a
complementary signal to the tool-call eval — a quant that holds its KLD
floor and its tool-calling accuracy but tanks on reasoning questions has
lost something we care about.

This module covers two concerns:

  * The prompt format and answer-extraction logic (pure, easy to unit-test).
  * The orchestrator ``run_mmlu_pro_eval`` that spawns a llama-server
    (or reuses one via ``--base-url``) and scores a holdout sample by sample.

The holdout itself is built by ``scripts/build_mmlu_pro_holdout.py``, which
samples N test questions per subject and pulls K shots from the dev split.
The format is documented at :func:`load_holdout`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

from quant_tuner.eval.server import running_server
from quant_tuner.eval.toolcall import Sampling, call_model

_LETTERS = "ABCDEFGHIJ"  # MMLU-Pro maxes out at 10 options


# ---------------------------------------------------------------------------
# Holdout schema
# ---------------------------------------------------------------------------


@dataclass
class MmluProSample:
    """One MMLU-Pro question, stripped to the fields the eval needs."""

    question_id: int | str
    subject: str
    question: str
    options: list[str]
    answer: str              # canonical answer letter, e.g. "B"
    answer_index: int        # 0-based index into options
    cot_content: str = ""    # ground-truth chain-of-thought (used only by shots)

    @classmethod
    def from_dict(cls, d: dict) -> MmluProSample:
        return cls(
            question_id=d["question_id"],
            subject=d["subject"],
            question=d["question"],
            options=list(d["options"]),
            answer=d["answer"],
            answer_index=int(d["answer_index"]),
            cot_content=d.get("cot_content", "") or "",
        )

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "subject": self.subject,
            "question": self.question,
            "options": self.options,
            "answer": self.answer,
            "answer_index": self.answer_index,
            "cot_content": self.cot_content,
        }


@dataclass
class MmluProHoldout:
    """Holdout file: a few-shot pool per subject + the actual test samples."""

    subjects: list[str]
    n_per_subject: int
    n_shot: int
    seed: int
    shots: dict[str, list[MmluProSample]]
    samples: list[MmluProSample]

    @classmethod
    def from_dict(cls, d: dict) -> MmluProHoldout:
        return cls(
            subjects=list(d["subjects"]),
            n_per_subject=int(d["n_per_subject"]),
            n_shot=int(d["n_shot"]),
            seed=int(d["seed"]),
            shots={
                subj: [MmluProSample.from_dict(s) for s in samples]
                for subj, samples in d["shots"].items()
            },
            samples=[MmluProSample.from_dict(s) for s in d["samples"]],
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


def load_holdout(path: Path) -> MmluProHoldout:
    """Load an MMLU-Pro holdout JSON file."""
    with path.open() as f:
        return MmluProHoldout.from_dict(json.load(f))


def save_holdout(holdout: MmluProHoldout, path: Path) -> None:
    path.write_text(json.dumps(holdout.to_dict(), indent=2))


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert taking a multiple-choice exam. For each question, "
    "consider every option carefully and respond with only the letter of "
    "the best answer (A, B, C, …)."
)


def format_question_block(sample: MmluProSample) -> str:
    """Render one question as a single user-turn string (no answer)."""
    lines = [f"Subject: {sample.subject}", f"Question: {sample.question.strip()}"]
    lines.append("Options:")
    for i, opt in enumerate(sample.options):
        lines.append(f"({_LETTERS[i]}) {opt}")
    return "\n".join(lines)


def build_messages(
    target: MmluProSample,
    shots: list[MmluProSample],
    *,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
) -> list[dict]:
    """Build the OpenAI chat-completions messages for a few-shot prompt.

    Each demonstration shot becomes a user turn (the question) followed by an
    assistant turn (the bare answer letter). The final user turn is the target
    question; the model is expected to respond with just the answer letter.
    """
    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    for shot in shots:
        msgs.append({"role": "user", "content": format_question_block(shot)})
        msgs.append({"role": "assistant", "content": f"Answer: {shot.answer}"})
    msgs.append({"role": "user", "content": format_question_block(target)})
    return msgs


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


# Regex preferences, applied in order. Each pattern requires the captured
# letter to be either parenthesized or followed by a word boundary, so a
# lowercase ``i`` from a stray "is" cannot masquerade as an answer.
#   1. "Answer: X" / "Answer = (X)" — explicit, delimiter required
#   2. "(X)" anywhere — lettered choice
#   3. "answer is X" / "the answer is (X)"
#   4. Standalone capital letter — "A" and "I" excluded (ordinary English
#      words: "A common approach…", "I think…"); they are only accepted by
#      the letter-only-line pattern below.
#   5. A line consisting solely of one letter ("B", "(A)", "I.")
_ANSWER_PATTERNS = [
    re.compile(r"(?i)answer\s*[:=\-]\s*\(?([A-J])\)?(?=\W|$)"),
    re.compile(r"\(([A-J])\)"),
    re.compile(r"(?i)answer\s+is\s+\(?([A-J])\)?(?=\W|$)"),
    re.compile(r"\b([B-HJ])\b"),
    re.compile(r"(?m)^\s*\(?([A-J])\)?\s*\.?\s*$"),
]


def parse_answer(text: str, n_options: int) -> str | None:
    """Extract the predicted answer letter from a completion.

    ``n_options`` caps which letters are valid (so a 4-option question
    rejects an "F" prediction). Returns the predicted letter or ``None`` if
    no in-range letter could be extracted.

    Within each pattern the LAST in-range match wins: reasoning models revise
    themselves ("Answer: C? No, wait… Answer: B"), and the completion fed in
    here includes the chain-of-thought, so the final occurrence is the
    model's actual answer.
    """
    if not text:
        return None
    valid = set(_LETTERS[:n_options])
    for pat in _ANSWER_PATTERNS:
        for m in reversed(list(pat.finditer(text))):
            if m.group(1).upper() in valid:
                return m.group(1).upper()
    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_one(
    client: OpenAI,
    target: MmluProSample,
    shots: list[MmluProSample],
    sampling: Sampling,
    *,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    model_name: str = "local",
) -> dict:
    """Run one MMLU-Pro question through the model and score it."""
    messages = build_messages(target, shots, system_prompt=system_prompt)
    try:
        resp = call_model(client, messages, tools=[], sampling=sampling,
                          model_name=model_name)
        msg = resp.choices[0].message
        content = msg.content or ""
        # Reasoning models (e.g. Qwen3/DeepSeek-style) emit <think>…</think>;
        # llama-server splits that into `reasoning_content` and leaves `content`
        # empty until thinking closes. Concatenate so parse_answer can find the
        # trailing "Answer: X" regardless of which field it lands in.
        reasoning = getattr(msg, "reasoning_content", None) or ""
        combined = (reasoning + "\n" + content) if reasoning else content
    except Exception as e:
        return {
            "question_id": target.question_id,
            "subject": target.subject,
            "truth": target.answer,
            "pred": None,
            "correct": False,
            "raw": "",
            "error": str(e),
        }
    pred = parse_answer(combined, n_options=len(target.options))
    return {
        "question_id": target.question_id,
        "subject": target.subject,
        "truth": target.answer,
        "pred": pred,
        "correct": pred == target.answer,
        "raw": combined[-400:],
    }


# ---------------------------------------------------------------------------
# Aggregation + orchestrator
# ---------------------------------------------------------------------------


@dataclass
class MmluProSummary:
    """Aggregate metrics returned by :func:`run_mmlu_pro_eval`."""

    model: str
    n_total: int
    n_correct: int
    accuracy: float
    by_subject: dict[str, dict] = field(default_factory=dict)
    n_unparseable: int = 0
    per_sample: list[dict] = field(default_factory=list)


def _aggregate(model_label: str, per_sample: list[dict]) -> MmluProSummary:
    by_subj: dict[str, list[dict]] = {}
    for r in per_sample:
        by_subj.setdefault(r["subject"], []).append(r)
    by_subject_summary = {
        subj: {
            "n": len(rs),
            "correct": sum(1 for r in rs if r["correct"]),
            "accuracy": sum(1 for r in rs if r["correct"]) / len(rs) if rs else 0.0,
            "unparseable": sum(1 for r in rs if r["pred"] is None),
        }
        for subj, rs in by_subj.items()
    }
    n_total = len(per_sample)
    n_correct = sum(1 for r in per_sample if r["correct"])
    n_unparseable = sum(1 for r in per_sample if r["pred"] is None)
    return MmluProSummary(
        model=model_label,
        n_total=n_total,
        n_correct=n_correct,
        accuracy=(n_correct / n_total) if n_total else 0.0,
        by_subject=by_subject_summary,
        n_unparseable=n_unparseable,
        per_sample=per_sample,
    )


def run_mmlu_pro_eval(
    holdout_path: Path,
    *,
    model_path: Path | None = None,
    base_url: str | None = None,
    sampling: Sampling | None = None,
    model_label: str | None = None,
    model_name: str = "local",
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    ctx: int = 8192,
    ngl: int = 99,
    server_log_path: Path | None = None,
    server_startup_timeout: float = 120.0,
    chat_template_kwargs: str | None = None,
    per_sample_log: Path | None = None,
    progress: bool = False,
    api_key: str = "sk-no-key",
) -> MmluProSummary:
    """Score every sample in the holdout and return aggregate accuracy.

    Either ``model_path`` (spawn a server) **or** ``base_url`` (reuse one)
    must be provided. Sampling defaults to greedy (``temperature=0``); at
    higher temperatures the multi-rep runner can compute mean ± stdev across
    independent evaluations.
    """
    if (model_path is None) == (base_url is None):
        raise ValueError("provide exactly one of model_path or base_url")

    sampling = sampling or Sampling()
    holdout = load_holdout(holdout_path)
    label = model_label or (model_path.name if model_path else "remote")
    n_shot = holdout.n_shot

    log_fh = per_sample_log.open("w") if per_sample_log is not None else None

    def _run_against(url: str) -> MmluProSummary:
        client = OpenAI(base_url=url, api_key=api_key)
        rows: list[dict] = []
        for i, target in enumerate(holdout.samples, 1):
            shots = holdout.shots.get(target.subject, [])[:n_shot]
            row = score_one(client, target, shots, sampling,
                           system_prompt=system_prompt, model_name=model_name)
            rows.append(row)
            if log_fh is not None:
                log_fh.write(json.dumps(row) + "\n")
            if progress:
                mark = "✓" if row["correct"] else ("?" if row["pred"] is None else "✗")
                print(
                    f"[{i}/{len(holdout.samples)}] {target.subject:20s} "
                    f"truth={target.answer} pred={row['pred']} {mark}",
                    flush=True,
                )
        return _aggregate(label, rows)

    try:
        if base_url is not None:
            return _run_against(base_url)
        with running_server(
            model_path, ctx=ctx, ngl=ngl,
            log_path=server_log_path,
            startup_timeout=server_startup_timeout,
            chat_template_kwargs=chat_template_kwargs,
        ) as url:
            return _run_against(url)
    finally:
        if log_fh is not None:
            log_fh.close()


def render_summary(summary: MmluProSummary) -> str:
    """Format a summary for human-readable stdout."""
    lines = [
        "=" * 60,
        f"Model: {summary.model}",
        f"  Overall accuracy:  {summary.n_correct}/{summary.n_total}  "
        f"({summary.accuracy:.3f})",
    ]
    if summary.n_unparseable:
        lines.append(f"  Unparseable:       {summary.n_unparseable}")
    if len(summary.by_subject) > 1 or list(summary.by_subject) != [summary.model]:
        lines.append("\n  Per-subject:")
        for subj in sorted(summary.by_subject):
            s = summary.by_subject[subj]
            lines.append(
                f"    {subj:20s} {s['correct']}/{s['n']}  ({s['accuracy']:.3f})"
                + (f"  [{s['unparseable']} unparseable]" if s['unparseable'] else "")
            )
    return "\n".join(lines)
