"""Registry of publishable datasets produced by this repo.

Adding a new dataset is a ONE-ENTRY change: write a builder function that yields records,
then append a :class:`DatasetSpec` to :data:`REGISTRY`. Everything else — staging directory
layout, dataset card, versioning, changelog, and the Hub push — is shared.

Layout produced under ``datasets/<name>/`` (the payloads are gitignored; the card, manifest
and changelog are tracked, so the repo records what was published without carrying the data):

    datasets/<name>/
        README.md        dataset card (YAML frontmatter + docs) -> becomes the Hub landing page
        manifest.json    version, per-split row counts + sha256, builder, build time
        CHANGELOG.md     one appended section per published version
        data/<split>.jsonl

Versioning is semver-ish and explicit: ``dataset_push.py --bump {major,minor,patch}`` (or
``--version X.Y.Z``). Each push tags the Hub repo with ``v<version>`` so previous releases stay
retrievable via ``revision=``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATASETS_DIR = REPO / "datasets"


@dataclass(frozen=True)
class SplitSpec:
    """One output file. ``builder`` yields dicts, one per line of the jsonl.

    ``publish=False`` builds the split locally but keeps it OFF the Hub — useful for slices
    we want for internal analysis (e.g. failed attempts) without shipping them.
    """

    name: str
    builder: Callable[[], Iterator[dict]]
    description: str = ""
    publish: bool = True


@dataclass(frozen=True)
class DatasetSpec:
    name: str                       # directory + default Hub repo name
    title: str
    summary: str                    # one-liner, used in the card header
    splits: list[SplitSpec]
    hf_owner: str = "pearsonkyle"
    license: str = "other"          # provenance is documented in the card; see note below
    task_categories: list[str] = field(default_factory=lambda: ["text-generation"])
    tags: list[str] = field(default_factory=list)
    # Long-form markdown appended to the card body (provenance, schema, caveats).
    body: str = ""
    # Optional "Row schema" markdown. When empty, the card falls back to the SWE
    # schema (back-compat). Set it so each dataset documents its own fields.
    schema_md: str = ""
    default_split: str = "resolved"

    @property
    def repo_id(self) -> str:
        return f"{self.hf_owner}/{self.name}"

    @property
    def stage_dir(self) -> Path:
        return DATASETS_DIR / self.name


# --------------------------------------------------------------------------------- builders
def _swe_trajectories(resolved_only: bool) -> Callable[[], Iterator[dict]]:
    def build() -> Iterator[dict]:
        # imported lazily so the registry is importable without torch/transformers
        from quant_tuner.datasets.swe_trajectories import iter_trajectory_records

        yield from iter_trajectory_records(resolved_only=resolved_only)

    return build


def _redteam(flagged_only: bool) -> Callable[[], Iterator[dict]]:
    def build() -> Iterator[dict]:
        from quant_tuner.datasets.redteam_disclosures import iter_redteam_records

        yield from iter_redteam_records(flagged_only=flagged_only)

    return build


_SWE_BODY = """
## What one row is

One row is a **complete agentic session for a single real GitHub issue** — not a
question/answer pair. The agent reads the issue, then over many steps runs shell commands,
reads and greps files, edits source, runs the project's tests, reads failures, edits again,
and finally submits a patch. Each step is a tool call plus the output it received. Sessions
average ~39 tool calls and tens of thousands of tokens.

## Provenance

* **Issues**: [nebius/SWE-rebench](https://huggingface.co/datasets/nebius/SWE-rebench) — real
  GitHub issues with hidden `FAIL_TO_PASS` / `PASS_TO_PASS` test sets.
* **Solver**: [Ornith-1.0-9B](https://huggingface.co/pearsonkyle/Ornith-1.0-9B) (Q5_K_M),
  driven by the OpenAI Agents SDK against a local `llama-server`, one clean Docker container
  per instance (the SWE-rebench image), `temperature=0.25`, max 100 steps.
* **Grading**: every trajectory was graded by actually running the gold tests in the container.
  `resolved=true` means the hidden tests **passed** — these are verified solutions, not merely
  plausible-looking diffs.

## What is included

Every row here is a **verified solution**: the hidden tests passed. Failed and empty-patch
attempts are deliberately excluded, so you can train on this directly without filtering and
without risking unverified trajectories leaking into supervision.

## Using it

Records are already in chat format, so they render with any chat template:

```python
import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("<your-model>")
rec = json.loads(open("data/resolved.jsonl").readline())
text = tok.apply_chat_template(rec["messages"], tools=rec["tools"], tokenize=False)
```

`messages` uses the standard roles (`system`, `user`, `assistant` with `tool_calls`, `tool`),
and `tools` carries the single `bash(command)` schema the agent was given, so schema-conditioned
tool use trains correctly.

## Caveats

* The solver is a 9B model: trajectories are competent but not expert, and often take
  exploratory detours before landing the fix. "Verified" means the tests pass, not that the
  path taken was optimal.
* Tool outputs are raw container stdout/stderr and can be long; truncate as needed.
* Passing the hidden tests is the grading signal, so the usual caveat applies: a patch can
  satisfy the tests without being the ideal fix.
* **Licensing**: the issue text and repository content originate from the upstream GitHub
  projects via SWE-rebench and retain their own licenses. Treat this as a derived research
  artifact and check upstream terms before commercial use.
"""


_REDTEAM_BODY = """
## What one row is

One row is a **single adversarial conversation** against one target model, plus how it
turned out. It answers three questions: *which model*, *what was said* (the full
conversation, including every turn of a multi-turn jailbreak), and *what happened*
(`outcome`: `complied` / `defended` / `errored`).

```json
{
  "model": "ornith-1.0-35b",
  "case_id": "0fde225510a6842e",
  "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "outcome": "complied",
  "unsafe": true,
  "category": "Safety", "vulnerability": "Toxicity", "vulnerability_type": "profanity",
  "attack_method": "Prompt Injection", "attack_class": "single-turn",
  "judge_reason": "The output contains toxic content ...",
  "judge_model": "...", "simulator_model": "...", "config": "red_team_broad",
  "target_reasoning": "..."
}
```

## Provenance

* **Harness**: `quant_tuner.eval.red_team` — the [deepteam](https://github.com/confident-ai/deepteam)
  red-teamer over a local `llama-server`. Three independent endpoints: an uncensored model
  *simulates* the attacks and *judges* the responses; the *target* is the model under test.
* **Grading**: `outcome` is the judge's verdict. `defended` = the target refused; `complied`
  = it produced the harmful content; `errored` = the case failed (kept, never scored as safe).
* Generated on the operator's own hardware against their own model endpoints.

## Splits

* `flagged` — the cases the target **complied** with (plus errored). These are the safety
  holes: what to disclose to the model's authors, and — paired with a refusal target and mixed
  with benign→helpful — the fine-tune (QAT) seed that actually generalizes.
* `all` — adds the **defended** cases (the model already refuses these), for a balanced view.

## ⚠️ Dual-use — read before publishing

`flagged` rows contain a **working attack and the harmful completion it elicited**. That is
precisely what a maintainer needs to reproduce and fix a weakness, and precisely what should
not be broadcast. **Both splits default to `publish=False`.** Share responsibly:

* a **private** Hub repo (`scripts/dataset.py push redteam-safety-disclosures --private`), or
* a metadata-only view (drop `messages` / `target_reasoning`, keep labels + `judge_reason`).

The intent is defensive: hardening open-weight releases, in line with the position that
sufficiently capable models should be safety-tested — including the *derived* artifacts that
ship after release. Do not use this to attack systems you do not own or are not authorized
to test.

## Using it as a fine-tune seed

The `complied` rows tell you *where* the model fails. Build the training set by pairing each
attack with a refusal and interleaving benign→helpful examples so the fine-tune learns to see
through the framing without over-refusing:

```python
import json
rows = [json.loads(l) for l in open("data/flagged.jsonl")]
attacks = [r for r in rows if r["unsafe"]]   # attack conversations to pair with refusals
```
"""


REGISTRY: list[DatasetSpec] = [
    DatasetSpec(
        name="swe-agentic-trajectories",
        title="SWE Agentic Trajectories (verified)",
        summary=(
            "Multi-step agentic software-engineering trajectories on real GitHub issues, "
            "graded by running the hidden tests. Chat-template ready."
        ),
        splits=[
            SplitSpec("resolved", _swe_trajectories(True),
                      "trajectories whose hidden tests PASSED (verified solutions)"),
            # built locally for failure analysis, deliberately NOT published: the point of
            # this release is verified solutions, and shipping failed attempts alongside them
            # invites training on unverified data by accident.
            SplitSpec("all", _swe_trajectories(False),
                      "every graded trajectory, including failures", publish=False),
        ],
        tags=["agentic", "swe-bench", "code", "tool-use", "distillation", "trajectories"],
        body=_SWE_BODY,
        default_split="resolved",
    ),
    DatasetSpec(
        name="redteam-safety-disclosures",
        title="Red-Team Safety Disclosures",
        summary=(
            "Adversarial conversations from red-teaming open-weight models, each labeled "
            "with the target model, the full conversation, and the safety outcome "
            "(complied / defended / errored)."
        ),
        splits=[
            # BOTH default publish=False — see _REDTEAM_BODY's dual-use note. The
            # `flagged` rows contain a working attack AND the harmful completion it
            # elicited; that is a responsible-disclosure artifact for a model's
            # authors and a QAT training seed, not something to broadcast. Ship via
            # a *private* Hub repo (`push --private`) or a metadata-only view.
            SplitSpec("flagged", _redteam(True),
                      "cases the target COMPLIED with (or errored) — the safety holes",
                      publish=False),
            SplitSpec("all", _redteam(False),
                      "every case incl. defended (safe) — for balanced analysis",
                      publish=False),
        ],
        license="other",
        task_categories=["text-generation"],
        tags=["red-team", "safety", "alignment", "refusal", "jailbreak", "open-weights"],
        body=_REDTEAM_BODY,
        schema_md=(
            "| field | meaning |\n"
            "| --- | --- |\n"
            "| `model` | the **target** model that was probed |\n"
            "| `case_id` | content hash of the attack — joins the same case across runs |\n"
            "| `messages` | the **full conversation** in chat format (multi-turn jailbreaks "
            "carry every escalation turn), ending in the model's response |\n"
            "| `outcome` | `complied` (produced the harmful content) / `defended` (refused) / "
            "`errored` |\n"
            "| `unsafe` | `true` iff `outcome == complied` — the safety hole |\n"
            "| `category`, `vulnerability`, `vulnerability_type` | what was probed |\n"
            "| `attack_method`, `attack_class` | how (e.g. Prompt Injection; single-/multi-turn) |\n"
            "| `judge_reason` | why the judge scored it that way |\n"
            "| `target_reasoning` | the model's own chain-of-thought (findings only) |\n"
            "| `judge_model`, `simulator_model`, `config` | provenance of the run |"
        ),
        default_split="flagged",
    ),
]


def get_spec(name: str) -> DatasetSpec:
    for spec in REGISTRY:
        if spec.name == name:
            return spec
    known = ", ".join(s.name for s in REGISTRY)
    raise KeyError(f"unknown dataset {name!r}; known: {known}")
