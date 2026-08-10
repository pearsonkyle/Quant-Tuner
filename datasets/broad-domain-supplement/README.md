---
license: cc-by-4.0
task_categories:
- text-generation
- question-answering
tags:
- calibration
- quantization
- imatrix
- awq
- gptq
- mtp
- speculative-decoding
- instruction-tuning
- synthetic
- multi-domain
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: corpus
    path: data/corpus.jsonl
  - split: instruct
    path: data/instruct.jsonl
---

# Broad-Domain Calibration & Instruction Supplement

~1M tokens of hand-authored text across 192 subjects in 9 areas, built to serve three jobs from one source: quantization calibration, MTP draft-head training (on a disjoint half), and light instruction tuning.

**Version `0.1.0`** · built 2026-08-09T17:45:53

| split | rows | tokens~ | size | contents |
| --- | ---: | ---: | ---: | --- |
| `corpus` | 5,536 | 969,606 | 5.4 MB | raw authored samples + provenance; carries the calib/mtp `half` label |
| `instruct` | 5,536 | 1,084,399 | 6.4 MB | the same samples as chat-format prompt/response pairs |

### Topic distribution

| area | subjects | samples | tokens~ | share |
| --- | ---: | ---: | ---: | ---: |
| `data_science_ml` | 27 | 861 | 148,041 | 15.3% |
| `software_web` | 24 | 699 | 118,985 | 12.3% |
| `humanities_business` | 24 | 633 | 116,425 | 12.0% |
| `math` | 22 | 614 | 106,933 | 11.0% |
| `physics` | 21 | 616 | 106,368 | 11.0% |
| `embedded_hardware` | 19 | 571 | 99,046 | 10.2% |
| `earth_life_sciences` | 19 | 520 | 97,693 | 10.1% |
| `generative_art` | 19 | 539 | 90,941 | 9.4% |
| `astronomy_space` | 17 | 483 | 85,174 | 8.8% |
| **total** | 192 | **5,536** | **969,606** | 100% |

### Sample registers

| register | samples | share |
| --- | ---: | ---: |
| `prose` | 4,245 | 76.7% |
| `table` | 971 | 17.5% |
| `qa` | 224 | 4.0% |
| `transcript` | 96 | 1.7% |

### Disjoint halves

Every row carries `half`, a **deterministic, non-overlapping** assignment (see below). Filter on it; do not re-split.

| half | samples | intended use |
| --- | ---: | --- |
| `calib` | 2,704 | quantization calibration (imatrix / AWQ / GPTQ) |
| `mtp` | 2,832 | MTP draft-head training |

## What this is for

A quantization calibration corpus is only as good as its coverage: `llama-imatrix`, AWQ and
GPTQ all decide *which weights matter* from the activations a corpus produces, so whatever the
corpus never exercises gets quantized on the assumption that it does not matter. The usual
mixes — wiki text plus whatever logs happen to be available — are narrow in a way that is
invisible until the quant is worse at something the corpus never covered.

This is a deliberately **broad** supplement, hand-authored across 9 areas and 192 subjects, to
sit alongside a domain corpus rather than replace it. It was written to serve three jobs from
one source:

1. **Quantization calibration** — feed the `calib` half to `llama-imatrix` / AWQ / GPTQ.
2. **MTP draft-head training** — train on the `mtp` half, next-token.
3. **Light instruction tuning** — the `instruct` split, already in chat format.

## The two halves are disjoint, on purpose

Every corpus row carries `half`, either `calib` or `mtp`. The assignment is deterministic and
the two sets **never overlap**. This matters for a specific reason: a draft head trained on the
same text used to calibrate the quant it drafts for would show an inflated acceptance rate,
because part of what you would be measuring is memorization rather than draft quality. Keeping
them apart is what makes an MTP acceptance number mean something.

The split is seeded **per source file**, not globally, so adding new subjects later never
reshuffles the existing assignment — anything already calibrated or trained on stays valid.

## How it was written

Hand-authored, one file per subject, in four deliberately mixed registers so the activation
statistics are not all from one kind of text:

| register | what it is |
| --- | --- |
| `prose` | expository explanation under a section heading |
| `table` | indented term/definition reference blocks — dense, low-redundancy token patterns |
| `qa` | exam-style question with options, reasoning, and a stated answer |
| `transcript` | short illustrative `[user]` / `[assistant]` / `[tool]` dialogues |

There are **no raw chat-control tokens** anywhere in the text (`<|im_start|>` and friends are
linted against). That is deliberate: `llama-perplexity` has no `--parse-special`, so a marker
embedded in the text tokenizes as a control token on one stack and as plain BPE on the other,
which quietly makes PPL/KLD numbers incomparable. This corpus is safe to use as an eval file.

## Using it

**Calibration corpus** — write the `calib` half out as flat text:

```python
from datasets import load_dataset

ds = load_dataset("pearsonkyle/broad-domain-supplement", split="corpus")
calib = ds.filter(lambda r: r["half"] == "calib")
with open("corpus.broad.calib.txt", "w") as f:
    f.write("\n\n".join(calib["text"]))
```

```bash
llama-imatrix -m model-F16.gguf -f corpus.broad.calib.txt -o imatrix.gguf -c 4096
```

Interleave it with your in-domain corpus rather than concatenating: a token-budgeted
calibrator samples the file, and a large block at the head can eat the whole budget.

**MTP draft-head training** — the disjoint half, next-token:

```python
mtp = ds.filter(lambda r: r["half"] == "mtp")
text = "\n\n".join(mtp["text"])      # ~500k tokens
```

**Instruction tuning** — already chat-shaped:

```python
from transformers import AutoTokenizer

inst = load_dataset("pearsonkyle/broad-domain-supplement", split="instruct")
tok = AutoTokenizer.from_pretrained("<your-model>")
rendered = tok.apply_chat_template(inst[0]["messages"], tokenize=False)

# authored prompts only (the question was written as a question, not templated):
authored = inst.filter(lambda r: r["prompt_source"] == "authored")
```

**Filtering by topic** — every row carries `area` and `subject`:

```python
ml = ds.filter(lambda r: r["area"] == "data_science_ml")
```

## Read this before using `instruct`

The `instruct` split's prompts come from two different places and the difference matters:

* **`prompt_source: "authored"`** (~6%) — the `qa` and `transcript` rows. The question or user
  turn was *written as a prompt*. These are genuine instruction data.
* **`prompt_source: "templated"`** (~94%) — the `prose` and `table` rows. The source text was
  written as continuous exposition, and the prompt is **generated** from the section heading
  and subject using a small set of templates. The *responses* are hand-written; the *questions*
  are not.

Templated prompts are fine for light instruction tuning and for teaching a model to answer
topically on demand. They are repetitive by construction, and a model trained on them heavily
will learn the template. If you want prompt diversity, filter to `authored`, rewrite the
prompts, or mix this with a real instruction set — do not treat all 5.5k rows as if a person
wrote 5.5k distinct questions. This is stated plainly because a dataset that quietly presents
templated prompts as authored ones is the kind of thing that is discovered later, in results.

## Caveats

* **Token counts are estimates.** `est_tokens` uses a measured 3.70 chars/token ratio, not a
  real tokenizer. Expect a few percent of drift; recount with your own tokenizer if it matters.
  Two figures in the table differ for real reasons rather than by mistake: `instruct` totals
  *more* than `corpus` because it counts the generated prompts as well as the responses, and
  both sit slightly under the ~1.0M raw-file figure because section headers, the per-file
  metadata block, and blank separator lines are not part of any sample.
* **Single author, single voice.** One person wrote all of it, so it is stylistically
  consistent in a way a scraped corpus is not. Good for controlled calibration, and it means
  the corpus does not represent stylistic diversity — do not use it to measure that.
* **Breadth over depth.** Each subject is a competent overview at roughly 5k tokens, not
  expert-level treatment. It is written to exercise vocabulary and reasoning patterns across
  many domains, which is what calibration needs; it is not a reference text.
* **`transcript` tool calls were never executed.** They are illustrative dialogues written to
  look like tool use, kept as literal assistant text rather than lifted into a structured
  `tool_calls` field, because presenting authored text as a captured trace would be misleading.
* No claim is made that the content is error-free. It is a written corpus, not a verified one.

## Row schema

Shared by both splits:

| field | meaning |
| --- | --- |
| `id` | stable content hash of the sample |
| `area`, `subject` | directory-level topic and subject file (e.g. `physics` / `quantum_information`) |
| `area_title`, `subject_title` | human-readable forms |
| `section` | the `## Section` heading the sample sits under |
| `register` | `prose` / `table` / `qa` / `transcript` — how it is written |
| `half` | **`calib` or `mtp` — disjoint.** Filter on this; do not re-split |
| `source_file` | path within `calibration_supplements/broad/` |
| `n_chars`, `est_tokens` | size; tokens are a 3.70 chars/token **estimate** |

`corpus` split adds:

| field | meaning |
| --- | --- |
| `text` | the sample as authored, section heading included |

`instruct` split adds:

| field | meaning |
| --- | --- |
| `messages` | chat-format turns (`user` / `assistant`, plus `tool` for transcripts) |
| `prompt_source` | `authored` (the prompt is from the source) or `templated` (generated from the heading — see the note above) |
| `n_turns` | message count |

## Reproducing

Generated with [Quant-Tuner](https://github.com/pearsonkyle/Quant-Tuner); see
`docs/ternary_qat.md` for the end-to-end pipeline and
`src/quant_tuner/datasets/` for the exact builder used to publish this.
