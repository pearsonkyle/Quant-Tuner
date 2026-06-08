# MMMU-category calibration / eval supplements

Domain-knowledge text corpora keyed to the **MMMU** taxonomy (6 disciplines / 30 subjects;
see MMMU, arXiv 2311.16502, https://mmmu-benchmark.github.io/). MMMU is multimodal and
exam-derived; here we adapt only its **taxonomy** to plain text — images that the real
benchmark would show are rendered as textual `Figure:` descriptions.

These files broaden calibration/eval beyond the code-only `calibration_supplement.txt` at
the repo root, which has no coverage of non-code knowledge domains.

## Files

One file per MMMU discipline (this round):

| File | Subjects covered |
| ---- | ---------------- |
| `art_and_design.txt` | Art, Art Theory, Design, Music |
| `business.txt` | Accounting, Economics, Finance, Management, Marketing |
| `science.txt` | Biology, Chemistry, Geography, Math, Physics |
| `health_and_medicine.txt` | Basic Medical Science, Clinical Medicine, Diagnostics & Lab Medicine, Pharmacy, Public Health |
| `humanities_and_social_science.txt` | History, Literature, Sociology, Psychology |
| `tech_and_engineering.txt` | Agriculture, Architecture & Engineering, Computer Science, Electronics, Energy & Power, Materials, Mechanical Engineering |

Each file is ~12–15k tokens.

### Future 30-subject expansion

Use a per-discipline subdirectory, one file per subject:

```
calibration_supplements/mmmu/<discipline>/<subject>.txt
# e.g. tech_and_engineering/computer_science.txt
```

## Format spec

- Plain UTF-8. **Samples are separated by a blank line** (matches the `\n\n` join in
  `data/split.write_corpus`). The whole file is consumed as one continuous token stream by
  imatrix / AWQ / GPTQ and by `llama-perplexity` — there is no line-delimited requirement.
- Each file opens with a short header block (discipline + subjects covered).
- Content is organized into `## Subject` sections. Within each section, blocks rotate
  through four formats:
  1. **Expository prose** — textbook-style concept explanation (natural language).
  2. **Exam-style Q&A** — A–D multiple choice + worked reasoning + answer (MMMU shape),
     with textual `Figure:` descriptions where the benchmark would show an image.
  3. **Domain notation / structured data** — the symbolic content unique to the domain
     (chemical equations, financial tables, harmony notation, formulae, terminology, …).
  4. **Agentic tool-call transcript (~20%)** — a short multi-turn `[user]/[assistant]/[tool]`
     session that solves a domain task via tools, matching the real CLI-log distribution.
- **Do not embed raw special tokens** (`<|im_start|>`, `<|im_end|>`, etc.). Agentic samples
  use a readable role-tagged + JSON-tool-call representation so the same file is safe for
  PPL eval (no special-token skew) and for calibration breadth.

## Usage

### Per-category eval (primary)

Run the bench once per discipline; each produces per-category BPW/KLD/PPL vs the F16 ref:

```bash
uv run quant-tuner bench \
    --quant out/run/gguf/Q4_K_M.gguf \
    --reference out/run/gguf/F16.gguf \
    --eval calibration_supplements/mmmu/science.txt \
    --out out/run/eval_mmmu/science.csv
```

### Calibration breadth (secondary)

A file may be pointed at a recipe's `data.supplement`. **Disjointness caveat:** per the
repo's invariant (CLAUDE.md), eval must not overlap calibration. Do not use the *same* MMMU
file as both supplement and eval in one run — keep the root `calibration_supplement.txt` as
the calibration-breadth file and reserve these for eval by default.
