# Broad-domain calibration / MTP supplement

Hand-authored plain-text corpus spanning the topic range this repo's calibration data
misses. It exists because the current calibration mix — agentic tool-call logs plus
`wiki.test.raw`, soon plus a 20-language agentic trajectory set — is narrow in a
specific way: heavy on code and tool-calling, heavy on encyclopedic prose, and thin on
the *working* registers of technical domains (derivations, datasheet reasoning,
notation, diagnostic dialogue).

Target: **1,000,000+ tokens.** Track progress with `scripts/build_supplement.py stats`.

Language coverage is deliberately **not** a goal here — the 20-language trajectory
dataset covers that axis. This corpus is English, and spends its budget on topic and
register breadth instead.

## Two consumers, disjoint halves

The corpus serves two purposes that must not share samples:

- **Calibration breadth** — extra distribution for `llama-imatrix` / AWQ / GPTQ
  alongside the logs and wiki.
- **MTP training** — next-token training data for a multi-token-prediction draft head.

`build_supplement.py build` splits every source file's samples 50/50 into
`supplement.calib.txt` and `supplement.mtp.txt`. The split is:

- **at sample granularity** (blank-line-delimited blocks),
- **stratified per file**, so both halves carry every subject in proportion,
- **seeded per file path**, so adding a new file never reshuffles the assignment of an
  existing one. A global shuffle would migrate samples across halves on every batch and
  silently contaminate anything already trained or calibrated on.

The build asserts the two halves are disjoint and records a sha256 of each in
`supplement_manifest.json`.

### Disjointness caveat vs. `../mmmu/`

`calibration_supplements/mmmu/combined.txt` is currently used as the **AWQ cv-gate
validation** corpus (`scripts/exp040_release_jackrong27b.py`,
`scripts/exp054_qwythos.py`). This tree is separate from it and neither half should be
pointed at an eval that also uses the mmmu files. Per the repo invariant in CLAUDE.md,
re-check every eval that touches a slice before repurposing it.

## Taxonomy

Nine areas, balanced. Each subject file is ~6-8k tokens. **Tier 1** is the foundational
sweep (complete); **tier 2** extends each area toward the 1M-token target with applied,
adjacent, and cross-cutting subjects.

| Area | Tier 1 (complete) | Tier 2 (extension) |
| ---- | ----------------- | ------------------ |
| `data_science_ml/` | statistics and inference, deep learning, classical ML, LLMs and transformers, MLOps and serving, data engineering, time series, experimentation, feature engineering, recommender systems, computer vision, reinforcement learning | NLP beyond LLMs, graph ML, anomaly detection, causal ML, bayesian modeling, model interpretability, speech and audio ML, ML systems on-device |
| `math/` | linear algebra, calculus and analysis, probability theory, optimization, numerical methods, discrete math, differential equations, information theory, graph theory | abstract algebra, topology and geometry, mathematical logic, complex analysis, category theory, cryptographic math |
| `physics/` | classical mechanics, electromagnetism, thermodynamics, quantum mechanics, relativity, optics and photonics, condensed matter, fluid dynamics, nuclear and particle | acoustics and waves, plasma physics, biophysics, computational physics, metrology and units |
| `astronomy_space/` | observational astronomy, astrophysics, planetary science and exoplanets, cosmology, spacecraft engineering, orbital mechanics, remote sensing, mission operations | radio astronomy, high-energy astrophysics, space environment, launch systems, planetary defense |
| `software_web/` | frontend, backend and APIs, databases, distributed systems, devops, networking, security, performance engineering, testing, type systems | compilers and interpreters, operating systems, computer architecture, algorithms and data structures, API and system design, developer experience |
| `embedded_hardware/` | microcontrollers, digital electronics, analog electronics, signal processing, sensors, PCB design, robotics and control, FPGA and HDL | power electronics, RF and wireless, communication buses, motor drives, embedded security, mechanical design for electronics |
| `generative_art/` | creative coding, shaders and GPU graphics, procedural generation, diffusion models, audio synthesis, color and design theory, computational geometry | physical computing and installation, typography and layout, animation and motion, data visualization, interactive systems |
| `earth_life_sciences/` | biology and genetics, chemistry, geology, climate science, ecology, materials science, neuroscience | physiology and medicine, immunology and disease, biochemistry, oceanography, atmospheric science, agriculture and food systems |
| `humanities_business/` | history, literature and rhetoric, economics, finance, psychology, law and policy, philosophy of science | management and organizations, marketing and strategy, accounting, sociology, education and pedagogy, ethics, negotiation and decision-making |

## Format spec

Inherited from `../mmmu/README.md`, with one addition (heading merge).

- Plain UTF-8, LF endings, no tabs, trailing newline.
- **Samples are separated by exactly one blank line.** The whole file is consumed as a
  continuous token stream.
- Each file opens with a `====`-fenced header block. The builder strips it — it is
  metadata for a human reader, not a training sample.
- Content is organized into `## Subject` sections. A bare `## Heading` line is merged
  into the block that follows it, so the heading travels with its content through the
  split.
- Within each section, blocks rotate through four registers:
  1. **Expository prose** — how a practitioner explains the concept.
  2. **Exam-style Q&A** — A-D multiple choice plus worked reasoning and an answer.
  3. **Domain notation / structured data** — the symbolic content unique to the domain:
     equations, register maps, tables, config blocks, formula references.
  4. **Agentic tool-call transcript (~20%)** — a short multi-turn `[user]/[assistant]/
     [tool]` session solving a domain task, matching the real CLI-log distribution.
- **No raw special tokens** (`<|im_start|>`, `<s>`, `<start_of_turn>`, …). These files
  are used for PPL/KLD eval as well as calibration, and `llama-perplexity` has no
  `--parse-special`, so an embedded marker tokenizes as plain BPE on one stack and as a
  control token on the other. `build_supplement.py lint` enforces this.

## Commands

```bash
python3 scripts/build_supplement.py stats -v      # per-subject token accounting
python3 scripts/build_supplement.py lint          # format checks
python3 scripts/build_supplement.py build --out out/supplement

# exact token counts instead of the 3.7 chars/token estimate
python3 scripts/build_supplement.py stats --tokenizer /path/to/hf/snapshot
```
