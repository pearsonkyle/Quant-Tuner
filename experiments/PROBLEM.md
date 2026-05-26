# improving quantization performance

- **Created:** 2026-05-25T17:04:13+00:00
- **Last updated:** 2026-05-25 (after exp-002 + outlier extension w/ HF↔GGUF mapping fix)

## Problem statement

Quantizing to Q4_K_M loses signal unevenly across tensors. Calibrating
with an importance matrix (imatrix) recovers some of that signal, but
the size of the recovery — and whether it transfers to actual downstream
task quality — depends on three knobs we control:

1. **The calibration corpus** (what activation distribution we record).
2. **The re-weighting method** (how we combine the recorded statistics
   into a per-tensor importance score).
3. **The granularity** at which the importance is expressed
   (per-channel vs per-quantization-block).

Working on the **Qwen3.5-9B family** (`Qwen/Qwen3.5-9B`,
`Tesslate/OmniCoder-9B`, `Jackrong/Qwopus3.5-9B-Coder`), we want to
understand which of these knobs actually matters, on which metric,
and where the imatrix idea hits its ceiling.

## Shared methodology

- **Target quant**: Q4_K_M (~5.0 BPW). Bit-rate is held fixed for the
  whole research so the imatrix story is isolated from the bit-rate
  story.
- **Bench metrics**: BPW, file size, PPL, mean/median KLD, same_top_p
  (the fraction of tokens where the quant's top prediction matches F16).
  Driven by `quant_tuner.bench.runner.bench_one` with `suite="kld"`.
- **Task-level metrics** (added in exp-002): `tool_selection_acc`,
  `param_acc_mean`, `schema_valid_rate`, `continuation_type_match_rate`,
  `recovery_appropriate_rate`. Driven by `eval.toolcall` via
  `scripts/run_toolcall_reps.py` (5 reps × 25-session holdout, stochastic
  sampling at T=0.6 / top_p=0.95 / top_k=20). Reported both as
  **mean ± stdev** (expected single-rollout performance) and **pass@5**
  (per-turn best-of-5, the deployment-with-retries scenario;
  retroactively computable via `scripts/compute_toolcall_passat5.py`).
- **Data invariants** (from CLAUDE.md, preserved across every experiment):
  `logtrain.jsonl` is split 80/10/10 with seed=42 into `train` / `test` /
  `holdout`. Calibration uses `train`, KLD eval uses `holdout`, tool-call
  eval uses `test`. The three slices stay disjoint.
- **Re-weighting "vanilla"** = whatever `llama-imatrix` writes (E[a²]
  per input channel, no post-processing). Re-weighting "output-aware"
  = the `hybrid_custom` variant in `calibrate/imatrix.py`
  (`max(L1(E[a²]), L1(‖W‖²·E[a²]))`).

## Headline findings to date

These are the cross-experiment takeaways. Read the per-experiment
README for the underlying numbers.

1. **Any imatrix beats no imatrix on distribution-shape metrics.**
   Calibration (any corpus, any variant) buys ~40% lower KLD and ~+2.5
   pts same_top_p vs an uncalibrated Q4_K_M.
2. **But task-level eval flips this for schema-valid JSON.** Every
   imatrix variant tested **regresses** `schema_valid_rate` relative
   to no calibration — vanilla custom −3.4 pts, wiki −1.9 pts,
   `output_aware` −5.6 pts. KLD averages over the whole token
   distribution and hides this; structural tokens like `{`, `}`, `"`,
   key names are rare and high-leverage, and the imatrix appears to be
   spending bit-budget elsewhere.
3. **Corpus identity is nearly inert** (exp-001). Custom (`logtrain.jsonl`
   + `calibration_supplement.txt`) and wiki (`wiki.test.raw`) differ by
   ≤ 2% on KLD and ≤ 0.15 pts on same_top_p across all three models.
4. **`output_aware` is a wash or worse on every metric** we can measure
   (exp-002). It marginally improves PPL, slightly worsens KLD, and
   either ties or trails vanilla on all four task-level metrics.
   Original hypothesis (that output-contribution `‖W[:,c]‖²·E[a²]`
   captures channels vanilla misses) is **refuted on this model at
   this bit-rate**.
5. **PPL is the wrong primary metric here.** The `none` row has the
   *lowest* PPL on every model (exp-001) and the *highest*
   `schema_valid_rate` (exp-002), but the *worst* KLD and same_top_p.
   PPL alone would point you at the artifact that's worst by every
   other signal.
6. **`outlier_l4` is the best variant tested on every distribution-shape
   metric, and the best imatrix variant on schema_valid.** Full-mapping
   numbers on Jackrong: KLD 0.499 (best), same_top_p 90.66 (best),
   schema_valid 0.928 mean (best among imatrix cells; vs `none` 0.939,
   vanilla 0.905, output_aware 0.883). Closes ~70% of the schema_valid
   regression that vanilla imatrix introduced on single-rollout. The
   heavy-tail signal (`√E[a⁴]`) hypothesis is **supported**.
6b. **The verdict depends on the metric definition.** On per-turn pass@5
   (best-of-5 deployment scenario, computed retroactively via
   `scripts/compute_toolcall_passat5.py`), outlier_l4 reaches
   schema_valid 0.948 — narrowly above `none` (0.943) and the best of
   any variant. Interpretation: outlier_l4 produces a wider distribution
   of correct outputs across stochastic samples than `none` does, so
   with retries it actually wins, but per-rollout reliability still
   trails by ~1 pt on single-rollout. The pass@5 numbers also crown
   `outlier_l4` (best schema_valid pass@5) and `wiki` (best param_acc
   pass@5), refining the single-rollout picture.
8. **F16 is not the task-level ceiling on every metric.** F16 schema_valid
   (mean 0.911, pass@5 0.906) is *below* `none` Q4 (0.939 / 0.943),
   `outlier_l4` (0.928 / 0.948), and `wiki` (0.920 / 0.947). F16
   continuation_match (0.929) is below every Q4 cell except `output_aware`.
   F16 only wins `param_acc` (mean 0.340, pass@5 0.382 — the best of all
   models). Interpretation: at T=0.6 / top_p=0.95, F16's sharper output
   distribution occasionally samples low-probability structural tokens
   that break JSON validity; quantization noise acts as accidental
   regularization on this workload. **Consequence**: matching F16 on
   distribution shape (which is what KLD/same_top_p measure) is not the
   same as maximizing downstream task quality. The right deployment
   target is the artifact that wins the metric you care about, not the
   one that best mirrors F16.
7. **The HF↔GGUF mapping for `models/hf_gguf_map.py` was incomplete for
   Qwen3-Next-style hybrid blocks** (`linear_attn.in_proj_*` /
   `out_proj`). The first outlier build hooked only 128/248 tensors
   (52%); after adding 5 regex rules, coverage is 100% non-SSM and
   every `outlier_l4` metric improved (KLD −1.0%, same_top_p +0.18 pts,
   schema_valid +1.4 pts). Worth checking the same fix gives 100%
   coverage on `Qwen/Qwen3.5-9B` and `Tesslate/OmniCoder-9B` before
   running outlier variants on them.

## Active hypothesis

The original schema-validity hypothesis ("can we keep KLD gains
without losing schema_valid?") is substantively closed: `outlier_l4` +
the HF↔GGUF mapping fix produces a Q4_K_M GGUF that ties-or-beats
`none` on schema_valid pass@5 (0.948 vs 0.943), is the **best on KLD,
same_top_p, and schema_valid pass@5 among all variants tested**, and
trails on single-rollout schema_valid by ~1 pt rather than 3-6.

The fp16 baseline finding (#8 in Headline findings) adds a nuance:
F16 itself is below several Q4 cells on schema_valid and
continuation_match. So "match F16" is not the right deployment target
on this workload. The deployment target should be "win the metric you
actually care about". For tool-call agents where JSON validity matters
most, `outlier_l4` Q4_K_M is the best artifact tested; for argument
fidelity (param_acc), `fp16` still wins.

The current question is whether **the outlier_l4 win generalizes** and
is worth promoting:

> Does the same outlier_l4 + mapping-fix recipe ties-or-beats `none`
> on schema_valid pass@5 for `Qwen/Qwen3.5-9B` and
> `Tesslate/OmniCoder-9B`? If yes, promote it as the default imatrix
> recipe.

Concrete moves (in priority order; full list with rationale in
`experiments/002-imatrix-with-additional-term/README.md` → Next steps):

1. **Generalization check** — verify the mapping fix gives 100%
   coverage on the other two models, then run forward_stats →
   outlier_l4 imatrix → bench → 5-rep tool-call eval for each. ~2 h
   per model.
2. **Promote** if generalization holds — flip
   `src/quant_tuner/recipes/q4_k_m_imatrix.yaml` from `hybrid_custom`
   to `outlier_l4`.
3. **Exp-004 combiner sweep** — now most informative as a test of
   "does combining `E[a²]` with `√E[a⁴]` beat outlier_l4 alone?"
   (one-line edit to use the heavy-tail term as second signal).
4. **fp16 task-level baseline** (in progress, `bjmrbbllp`) — establishes
   the absolute ceiling so all gaps are interpretable as fractions of
   the quantization headroom.

Lower priority now that the original gap is closed:
- **Exp-003 (token sensitivity)** — only interesting on outlier_l4 to
  test heavy-tail convergence stability.
- **Exp-005 (block-aware aggregation)** — keep scaffolded as fallback
  if outlier_l4 fails to generalize.
- **Bit-rate comparison (Q5_K_S vs Q4_K_M)** — was framed as "fallback
  if calibration is capped"; now an independent question worth its own
  experiment number.

## State of experiments

Status as of 2026-05-25. See `research.json` for the canonical state and
each experiment's `README.md` for details.

| #   | Title                                | Scope                          | Status      | Headline result                                 |
|-----|--------------------------------------|--------------------------------|-------------|-------------------------------------------------|
| 001 | imatrix with custom data             | 3 models × 3 corpora            | **done**    | Corpus barely moves KLD/same_top_p; PPL ⟂ KLD   |
| 002 | output-aware re-weighting            | Jackrong + tool-call eval       | **done**    | `output_aware` regresses schema_valid 5.6 pts   |
| 002+| outlier_l4 / outlier_max + mapping fix | Jackrong, extends exp-002     | **done**    | outlier_l4 best on KLD/same_top_p/schema; partial schema rescue (−1.1 vs none) |
| 003 | imatrix token sensitivity            | Jackrong, 3 datasets × 3 tok × 3 ctx | **planned** | _not yet run_                                   |
| 004 | imatrix combiner sweep               | Jackrong, 5 combiner variants   | **planned** | _scaffolded_                                    |
| 005 | block-aware imatrix aggregation      | Jackrong, 5 cells (max/mean × 256/32) | **planned** | _scaffolded_                                    |
| 006 | outlier_max rescue + heavy-tail mixing | Jackrong, 7 cells + diagnostic | **planned** | _scaffolded; diagnostic refutes initial hypothesis — see README_ |

## Reference artifacts (do not regenerate)

These are expensive to produce and are intentionally shared across
experiments — point new experiments at them rather than rebuilding.

| Path | What | Cost to rebuild |
|------|------|-----------------|
| `out/exp-001/{slug}/model-f16.gguf` | F16 GGUF (per model) | HF download + conversion, ~5 min per model |
| `out/exp-001/{slug}/model_extracted/` | Text-only HF dir (per model) | HF download |
| `out/exp-001/{slug}/corpus.train.txt` | 500K-token custom calibration corpus | ~30 s per model |
| `out/exp-001/{slug}/corpus.eval.txt` | 50K-token held-out eval set (KLD) | ~30 s per model |
| `out/exp-001/{slug}/imatrix-custom.gguf` | Vanilla `E[a²]` imatrix on custom corpus | ~19 min per model (llama-imatrix) |
| `out/exp-001/{slug}/imatrix-wiki.gguf` | Vanilla `E[a²]` imatrix on wiki.test.raw | ~10 min per model |
| `out/exp-001/{slug}/baseline.kld` | F16 KLD reference | ~1 min per model |
| `out/exp-001/wiki/wiki.test.raw` | wikitext-2 raw test (~280K tokens) | trivial — HF datasets |
| `out/exp-002/{slug}/imatrix-output_aware.gguf` | output_aware re-weighting of vanilla custom | ~10 s |
| `out/exp-002/{slug}/forward_stats.npz` | E[a⁴] + max|a| from 50K-token forward pass (Jackrong only, full-mapping) | ~2.3 min on MPS |
| `out/exp-002/{slug}/imatrix-outlier_l4.gguf` | `√E[a⁴]` reranking on Jackrong (full-mapping) | ~10 s |
| `out/exp-002/{slug}/imatrix-outlier_max.gguf` | `max|a_c|` reranking on Jackrong (full-mapping) | ~10 s |
| `out/exp-002/toolcall_holdout.jsonl` | 25-session tool-call holdout from test slice | trivial |

`{slug}` = the HF repo ID with `/` replaced by `__` (e.g.
`Jackrong__Qwopus3.5-9B-Coder`).

## Where things live

```
experiments/
├── research.json                    # canonical experiment index (managed by `researcher` CLI)
├── PROBLEM.md                       # this file — read first
└── NNN-<slug>/
    ├── README.md                    # summary, approach, metrics, observations, analysis
    └── REPRODUCE.md                 # env, prereqs, steps, expected output

scripts/
├── run_exp001.py                    # driver (3 models × 3 corpora)
├── run_exp002_outliers.py           # outlier_l4 / outlier_max extension
├── run_exp003.py                    # token sensitivity sweep
├── run_exp004.py                    # combiner sweep
├── run_exp005.py                    # block-aware aggregation
├── build_exp002_toolcall_holdout.py # tool-call holdout from test slice
├── render_exp001_table.py           # CSV → markdown for exp-001
└── render_exp003_tables.py          # CSV → markdown for exp-003
                                     # (run_toolcall_reps.py is pre-existing)

out/
├── exp-001/, exp-002/, ...          # per-experiment workspaces
└── {slug}/                          # per-model artifacts inside each
```

## Workflow (standard scaffold)

1. **Hypothesize** — propose falsifiable statements; record via
   `researcher hypo add`.
2. **Develop** — `git checkout -b exp/NNN-slug`, scaffold the dir,
   write `REPRODUCE.md` as you build, not after.
3. **Test** — run per REPRODUCE; record numbers into the README metrics
   table; flip status via `researcher exp status`.
4. **Analyze** — write *Observations* (descriptive) and *Analysis*
   (interpretive) sections. Cross-reference other experiments' tables.
5. **Iterate** — refine, propose new hypotheses, or stop. Capture the
   decision in *Next steps*.

When you finish an experiment, also update the **State of experiments**
table and **Headline findings** section of this file so the next
session has the current picture.
