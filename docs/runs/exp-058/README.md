# exp-058 run records (ternary-QAT anchor arc)

Small, tracked copies of the per-run measurement artifacts for anchor6–anchor10
(the big latents/checkpoints are deleted or live only on the training box; the
GGUF deliverables are `out/exp-057/Ternary-Bonsai-8B-{vanilla,anchor9,anchor10}-Q2_0.gguf`).
Each run dir holds: `notes.md` (design + verdict), `run_config.json`, `train.log`,
`metrics.jsonl`, `telemetry/` (parsed steps/flips/stop-probe/val CSVs),
`report.html` (rendered by `scripts/qat_report.py`), and — from anchor9 on —
`rep_measure.json` (P(repeat) vs k on the real-material bank).

**anchor10 is the shipped recipe** (see CLAUDE.md "working recipe" and
`docs/ternary_qat_curriculum.md` — the repetition arc). Its `notes.md` carries the
final VERDICT and the both-levers 2×2.

`inputs/` holds the exact training inputs the anchor10 command expects at
`out/exp-058/kd/` on a new machine (copy them there, or point `REP_BANK`/`REP_TRAJ`
env at these paths):
- `rep_bank.json` — 200 real (tool_call, response) pairs + 60 tasks from the corpus
  (`scripts/build_rep_bank.py`)
- `rep_traj_contexts.jsonl` — 36 harvested full-prefix loop contexts from 6 instances
  (`tools/swe_mimic/gen_harvest.sh` → `scripts/build_rep_traj_contexts.py`)
- `teacher_probe_32b.json` — the 32B teacher's stop-probe asymptotes
  (`scripts/teacher_stop_probe.py`)

Not tracked (too big, rebuildable): the 32B forced-stop KD table
`out/exp-058/kd/ourssft_32b_topk64_fs151645.pt` (2.2 GB — rebuild with
`qat/kd_precompute.py` from the corpus + `SWE-Lego/SWE-Lego-Qwen3-32B`), the SFT
corpus tensors (`scripts/build_sft_qat_corpus.py`), and the trained latents.
