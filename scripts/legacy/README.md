# Legacy red-team scripts

These are the **original prototypes** the current red-team harness grew out of.
They are committed for provenance and so a checkout on another machine has them,
but they are **superseded** — do not reach for them to reproduce current results.

**To reproduce the results in `docs/benchmarks.md#red-team-safety`, use
`scripts/reproduce_redteam.sh`** (which drives `scripts/eval_redteam.py` +
`scripts/redteam_ladder.py`). That is the maintained path.

| File | What it was | Superseded by |
| --- | --- | --- |
| `red_team_runner.py` | The original **Hydra**-based monolith. Needs `pip install hydra-core omegaconf` (deliberately dropped from the `redteam` extra when the harness was modularized). | `eval/red_team.py` + `scripts/eval_redteam.py` |
| `eval_redteam_frozen.py` | Standalone frozen-bank driver. | `eval_redteam.py --frozen-bank` (built in) |
| `build_redteam_full_summary.py` | Wilson-CI markdown summary of a full sweep. References old `gemma_full_results.csv` / `run_full.log` paths. | `render_summary` / `render_reps_table` in `eval/red_team.py` |
| `redteam_compare.sh` | gemma-vs-gemma shell wrapper (remote vLLM, minimax judge). | `reproduce_redteam.sh` (parameterized) |

Kept as-is (not linted, not tested, not import-clean without the extra deps).
