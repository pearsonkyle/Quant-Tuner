# SWE-rebench mimic harness (Docker-free)

The agentic eval + episode-harvest harness used throughout the ternary-QAT arc
(anchor6–anchor10). It lived at `/workspace/swe-mimic/` on the training box; this
copy makes the work reproducible on a new machine. **It is a smoke test, not
SWE-rebench** — see the header of `run_agent.py` for exactly what differs from
the official containerized harness and why the numbers are not comparable.

## Setup on a new machine

```bash
mkdir -p /workspace/swe-mimic && cp tools/swe_mimic/* /workspace/swe-mimic/
cd /workspace/swe-mimic
python3 -m venv .venv && .venv/bin/pip install openai-agents openai datasets requests
# Stage the eval instance (dask) + the 6 harvest instances:
bash setup_instance.sh instance.json          # → work/<instance_id>/ (clone @ base_commit, venv, test_patch, golden gate)
bash setup_harvest.sh                          # → the 6 non-eval harvest workspaces
```

Instance specs come from `nebius/SWE-rebench`; `instance.json` is the recorded
dask eval instance. The harvest pool (12rambau__sepal_ui-814, ASPP__pelita-875,
Azure__pykusto-159, CS-SI__eodag-490, DataDog__datadogpy-625, Duke-GCB__lando-194)
is deliberately **disjoint from the eval instance** — episodes harvested there feed
training (`scripts/build_rep_traj_contexts.py`), never grading.

## The three entry points

- `run_agent.py` — one episode: OpenAI Agents SDK → local llama-server → local repo
  workspace, graded by the instance's own `test_cmd` over recorded F2P/P2P ids. The
  golden gate (F2P fail + P2P pass before the agent runs) aborts with exit 2 on a
  broken workspace instead of fabricating a score.
- `run_all_quants.sh` — the eval sweep over a set of GGUFs (one llama-server each).
  `TEMP=0.7` env passes `--temperature`; the anchor10 verdict was measured with the
  2×2 {anchor9,anchor10}×{T0.25,T0.7} sweep here. Serve winners at **T=0.7,
  top_p=0.95, no repeat/presence penalties**.
- `gen_harvest.sh` — bare-model episode generation over the harvest workspaces
  (no penalties: loops are the harvest target). Output trajectories
  (`work/.../traj_<LABEL>-*.json`) feed `scripts/build_rep_traj_contexts.py`.

`harvest_results.csv` / `swe_mimic_ternary.csv` are the recorded results of the
published arc (kept for the write-up).
