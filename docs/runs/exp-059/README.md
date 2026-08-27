# exp-059 — the scaled coder-QAT campaign (coder1 → coder2 → coder3)

Goal: a coder version of Ternary-Bonsai-8B trained on ~80M tokens of multi-language,
multi-agent SWE trajectories (sft_v2) with the settled anchor10 prescription, to better
use tools and reasoning. Three runs, two mid-flight interventions, one completed
artifact. Per-run detail in kd32b-full-coder{1,2,3}/notes.md. **To start the next run,
read SPINUP.md — it is the complete restart path.**

## The arc in one table

| run | corpus | accum | fate | what it taught |
|---|---|---|---|---|
| coder1 | universal-v2 + raw sft_v2 (97M tok) | 1 | killed @905/2929 | 55% /testbed monoculture → model learns the path VALUE, not the prompt lookup. CE/probe blind to it; only the agentic sidecar saw it. |
| coder2 | + path-diversified (coder-v2p) | 1 | killed @~830/2974 | Diversification fixed the value; the deeper failure was TOTAL DRIFT — flips 9.1% @s800 vs anchor10's ~2%/run. LR can't drop (flip floor ~3e-4) → the lever is optimizer-step count. |
| coder3 | same as coder2 | 4 | **completed 743/743** | Accum-4 restores the validated ~600-step optimizer trajectory. Healthiest telemetry of the campaign; artifact grounded + mostly loop-free but never self-concludes when stuck. |

## Campaign laws (each cost GPU-days to learn)

1. **Bench agentically DURING training.** Masked-CE, val, and the stop probe were all
   uninformative or misleading three times. scripts/qat_benchwatch.sh (every ~200
   accum-steps: CPU export → probe → 3 graded T=0.7 episodes) is now standing
   infrastructure. The counters that matter: testbed_cmds, unique-command ratio,
   repeat streaks, edit/pytest commands, self-termination, out_tok vs cap.
2. **Scaling data 5x at fixed LR ≈ 5x total drift.** Ternary LR is pinned by the flip
   floor (~3e-4), so scale optimizer-step count DOWN with GRAD_ACCUM as data scales up.
   Target anchor10's ~600 steps: accum ≈ n_windows / 600.
3. **Accum needs GradOffload on this card** (train.py, --accum-offload auto): grads
   resident across micro-batches cost +2.1 GiB over the accum-1 peak = 2 OOMs.
   Also QAT_LOGIT_CHUNK=512 + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
4. **Trajectory data carries environment constants** (checkout roots, scaffold quirks).
   Census them BEFORE training (path census in SPINUP.md); diversify what is constant
   (scripts/path_diversify_sft.py — deterministic, provenance in corpora/).
5. **Judge checkpoints only at probe-clean moments and only at T=0.7.** The s200 GGUF
   snapshotted a 25-step steering-corrected transient; the chain's T=0.25 mimic
   reproduces the known sharpening loop even for anchor10.
6. **Resolved-only trajectories teach "never give up".** coder3 explores coherently for
   60 turns and never concludes. Next-arc candidate: a conclude-when-stuck curriculum
   slice (truncated episodes ending in a summarize-and-stop turn).
7. **Path literal typos at long context ("swe-micic") are the 2-bit fidelity signature**
   — improved with decay (s400→s600→final) but not eliminated at 743 steps.

## Where things stand (2026-08-27)

- **Artifact**: out/exp-057/Ternary-Bonsai-8B-coder3-Q2_0.gguf (2.03 GiB) + final
  latents out/exp-059/kd32b-full-coder3/trained_latents.pt (28 GB, on-disk only).
- Full eval: stop probe textbook; in-dist after_tool_call median 0.95; T=0.7 mimic =
  grounded, no loops >6, but 3/3 hit the turn cap with no patch (see coder3 notes).
- **Not yet done**: multi-instance eval (SWE-rebench holdout50 w/ Docker) for a real
  pass_rate — the single-instance mimic is a smoke test every model fails.
- Next-arc options (deliberately not started): (a) conclude-when-stuck curriculum +
  rerun; (b) rep-bank refresh from coder3's own harvested loop states
  (trajectories/ here is the harvest material); (c) multi-instance eval first.

## What is in this directory

- kd32b-full-coder{1,2,3}/ — train logs (incl. OOM logs), run_configs, metrics,
  telemetry CSVs, report.html, notes.md per run.
- corpora/ — corpus build logs (raw + path-diversified) and the sft_v2_pathdiv
  provenance README (SHA-256s, determinism, regenerate command).
- ops/ — the exact chain/watcher scripts as run (chain.sh…chain4.sh, benchwatch,
  sidecar, report-watch launcher) + the full chain.log. Generalized, tracked versions:
  scripts/qat_benchwatch.sh, scripts/qat_sidecar_bench.sh, scripts/run_kd_anchor_qat.sh.
- eval/ — benchwatch + sidecar CSVs, the arc mimic CSV, anomaly analysis.
- trajectories/ — every CODER* mimic trajectory/patch/result (harvest material).

Not tracked (sizes): latents, GGUFs, KD tables, corpus .pt files. Rebuild paths in
SPINUP.md; the KD table is ~9.5 h GPU, everything else is minutes.
