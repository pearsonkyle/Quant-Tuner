# coder1 — STOPPED at step 905/2929 (accum 1, raw sft_v2 corpus)

First scaled coder run: anchor10 recipe unchanged, corpus = qwen3-universal-v2 +
raw sft_v2 (97.0M tok, 2,929 windows @ 32768), 32B forced-stop KD table, accum 1.

- Launch OOM (fragmentation, 7 GiB reserved-unallocated) → fixed with
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (train.oom1.log).
- Telemetry looked "fine-ish" (val plateau 1.4-1.5, probe textbook) but the
  **s800 CPU sidecar bench caught the real state**: 3 episodes, 3 failure modes —
  /testbed fixation against an explicit "there is NO /testbed" prompt (23 steps,
  17 nonzero), an 8k-token no-tool-call ramble, garbled path literals
  ("/workspace/swe-mirir.py"). Probe stayed textbook throughout — the probe and
  the agent trajectory are each blind in one direction (curriculum's law, again).
- Root cause (measured): 55% of sft_v2 conversations command into /testbed
  (all 1,668 sweagent rows), ~30% into /workspace (all openhands rows). The
  checkout root is a near-constant VALUE, so the student learns the value, not
  the prompt-lookup. → scripts/path_diversify_sft.py (f78c060).

Killed at step 905; step-800 GGUF + trajectories preserved (trajectories/,
eval/sidecar_coder1... rows in swe_mimic CSVs). Latents deleted.
