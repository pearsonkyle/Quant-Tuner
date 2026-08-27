# coder2 — STOPPED at step ~830/2974 (accum 1, path-diversified corpus)

Same recipe, corpus rebuilt from path-diversified data (coder-v2p: /testbed
1,763→217 convs, ~uniform over 11 roots, grounding preserved). New KD table
(coder2_32b_topk64_fs151645, 25.2M positions, coverage 0.998).

- benchwatch s400: episodes ACT but flail — 58/60 commands into /testbed *in
  spite of* the diversified corpus, plus "/test/repos/repo" (a blended
  hallucination of training roots). Grounding of the VALUE fixed; in-context
  copying itself degraded.
- benchwatch s800: near-mute — ≤2 tool calls per episode, 8k-token rambles,
  "cd /workspace/swe-micode". Worse than s400: monotone decline in-plateau.
- The decisive number: **flips 9.1%** in the leading tensor at s800 vs
  anchor10's ~2% for a WHOLE run; val 1.5-1.7 and climbing. Total code drift
  ~5x the validated envelope. LR cannot drop (flip floor ~3e-4 — below it codes
  never flip), so the drift lever is optimizer-step COUNT → grad accumulation.

Killed at ~830. Its corpus + KD table are the coder3 inputs (unchanged).
s400/s800 GGUFs + trajectories preserved; latents deleted.
