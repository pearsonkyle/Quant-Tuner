# coder3 — COMPLETED 743/743 (accum 4, path-diversified corpus) — the artifact

TAG=coder3 GRAD_ACCUM=4 on coder2's corpus + KD table: 743 optimizer steps over
all 2,974 windows ≈ anchor10's validated ~600-step optimizer trajectory with 4x
tokens per update. Needed two trainer changes (both committed):
GradOffload (7f98c65 — accum keeps ~28 GB of grads resident through
micro-batches; CPU-stash restores the accum-1 peak, bit-exact) and
QAT_LOGIT_CHUNK env. Peak 91.4-91.6 GiB, 313 s/step, zero OOMs after that.

Healthiest telemetry of the campaign:
- loss 1.90→0.51, KD KL →0.27, val CE monotonically 0.79→0.62 (coder2: 1.5+ and
  climbing at equal data), flips 2.8/1.7% (the anchor10 band), gnorm calm.
- ONE termination event: step-200 probe sentence_period 0.12 / newline 0.099
  (PROBE-WARN strike 1/2) — the steering hinge fired (st=4.09 that step) and
  the probe was fully back in band by 225. The s200 GGUF snapshots the wobble
  (probe 0.144, mute episode) — evidence for "export BETWEEN probe intervals
  can capture a transient"; disregard s200 as an artifact.
- benchwatch trend across decay: s400 12% unique cmds / streak 46 / ctx-death →
  s600 56% unique / streak 11 / first edit+pytest cmds / 1 of 3 self-terminated.

Final artifact out/exp-057/Ternary-Bonsai-8B-coder3-Q2_0.gguf (2.03 GiB):
- GGUF stop probe textbook (diagnostics ≤4e-5, after_tool_call 0.99997).
- In-dist stop: after_tool_call median 0.95 (vanilla 0.99, p10 0.78 vs 0.98).
- Mimic @ T=0.25 (chain default): loop 35x — SAME as anchor10 at T=0.25; not a
  regression, the 2x2's known sharpening collapse. Judge at T=0.7 only.
- Mimic @ T=0.7 x3: grounded (0 /testbed), 70% unique commands, streaks ≤6,
  ZERO edit commands, 3/3 hit the 60-turn cap — never concludes on its own.
  Two residual gaps: (1) never-terminates-when-stuck is plausibly ON-distribution
  for a resolved-only corpus (training episodes grind until success); (2) path
  literal typos ("swe-micic") = long-context fidelity at 2 bits.

Verdict: telemetry-healthy, grounded, mostly loop-free; NOT yet a better agent
than anchor10 on the smoke (anchor10-T07 self-terminates; coder3-T07 doesn't).
