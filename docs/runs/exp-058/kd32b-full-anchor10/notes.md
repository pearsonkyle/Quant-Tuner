## anchor10 — harvested-episode repetition steering

Single addition over anchor9: `--steer-rep-traj` — 4 full-prefix contexts per
application, every 4th step, harvested from real bare-model episodes on 6
training-pool instances (36 available, sampled at seed 31). Cap 0.6 unchanged;
bank hinge (k=1..5) kept.

Why: the state-dependence ladder. anchor9 suppressed constructed states completely
(0.08-0.15 at any depth) but the real episode state read 0.96 -> looped 29x; a
truncated real prefix collapses to 0.06, so the full history IS the state. The
harvested contexts read 0.71 mean / 0.99 max on anchor9 — gradient now flows there.

Success = rt= active-then-declining; rp=/probes as anchor9; bare mimic streak << 29
(ideally clean); harvested-context remeasure at/below cap; held-out bank + real-
episode series on the report panel.

## VERDICT (post-run, 2026-08-21)

The 2x2 that closes the repetition arc (all bare-model, single dask episode):

|                       | T=0.25            | T=0.7                                  |
|-----------------------|-------------------|----------------------------------------|
| anchor9 (P_real~0.96) | loop 29x          | loop 11x, 31/48 malformed, server err  |
| anchor10 (P_real<=0.6)| loop 43x          | CLEAN: streak 1, 0 malformed, self-term|

Both levers necessary: the harvested-state hinge caps P(repeat) in real episode
states (verified on the UNSEEN dask state: 0.96 -> 0.53-0.59 flat), and T=0.7 stops
low-temperature sharpening from collapsing onto the argmax anyway. anchor9's T=0.7
arm also shows why capped confidence matters beyond loops: uncapped models need low
T for command syntax (31 malformed at 0.7), which is exactly the regime that loops.

Endpoint quality: best val of the ladder (0.7411), 24/24 probes, KD KL 0.29.
Serving recipe for this model: plain sampling at temperature 0.7, top_p 0.95 — no
repeat/presence penalties needed.
