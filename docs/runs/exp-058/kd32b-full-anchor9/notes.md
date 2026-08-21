## anchor9 — bounded repetition hinge on REAL-material contexts (hinge-only)

Changes from anchor8: rep contexts from the corpus bank (real calls/outputs/tasks,
n=10, k=1..5 — entry states included), cap 0.6 (vanilla's real-material level).
NO rep teacher-KL: the capture measured the 32B at 0.79-0.99 P(verbatim repeat) at
k>=2 and 0.49-0.93 at k=1 — teacher-forced pattern continuation is competent-LM
behavior, so a KL toward it would teach copying. The hinge is bounded: it only fires
above 0.6, so legitimate retries (teacher-endorsed) are untouched; the 0.96-0.98
near-determinism that sustains 56-identical-round loops becomes unreachable.

Why: anchor7/8 escalate to 0.96-0.98 on these states (vanilla flat 0.55) and anchor8
proved synthetic contexts don't transfer (inverted its own curve, moved real states
by 0.02, looped 56x).

Success = rp= hot as escalation develops, then suppressed; 24/24 probes; bank
remeasure capped near 0.6; bare mimic loop shorter/gone. The mimic is closed-loop
ground truth — the probes cannot see sampling dynamics.
