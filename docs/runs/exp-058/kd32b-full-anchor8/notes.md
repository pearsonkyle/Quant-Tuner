## anchor8 — repetition steering trained where the escalation lives

Single-variable change from anchor7: `--steer-rep-k 2,3,4,5 --steer-rep-cap 0.45
--steer-rep-weight 0.1`. Everything else identical (32B forced-stop table, anchor
1.0/0.1, steer 0.1, clip 0.25, lr 5e-4 group-scaled, patience-2 guards).

Why: anchor7's rp= read 0.0000 all run — at the v1 contexts (k=1) the model sits at
~0.33 P(repeat), under the 0.5 cap, so the loss never fired; the mimic then looped
59x. Measured escalation on anchor7 latents: 0.33→0.52 over k=1..5 (vanilla flat
~0.35). The cap moves to vanilla's own level and the contexts to k=2-5: penalize the
escalation, not repetition per se.

Success = rp= NONZERO early (the hinge must actually fire now) and trending down; the
same 24/24 probe record; and the bare-mimic loop shorter or gone. Post-run, re-run
scripts/measure_repeat_prob.py — anchor8 should be flat-in-k like vanilla. Serving
fallback if partial: --repeat-penalty 1.3 --repeat-last-n 2048 (NO presence-penalty —
measured: presence mutes anchor7 to zero tool calls; repeat-penalty alone gives the
clean 5-step episode).
