## The first complete artifact

anchor6 ran all 613 steps (1.0355 epochs) with a PERFECT probe record: all 24 probes at
exactly 0.0000 diagnostic / 1.0000 control. Config: KD (forced-stop table, alpha 0.5,
tail-bucket KL) + one-sided per-side-margin stop anchor (beta 0.2, 1.0/0.1 nats) +
termination steering (--steer-weight 0.1: 8 probe-family contexts every step) +
--clip-norm 0.25 + lr-scale group-scale + lr 5e-4 + patience-2 guards (never fired).

Final flips 1.07-2.37% across all eight tracked tensors (v_proj alive at 1.07%);
val 0.745 = the dense control's endpoint; loss 2.303 -> 0.405.

Designed from the dense-control decomposition: steering counters the objective's slow
control-face leak with every-step gradient; clip 0.25 damps the waves that leak
through the ternary quantization filter; the anchor pins the stop policy on-corpus.
