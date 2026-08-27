
## 2026-08-22 16:20 — the stop anchor, pre-registered before launch

**What the read-outs say so far** (real held-out corpus, 40 sampled stop targets,
2048 tokens of context each; `scripts/gemma4_stop_table.py`):

| arm | commit | P(stop)@stop | elsewhere | ratio |
|---|---|---|---|---|
| shipped | 35.0% | 0.3452 | 1.7e-04 | 2054 |
| untrained-ternary | 15.0% | 0.2418 | 6.4e-04 | 379 |
| dense-ft | 2.5% | 0.1327 | 1.1e-03 | 123 |
| ce-only | 0.0% | 0.0265 | 2.9e-04 | 91 |
| self-KD | 2.5% | 0.0788 | 9.1e-04 | 87 |
| self-KD + stop-weight 5.5 | 5.0% | 0.0781 | 4.3e-05 | 1798 |

Two conclusions I had not drawn cleanly before:

1. **The commitment deficit is not quantization-specific.** A *dense* fine-tune on this
   corpus lands at 2.5%, the same place ternary self-KD lands. Whatever destroys the
   commitment does it to a bf16 model too. (`dense-sw5.5` in the running sweep is still
   the pre-registered falsification and still gets to run — I am not dropping a test
   because I think I know its answer.)
2. **stop-weight is the wrong lever, and the sweep already proved it.** 5.5 bought the
   discrimination and no commitment; 16 bought neither and cost the model its learning
   (val 2.7072 -> 2.6457, essentially flat, vs 5.5's 2.4918 -> 2.0478).

**The teacher is not the problem.** Measured on the corpus the student trains on
(`kd/stop_signal_e4bself.json`): the self-KD table's teacher assigns mean **0.530** at
the 6,300 real stop targets and 0.00053 at the other 6,118,196 rows — ratio 993. The
signal is in the table. It is 0.103% of the rows, so the KL average never feels it.

**Hypothesis.** The lever that fits is the one already wired and left inert:
`--stop-anchor` at 0.2. Its own telemetry says the mechanism works and the gain is
~100x short — `an` climbs 0.0008 -> 0.15-0.47 across 60 steps (the student drifting off
the teacher's stop level, hinge engaging) while contributing 0.04 against a loss of
1.2-2.0.

The anchor differs from stop-weight in exactly the way that matters here: it is
**one-sided and saturating**. It pushes only until the student is within `margin_hi`
(0.1 nat) of the *teacher's own* log P(stop) at that position and then goes silent, so a
large beta converges to shipped behaviour instead of overshooting into a stop-happy
model. beta=8 puts the initial contribution (~1.6) alongside CE.

**Pre-registered criteria** for `anchor8-sw5.5` and `anchor8` (60 steps each):
- **Pass**: commitment >= 15% (the untrained-ternary level — i.e. training no longer
  destroys the stopping policy) with val masked-CE <= 2.10 and ratio >= 1000.
- **Partial**: commitment 8-15% at val <= 2.10. Worth carrying into the full stage with
  a larger beta.
- **Fail**: commitment < 8%, or val > 2.15. Then the anchor is not the lever either, and
  the honest next move is the corpus (gemma renders a whole tool exchange as ONE model
  turn -> 1 stop target per 972 supervised tokens; splitting those turns is a corpus fix,
  not a loss fix).
Second arm changed before launch from `anchor8` (anchor alone, to see whether its
continue-side hinge retires stop-weight) to **`anchor25-sw5.5`**. beta=8 is one
estimate's first guess and the value it replaces was 40x too weak, so bracketing beta
is worth more tonight than attributing a hyperparameter: if 8 is still too weak, 25
answers it in the same night; if 8 works, 25 says whether more helps or hurts.

Capability result is unchanged and still the headline: at matched training, ternarizing
six layers costs nothing measurable (held-out masked CE dense 1.7796 vs ternary 1.7290).

### The corpus fallback above is wrong — checked before proposing it

I wrote that if the anchor fails, the next move is the corpus, on the theory that gemma
renders a whole tool exchange as one model turn. **It does not.** Decoded from the packed
corpus (`ids`/`labels`, window 0):

```
... | GGUF at 4.75M params | **10.2 MB** ... |<turn|>\n<|turn>user\nhelp
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^.....      <- 106 labeled
... overhead |<turn|>\n<|turn>user\n...smaller??!<turn|>\n<|turn>model\n<|tool_call>
                                                 ^ not labeled ^ labeled again
```

The template stops at every assistant turn including tool calls, the corpus labels
exactly those, and it masks the `<end_of_turn>` that closes a user/tool turn — 13,273
id-106 tokens present, **6,300 labeled, 6,973 correctly masked**. So 1 stop per 972
supervised tokens is not a rendering defect; it is what this data's assistant turns
actually average, and re-balancing it by up-weighting would teach the model to stop more
often than the dialect calls for.

That is the argument for a saturating objective rather than a weighted one, and it is
why `--stop-anchor` (converges to the *teacher's* per-position level and then goes
silent) is the right shape where `--stop-weight` (pushes forever) is not. If the anchor
fails at beta=8 the next move is a larger beta, not a different corpus.

## 2026-08-22 16:30 — paused, GPU released

Stopped for the GPU to be used elsewhere; resume notes in `docs/gemma4_ternary/HANDOFF.md`.

`a75-sw5.5-lr2e-4` was killed at step 21/60. The trainer caught the signal and saved
cleanly (`trained_latents.step21.pt` with flip telemetry: ~2.06% flips at layer 8
q_proj, density 58.5 -> 58.3%, scale-drift 1.43%), but 21 steps is not comparable to the
60-step arms — **treat the arm as not run**. Its question (does kd-alpha 0.75 pull
commitment toward the dense distribution?) is superseded by the anchor arms.

The anchor arms (beta 8 and 25) were queued and never started. They are the first thing
to run on the new machine; criteria are pre-registered above and unchanged.

## 2026-08-22 16:45 — sw16's read-out landed after the pause, and it corrects me

`sw16-lr2e-4: commit 17.5%  P(stop)@stop 0.2706  elsewhere 6.45e-04  ratio 419`

I had written that stop-weight 16 "bought neither" half. **That is wrong as stated** — it
read the highest commitment of any trained arm, above the 15% bar I pre-registered for
the anchor. But it is not the win the number looks like, and the reason is visible in
every other column.

All four of its stop metrics land on **untrained-ternary**, not partway between:

| | commit | P(stop)@stop | elsewhere | ratio |
|---|---|---|---|---|
| untrained-ternary | 15.0% | 0.2418 | 6.38e-04 | 379 |
| sw16 | 17.5% | 0.2706 | 6.45e-04 | 419 |

At n=40 probe points, 17.5% vs 15.0% is 7 targets vs 6. Nothing here is distinguishable
from the model before training.

**What it actually did was refuse to learn.** Validation is unweighted masked CE
(`run_validation` passes no `weights=`), so the numbers are comparable across arms, and
sw16's barely moved: 2.7072 → 2.6457, against sw5.5's 2.4918 → 2.0478. It also flipped
20–25% fewer codes (layer-0 q_proj 2.28% vs sw5.5's 2.72%).

### The real finding: commitment and capability trade off monotonically on this corpus

| arm | final val CE | flips (L0 q_proj) | commit |
|---|---|---|---|
| untrained-ternary | — | 0% | 15.0% |
| stop-weight 16 | 2.6457 | 2.28% | 17.5% |
| stop-weight 5.5 | 2.0478 | 2.72% | 5.0% |
| self-KD (100 steps) | 1.9948 | 4.14% | 2.5% |
| dense fine-tune | 1.7796 | (dense) | 2.5% |
| CE only | 1.7290 | 2.65% | 0.0% |

**The ordering by val CE is the exact reverse of the ordering by commitment**, and it
does not care about the grid — the dense fine-tune sits right next to CE-only. Fitting
this corpus destroys the stopping policy in proportion to how well you fit it. Every
"lever" tried so far, stop-weight included, has only moved the model along this curve;
stop-weight 16 bought its commitment by declining to train.

### This sharpens the anchor criterion, and it is the version to use

The pass bar is no longer "commitment ≥ 15%" — sw16 clears that trivially by
under-training. It is:

> **commitment ≥ 15% AND val masked-CE ≤ 2.10** — i.e. *off* the curve, not further
> along it. Anything that buys commitment by raising val CE is not a result.

This is exactly what a saturating anchor should be able to do and a weight cannot: it
adds force only at stop targets and only until the student reaches the teacher's own
level there, so it has no mechanism for degrading the model's general fit the way a 16×
CE reweighting does. If the anchor also only walks the curve, the honest conclusion is
that this corpus and this objective cannot both be satisfied, and the next move is a
mixed objective (e.g. holding the shipped model's full distribution at stop targets
while fitting content elsewhere), not another scalar.
