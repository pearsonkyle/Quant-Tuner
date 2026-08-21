# Session brief: gemma-4-E4B from-scratch ternarization — stage 1 go/no-go

You are running the FIRST TRAINING TEST of ternarizing `google/gemma-4-E4B-it-qat-q4_0-unquantized`
from scratch (no native ternary weights — the opposite of the Bonsai fine-tune). Everything
you need is staged and documented; your job is to execute stage 1, keep it honest with the
measurement discipline below, and answer one question:

> **Does QAT recover a stage's ternarization damage before the next stage compounds on it?**
> The no-training cumulative damage curve doubles every ~6 layers. If a trained stage-1
> cannot pull its own damage back down, the schedule buys nothing and the honest verdict is
> that fully-ternary E4B is out of reach — write that up and stop.

## Read first, in this order
1. `docs/gemma4_ternary_reproduce.md` — the runbook. §7 is your exact recipe (KD table with
   `--include-ids 106`, teacher probe, stage-1 train command with gemma-scaled aborts).
   §§1–4 are measurements already done — do not redo them, read their numbers.
2. `docs/gemma4_ternary_feasibility.md` — the loss-stack rationale and what transfers from
   Bonsai as MECHANISM vs what must be RE-MEASURED (short answer: every constant).
3. `docs/ternary_qat_curriculum.md`, final section ("anchor7–anchor10: the repetition arc")
   — findings newer than the gemma docs; the transferable ones are restated below.
4. `out/gemma4-ternary/` — staged artifacts: corpus (`corpus_sft_gemma4_32768.pt` + val),
   damage scans (`layer_damage.json` has the stage order), `stop_baseline.json`
   (diagnostic 0.00274 / control 0.070 — YOUR reference lines, not Bonsai's).

## Hard constraints (each one was paid for)
- **`--steer-weight 0` and NO `--steer-rep-*` for gemma.** The steering context classes
  are Qwen-dialect (their control class is *inverted* under gemma's template) and the
  repetition machinery's banks are Qwen-rendered. Port per `PROBE_SPECS` before enabling.
- **Never add a rep/loop teacher-KL** in any dialect: dense teachers assign 0.79–0.99 to a
  verbatim repeat under forcing (in-context pattern continuation) — distilling those
  states teaches copying. Bounded one-sided hinges at the untrained model's own level only.
- **Abort thresholds come from gemma's baseline** (0.03 diagnostic ceiling / 0.01 control
  floor), and gemma's control headroom is only ~25× — treat the control abort as a
  last-ditch floor and verify any control movement against a generated trajectory.
- **`--dense-kind down_proj` stays dense** (KLD 1.20 vs 0.15 for q_proj), and stage 1 is
  layers `0,1,2,3,7,8` from `layer_damage.json["layer_order"]`.
- **fp32 latents/compute, adafactor, group-scale lr, clip 0.25** — lr 5e-4 is Bonsai's
  number and only a first guess: run a 60-step A/B arm before committing a full schedule.
- **Read code-flip telemetry, not loss.** A ternary model only learns by flipping codes;
  scale drift makes loss fall while nothing real changes. The report's flip panels work
  (the Δ-field parser bug is fixed); watch flips per stage next to the damage number.

## Ops rules from the Bonsai arc (all bitten at least once)
- One GPU on this box, shared with another session preparing a large Bonsai run — check
  `nvidia-smi` before every launch and before ANY concurrent torch eval; Bonsai training
  peaked at 91.4/95 GiB, so mid-run benching runs on CPU (see the ckpt-eval sidecar
  pattern in the curriculum doc).
- NEVER edit a `.sh` a live bash is executing (byte-offset corruption; check
  `ps -eo args | grep <script>` first). Bracket pkill/pgrep patterns (`qat[.]train`) and
  keep unbracketed literals out of the same command.
- The runner (`run_kd_anchor_qat.sh`) propagates the trainer's rc; chains must gate on it
  AND on `PROBE-ABORT` in the log. New step-line telemetry fields REQUIRE the
  `parse_qat_log.py` regex + a test in the same commit (six precedents).
- Memory traps already fixed — keep them fixed: span-only lm_head for any per-position
  loss, position-chunked fp32 softmax in KD precompute, unpadded per-row forwards for
  long-context losses (a float mask forces math-path SDPA).
- Launch long work as background chains with a Monitor tailing the log; arm
  `scripts/qat_report_watch.sh <run_dir>` so the run's `report.html` live-updates
  (teacher probe json → dotted asymptotes; `notes.md` → Findings section).

## Deliverables
1. Stage-1 verdict vs the §7 open question, stated with numbers: per-stage damage before
   vs after training (output-space KLD, same probe as `layer_damage.json`), flip
   telemetry, val trend, stop probe vs `stop_baseline.json`.
2. The live `report.html` for the run, `notes.md` with hypothesis + criteria written
   BEFORE launch, and every new script committed with the findings in the message.
3. If stage 1 recovers: proceed to stage 2 (`05,06,36,37,38,39`) with the same
   discipline. If not: one diagnostic iteration (lr A/B, longer stage, or wider dense
   set) before calling it — then the honest write-up either way.

Work autonomously; keep a terse running log in the run's notes.md. The measurement is the
product — a null result cleanly measured is a success.
