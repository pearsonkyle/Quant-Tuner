## anchor7 — 32B teacher + repetition steering

Two deltas from anchor6 (the first surviving run), everything else identical:

- **KD table from SWE-Lego-Qwen3-32B** (was 8B). Same tokenizer id→string map
  (verified by the precompute gate), forced-stop support (`--include-ids 151645`),
  top-64, T=1 tail-bucket KL. Hypothesis: a stronger agentic teacher moves capability,
  not just termination — read `coverage` at startup and the KL magnitude vs anchor6.
- **Repetition steering ON, gently**: `--steer-rep-weight 0.05 --steer-rep-cap 0.5`
  (`rp=` in step lines). Trains the anchor6 frontier failure (49× verbatim command
  repeat, mitigated at serving time by penalties) into the weights. One-sided hinge:
  silent unless the verbatim repeat's mean per-token P exceeds 0.5.

Success = probe record like anchor6 (diag 0.00 / control ≥0.95), `rp=` trending down
or staying silent, and the mimic episode clean **without** the server-side penalties.
