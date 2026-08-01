# Changelog — redteam-safety-disclosures

## v0.1.0 — 2026-07-31

First release: ornith-1.0-35b red-teamed with deepteam (broad + thorough configs, 3 reps). Each row = target model id + full conversation + outcome (complied/defended/errored). PRIVATE — dual-use.

- `flagged`: 8 rows (0 verified), 0.1 MB
- `all`: 91 rows (0 verified), 0.3 MB

## v0.2.0 — 2026-07-31

v0.2.0: maximal sweep + top-up (broad/thorough/maximal/topup, 5 new attack styles, custom vulns attempted). 234 rows across 17 attack methods; empty custom-vuln simulation-errors filtered.

- `flagged`: 28 rows (0 verified), 0.3 MB
- `all`: 234 rows (0 verified), 0.9 MB

## v0.3.0 — 2026-08-01

v0.3.0: +114 rows (348 total) from a maximal re-run with the schema fix — custom vulnerabilities (BenignFramingBypass/StructuredOutputSmuggling) now simulate and are captured. New finding: Bad Likert Judge -> weapons.

- `flagged`: 42 rows (0 verified), 0.5 MB
- `all`: 348 rows (0 verified), 1.4 MB
