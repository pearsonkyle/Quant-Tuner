# sft_v2_pathdiv.jsonl.gz — path-diversified SWE trajectories

Derived 2026-08-23 from `sft_v2.jsonl.gz` (3,213 resolved-only agentic coding
trajectories, 80.7M tokens; sources open-swe-traces + multi-harness-swe) by
`Quant-Tuner/scripts/path_diversify_sft.py` (repo commit f78c060).

Why: 55% of the original conversations command into /testbed (sweagent) and ~30%
into /workspace (openhands). Trained on that, Ternary-Bonsai coder1@step800 emitted
/testbed against a prompt explicitly stating the repo lived elsewhere — the checkout
path had been learned as a constant, not read from the prompt.

What changed: each conversation's checkout root is rewritten to a per-conversation
sample over ~11 realistic roots (/srv/ci/<repo>, /home/dev/<repo>, /data/repos/<repo>,
…), applied consistently across the task prompt, assistant commands, tool-call
arguments and tool outputs, so prompt→path grounding is preserved while the value
varies. 12% of conversations keep their original root. Only root-position tokens are
rewritten (/testbed matches; /opt/testbed does not). Rewrite is DETERMINISTIC
(seeded per instance_id): re-running the script on the original reproduces this file
byte-identically.

Result census: /testbed 1,763 → 217 conversations; root mentions ~uniform across
11 roots. Rewritten: 1,553 testbed-mode + 722 workspace-mode; 313 kept; 625 had no
absolute root.

Row schema: unchanged from sft_v2 (id, source, split, messages, tools, meta, n_*).
Splits and instance_ids unchanged — eval-disjointness guarantees carry over.

sha256:
  b26f3df1a8660ece70d3a097562531b70c715e7a459b788860b9b9323469c4b7  sft_v2.jsonl.gz (source)
  c65abc4173b2ab3cfad4075cd87a2ded33fba259db3e5885d5422750460fb2ea  sft_v2_pathdiv.jsonl.gz

Regenerate anywhere:
  python Quant-Tuner/scripts/path_diversify_sft.py --inp sft_v2.jsonl.gz --out sft_v2_pathdiv.jsonl.gz
