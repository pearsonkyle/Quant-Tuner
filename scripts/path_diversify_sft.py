"""Path-diversify SWE trajectory conversations for QAT.

Measured on sft_v2 (2026-08-23): 55% of conversations command into /testbed and ~30%
into /workspace — the checkout root is a near-constant, so a student learns the *value*
instead of reading the path from the prompt (observed: coder1-s800 ignored an explicit
"there is NO /testbed" instruction). This rewrites each conversation's checkout root to
a per-conversation deterministic sample (seeded on instance_id), consistently across
system/user/assistant content AND tool-call arguments AND tool outputs, so the
prompt-states-path -> commands-use-path grounding is preserved while the value varies.

Only root-position tokens are rewritten ('/testbed' matches, '/opt/testbed' does not).
Conversations with no dominant root are passed through untouched; a KEEP share retains
the original root so real-world conventions stay represented.
"""
from __future__ import annotations
import argparse, gzip, json, random, re, sys
from collections import Counter

TESTBED = re.compile(r"(?<![\w-])/testbed\b")     # root IS the checkout dir
WORKSPACE = re.compile(r"(?<![\w-])/workspace\b")  # base dir containing the repo dir
FULL_TEMPLATES = ["/workspace/{r}", "/home/dev/{r}", "/srv/ci/{r}", "/opt/work/{r}",
                  "/build/{r}", "/data/repos/{r}", "/mnt/work/{r}", "/projects/{r}",
                  "/var/ci/{r}"]
BASE_TEMPLATES = ["/srv/ci", "/home/dev", "/opt/work", "/data/repos", "/mnt/work",
                  "/projects", "/build", "/var/ci", "/code"]
KEEP_SHARE = 0.12

def _walk(x, fn):
    if isinstance(x, str): return fn(x)
    if isinstance(x, list): return [_walk(v, fn) for v in x]
    if isinstance(x, dict): return {k: _walk(v, fn) for k, v in x.items()}
    return x

def rewrite_row(row: dict, stats: Counter) -> dict:
    text = json.dumps(row["messages"])
    n_tb, n_ws = len(TESTBED.findall(text)), len(WORKSPACE.findall(text))
    if n_tb == 0 and n_ws == 0:
        stats["no_root"] += 1; return row
    rng = random.Random(f"pathdiv:{row['meta']['instance_id']}")
    if rng.random() < KEEP_SHARE:
        stats["kept_original"] += 1; return row
    repo = (row["meta"].get("repo") or "repo").split("/")[-1].replace(" ", "-")
    if n_tb >= n_ws:
        new = rng.choice(FULL_TEMPLATES).format(r=repo)
        pat, mode = TESTBED, "testbed"
        # avoid a no-op when the sampled root collides with the secondary root
        if new.startswith("/workspace") and n_ws:
            new = "/srv/ci/" + repo
    else:
        new = rng.choice(BASE_TEMPLATES)
        pat, mode = WORKSPACE, "workspace"
    fn = lambda s: pat.sub(new, s)
    row = dict(row)
    row["messages"] = _walk(row["messages"], fn)
    if row.get("tools"): row["tools"] = _walk(row["tools"], fn)
    stats[f"rewritten_{mode}"] += 1; stats[f"root:{new}"] += 1
    return row

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    stats: Counter = Counter()
    op = gzip.open if a.out.endswith(".gz") else open
    ip = gzip.open if a.inp.endswith(".gz") else open
    with ip(a.inp, "rt") as f, op(a.out, "wt") as g:
        for line in f:
            line = line.strip()
            if not line: continue
            g.write(json.dumps(rewrite_row(json.loads(line), stats), ensure_ascii=False) + "\n")
    for k, v in sorted(stats.items()): print(f"{k:>28}  {v}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
